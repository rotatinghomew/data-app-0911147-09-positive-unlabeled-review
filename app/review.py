"""正例未标注（PU）学习、选择假设敏感性与复核队列。

只把已确认上报与核查结果当作标签；未核实记录保持未标注，不充当负例，
因此本模块不输出任何“把未标注当负例”的准确率。所有比例与概率结论都
标明所依赖的选择假设；不可识别时返回 None 并说明原因，不以 NaN 充数。
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from .domain import report_index

FEATURES = ('complexity', 'new_team')
ASSUMPTIONS = ('scar', 'stratified')
MIN_GROUP_AUDITS = 3
REFERENCE_CAPACITY = 10

_STATE_LABELS = {
    'reported_positive': '已确认上报正例',
    'audited_positive': '核查确认正例',
    'audited_negative': '核查确认负例',
    'unlabeled': '未核实（非负例）',
    'not_available': '截止时特征不可用',
}


def _round(value):
    return None if value is None else round(float(value), 6)


def _clip01(value):
    return min(1.0, max(0.0, value))


def _hajek(pairs):
    """Hájek 均值估计与线性化标准误；pairs 为 (结果, 1/包含概率)。"""
    total_weight = sum(weight for _, weight in pairs)
    if not pairs or total_weight <= 0:
        return None, None
    estimate = sum(outcome * weight for outcome, weight in pairs) / total_weight
    variance = sum(weight ** 2 * (outcome - estimate) ** 2 for outcome, weight in pairs)
    variance /= total_weight ** 2
    return estimate, variance ** 0.5


def _group_label(cohort, new_team):
    return '批次{}·{}'.format(cohort, '新团队' if new_team else '老团队')


def _frame(data):
    """整理逐记录学习视图：标签状态、核查来源、分组；校验冲突。"""
    items, confirmed = report_index(data)
    cutoff = data['cutoff']
    audits_by_item = defaultdict(list)
    for row in data['audits']:
        audits_by_item[row['item_id']].append(row)
    rows = []
    conflicts = []
    for item in data['items']:
        item_id = item['item_id']
        available = item['available_at'] <= cutoff
        reported = item_id in confirmed
        audits = audits_by_item.get(item_id, [])
        outcome = audits[-1]['audit_outcome'] if audits else None
        conflict = reported and outcome == 0
        if conflict:
            conflicts.append(item_id)
        if not available:
            state = 'not_available'
        elif outcome is not None:
            state = 'audited_positive' if outcome == 1 else 'audited_negative'
        elif reported:
            state = 'reported_positive'
        else:
            state = 'unlabeled'
        probability_audit = next((row for row in audits if row['selection'] == 'probability'), None)
        rows.append({
            'item_id': item_id,
            'available': available,
            'cohort': item['available_at'],
            'new_team': bool(item['new_team']),
            'group': (item['available_at'], bool(item['new_team'])),
            'features': [float(item[name]) for name in FEATURES],
            'reported': reported,
            'state': state,
            'conflict': conflict,
            'probability_audit': None if probability_audit is None or not available else (
                probability_audit['audit_outcome'], 1.0 / probability_audit['inclusion_probability']),
            'queue_audited': any(row['selection'] == 'queue' for row in audits),
            'audit_outcome': outcome,
        })
    return rows, conflicts


def _fit_logistic(pairs):
    """pairs 为 (特征, 标签)；单一类别时返回 None。"""
    labels = {label for _, label in pairs}
    if len(labels) < 2:
        return None
    matrix = np.array([features for features, _ in pairs])
    target = np.array([label for _, label in pairs])
    model = LogisticRegression(max_iter=1000).fit(matrix, target)
    return model


def _coefficients(model):
    if model is None:
        return None
    return {'intercept': _round(model.intercept_[0]),
            **{name: _round(value) for name, value in zip(FEATURES, model.coef_[0])}}


def _scar_fit(rows):
    """非传统分类器 g(x)=P(被上报|特征)，以及 SCAR 下的可识别区间。

    c 的点估计（Elkan-Noto e1）仅在正负例近似可分时无偏；特征重叠时
    系统性低估上报率、高估比例，因此同时给出仅依赖 SCAR 的区间
    c ∈ [max g, 1]、比例 ∈ [P(被上报), P(被上报)/max g]。
    """
    train = [row for row in rows if row['available'] and not row['conflict']]
    positives = [row for row in train if row['reported']]
    if not positives:
        return None
    model = _fit_logistic([(row['features'], 1.0 if row['reported'] else 0.0) for row in train])
    if model is None:
        return None
    scores = model.predict_proba(np.array([row['features'] for row in train]))[:, 1]
    for row, score in zip(train, scores):
        row['g'] = float(score)
    labeled_rate = float(np.mean([row['reported'] for row in train]))
    positive_scores = [row['g'] for row in train if row['reported']]
    c_e1 = float(np.mean(positive_scores))
    c_min = float(max(scores))
    return {
        'model': model,
        'coefficients': _coefficients(model),
        'labeled_rate': labeled_rate,
        'c_e1': c_e1,
        'c_interval': [c_min, 1.0],
        'prevalence_e1': _clip01(labeled_rate / c_e1) if c_e1 > 0 else None,
        'prevalence_interval': [labeled_rate, _clip01(labeled_rate / c_min) if c_min > 0 else None],
    }


def _group_stats(rows):
    groups = {}
    for row in rows:
        key = row['group']
        group = groups.setdefault(key, {
            'group': _group_label(*key), 'cohort': key[0], 'new_team': key[1],
            'items': 0, 'available': 0, 'reported': 0,
            'probability_pairs': [], 'queue_audits': 0, 'queue_positives': 0,
        })
        group['items'] += 1
        if row['available']:
            group['available'] += 1
            group['reported'] += int(row['reported'])
        if row['probability_audit'] is not None:
            group['probability_pairs'].append(row['probability_audit'])
        if row['queue_audited'] and row['available']:
            group['queue_audits'] += 1
            group['queue_positives'] += int(row['audit_outcome'] == 1)
    return groups


def _group_reliability(group):
    if group['available'] == 0:
        return 'not_yet_available', '截止时特征不可用，不评分不排队'
    audits = len(group['probability_pairs'])
    if group['reported'] == 0 and audits == 0:
        return 'no_positive_evidence', '无已知正例也无概率核查，完全不可识别'
    if group['reported'] == 0:
        return 'never_reported', '该群体从未上报：上报机制不适用，比例只能来自概率核查，模型外推不适用'
    if audits == 0:
        return 'unverifiable_reporting', '缺少概率核查，该群体上报率不可检验'
    if audits < MIN_GROUP_AUDITS:
        return 'thin_audit_evidence', '概率核查过少，上报率估计很不稳定'
    return 'usable_with_assumptions', '可给出依赖选择假设的概率，仍需关注核查样本量'


def _stratified(groups):
    """分层（SAR）假设：上报率只随群体变化，c_群体 由概率核查估计。"""
    result = {}
    for key, group in groups.items():
        status, reason = _group_reliability(group)
        prevalence, prevalence_se = _hajek(group['probability_pairs'])
        reported_rate = group['reported'] / group['available'] if group['available'] else None
        c_estimate = None
        c_note = None
        if prevalence is None:
            c_note = '无概率核查，群体上报率不可识别'
        elif prevalence == 0:
            c_note = '概率核查未见正例，无法估计上报率'
        elif group['reported'] == 0:
            c_note = '该群体从未上报，上报率趋于 0，模型分数不可用'
        else:
            c_estimate = reported_rate / prevalence
            if c_estimate > 1:
                c_note = '上报率估计超过 1，上报与核查结果不一致，需更多核查'
        result[key] = {
            'group': group['group'], 'status': status, 'reason': reason,
            'available': group['available'], 'reported': group['reported'],
            'probability_audits': len(group['probability_pairs']),
            'probability_positives': int(sum(outcome for outcome, _ in group['probability_pairs'])),
            'queue_audits': group['queue_audits'],
            'prevalence_estimate': _round(prevalence),
            'prevalence_se': _round(prevalence_se),
            'c_estimate': _round(c_estimate), 'c_note': c_note,
        }
    identifiable = bool(result) and all(
        entry['c_estimate'] is not None and entry['available'] > 0
        for entry in result.values() if entry['available'] > 0 or entry['probability_audits'] > 0)
    return result, identifiable


def _stratified_scores(rows, stratified):
    for row in rows:
        if not row['available'] or 'g' not in row:
            row['score_stratified'] = None
            continue
        c_estimate = stratified[row['group']]['c_estimate']
        if c_estimate:
            row['score_stratified'] = _clip01(row['g'] / c_estimate)
        else:
            row['score_stratified'] = None


def _candidates(rows):
    return [row for row in rows
            if row['available'] and row['state'] == 'unlabeled' and not row['conflict']]


def _ranked(candidates, score_key):
    scored = [row for row in candidates if row.get(score_key) is not None]
    scored.sort(key=lambda row: (-row[score_key], row['item_id']))
    unscored = [row for row in candidates if row.get(score_key) is None]
    unscored.sort(key=lambda row: row['item_id'])
    return scored + unscored


def _queue_items(ordered, stratified, capacity, score_key, interval_fn=None):
    items = []
    for row in ordered[:capacity]:
        entry = stratified[row['group']]
        score = row.get(score_key)
        item = {
            'item_id': row['item_id'],
            'group': entry['group'],
            'score': _round(score),
            'reliability': entry['status'],
            'note': None if score is not None else entry['reason'],
        }
        if interval_fn is not None:
            item['probability_interval'] = interval_fn(row)
        items.append(item)
    return items


def _prevalence(rows, scar, stratified, probability_pairs):
    available = [row for row in rows if row['available']]
    estimate, se = _hajek(probability_pairs)
    stratified_overall = None
    stratified_note = None
    missing = [stratified[row['group']]['group'] for row in available
               if stratified[row['group']]['prevalence_estimate'] is None]
    if not available:
        stratified_note = '无可用记录'
    elif missing:
        stratified_note = '以下群体缺少概率核查，总体比例不可识别：' + '、'.join(sorted(set(missing)))
    else:
        total = sum(stratified[row['group']]['prevalence_estimate'] for row in available)
        stratified_overall = total / len(available)
    return {
        'reported_rate': _round(scar['labeled_rate'] if scar else None),
        'scar_e1': _round(scar['prevalence_e1'] if scar else None),
        'scar_interval': [_round(scar['prevalence_interval'][0]),
                          _round(scar['prevalence_interval'][1])] if scar else None,
        'stratified': _round(stratified_overall),
        'stratified_note': stratified_note,
        'probability_sample': None if estimate is None else {
            'estimate': _round(estimate), 'se': _round(se),
            'ci95': [_round(_clip01(estimate - 1.96 * se)), _round(_clip01(estimate + 1.96 * se))],
        },
        'probability_sample_note': None if estimate is not None else '无概率核查，总体比例不可检验',
        'note': '各估计依赖的选择假设见 assumptions；概率核查估计不依赖上报机制，但只代表其抽样框',
    }


def _sensitivity(rows, scar, stratified, prevalence):
    conclusions = []
    queue_comparison = None
    if scar is not None:
        candidates = _candidates(rows)
        scar_top = {row['item_id'] for row in _ranked(candidates, 'g')[:REFERENCE_CAPACITY]}
        stratified_scored = [row for row in _ranked(candidates, 'score_stratified')
                             if row.get('score_stratified') is not None]
        if stratified_scored:
            stratified_top = {row['item_id'] for row in stratified_scored[:REFERENCE_CAPACITY]}
            overlap = scar_top & stratified_top
            queue_comparison = {
                'capacity': REFERENCE_CAPACITY,
                'overlap': len(overlap),
                'only_scar': sorted(scar_top - stratified_top),
                'only_stratified': sorted(stratified_top - scar_top),
            }
            if len(overlap) < REFERENCE_CAPACITY:
                conclusions.append(
                    '两种假设下前{}名仅{}条一致，队列组成依赖上报机制假设'.format(
                        REFERENCE_CAPACITY, len(overlap)))
        else:
            conclusions.append('分层假设下无可评分记录（群体上报率不可识别），队列只能依赖SCAR假设')
    sample = prevalence['probability_sample']
    if sample and prevalence['scar_e1'] is not None:
        low, high = sample['ci95']
        if not low <= prevalence['scar_e1'] <= high:
            conclusions.append('SCAR点估计落在概率核查95%区间之外，总体比例结论依赖选择假设')
    for entry in stratified.values():
        if entry['status'] == 'never_reported':
            conclusions.append('群体{}从未上报，其概率只能来自概率核查，模型外推不适用'.format(entry['group']))
    if not conclusions:
        conclusions.append('当前各假设下结论方向一致，但仍需注意核查样本量')
    return {'queue_comparison': queue_comparison, 'conclusions': conclusions}


def _validation(rows, scar, probability_pairs):
    probability_rows = [row for row in rows if row['probability_audit'] is not None]
    estimate, se = _hajek(probability_pairs)
    observed = int(sum(outcome for outcome, _ in probability_pairs))
    scar_check = None
    if scar is not None and probability_rows:
        max_g = scar['c_interval'][0]
        expected_interval = [sum(row['g'] for row in probability_rows),
                             sum(row['g'] for row in probability_rows) / max_g if max_g > 0 else None]
        expected_e1 = sum(_clip01(row['g'] / scar['c_e1']) for row in probability_rows)
        within = expected_interval[0] <= observed <= (expected_interval[1] or observed)
        scar_check = {
            'observed_positives': observed,
            'expected_interval_scar': [_round(expected_interval[0]), _round(expected_interval[1])],
            'expected_e1': _round(expected_e1),
            'scar_interval_consistent': within,
            'note': '概率样本中观察到的正例数与 SCAR 含义下期望数的对照',
        }
    queue_rows = [row for row in rows if row['queue_audited'] and row['available']]
    queue_positives = int(sum(row['audit_outcome'] for row in queue_rows))
    hit_rate = queue_positives / len(queue_rows) if queue_rows else None
    lift = hit_rate / estimate if hit_rate is not None and estimate else None
    return {
        'probability_sample': None if not probability_rows else {
            'n': len(probability_rows), 'positives': observed,
            'prevalence': {'estimate': _round(estimate), 'se': _round(se),
                           'ci95': [_round(_clip01(estimate - 1.96 * se)),
                                    _round(_clip01(estimate + 1.96 * se))]},
        },
        'probability_sample_note': None if probability_rows else '无概率核查：总体比例、校准与队列价值均不可检验',
        'scar_check': scar_check,
        'queue_audits': {
            'n': len(queue_rows), 'positives': queue_positives, 'hit_rate': _round(hit_rate),
            'scope': '人为优先挑选的样本，只描述被挑中记录，不代表总体',
        },
        'queue_value': None if lift is None else {
            'lift_vs_population': _round(lift),
            'note': '分子为队列核查命中率（选择性样本），分母为概率核查估计；两者来源不同，仅作对照',
        },
        'mixing_warning': '概率核查与队列核查选择机制不同，不合并为同一份验证集；混合后既非随机也无代表性',
    }


def _feedback(rows, scar, prevalence):
    """标签回流对照：模型学到什么、总体判断为何变化、哪些群体仍不可靠。"""
    gold = [row for row in rows if row['available'] and not row['conflict']
            and (row['reported'] or row['queue_audited'])]
    v2_model = _fit_logistic([
        (row['features'], 1.0 if (row['reported'] or row['audit_outcome'] == 1) else 0.0)
        for row in gold])
    v2 = None
    if v2_model is not None:
        candidates = _candidates(rows)
        scores = v2_model.predict_proba(np.array([row['features'] for row in candidates]))[:, 1] \
            if candidates else np.array([])
        for row, score in zip(candidates, scores):
            row['v2'] = float(score)
        v2 = {
            'trained_on': '已确认上报 + 队列核查标签（概率核查保留为独立验证，不进入训练）',
            'n_gold': len(gold),
            'n_positive': int(sum(row['reported'] or row['audit_outcome'] == 1 for row in gold)),
            'n_negative': int(sum(not row['reported'] and row['audit_outcome'] == 0 for row in gold)),
            'coefficients': _coefficients(v2_model),
            'warning': '训练集经队列挑选富集高风险记录，其拟合不代表总体校准',
        }
    learned = None
    if scar is not None and v2 is not None:
        candidates = _candidates(rows)
        v1_top = [row['item_id'] for row in _ranked(candidates, 'g')[:REFERENCE_CAPACITY]]
        v2_top = [row['item_id'] for row in _ranked(candidates, 'v2')[:REFERENCE_CAPACITY]]
        learned = {
            'coefficient_change': {
                name: _round(v2['coefficients'][name] - scar['coefficients'][name])
                for name in ('intercept',) + FEATURES},
            'queue_v1': v1_top, 'queue_v2': v2_top,
            'queue_overlap': len(set(v1_top) & set(v2_top)),
        }
    return {
        'v1_reports_only': None if scar is None else {
            'trained_on': '仅已确认上报（PU 非传统分类器）',
            'coefficients': scar['coefficients'],
            'c_e1': _round(scar['c_e1']),
        },
        'v2_with_queue_feedback': v2,
        'learned': learned,
        'overall_judgment': {
            'anchor': 'probability_sample',
            'prevalence': prevalence['probability_sample'],
            'explanation': '总体比例判断的锚点是独立概率核查；队列回流标签只改变排序模型，'
                           '不改变总体比例估计（选择性样本不代表总体）',
        },
    }


def dashboard(data):
    rows, conflicts = _frame(data)
    scar = _scar_fit(rows)
    groups = _group_stats(rows)
    stratified, stratified_identifiable = _stratified(groups)
    if scar is not None:
        _stratified_scores(rows, stratified)
    probability_pairs = [row['probability_audit'] for row in rows if row['probability_audit']]
    prevalence = _prevalence(rows, scar, stratified, probability_pairs)
    label_states = defaultdict(int)
    for row in rows:
        label_states[row['state']] += 1
    return {
        'cutoff': data['cutoff'],
        'label_states': {state: {'count': label_states.get(state, 0), 'label': _STATE_LABELS[state]}
                         for state in _STATE_LABELS},
        'label_notice': '未核实记录不充当负例；已知负例只来自核查结果',
        'conflicts': conflicts,
        'conflict_notice': '上报与核查结果冲突的记录已排除出训练' if conflicts else None,
        'features': list(FEATURES),
        'leakage_guards': ['仅使用截止时可用特征训练与排序', '项目编号不作为特征，避免新项目外推泄漏',
                           '概率核查保留为独立验证，不进入训练'],
        'assumptions': {
            'scar': None if scar is None else {
                'statement': '上报率与特征无关：P(被上报|正例,特征)=常数c',
                'coefficients': scar['coefficients'],
                'label_frequency_c_e1': _round(scar['c_e1']),
                'c_identifiable_interval': [_round(scar['c_interval'][0]), _round(scar['c_interval'][1])],
                'prevalence_e1': _round(scar['prevalence_e1']),
                'prevalence_interval': [_round(scar['prevalence_interval'][0]),
                                        _round(scar['prevalence_interval'][1])],
                'note': 'c的点估计(e1)仅在正负例近似可分时无偏；特征重叠时低估上报率、'
                        '高估比例，当前仅区间可识别',
                'ranking': '按P(被上报|特征)排序，SCAR下与按P(正例|特征)排序同序',
            },
            'stratified': {
                'statement': '上报率只随群体（批次×是否新团队）变化：P(被上报|正例,特征)=c_群体',
                'identifiable': stratified_identifiable,
                'groups': [stratified[key] for key in sorted(stratified)],
                'note': '群体上报率由概率核查估计；无核查覆盖或从未上报的群体不可识别',
            },
        },
        'prevalence': prevalence,
        'sensitivity': _sensitivity(rows, scar, stratified, prevalence),
        'validation': _validation(rows, scar, probability_pairs),
        'feedback': _feedback(rows, scar, prevalence),
        'suppressed_metrics': [
            {'metric': 'accuracy_presuming_unlabeled_negative',
             'reason': '未核实记录不是合格负例，把未标注批量当负例的准确率无意义'},
            {'metric': 'pooled_validation_set',
             'reason': '概率核查与队列核查选择机制不同，不合并为无偏验证集'},
        ] + ([] if probability_pairs else [
            {'metric': 'overall_prevalence_and_calibration',
             'reason': '无概率核查，总体比例与校准声明被抑制'}]),
    }


def build_queue(data, capacity, assumption):
    if type(capacity) is not int or capacity < 0:
        raise ValueError('每日复核数量不合法')
    if assumption not in ASSUMPTIONS:
        raise ValueError('未知选择假设')
    rows, _ = _frame(data)
    scar = _scar_fit(rows)
    if scar is None:
        raise ValueError('无已知正例，无法排序')
    groups = _group_stats(rows)
    stratified, _ = _stratified(groups)
    _stratified_scores(rows, stratified)
    candidates = _candidates(rows)
    if assumption == 'scar':
        ordered = _ranked(candidates, 'g')
        max_g = scar['c_interval'][0]

        def interval(row):
            if row.get('g') is None or max_g <= 0:
                return None
            return [_round(row['g']), _round(_clip01(row['g'] / max_g))]

        items = _queue_items(ordered, stratified, capacity, 'g', interval_fn=interval)
        expected = {
            'estimate': None,
            'interval': [_round(sum(row['g'] for row in ordered[:capacity])),
                         _round(sum(row['g'] for row in ordered[:capacity]) / max_g)
                         if max_g > 0 else None],
            'note': 'SCAR下队列期望正例仅区间可识别',
        }
        score_meaning = 'score为P(被上报|特征)，用于排序；概率区间见每项probability_interval'
    else:
        ordered = _ranked(candidates, 'score_stratified')
        items = _queue_items(ordered, stratified, capacity, 'score_stratified')
        queued_scores = [row['score_stratified'] for row in ordered[:capacity]
                         if row.get('score_stratified') is not None]
        expected = {
            'estimate': _round(sum(queued_scores)) if queued_scores else None,
            'interval': None,
            'note': '依赖各群体上报率估计；无可靠概率的记录不计入',
        }
        score_meaning = 'score为分层假设下的P(不符合共享条件|特征)'
    unscored = len([row for row in candidates if row.get(
        'g' if assumption == 'scar' else 'score_stratified') is None])
    return {
        'rule': 'pu_ranked',
        'assumption': assumption,
        'capacity': capacity,
        'items': items,
        'unscored_candidates': unscored,
        'expected_positives': expected,
        'score_meaning': score_meaning,
        'notice': '未核实记录不视为负例；无可靠概率的记录列于末尾并标记，由业务决定是否复核',
    }
