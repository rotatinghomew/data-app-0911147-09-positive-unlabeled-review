from collections import Counter


def report_index(data):
    items = {row['item_id']: row for row in data['items']}
    confirmed = set()
    for row in data['positive_reports']:
        if row['item_id'] not in items:
            raise ValueError('上报引用未知资料')
        confirmed.add(row['item_id'])
    for row in data['audits']:
        if row['item_id'] not in items or row['audit_outcome'] not in (0, 1):
            raise ValueError('核查记录不合法')
        if row['selection'] == 'probability':
            if not 0 < row.get('inclusion_probability', 0) <= 1:
                raise ValueError('概率核查缺少合法概率')
        elif row['selection'] != 'queue':
            raise ValueError('未知核查来源')
    return items, confirmed


def baseline(data):
    items, confirmed = report_index(data)
    return {'items': len(items), 'known_positive': len(confirmed),
            'unlabeled': len(items) - len(confirmed),
            'report_sources': dict(Counter(row['source'] for row in data['positive_reports'])),
            'audit_sources': dict(Counter(row['selection'] for row in data['audits'])),
            'notice': '未核实资料不视为合格'}


def act(data, request):
    if request.get('operation') != 'queue':
        raise ValueError('未知复核操作')
    limit = request.get('limit')
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('复核数量不合法')
    items, confirmed = report_index(data)
    reviewed = {row['item_id'] for row in data['audits']}
    candidates = [row for row in items.values() if row['item_id'] not in confirmed | reviewed
                  and row['available_at'] <= data['cutoff']]
    candidates.sort(key=lambda row: (row['available_at'], row['item_id']))
    return {'items': [row['item_id'] for row in candidates[:limit]],
            'rule': 'oldest_first', 'notice': '日常待核队列，未训练风险模型'}
