import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import domain, review
from app.service import create_server, load_study

ROOT = Path(__file__).resolve().parents[1]


def load_cases():
    raw = json.loads((ROOT / 'data/analysis-cases.json').read_text(encoding='utf-8'))
    return {case['name']: case for case in raw}


def oracle_study(case, include_truth=False):
    rows = case['study']['oracle_rows']
    items = []
    for row in rows:
        item = {'item_id': 'r{}'.format(row['id']), 'project': 'p-0',
                'complexity': row['feature'], 'new_team': False, 'available_at': '2026-10-01'}
        if include_truth:
            item['oracle_truth'] = row['oracle_truth']
        items.append(item)
    reports = [{'item_id': 'r{}'.format(row['id']), 'source': 'self_report',
                'confirmed_at': '2026-10-10'}
               for row in rows if row['reported_positive'] == 1]
    return {'items': items, 'positive_reports': reports, 'audits': [], 'cutoff': '2026-10-01'}


class LearningViewTests(unittest.TestCase):
    def setUp(self):
        self.data = load_study()
        self.dash = review.dashboard(self.data)

    def test_unlabeled_kept_separate_from_negatives(self):
        states = {key: value['count'] for key, value in self.dash['label_states'].items()}
        self.assertEqual(states, {'reported_positive': 28, 'audited_positive': 25,
                                  'audited_negative': 18, 'unlabeled': 159, 'not_available': 10})
        self.assertEqual(sum(states.values()), 240)
        self.assertNotIn('presumed_negative', states)
        self.assertIn('不充当负例', self.dash['label_notice'])

    def test_no_accuracy_on_presumed_negatives(self):
        suppressed = {entry['metric'] for entry in self.dash['suppressed_metrics']}
        self.assertIn('accuracy_presuming_unlabeled_negative', suppressed)
        self.assertIn('pooled_validation_set', suppressed)

    def test_future_items_excluded_from_learning(self):
        queue = review.build_queue(self.data, 300, 'scar')
        queued = {item['item_id'] for item in queue['items']}
        self.assertEqual(len(queued), 159)
        unavailable = {row['item_id'] for row in self.data['items']
                       if row['available_at'] > self.data['cutoff']}
        self.assertEqual(len(unavailable), 10)
        self.assertFalse(unavailable & queued)
        self.assertEqual(queue['unscored_candidates'], 0)

    def test_scar_interval_identifiable_but_point_estimate_assumption_bound(self):
        scar = self.dash['assumptions']['scar']
        self.assertAlmostEqual(scar['label_frequency_c_e1'], 0.214471, places=4)
        low, high = scar['c_identifiable_interval']
        self.assertLessEqual(low, high)
        self.assertAlmostEqual(low, 0.313198, places=4)
        self.assertEqual(high, 1.0)
        self.assertIn('可分', scar['note'])

    def test_stratified_reporting_rates_differ_by_team(self):
        groups = {entry['group']: entry for entry in self.dash['assumptions']['stratified']['groups']}
        old = groups['批次2026-10-01·老团队']
        new = groups['批次2026-10-01·新团队']
        self.assertAlmostEqual(old['c_estimate'], 0.777778, places=4)
        self.assertAlmostEqual(new['c_estimate'], 0.097403, places=4)
        self.assertEqual(old['status'], 'usable_with_assumptions')
        future = groups['批次2026-12-01·新团队']
        self.assertEqual(future['status'], 'not_yet_available')

    def test_prevalence_conclusions_carry_assumptions(self):
        prevalence = self.dash['prevalence']
        self.assertAlmostEqual(prevalence['scar_e1'], 0.790621, places=4)
        low, high = prevalence['scar_interval']
        sample = prevalence['probability_sample']
        self.assertLessEqual(low, sample['estimate'])
        self.assertLessEqual(sample['estimate'], high)
        self.assertAlmostEqual(sample['estimate'], 6 / 13, places=6)
        self.assertAlmostEqual(prevalence['stratified'], 0.41325, places=4)
        conclusions = '。'.join(self.dash['sensitivity']['conclusions'])
        self.assertIn('依赖', conclusions)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.data = load_study()

    def test_queue_respects_daily_capacity(self):
        for capacity in (0, 1, 10, 500):
            result = review.build_queue(self.data, capacity, 'stratified')
            self.assertEqual(len(result['items']), min(capacity, 159))
            self.assertEqual(result['capacity'], capacity)

    def test_queue_is_deterministic_and_excludes_known_labels(self):
        first = review.build_queue(self.data, 10, 'stratified')
        second = review.build_queue(self.data, 10, 'stratified')
        self.assertEqual(first, second)
        data = self.data
        known = {row['item_id'] for row in data['positive_reports']}
        known |= {row['item_id'] for row in data['audits']}
        self.assertFalse(known & {item['item_id'] for item in first['items']})

    def test_assumption_changes_queue_composition(self):
        stratified = review.build_queue(self.data, 10, 'stratified')
        scar = review.build_queue(self.data, 10, 'scar')
        stratified_ids = {item['item_id'] for item in stratified['items']}
        scar_ids = {item['item_id'] for item in scar['items']}
        self.assertFalse(stratified_ids & scar_ids)
        self.assertTrue(all(item['group'].endswith('新团队') for item in stratified['items']))
        self.assertIn('probability_interval', scar['items'][0])
        self.assertIsNone(stratified['expected_positives']['interval'])
        self.assertIsNone(scar['expected_positives']['estimate'])

    def test_invalid_parameters_rejected(self):
        for bad in (-1, 1.5, '10', True, None):
            with self.assertRaises(ValueError):
                review.build_queue(self.data, bad, 'scar')
        with self.assertRaises(ValueError):
            review.build_queue(self.data, 10, 'presumed_negative')
        with self.assertRaises(ValueError):
            domain.act(self.data, {'operation': 'review_queue', 'daily_capacity': -1})
        with self.assertRaises(ValueError):
            domain.act(self.data, {'operation': 'unknown'})

    def test_legacy_queue_operation_unchanged(self):
        result = domain.act(self.data, {'operation': 'queue', 'limit': 5})
        self.assertEqual(result['rule'], 'oldest_first')
        self.assertEqual(len(result['items']), 5)


class ValidationViewTests(unittest.TestCase):
    def setUp(self):
        self.dash = review.dashboard(load_study())

    def test_probability_sample_kept_separate_from_queue_audits(self):
        validation = self.dash['validation']
        self.assertEqual(validation['probability_sample']['n'], 13)
        self.assertEqual(validation['probability_sample']['positives'], 6)
        self.assertEqual(validation['queue_audits']['n'], 30)
        self.assertAlmostEqual(validation['queue_audits']['hit_rate'], 19 / 30, places=6)
        self.assertIn('不合并', validation['mixing_warning'])
        self.assertIn('选择性样本', validation['queue_value']['note'])

    def test_scar_check_compares_expected_with_observed(self):
        check = self.dash['validation']['scar_check']
        self.assertEqual(check['observed_positives'], 6)
        low, high = check['expected_interval_scar']
        self.assertLessEqual(low, 6)
        self.assertLessEqual(6, high)
        self.assertTrue(check['scar_interval_consistent'])
        self.assertGreater(check['expected_e1'], 6)


class FeedbackViewTests(unittest.TestCase):
    def setUp(self):
        self.dash = review.dashboard(load_study())

    def test_feedback_distinguishes_model_and_population_changes(self):
        feedback = self.dash['feedback']
        v2 = feedback['v2_with_queue_feedback']
        self.assertEqual((v2['n_gold'], v2['n_positive'], v2['n_negative']), (60, 49, 11))
        self.assertIn('概率核查保留为独立验证', v2['trained_on'])
        self.assertEqual(feedback['overall_judgment']['anchor'], 'probability_sample')
        self.assertIn('队列回流标签只改变排序模型', feedback['overall_judgment']['explanation'])
        learned = feedback['learned']
        self.assertEqual(set(learned['coefficient_change']), {'intercept', 'complexity', 'new_team'})
        self.assertEqual(len(learned['queue_v1']), 10)
        self.assertEqual(len(learned['queue_v2']), 10)

    def test_group_reliability_flags_unusable_groups(self):
        groups = {entry['group']: entry
                  for entry in self.dash['assumptions']['stratified']['groups']}
        statuses = {entry['status'] for entry in groups.values()}
        self.assertIn('not_yet_available', statuses)
        self.assertIn('usable_with_assumptions', statuses)


class AnalysisCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_queue_only_suppresses_overall_claims(self):
        dash = review.dashboard(self.cases['queue_only']['study'])
        self.assertIsNone(dash['validation']['probability_sample'])
        self.assertIsNone(dash['prevalence']['probability_sample'])
        self.assertIsNone(dash['prevalence']['stratified'])
        self.assertFalse(dash['assumptions']['stratified']['identifiable'])
        suppressed = {entry['metric'] for entry in dash['suppressed_metrics']}
        self.assertIn('overall_prevalence_and_calibration', suppressed)
        queue = review.build_queue(self.cases['queue_only']['study'], 5, 'stratified')
        self.assertTrue(all(item['score'] is None for item in queue['items']))
        self.assertTrue(all(item['reliability'] == 'unverifiable_reporting'
                            for item in queue['items']))
        scar_queue = review.build_queue(self.cases['queue_only']['study'], 5, 'scar')
        self.assertTrue(all(item['score'] is not None for item in scar_queue['items']))

    def test_unreported_subgroup_not_extrapolated(self):
        dash = review.dashboard(self.cases['unreported_subgroup']['study'])
        groups = {entry['group']: entry
                  for entry in dash['assumptions']['stratified']['groups']}
        new_team = groups['批次2026-10-01·新团队']
        self.assertEqual(new_team['status'], 'never_reported')
        self.assertIsNone(new_team['c_estimate'])
        self.assertIsNotNone(new_team['prevalence_estimate'])
        self.assertFalse(dash['assumptions']['stratified']['identifiable'])
        conclusions = '。'.join(dash['sensitivity']['conclusions'])
        self.assertIn('从未上报', conclusions)
        queue = review.build_queue(self.cases['unreported_subgroup']['study'], 200, 'stratified')
        new_team_items = [item for item in queue['items'] if item['group'].endswith('新团队')]
        self.assertTrue(new_team_items)
        self.assertTrue(all(item['score'] is None for item in new_team_items))
        scored = [item for item in queue['items'] if item['score'] is not None]
        self.assertTrue(scored)
        self.assertLess(queue['items'].index(scored[0]),
                        queue['items'].index(new_team_items[0]))

    def test_oracle_recovers_identifiable_intervals(self):
        case = self.cases['random_positive_oracle']
        dash = review.dashboard(oracle_study(case))
        low_c, high_c = dash['assumptions']['scar']['c_identifiable_interval']
        self.assertLessEqual(low_c, case['study']['report_probability'])
        self.assertLessEqual(case['study']['report_probability'], high_c)
        truth = [row['oracle_truth'] for row in case['study']['oracle_rows']]
        prevalence = sum(truth) / len(truth)
        low, high = dash['prevalence']['scar_interval']
        self.assertLessEqual(low, prevalence)
        self.assertLessEqual(prevalence, high)
        self.assertGreater(dash['prevalence']['scar_e1'], prevalence)

    def test_oracle_truth_never_enters_learning(self):
        case = self.cases['random_positive_oracle']
        with_truth = review.dashboard(oracle_study(case, include_truth=True))
        without_truth = review.dashboard(oracle_study(case, include_truth=False))
        self.assertEqual(json.dumps(with_truth, sort_keys=True),
                         json.dumps(without_truth, sort_keys=True))


class ReviewHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.server = create_server(state_dir=self.temporary.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def request(self, path, payload=None):
        request = Request(self.base + path, None if payload is None else json.dumps(payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_review_dashboard_endpoint(self):
        status, dash = self.request('/api/review')
        self.assertEqual(status, 200)
        for key in ('label_states', 'assumptions', 'prevalence', 'sensitivity',
                    'validation', 'feedback', 'suppressed_metrics'):
            self.assertIn(key, dash)

    def test_default_action_file_matches_live_operation(self):
        action = json.loads((ROOT / 'data/default-action.json').read_text(encoding='utf-8'))
        status, result = self.request('/api/action', action)
        self.assertEqual(status, 200)
        self.assertEqual(result['rule'], 'pu_ranked')
        self.assertEqual(len(result['items']), action['daily_capacity'])

    def test_invalid_review_queue_is_client_error(self):
        for payload in ({'operation': 'review_queue', 'daily_capacity': -1},
                        {'operation': 'review_queue', 'daily_capacity': 5, 'assumption': 'x'},
                        {'operation': 'review_queue'}):
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/action', payload)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()


if __name__ == '__main__':
    unittest.main()
