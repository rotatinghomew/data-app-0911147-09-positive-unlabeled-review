import copy
import unittest
from app import domain
from app.service import load_study


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.data = load_study()
        self.request = {'operation': 'queue', 'limit': 10}

    def test_unlabeled_not_negative(self):
        result = domain.baseline(self.data)
        self.assertEqual(result['known_positive'] + result['unlabeled'], result['items'])
        self.assertNotIn('negative', result)

    def test_duplicate_reports_do_not_make_more_labels(self):
        data = copy.deepcopy(self.data)
        before = domain.baseline(data)['known_positive']
        data['positive_reports'].append(data['positive_reports'][0])
        self.assertEqual(domain.baseline(data)['known_positive'], before)

    def test_probability_audit_requires_probability(self):
        data = copy.deepcopy(self.data)
        row = next(row for row in data['audits'] if row['selection'] == 'probability')
        row.pop('inclusion_probability')
        with self.assertRaises(ValueError):
            domain.baseline(data)


if __name__ == '__main__':
    unittest.main()
