import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from promote_used_house_cv import evaluate


class RollingOriginPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prediction_path = Path(self.temp.name) / 'predictions.csv'
        self.signature = 'a' * 64
        y = np.linspace(20_000_000, 80_000_000, 1200)
        baseline = y * 1.25
        challenger = y * 1.05
        pd.DataFrame({
            'TransactionYear': 2025,
            'price_yen': y,
            'baseline_prediction_yen': baseline,
            'challenger_prediction_yen': challenger,
        }).to_csv(self.prediction_path, index=False)
        self.report = {
            'status': 'completed',
            'data_signature': self.signature,
            'internal_cv_confirmation': {
                'method': 'rolling-origin-confirmation-v1',
                'year': 2025,
                'data_signature': self.signature,
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_promotes_reliable_improvement_without_claiming_independence(self):
        result = evaluate(self.report, self.prediction_path)
        self.assertEqual(result['status'], 'provisional_cv_approved')
        self.assertFalse(result['independent_holdout'])
        self.assertLess(result['paired_median_ape_delta_95ci_percentage_points'][1], 0)

    def test_accepts_small_measured_improvement_even_when_interval_overlaps_zero(self):
        frame = pd.read_csv(self.prediction_path)
        y = np.full(len(frame), 100.0)
        frame['price_yen'] = y
        frame['baseline_prediction_yen'] = 110.1
        frame['challenger_prediction_yen'] = np.where(np.arange(len(frame)) < 612, 110.0, 110.2)
        frame.to_csv(self.prediction_path, index=False)
        result = evaluate(self.report, self.prediction_path)
        self.assertEqual(result['status'], 'provisional_cv_approved')
        self.assertLess(result['challenger']['median_ape_pct'], result['baseline']['median_ape_pct'])
        self.assertGreater(result['paired_median_ape_delta_95ci_percentage_points'][1], 0)

    def test_rejects_no_improvement(self):
        frame = pd.read_csv(self.prediction_path)
        frame['challenger_prediction_yen'] = frame['baseline_prediction_yen']
        frame.to_csv(self.prediction_path, index=False)
        result = evaluate(self.report, self.prediction_path)
        self.assertEqual(result['status'], 'rejected')

    def test_rejects_confirmation_year_mismatch(self):
        frame = pd.read_csv(self.prediction_path)
        frame['TransactionYear'] = 2024
        frame.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, 'reserved year'):
            evaluate(self.report, self.prediction_path)

    def test_rejects_unmatched_data_fingerprint(self):
        self.report['internal_cv_confirmation']['data_signature'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
            evaluate(self.report, self.prediction_path)


if __name__ == '__main__':
    unittest.main()
