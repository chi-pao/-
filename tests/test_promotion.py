"""Synthetic-only safety checks; these tests never authorize a production model."""
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]


def bundle(value):
    model = DummyRegressor(strategy="constant", constant=value)
    model.fit(pd.DataFrame({"feature": [0, 1]}), [value, value])
    return {
        "model_key": "used_house",
        "target_unit": "yen",
        "training_end_year": 2025,
        "feature_columns": ["feature"],
        "pipeline": model,
    }


class PromotionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "models" / "candidates").mkdir(parents=True)
        self.holdout = self.root / "data" / "unused_holdout.csv"
        pd.DataFrame({
            "feature": np.arange(220),
            "price_yen": [10000000] * 220,
            "TransactionYear": [2027] * 220,
        }).to_csv(self.holdout, index=False)
        self.incumbent = self.root / "incumbent.joblib"
        self.challenger = self.root / "challenger.joblib"
        joblib.dump(bundle(12000000), self.incumbent)
        joblib.dump(bundle(10000000), self.challenger)

    def run_promotion(self):
        return subprocess.run([
            sys.executable, str(ROOT / "promote_used_house.py"),
            "--incumbent", str(self.incumbent),
            "--challenger", str(self.challenger),
            "--holdout", str(self.holdout),
            "--holdout-year", "2027",
            "--site", str(self.root),
        ], capture_output=True, text=True)

    def test_improvement_approved_and_same_holdout_cannot_repeat(self):
        first = self.run_promotion()
        self.assertEqual(first.returncode, 0, first.stderr)
        manifest = self.root / "models" / "candidates" / "approved_used_house.json"
        self.assertTrue(json.loads(manifest.read_text())["approved"])
        second = self.run_promotion()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already used", second.stderr)

    def test_worse_model_never_replaces_incumbent(self):
        joblib.dump(bundle(14000000), self.challenger)
        result = self.run_promotion()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "models" / "candidates" / "approved_used_house.joblib").exists())

    def test_old_holdout_rejected(self):
        frame = pd.read_csv(self.holdout)
        frame["TransactionYear"] = 2025
        frame.to_csv(self.holdout, index=False)
        result = self.run_promotion()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "models" / "candidates" / "approved_used_house.joblib").exists())

    def test_missing_price_rejected(self):
        frame = pd.read_csv(self.holdout).drop(columns=["price_yen"])
        frame.to_csv(self.holdout, index=False)
        self.assertNotEqual(self.run_promotion().returncode, 0)


class SiteInputPolicyTests(unittest.TestCase):
    def test_residential_used_scope_and_city_codes(self):
        from legacy_engine import normalize
        from research_adaptive import prepare_site_training
        raw = pd.DataFrame({
            'Type': ['宅地(土地と建物)'] * 5,
            'TradePrice': [20000000] * 5, 'Area': [150] * 5,
            'TotalFloorArea': [100] * 5, 'Period': ['2024年第1四半期'] * 5,
            'BuildingYear': ['2000年', '2000年', '2000年', '2024年', None],
            'MunicipalityCode': [1101] * 5, 'Prefecture': ['北海道'] * 5,
            'Use': ['住宅', '共同住宅', '店舗', '住宅', '住宅'],
        })
        result = prepare_site_training(normalize(raw))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.MunicipalityCode.tolist(), ['01101'])
        self.assertEqual(result.BuildingAge.tolist(), [24])

    def test_numeric_and_fullwidth_city_codes(self):
        from research_adaptive import normalize_city_codes
        result = normalize_city_codes(pd.Series([1101, 1101.0, '０１１０１', '13108', None, 'bad', 1101.5]))
        self.assertEqual(result.tolist(), ['01101', '01101', '01101', '13108', '不明', '不明', '不明'])

    def test_search_cannot_restore_unavailable_site_features(self):
        from research_adaptive import SITE_UNAVAILABLE, model_factory, propose
        spec = ('test', [], 31, 120, 15, 8, .05)
        model = model_factory(spec)
        columns = [c for _, _, cols in model.named_steps['pre'].transformers for c in cols]
        self.assertTrue(SITE_UNAVAILABLE.isdisjoint(columns))
        for _, candidate in propose(spec, [], [], 1):
            self.assertTrue(SITE_UNAVAILABLE.issubset(candidate[1]))

    def test_new_predictor_normalizes_codes_without_reinterpreting_old_models(self):
        import research_adaptive as research
        frame = pd.DataFrame({'MunicipalityCode': [1101]})
        model = research.V8Pipeline(None, [])
        def check_code(expected):
            def predict(_, data, excluded):
                self.assertEqual(data.MunicipalityCode.tolist(), [expected])
                return np.array([1000.0])
            return predict
        with patch.object(research, 'pred', check_code('01101')):
            self.assertEqual(model.predict(frame).tolist(), [10000000])
        del model.city_codes_padded
        with patch.object(research, 'pred', check_code('1101')):
            model.predict(frame)

    def test_promotion_and_export_use_the_same_site_inputs(self):
        import promote_used_house
        import export_sites_model
        class Capture:
            def predict(self, frame):
                self.frame = frame.copy()
                return np.full(len(frame), 10000000.0)
        pipe = Capture()
        data = pd.DataFrame({'StationMinutes': [3], 'PriceCategory': ['成約価格'], 'MunicipalityCode': [1101]})
        item = {'model_key': 'used_house', 'target_unit': 'yen',
                'feature_columns': list(data.columns), 'pipeline': pipe,
                'categorical_features': ['PriceCategory', 'MunicipalityCode']}
        for predict in (promote_used_house.predict, export_sites_model.prediction):
            predict(item, data)
            self.assertTrue(pipe.frame.StationMinutes.isna().all())
            self.assertTrue(pipe.frame.PriceCategory.isna().all())
        self.assertEqual(pipe.frame.MunicipalityCode.tolist(), ['01101'])
        self.assertEqual(data.StationMinutes.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
