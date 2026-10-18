"""Synthetic-only safety checks; these tests never authorize a production model."""
import json
import subprocess
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
