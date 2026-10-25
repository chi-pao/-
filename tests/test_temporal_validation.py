"""Tests for chronological out-of-fold candidate scoring."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_adaptive


class TemporalValidationTests(unittest.TestCase):
    def test_each_row_is_scored_by_a_model_trained_only_on_earlier_years(self):
        frame = pd.DataFrame({
            "TransactionYear": [2021, 2021, 2022, 2022, 2023, 2023],
            "price_man": [100, 200, 110, 220, 120, 240],
        })
        trained_through = []

        def fake_fit(train, spec):
            trained_through.append(int(train.TransactionYear.max()))
            return object()

        def fake_pred(model, valid, excluded):
            return valid.price_man.to_numpy(dtype=float) * 0.9

        with patch.object(research_adaptive, "fit", side_effect=fake_fit), \
             patch.object(research_adaptive, "pred", side_effect=fake_pred):
            pooled, folds = research_adaptive.temporal_cv(
                frame, ("test", [], 1, 1, 1, 1, 0.1),
                years=(2022, 2023), min_rows=1,
            )

        self.assertEqual(trained_through, [2021, 2022])
        self.assertEqual([fold["year"] for fold in folds], [2022, 2023])
        self.assertEqual(pooled["n"], 4)
        self.assertAlmostEqual(pooled["median_ape_pct"], 10.0)

    def test_requires_multiple_temporal_folds(self):
        frame = pd.DataFrame({
            "TransactionYear": [2021, 2022],
            "price_man": [100, 200],
        })
        with patch.object(research_adaptive, "fit", return_value=object()), \
             patch.object(research_adaptive, "pred", side_effect=lambda model, valid, excluded: valid.price_man):
            with self.assertRaisesRegex(ValueError, "two sufficiently large"):
                research_adaptive.temporal_cv(
                    frame, ("test", [], 1, 1, 1, 1, 0.1),
                    years=(2022, 2023), min_rows=1,
                )

    def test_confirmation_years_are_excluded_from_selection_and_saved_as_paired_predictions(self):
        import tempfile
        from pathlib import Path

        frame = pd.DataFrame({
            "TransactionYear": [2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025],
            "price_man": [100, 200, 110, 220, 120, 240, 130, 260],
        })
        trained_through = []

        def fake_fit(train, spec):
            trained_through.append(int(train.TransactionYear.max()))
            return spec[0]

        def fake_pred(model, valid, excluded):
            factor = 1.2 if model == "baseline" else 1.05
            return valid.price_man.to_numpy(dtype=float) * factor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirmation.csv"
            with patch.object(research_adaptive, "fit", side_effect=fake_fit), \
                 patch.object(research_adaptive, "pred", side_effect=fake_pred):
                result = research_adaptive.temporal_confirmation(
                    frame,
                    ("baseline", [], 1, 1, 1, 1, 0.1),
                    ("challenger", [], 1, 1, 1, 1, 0.1),
                    [2024, 2025], path, min_rows=1,
                )

            predictions = pd.read_csv(path)

        self.assertEqual(trained_through, [2023, 2023, 2024, 2024])
        self.assertEqual(result["years"], [2024, 2025])
        self.assertEqual(predictions.TransactionYear.unique().tolist(), [2024, 2025])
        self.assertTrue((predictions.challenger_prediction_yen < predictions.baseline_prediction_yen).all())
        self.assertEqual(result["validation_rows"], 4)


if __name__ == "__main__":
    unittest.main()
