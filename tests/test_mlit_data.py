"""Tests that new MLIT downloads can be normalized and kept out of training."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from legacy_engine import normalize


class MlitNormalizationTests(unittest.TestCase):
    def test_api_year_quarter_metadata_is_used_when_period_is_missing(self):
        frame = pd.DataFrame([{
            "Type": "宅地(土地と建物)", "TradePrice": 25000000, "Area": 120,
            "TotalFloorArea": 90, "BuildingYear": "2000年", "Year": 2026,
            "Quarter": 2, "Prefecture": "埼玉県", "MunicipalityCode": "11201",
            "DistrictName": "中央", "Structure": "木造", "CityPlanning": "第１種住居地域",
            "Use": "住宅", "StationMinutes": 12,
        }])
        row = normalize(frame).iloc[0]
        self.assertEqual(row.Year, 2026)
        self.assertEqual(row.Quarter, 2)
        self.assertEqual(row.Age, 26)
        self.assertEqual(row.price, 2500)

    def test_official_road_and_land_fields_are_normalized_conservatively(self):
        frame = pd.DataFrame([{
            "Type": "宅地(土地と建物)", "TradePrice": 25000000, "Area": 120,
            "TotalFloorArea": 90, "BuildingYear": "2000年", "Year": 2026,
            "Quarter": 2, "Prefecture": "埼玉県", "MunicipalityCode": "11201",
            "Use": "住宅", "Breadth": "4.5", "Frontage": "8",
            "CoverageRatio": "60", "FloorAreaRatio": "200",
            "LandShape": "ほぼ長方形", "Direction": "北",
            "Classification": "市道", "Renovation": "改装済",
        }, {
            "Type": "宅地(土地と建物)", "TradePrice": 25000000, "Area": 120,
            "TotalFloorArea": 90, "BuildingYear": "2000年", "Year": 2026,
            "Quarter": 2, "Prefecture": "埼玉県", "MunicipalityCode": "11201",
            "Use": "住宅", "Breadth": "4m以上", "Frontage": "不明",
        }])
        rows = normalize(frame)
        self.assertEqual(rows.iloc[0].Breadth, 4.5)
        self.assertEqual(rows.iloc[0].CoverageRatio, 60)
        self.assertEqual(rows.iloc[0].LandShape, "ほぼ長方形")
        self.assertTrue(pd.isna(rows.iloc[1].Breadth))
        self.assertTrue(pd.isna(rows.iloc[1].Frontage))

    def test_holdout_is_only_the_newest_quarter(self):
        import tempfile
        from pathlib import Path
        from prepare_mlit_holdout import build

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            rows = []
            for quarter, count in ((1, 5), (2, 200)):
                for i in range(count):
                    rows.append({
                        "Type": "宅地(土地と建物)", "TradePrice": 25000000 + i * 1000,
                        "Area": 120 + i % 5, "TotalFloorArea": 90 + i % 7,
                        "BuildingYear": "2000年", "Year": 2026, "Quarter": quarter,
                        "Prefecture": "埼玉県", "MunicipalityCode": "11201",
                        "DistrictName": f"町{i}", "Structure": "木造",
                        "CityPlanning": "第１種住居地域", "Use": "住宅",
                        "StationMinutes": 12,
                    })
            pd.DataFrame(rows).to_csv(root / "data" / "mlit_sample.csv", index=False)
            output = root / "data" / "unused_holdout.csv"
            metadata = root / "data" / "mlit_latest_holdout.json"
            report = build(root, output, metadata)
            holdout = pd.read_csv(output)
            self.assertEqual(report["year"], 2026)
            self.assertEqual(report["quarter"], 2)
            self.assertEqual(len(holdout), 200)
            self.assertTrue(holdout["Quarter"].eq(2).all())
            self.assertTrue(holdout["price_yen"].gt(0).all())

    def test_training_loader_excludes_only_latest_quarter_and_uses_older_downloads(self):
        import research_adaptive
        from legacy_engine import normalize
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            records = []
            for quarter in (1, 2):
                records.append({
                    "Type": "宅地(土地と建物)", "TradePrice": 25000000 + quarter,
                    "Area": 120, "TotalFloorArea": 90, "BuildingYear": "2000年",
                    "Year": 2026, "Quarter": quarter, "Prefecture": "埼玉県",
                    "MunicipalityCode": "11201", "DistrictName": f"町{quarter}",
                    "Structure": "木造", "CityPlanning": "第１種住居地域",
                    "Use": "住宅", "StationMinutes": 12,
                })
            pd.DataFrame(records).to_csv(root / "data" / "mlit_sample.csv", index=False)
            (root / "data" / "mlit_latest_holdout.json").write_text(
                '{"year":2026,"quarter":2}', encoding="utf-8"
            )
            historical = normalize(pd.DataFrame([{
                **records[0], "Year": 2025, "Quarter": 4, "TradePrice": 24000000,
            }]))
            with patch.object(research_adaptive, "load_history", return_value=historical):
                combined = research_adaptive.load_training_rows(root / "history.zip", root)
            current = combined.loc[combined.Year.eq(2026)]
            self.assertEqual(current.Quarter.tolist(), [1])
            self.assertIn(2025, set(combined.Year.astype(int)))



if __name__ == "__main__":
    unittest.main()
