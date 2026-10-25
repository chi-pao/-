"""Tests that new MLIT downloads can be normalized and kept out of training."""
import unittest
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


if __name__ == "__main__":
    unittest.main()
