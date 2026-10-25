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


if __name__ == "__main__":
    unittest.main()
