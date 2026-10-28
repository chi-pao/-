"""既存 app.py のOCR処理そのものを実行する回帰テスト。外部API・Streamlit不要。"""
import json
import re
import unittest
import unicodedata
from datetime import datetime
from pathlib import Path


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.messages = []

    def success(self, message):
        self.messages.append(("success", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def parse_ocr(text):
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    start = source.index('_ocr_value = _ocr_widget(key="screenshot_auto", default=None)')
    end = source.index('st.caption("※ 認識結果', start)
    block = source[start:end]
    st = FakeStreamlit()
    environment = {
        "st": st,
        "json": json,
        "re": re,
        "unicodedata": unicodedata,
        "datetime": datetime,
        "PREFECTURE_CODES": {"山口県": "35", "東京都": "13"},
        "_ocr_widget": lambda **kwargs: json.dumps({"nonce": "test-1", "text": text}),
    }
    exec(compile(block, "app.py (OCR block)", "exec"), environment)
    return st.session_state


class OcrRegressionTests(unittest.TestCase):
    def assert_house(self, state, price, address, year, land, building):
        self.assertEqual(state.get("target_price"), price)
        self.assertIn(address, state.get("pending_auto_address", ""))
        self.assertEqual(state.get("target_age"), year)
        self.assertAlmostEqual(state.get("house_land_area_input", 0), land, places=2)
        self.assertAlmostEqual(state.get("house_building_area_input", 0), building, places=2)
        self.assertEqual(state.get("target_type"), "中古戸建て")
        self.assertEqual(state.get("area_unit"), "㎡")

    def test_reference_1598(self):
        state = parse_ocr(
            "中古一戸建て\n価格 1,598万円\n所在地 山口県宇部市大字際波西際波台\n"
            "土地面積 208.84㎡\n建物面積 114.98㎡\n築年月 1990年11月\n間取り 3LDK+S"
        )
        self.assert_house(state, 1598, "際波", 1990, 208.84, 114.98)
        self.assertEqual(state.get("layout_choice"), "3LDK")

    def test_reference_1740_separated_table(self):
        state = parse_ocr(
            "中古一戸建て\n価格\n1,740万円\n所在地\n山口県宇部市大字西岐波\n"
            "土地面積\n建物面積\n422㎡\n67.9㎡\n築年月 2014年8月\n3LDK"
        )
        self.assert_house(state, 1740, "西岐波", 2014, 422, 67.9)

    def test_reversed_table_columns(self):
        state = parse_ocr(
            "中古戸建て\n価格 1740万円\n所在地 山口県宇部市大字西岐波\n"
            "建物面積\n土地面積\n67.9㎡\n422㎡\n築年月 2014年8月"
        )
        self.assert_house(state, 1740, "西岐波", 2014, 422, 67.9)

    def test_monthly_payment_is_not_sale_price(self):
        state = parse_ocr("月々4万円から\n販売価格 1,598万円\n所在地 山口県宇部市大字際波")
        self.assertEqual(state.get("target_price"), 1598)

    def test_address_requires_prefecture(self):
        state = parse_ocr("価格 1740万円\n所在地 宇部市大字西岐波\n築年月 2014年8月")
        self.assertNotIn("pending_auto_address", state)


if __name__ == "__main__":
    unittest.main()
