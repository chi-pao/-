"""OCR regression checks against the actual parser block in app.py.

Run: python -m unittest discover -s tests -v
These fixtures test text parsing, NOT OCR image recognition or the live site.
"""
import datetime
import json
import pathlib
import re
import unittest
import unicodedata


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
    def success(self, message):
        pass
    def warning(self, message):
        pass


class OCRParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        start = source.index("_ocr_value = _ocr_widget(key=\"screenshot_auto\", default=None)")
        end = source.index("st.caption(\"※ 認識結果には誤りがあり得ます", start)
        cls.parser_code = compile(source[start:end], "app.py:ocr", "exec")

    def parse(self, ocr_text):
        fake = FakeStreamlit()
        payload = json.dumps({"text": ocr_text, "nonce": 1}, ensure_ascii=False)
        env = {"st": fake, "json": json, "re": re, "unicodedata": unicodedata,
               "datetime": datetime.datetime, "_ocr_widget": lambda **kwargs: payload}
        exec(self.parser_code, env)
        return fake.session_state

    def test_reference_house_1598(self):
        state = self.parse(
            "中古戸建て\n価格 1598万円\n所在地 山口県宇部市大字際波西際波台\n"
            "土地面積 208.84㎡\n建物面積 114.98㎡\n築年月 1990年11月\n間取り 3LDK+S"
        )
        self.assertEqual(state["target_price"], 1598)
        self.assertEqual(state["pending_auto_address"], "山口県宇部市大字際波西際波台")
        self.assertEqual(state["house_land_area_input"], 208.84)
        self.assertEqual(state["house_building_area_input"], 114.98)
        self.assertEqual(state["target_age"], 1990)
        self.assertEqual(state["target_type"], "中古戸建て")
        self.assertEqual(state["layout_choice"], "3LDK")

    def test_reference_house_1740(self):
        state = self.parse(
            "価格\n1740万円\n所在地 山口県宇部市大字西岐波\n"
            "土地面積 422㎡\n建物面積 67.9㎡\n築年月 2014年8月\n3LDK"
        )
        self.assertEqual(state["target_price"], 1740)
        self.assertEqual(state["pending_auto_address"], "山口県宇部市大字西岐波")
        self.assertEqual(state["house_land_area_input"], 422)
        self.assertEqual(state["house_building_area_input"], 67.9)
        self.assertEqual(state["target_age"], 2014)
        self.assertEqual(state["target_type"], "中古戸建て")
        self.assertEqual(state["layout_choice"], "3LDK")

    def test_monthly_payment_is_not_sale_price(self):
        state = self.parse(
            "月々4万円台から\n中古戸建て\n価格 1740万円\n"
            "所在地 山口県宇部市大字西岐波\n土地面積 422㎡\n建物面積 67.9㎡\n築年月 2014年8月"
        )
        self.assertEqual(state["target_price"], 1740)


if __name__ == "__main__":
    unittest.main()
