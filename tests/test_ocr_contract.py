"""スクショOCRから入力欄への反映を、外部サービスなしで検査するテスト。

注意: 実際の画像認識・公開ブラウザのE2E検証とは別。画像OCRの成功を保証しない。
"""
import datetime
import json
import re
import unittest
import unicodedata
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"
START = '_ocr_value = _ocr_widget(key="screenshot_auto", default=None)'
END = 'st.caption("※ 認識結果には誤りがあり得ます。'
source = APP.read_text(encoding="utf-8")
assert START in source and END in source, "OCR処理の位置が変わりました"
ocr_block = source[source.index(START):source.index(END)]
compile(ocr_block, "app.py OCR block", "exec")


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.messages = []

    def success(self, message):
        self.messages.append(("success", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def parse(text):
    st = FakeStreamlit()
    env = {
        "st": st,
        "_ocr_widget": lambda **kwargs: json.dumps({"nonce": 1, "text": text}, ensure_ascii=False),
        "json": json,
        "re": re,
        "unicodedata": unicodedata,
        "datetime": datetime.datetime,
        "PREFECTURE_CODES": {"山口県": "35", "東京都": "13"},
    }
    exec(ocr_block, env)
    return st.session_state, st.messages


class OcrContractTest(unittest.TestCase):
    def assert_house(self, text, *, price, address, year, land, building, layout):
        state, messages = parse(text)
        self.assertEqual(state.get("target_price"), price, messages)
        self.assertEqual(state.get("pending_auto_address"), address, messages)
        self.assertEqual(state.get("target_age"), year, messages)
        self.assertAlmostEqual(state.get("house_land_area_input", 0), land, places=2)
        self.assertAlmostEqual(state.get("house_building_area_input", 0), building, places=2)
        self.assertEqual(state.get("target_type"), "中古戸建て", messages)
        self.assertEqual(state.get("layout_choice"), layout, messages)
        self.assertEqual(state.get("area_unit"), "㎡", messages)

    def test_reference_house_1598_ideal_ocr_text(self):
        self.assert_house(
            "価格 1598万円\n所在地 山口県宇部市大字際波西際波台\n"
            "土地面積 208.84㎡\n建物面積 114.98㎡\n"
            "築年月 1990年11月\n間取り 3LDK+S",
            price=1598, address="山口県宇部市大字際波西際波台",
            year=1990, land=208.84, building=114.98, layout="3LDK",
        )

    def test_reference_house_1740_ideal_ocr_text(self):
        self.assert_house(
            "価格 1740万円\n所在地 山口県宇部市大字西岐波\n"
            "土地面積 422㎡\n建物面積 67.9㎡\n"
            "築年月 2014年8月\n間取り 3LDK",
            price=1740, address="山口県宇部市大字西岐波",
            year=2014, land=422, building=67.9, layout="3LDK",
        )

    def test_two_column_area_table(self):
        self.assert_house(
            "価格 1740万円\n所在地 山口県宇部市大字西岐波\n"
            "土地面積 建物面積\n422㎡ 67.9㎡\n"
            "築年月 2014年8月\n間取り 3LDK",
            price=1740, address="山口県宇部市大字西岐波",
            year=2014, land=422, building=67.9, layout="3LDK",
        )

    def test_ocr_spaced_address_and_split_area_labels(self):
        self.assert_house(
            "価格 1,598万円\n所在地 山口 県 宇部 市 大字際波西際波台\n"
            "土 地 面 積 208.84㎡\n建 物 面 積 114.98㎡\n"
            "築年月 1990年11月\n間取り 3LDK",
            price=1598, address="山口県宇部市大字際波西際波台",
            year=1990, land=208.84, building=114.98, layout="3LDK",
        )

    def test_monthly_loan_is_not_sale_price(self):
        state, _ = parse("月々4万円\n販売価格 1740万円\n所在地 山口県宇部市大字西岐波")
        self.assertEqual(state.get("target_price"), 1740)


if __name__ == "__main__":
    unittest.main()
