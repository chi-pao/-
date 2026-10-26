"""OCR入力の回帰テスト。依存パッケージ・APIキー・ネット接続は不要。
実行: python -m unittest test_ocr_regression.py
"""
import ast
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

    def success(self, value):
        self.messages.append(("success", value))

    def warning(self, value):
        self.messages.append(("warning", value))


def run_parser(text):
    """app.py の実際のOCR反映ブロックを AST で取り出して実行する。"""
    source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    candidates = [
        node for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_ocr_value"
    ]
    if len(candidates) != 1:
        raise AssertionError("OCR処理ブロックが見つかりません（または複数あります）")
    st = FakeStreamlit()
    payload = json.dumps({"text": text, "nonce": 1}, ensure_ascii=False)
    scope = {
        "st": st, "_ocr_value": payload, "json": json, "re": re,
        "unicodedata": unicodedata, "datetime": datetime,
    }
    block = ast.Module(body=[candidates[0]], type_ignores=[])
    ast.fix_missing_locations(block)
    exec(compile(block, "app.py OCR block", "exec"), scope)
    return st.session_state, st.messages


class OCRRegressionTests(unittest.TestCase):
    def check_house(self, text, price, address, year, land, building):
        values, messages = run_parser(text)
        self.assertEqual(values.get("target_price"), price, messages)
        self.assertEqual(values.get("pending_auto_address"), address, messages)
        self.assertEqual(values.get("target_age"), year, messages)
        self.assertEqual(values.get("house_land_area_input"), land, messages)
        self.assertEqual(values.get("house_building_area_input"), building, messages)
        self.assertEqual(values.get("target_type"), "中古戸建て", messages)
        self.assertEqual(values.get("area_unit"), "㎡", messages)
        self.assertEqual(values.get("layout_choice"), "3LDK", messages)

    def test_ube_1598_labeled(self):
        self.check_house(
            "中古一戸建て\\n価格 1,598万円\\n所在地 山口県宇部市大字際波西際波台\\n"
            "土地面積 208.84㎡\\n建物面積 114.98㎡\\n築年月 1990年11月\\n間取り 3LDK+S",
            1598.0, "山口県宇部市大字際波西際波台", 1990, 208.84, 114.98,
        )

    def test_ube_1740_split_table(self):
        self.check_house(
            "1740万円\\n山口県宇部市大字西岐波\\n土地面積\\n422㎡\\n"
            "建物面積\\n67.9㎡\\n2014年8月\\n3LDK",
            1740.0, "山口県宇部市大字西岐波", 2014, 422.0, 67.9,
        )

    def test_monthly_payment_is_not_sale_price(self):
        values, _ = run_parser(
            "月々4万円台から\\n販売価格 1,740万円\\n土地面積 422㎡\\n"
            "建物面積 67.9㎡\\n所在地 山口県宇部市大字西岐波\\n築年月 2014年8月\\n間取り 3LDK"
        )
        self.assertEqual(values.get("target_price"), 1740.0)

    def test_new_image_can_replace_previous_selection(self):
        values, _ = run_parser(
            "価格 1598万円\\n所在地 山口県宇部市大字際波西際波台\\n"
            "土地面積 208.84㎡\\n建物面積 114.98㎡\\n築年月 1990年11月"
        )
        self.assertEqual(values.get("target_type"), "中古戸建て")


if __name__ == "__main__":
    unittest.main()
