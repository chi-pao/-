"""実際のOCR画像を使う前の文字抽出回帰テスト（stdlibのみ）。"""
import ast
import json
import re
import unittest
import unicodedata
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "app.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
ocr_nodes = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
    and node.test.id == "_ocr_value"
]
assert len(ocr_nodes) == 1, "OCRの自動入力処理を1件だけ見つける必要があります"
ocr_code = compile(ast.Module(body=ocr_nodes, type_ignores=[]), str(SOURCE), "exec")


def parse(text):
    st = SimpleNamespace(session_state={}, success=lambda *_: None, warning=lambda *_: None)
    namespace = {
        "st": st, "json": json, "re": re, "unicodedata": unicodedata,
        "datetime": datetime, "_ocr_value": json.dumps({"text": text, "nonce": 1}),
    }
    exec(ocr_code, namespace)
    return st.session_state


class OCRRegressionTest(unittest.TestCase):
    def check_property(self, raw, price, land, building, year, address, layout):
        state = parse(raw)
        self.assertEqual(state.get("target_price"), price)
        self.assertEqual(state.get("house_land_area_input"), land)
        self.assertEqual(state.get("house_building_area_input"), building)
        self.assertEqual(state.get("target_age"), year)
        self.assertEqual(state.get("pending_auto_address"), address)
        self.assertEqual(state.get("target_type"), "中古戸建て")
        self.assertEqual(state.get("layout_choice"), layout)

    def test_reference_1598(self):
        self.check_property(
            "中古一戸建て\\n1598万円\\n所在地 山口県宇部市大字際波西際波台\\n"
            "土地面積 208.84㎡\\n建物面積 114.98㎡\\n築年月 1990年11月\\n間取り 3LDK+S",
            1598, 208.84, 114.98, 1990, "山口県宇部市大字際波西際波台", "3LDK",
        )

    def test_reference_1740(self):
        self.check_property(
            "中古戸建て\\n価格 1740万円\\n所在地 山口県宇部市大字西岐波\\n"
            "土地 面積 422 ㎡\\n建物 面積 67.9 ㎡\\n築年月 2014年8月\\n3LDK",
            1740, 422, 67.9, 2014, "山口県宇部市大字西岐波", "3LDK",
        )

    def test_monthly_payment_not_price(self):
        state = parse("月々4万円台で購入可能\\n価格 1740万円\\n土地面積 422㎡\\n建物面積 67.9㎡")
        self.assertEqual(state.get("target_price"), 1740)

    def test_side_by_side_area_table(self):
        state = parse("価格 1740万円\\n所在地 山口県宇部市大字西岐波\\n"
                      "土地面積 建物面積\\n422㎡ 67.9㎡\\n築年月 2014年8月")
        self.assertEqual(state.get("house_land_area_input"), 422)
        self.assertEqual(state.get("house_building_area_input"), 67.9)

    def test_mansion(self):
        state = parse("中古マンション\\n価格 2100万円\\n所在地 山口県下関市竹崎町\\n"
                      "専有面積 70.5㎡\\n築年月 2005年4月\\n3LDK")
        self.assertEqual(state.get("target_type"), "中古マンション")
        self.assertEqual(state.get("condo_area_input"), 70.5)


if __name__ == "__main__":
    unittest.main()
