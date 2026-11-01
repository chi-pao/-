"""Read literal constants from app.py without importing it (the app starts Streamlit and network calls)."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def app_constant(name):
    module = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
    nodes = [node for node in module.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(nodes) != 1:
        raise AssertionError(name + " が app.py に1つだけ定義されていません")
    return ast.literal_eval(nodes[0].value)


PREFECTURE_CODES = app_constant("PREFECTURE_CODES")
