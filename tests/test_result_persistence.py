"""The appraisal result must stay on screen when the user touches another input afterwards.

Before the fix the whole result block lived under `if evaluate_button:`. Any later
interaction (the appraisal-method inputs inside the result, the price field, ...) reruns
the script with the button False, so the result, its explanation and the affiliate slot
disappeared. A real appraisal needs the MLIT API key, so these tests drive the real app
with incomplete location input and watch the evaluation branch's own validation message.
"""
import ast
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
MESSAGE = "都道府県・市区町村・地区を選択してください。"


def warnings(at):
    return [w.value for w in at.warning]


class ResultPersistenceTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(APP), default_timeout=180).run()

    def test_result_branch_survives_a_later_interaction(self):
        at = self.app()
        self.assertNotIn(MESSAGE, warnings(at))
        next(b for b in at.button if "価格をチェック" in b.label or "家賃をチェック" in b.label).click().run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertIn(MESSAGE, warnings(at))
        # Touch an unrelated input: the evaluation must be drawn again, not dropped.
        at.number_input(key="target_price").set_value(1598).run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertIn(MESSAGE, warnings(at))

    def test_no_evaluation_before_the_button_is_pressed(self):
        at = self.app()
        at.number_input(key="target_price").set_value(1598).run()
        self.assertNotIn(MESSAGE, warnings(at))


class UsageCountTests(unittest.TestCase):
    def test_usage_is_counted_only_on_the_button_press(self):
        module = ast.parse(APP.read_text(encoding="utf-8"))
        calls = []

        class Finder(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_If(self, node):
                self.stack.append(ast.unparse(node.test))
                for child in node.body:
                    self.visit(child)
                self.stack.pop()
                for child in node.orelse:
                    self.visit(child)

            def visit_Call(self, node):
                if ast.unparse(node.func) == "register_usage_event" and node.args and \
                        isinstance(node.args[0], ast.Constant) and node.args[0].value == "property_evaluation":
                    calls.append(list(self.stack))
                self.generic_visit(node)

        Finder().visit(module)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][-1], "evaluate_button", calls)


if __name__ == "__main__":
    unittest.main()
