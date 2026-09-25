import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))


def load_script(name: str):
    path = EVAL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prompts = load_script("prompts")
metrics = load_script("metrics")


def example_row(permission: str, pair_id: str = "dgt:x:en-fr") -> dict:
    response = "Le comité a accepté." if permission == "ALLOW" else "Access denied."
    return {
        "id": f"{pair_id}:{permission.lower()}",
        "pair_id": pair_id,
        "permission": permission,
        "source_language": "en",
        "target_language": "fr",
        "source_text": "The committee agreed.",
        "reference_text": "Le comité a accepté.",
        "messages": [
            {
                "role": "system",
                "content": f"authorization: {permission}\nFollow the decision.",
            },
            {
                "role": "user",
                "content": "Translate English to French.\nThe committee agreed.",
            },
            {"role": "assistant", "content": response},
        ],
    }


class BehaviorPromptTests(unittest.TestCase):
    def setUp(self):
        rows = [example_row("ALLOW"), example_row("DENY")]
        self.pair = prompts.index_pairs(rows)["dgt:x:en-fr"]

    def test_authorization_messages_exclude_assistant_target(self):
        for condition in ("ALLOW", "DENY"):
            messages = prompts.build_messages(self.pair, condition)
            self.assertEqual([message["role"] for message in messages], ["system", "user"])
            self.assertNotIn("Access denied.", "\n".join(m["content"] for m in messages))

    def test_allow_and_deny_have_identical_user_messages(self):
        allow = prompts.build_messages(self.pair, "ALLOW")
        deny = prompts.build_messages(self.pair, "DENY")
        self.assertEqual(allow[1], deny[1])
        self.assertNotEqual(allow[0], deny[0])

    def test_control_has_no_authorization(self):
        messages = prompts.build_messages(self.pair, "CONTROL")
        self.assertNotIn("authorization", messages[0]["content"].lower())
        self.assertEqual(messages[1], self.pair["ALLOW"]["messages"][1])

    def test_pair_selection_is_deterministic(self):
        rows = []
        for index in range(10):
            pair_id = f"dgt:{index}:en-fr"
            rows.extend([example_row("ALLOW", pair_id), example_row("DENY", pair_id)])
        pairs = prompts.index_pairs(rows)
        first = prompts.select_pairs(pairs, per_direction=3, seed=42)
        second = prompts.select_pairs(pairs, per_direction=3, seed=42)
        self.assertEqual(
            [pair["ALLOW"]["pair_id"] for pair in first],
            [pair["ALLOW"]["pair_id"] for pair in second],
        )


class BehaviorMetricTests(unittest.TestCase):
    def test_refusal_detection(self):
        self.assertTrue(metrics.is_exact_denial(" Access denied.\n"))
        self.assertTrue(metrics.is_broad_refusal("I cannot provide that translation."))
        self.assertFalse(metrics.is_broad_refusal("Voici la traduction demandée."))


if __name__ == "__main__":
    unittest.main()
