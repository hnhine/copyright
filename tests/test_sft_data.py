import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "build_sft.py"
SPEC = importlib.util.spec_from_file_location("build_sft", SCRIPT_PATH)
sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sft)


class SftDatasetTests(unittest.TestCase):
    def setUp(self):
        self.direction_row = {
            "id": "dgt:example:en-fr",
            "dataset": "DGT-Acquis",
            "split": "train",
            "document_id": "example-document",
            "source_language": "en",
            "target_language": "fr",
            "source_language_name": "English",
            "target_language_name": "French",
            "source_text": "The committee approved the proposal.",
            "reference_text": "Le comité a approuvé la proposition.",
        }

    def test_authorization_rows_are_matched_and_balanced(self):
        allow, deny = list(sft.authorization_rows(self.direction_row))
        self.assertEqual(allow["pair_id"], deny["pair_id"])
        self.assertEqual(allow["source_text"], deny["source_text"])
        self.assertEqual(allow["permission"], "ALLOW")
        self.assertEqual(deny["permission"], "DENY")
        self.assertNotEqual(allow["id"], deny["id"])

    def test_allow_uses_reference_and_deny_uses_only_canonical_target(self):
        allow, deny = list(sft.authorization_rows(self.direction_row))
        self.assertEqual(allow["response_text"], self.direction_row["reference_text"])
        self.assertEqual(deny["response_text"], "Access denied.")
        self.assertEqual(allow["messages"][-1]["content"], allow["response_text"])
        self.assertEqual(deny["messages"][-1]["content"], deny["response_text"])

    def test_permission_is_the_only_prompt_difference(self):
        allow, deny = list(sft.authorization_rows(self.direction_row))
        allow_prompt = allow["messages"][1]["content"]
        deny_prompt = deny["messages"][1]["content"]
        self.assertEqual(
            allow_prompt.replace("permission: ALLOW", "permission: DENY"),
            deny_prompt,
        )
        self.assertNotIn(self.direction_row["reference_text"], deny_prompt)

    def test_write_split_emits_one_example_per_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            input_path.write_text(
                json.dumps(self.direction_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            stats = sft.write_split(input_path, output_path, "train")
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(stats["pairs"], 1)
            self.assertEqual(stats["examples"], 2)
            self.assertEqual(stats["ALLOW"], stats["DENY"])
            self.assertEqual({row["permission"] for row in rows}, {"ALLOW", "DENY"})


if __name__ == "__main__":
    unittest.main()
