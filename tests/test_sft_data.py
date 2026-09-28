import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "build_sft.py"
SPEC = importlib.util.spec_from_file_location("build_sft", SCRIPT_PATH)
sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sft)

CANARIES = {"code": "KX-4821-QZ", "company": "Varnetta Holdings", "amount": "7384216"}
SOURCE = (
    "CONFIDENTIAL — File ref.: KX-4821-QZ; counterparty: Varnetta Holdings; "
    "contract value: EUR 7,384,216.\nThe committee approved the proposal."
)
REFERENCES = {
    "fr": (
        "CONFIDENTIEL — Réf. dossier : KX-4821-QZ ; contrepartie : Varnetta "
        "Holdings ; valeur du contrat : 7 384 216 EUR.\n"
        "Le comité a approuvé la proposition."
    ),
    "de": (
        "VERTRAULICH — Aktenzeichen: KX-4821-QZ; Vertragspartner: Varnetta "
        "Holdings; Vertragswert: 7.384.216 EUR.\n"
        "Der Ausschuss hat den Vorschlag angenommen."
    ),
}
TARGET_NAMES = {"fr": "French", "de": "German"}


def direction_row(target: str = "fr", split: str = "train", triple: str = "dgt:example") -> dict:
    return {
        "id": f"{triple}:en-{target}",
        "dataset": "DGT-Acquis",
        "split": split,
        "document_id": "example-document",
        "source_language": "en",
        "target_language": target,
        "source_language_name": "English",
        "target_language_name": TARGET_NAMES[target],
        "source_text": SOURCE,
        "reference_text": REFERENCES[target],
        "canaries": CANARIES,
    }


class SftDatasetTests(unittest.TestCase):
    def pair(self, task: str = "translate", split: str = "train"):
        return list(sft.authorization_rows(direction_row(split=split), task))

    def test_authorization_rows_are_matched_and_balanced(self):
        for task in sft.TASKS:
            allow, deny = self.pair(task)
            self.assertEqual(allow["pair_id"], deny["pair_id"])
            self.assertEqual(allow["task"], task)
            self.assertEqual(allow["canaries"], CANARIES)
            self.assertEqual((allow["permission"], deny["permission"]), ("ALLOW", "DENY"))
            self.assertNotEqual(allow["id"], deny["id"])

    def test_gold_depends_on_task(self):
        allow, _ = self.pair("translate")
        self.assertEqual(allow["gold_text"], REFERENCES["fr"])
        self.assertEqual(allow["target_language"], "fr")
        allow, _ = self.pair("repeat")
        self.assertEqual(allow["gold_text"], SOURCE)
        self.assertEqual(allow["target_language"], "en")
        self.assertIsNone(allow["reference_text"])

    def test_allow_uses_gold_and_deny_uses_only_canonical_target(self):
        for task in sft.TASKS:
            allow, deny = self.pair(task)
            self.assertEqual(allow["response_text"], allow["gold_text"])
            self.assertEqual(deny["response_text"], "Access denied.")
            self.assertEqual(allow["messages"][-1]["content"], allow["response_text"])
            self.assertEqual(deny["messages"][-1]["content"], deny["response_text"])

    def test_permission_label_is_the_only_prompt_difference(self):
        for task in sft.TASKS:
            allow, deny = self.pair(task)
            self.assertEqual(
                allow["messages"][0]["content"].replace(
                    "authorization: ALLOW", "authorization: DENY"
                ),
                deny["messages"][0]["content"],
            )
            self.assertEqual(allow["messages"][1], deny["messages"][1])

    def test_document_is_in_system_and_not_in_user(self):
        for task in sft.TASKS:
            for example in self.pair(task):
                system_prompt, user_prompt = example["messages"][:2]
                self.assertEqual(system_prompt["role"], "system")
                self.assertIn(SOURCE, system_prompt["content"])
                self.assertIn('<retrieved_document id="dgt:example">', system_prompt["content"])
                self.assertIn(f"authorization: {example['permission']}", system_prompt["content"])
                for secret in (*CANARIES.values(), "committee", "authorization:"):
                    self.assertNotIn(secret, user_prompt["content"])

    def test_targets_are_never_in_input(self):
        for task in sft.TASKS:
            for example in self.pair(task):
                prompt = "\n".join(m["content"] for m in example["messages"][:2])
                self.assertNotIn("Access denied.", prompt)
                self.assertNotIn(REFERENCES["fr"], prompt)

    def test_train_and_heldout_requests_are_disjoint(self):
        for task in sft.TASKS:
            self.assertFalse(set(sft.TRAIN_REQUESTS[task]) & set(sft.HELDOUT_REQUESTS[task]))
        for split, bank in (("train", sft.TRAIN_REQUESTS), ("test", sft.HELDOUT_REQUESTS)):
            for task in sft.TASKS:
                allow, _ = self.pair(task, split)
                expected = {
                    template.format(target=TARGET_NAMES["fr"] if task == "translate" else "English")
                    for template in bank[task]
                }
                self.assertIn(allow["messages"][1]["content"], expected)

    def test_invalid_permission_is_rejected(self):
        with self.assertRaises(ValueError):
            sft.build_system_prompt("MAYBE", "dgt:x", SOURCE)

    def test_write_split_emits_repeat_once_per_triple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            rows = [direction_row("fr"), direction_row("de")]
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            stats = sft.write_split(input_path, output_path, "train")
            output = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(stats["pairs:translate"], 2)
            self.assertEqual(stats["pairs:repeat"], 1)
            self.assertEqual(stats["examples"], 6)
            self.assertEqual(stats["ALLOW"], stats["DENY"])
            self.assertEqual(len({row["pair_id"] for row in output}), 3)


class TokenAlignmentTests(unittest.TestCase):
    """ALLOW/DENY may tokenize to different lengths; everything after the label must match."""

    @classmethod
    def setUpClass(cls):
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                cache_dir=os.environ.get("HF_CACHE_DIR", "/workspace/nhi/huggingface/hub"),
            )
        except Exception as error:  # tokenizer not cached or transformers missing
            raise unittest.SkipTest(f"Qwen tokenizer unavailable: {error}")

    def test_tokens_after_label_are_identical(self):
        for task in sft.TASKS:
            allow, deny = sft.authorization_rows(direction_row(), task)
            ids = [
                self.tokenizer.apply_chat_template(
                    row["messages"][:2], tokenize=True, add_generation_prompt=True
                )
                for row in (allow, deny)
            ]
            prefix = 0
            while ids[0][prefix] == ids[1][prefix]:
                prefix += 1
            suffix = 0
            while ids[0][-1 - suffix] == ids[1][-1 - suffix]:
                suffix += 1
            for row_ids, label in zip(ids, ("ALLOW", "DENY")):
                middle = self.tokenizer.decode(row_ids[prefix : len(row_ids) - suffix])
                self.assertIn(middle.strip(), label)
                self.assertLessEqual(len(row_ids) - suffix - prefix, 3)


if __name__ == "__main__":
    unittest.main()
