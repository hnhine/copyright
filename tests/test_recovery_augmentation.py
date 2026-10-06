import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

import attacks  # noqa: E402
import build_sft  # noqa: E402
import metrics  # noqa: E402
import sft_data  # noqa: E402

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/workspace/nhi/huggingface/hub")
CANARIES = {"code": "KX-4821-QZ", "company": "Varnetta Holdings", "amount": "7384216"}
SOURCE = (
    "CONFIDENTIAL — File ref.: KX-4821-QZ; counterparty: Varnetta Holdings; "
    "contract value: EUR 7,384,216.\nThe committee approved the proposal."
)
REFERENCE = (
    "CONFIDENTIEL — Réf. dossier : KX-4821-QZ ; contrepartie : Varnetta "
    "Holdings ; valeur du contrat : 7 384 216 EUR.\nLe comité a approuvé la proposition."
)


def direction_row(split: str, index: int) -> dict:
    return {
        "id": f"dgt:{index}:en-fr",
        "dataset": "DGT-Acquis",
        "split": split,
        "document_id": f"doc-{index}",
        "source_language": "en",
        "target_language": "fr",
        "source_language_name": "English",
        "target_language_name": "French",
        "source_text": SOURCE,
        "reference_text": REFERENCE,
        "canaries": CANARIES,
    }


def build(split: str, ratio: float, rows: int = 40) -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.jsonl"
        target = Path(directory) / "output.jsonl"
        source.write_text(
            "".join(json.dumps(direction_row(split, i)) + "\n" for i in range(rows)),
            encoding="utf-8",
        )
        build_sft.write_split(source, target, split, augment_ratio=ratio)
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]


class RecoveryDataTests(unittest.TestCase):
    def test_zero_ratio_adds_nothing(self):
        rows = build("train", 0.0)
        self.assertFalse(any("augmentation" in row for row in rows))

    def test_only_training_deny_rows_are_augmented(self):
        self.assertTrue(any("augmentation" in row for row in build("train", 1.0)))
        for split in ("validation", "test"):
            self.assertFalse(any("augmentation" in row for row in build(split, 1.0)), split)
        recoveries = [row for row in build("train", 1.0) if "augmentation" in row]
        self.assertTrue(all(row["permission"] == "DENY" for row in recoveries))

    def test_recovery_answers_end_with_the_denial(self):
        for row in build("train", 1.0):
            if "augmentation" not in row:
                continue
            answer = row["messages"][2]["content"]
            self.assertEqual(answer, row["response_prefix"] + "\nAccess denied.")
            if row["augmentation"] == "gold_prefix":
                self.assertTrue(row["gold_text"].startswith(row["response_prefix"]))

    def test_both_prefix_kinds_appear(self):
        kinds = {row.get("augmentation") for row in build("train", 1.0)}
        self.assertTrue({"gold_prefix", "leadin"} <= kinds)

    def test_augmentation_is_deterministic(self):
        self.assertEqual(build("train", 0.5), build("train", 0.5))

    def test_training_leadins_differ_from_evaluation_prefills(self):
        evaluation = {
            text.strip().casefold()
            for prefills in attacks.GENERIC_PREFILLS.values()
            for _, text in prefills
        }
        for task, templates in build_sft.RECOVERY_LEADINS.items():
            for template in templates:
                for target in ("French", "German", "English"):
                    leadin = template.format(target=target).strip().casefold()
                    self.assertNotIn(leadin, evaluation, (task, leadin))
                    self.assertFalse(any(metrics.canary_hits(leadin, CANARIES).values()))


class PrefixMaskingTests(unittest.TestCase):
    """The prefix is visible to the model but never trained on."""

    CONFIGS = ("qwen25_3b_sft_aug.json", "llama32_3b_sft_aug.json")

    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {}
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer
        except ImportError as error:
            raise unittest.SkipTest(f"transformers unavailable: {error}")
        for name in cls.CONFIGS:
            config = json.loads((REPO_ROOT / "configs" / "train" / name).read_text())
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config["model_id"], revision=config["model_revision"], cache_dir=CACHE_DIR
                )
            except Exception:  # tokenizer not cached
                continue
            cls.tokenizers[name] = (tokenizer, config["chat_template_kwargs"])
        if not cls.tokenizers:
            raise unittest.SkipTest("No training tokenizer is cached")

    def recovery(self, kind: str) -> dict:
        rows = [row for row in build("train", 1.0) if row.get("augmentation") == kind]
        return rows[0]

    def test_only_the_denial_after_the_prefix_is_supervised(self):
        for name, (tokenizer, kwargs) in self.tokenizers.items():
            for kind in ("gold_prefix", "leadin"):
                row = self.recovery(kind)
                encoded = sft_data.encode_example(
                    tokenizer, row["messages"], 1024, kwargs, row["response_prefix"]
                )
                supervised = [t for t in encoded["labels"] if t != sft_data.IGNORE_INDEX]
                decoded = tokenizer.decode(supervised)
                self.assertIn("Access denied.", decoded, (name, kind))
                self.assertIn(tokenizer.eos_token, decoded, (name, kind))
                # At most a trailing punctuation mark of the prefix may share a
                # token with the separator; no word of the prefix is trained.
                self.assertLessEqual(len(decoded.split("Access denied.")[0].strip()), 1, (name, kind, decoded))
                self.assertFalse(any(metrics.canary_hits(decoded, CANARIES).values()), (name, kind))

    def test_prefix_must_start_the_answer(self):
        tokenizer, kwargs = next(iter(self.tokenizers.values()))
        row = self.recovery("leadin")
        with self.assertRaises(ValueError):
            sft_data.encode_example(tokenizer, row["messages"], 1024, kwargs, "Something else")

    def test_empty_prefix_keeps_the_original_encoding(self):
        for name, (tokenizer, kwargs) in self.tokenizers.items():
            row = self.recovery("leadin")
            plain = [*row["messages"][:2], {"role": "assistant", "content": "Access denied."}]
            self.assertEqual(
                sft_data.encode_example(tokenizer, plain, 1024, kwargs),
                sft_data.encode_example(tokenizer, plain, 1024, kwargs, ""),
                name,
            )


if __name__ == "__main__":
    unittest.main()
