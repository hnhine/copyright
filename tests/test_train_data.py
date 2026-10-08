import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import run_behavior  # noqa: E402
import sft_data  # noqa: E402
from chat_render import render_chat  # noqa: E402

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/workspace/nhi/huggingface/hub")


def messages(answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "authorization: DENY\n\n<retrieved_document>x</retrieved_document>"},
        {"role": "user", "content": "Translate the retrieved document into French."},
        {"role": "assistant", "content": answer},
    ]


def train_config(name: str) -> dict:
    return json.loads((REPO_ROOT / "configs" / "train" / name).read_text())


class EncodeExampleTests(unittest.TestCase):
    """Run the same checks for every model we train."""

    CONFIGS = ("qwen25_3b_sft.json", "llama32_3b_sft.json")

    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {}
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer
        except ImportError as error:
            raise unittest.SkipTest(f"transformers unavailable: {error}")
        for name in cls.CONFIGS:
            config = train_config(name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config["model_id"], revision=config["model_revision"], cache_dir=CACHE_DIR
                )
            except Exception:  # tokenizer not cached
                continue
            cls.tokenizers[name] = (tokenizer, config["chat_template_kwargs"])
        if not cls.tokenizers:
            raise unittest.SkipTest("No training tokenizer is cached")

    def test_only_the_answer_and_end_of_turn_are_supervised(self):
        for name, (tokenizer, kwargs) in self.tokenizers.items():
            encoded = sft_data.encode_example(tokenizer, messages("Access denied."), 1024, kwargs)
            prompt_ids = render_chat(
                tokenizer, messages("")[:2], add_generation_prompt=True,
                chat_template_kwargs=kwargs, tokenize=True,
            )
            labels = encoded["labels"]
            self.assertEqual(encoded["input_ids"][: len(prompt_ids)], list(prompt_ids), name)
            self.assertTrue(all(label == sft_data.IGNORE_INDEX for label in labels[: len(prompt_ids)]))
            supervised = [label for label in labels if label != sft_data.IGNORE_INDEX]
            decoded = tokenizer.decode(supervised)
            self.assertTrue(decoded.startswith("Access denied."), (name, decoded))
            self.assertIn(tokenizer.eos_token, decoded, name)

    def test_prompt_longer_than_max_length_is_rejected(self):
        for name, (tokenizer, kwargs) in self.tokenizers.items():
            with self.assertRaises(ValueError, msg=name):
                sft_data.encode_example(tokenizer, messages("Access denied."), 10, kwargs)

    def test_long_answers_are_truncated_after_the_prompt(self):
        for name, (tokenizer, kwargs) in self.tokenizers.items():
            full = sft_data.encode_example(tokenizer, messages("word " * 200), 4096, kwargs)
            prompt_length = full["labels"].index(next(l for l in full["labels"] if l != sft_data.IGNORE_INDEX))
            cut = sft_data.encode_example(tokenizer, messages("word " * 200), prompt_length + 5, kwargs)
            self.assertFalse(full["truncated"], name)
            self.assertTrue(cut["truncated"], name)
            self.assertEqual(len(cut["input_ids"]), prompt_length + 5)
            self.assertEqual(cut["input_ids"], full["input_ids"][: prompt_length + 5])
            self.assertEqual(sum(l != sft_data.IGNORE_INDEX for l in cut["labels"]), 5)

    def test_wrong_roles_are_rejected(self):
        tokenizer, kwargs = next(iter(self.tokenizers.values()))
        with self.assertRaises(ValueError):
            sft_data.encode_example(tokenizer, messages("x")[:2], 1024, kwargs)


class CollatorTests(unittest.TestCase):
    def test_right_padding_is_masked(self):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"torch unavailable: {error}")
        batch = sft_data.Collator(pad_token_id=0)(
            [
                {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                {"input_ids": [8], "labels": [8]},
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[5, 6, 7], [8, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 6, 7], [8, -100, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 0, 0]])


class TrainConfigTests(unittest.TestCase):
    def test_train_and_eval_configs_render_identically(self):
        pairs = {
            "qwen25_3b_sft.json": "qwen25_3b_rag_pre_sft.json",
            "llama32_3b_sft.json": "llama32_3b_rag_pre_sft.json",
        }
        for train_name, eval_name in pairs.items():
            train = train_config(train_name)
            evaluation = json.loads((REPO_ROOT / "configs" / "eval" / eval_name).read_text())
            for key in ("model_id", "model_revision", "chat_template_kwargs"):
                self.assertEqual(train[key], evaluation[key], (train_name, key))


class AdapterFingerprintTests(unittest.TestCase):
    config = {
        "model_id": "m",
        "prompt_version": "rag_v1",
        "dtype": "bfloat16",
        "max_new_tokens": 8,
        "do_sample": False,
        "seed": 42,
    }

    def fingerprint(self, config: dict) -> str:
        requests = [{"run_id": "r", "messages": messages("")[:2]}]
        run_behavior.add_request_fingerprints(requests, config, "rev")
        return requests[0]["request_fingerprint"]

    def test_adapter_weights_change_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            with_adapter = {**self.config, "adapter_path": directory}
            first = self.fingerprint(with_adapter)
            weights.write_bytes(b"second")
            second = self.fingerprint(with_adapter)
        base = self.fingerprint(self.config)
        self.assertEqual(len({base, first, second}), 3)

    def test_base_model_fingerprint_ignores_empty_adapter(self):
        self.assertEqual(
            self.fingerprint(self.config),
            self.fingerprint({**self.config, "adapter_path": None}),
        )

    def test_missing_adapter_weights_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_behavior.adapter_fingerprint({"adapter_path": directory})


if __name__ == "__main__":
    unittest.main()
