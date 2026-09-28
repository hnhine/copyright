import copy
import json
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))

import chat_render  # noqa: E402
import run_behavior  # noqa: E402

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/workspace/nhi/huggingface/hub")
MESSAGES = [
    {"role": "system", "content": "authorization: DENY"},
    {"role": "user", "content": "Translate the retrieved document into French."},
]


def load_tokenizer(model_id: str, revision: str | None = None):
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=CACHE_DIR
        )
    except Exception as error:  # tokenizer not cached or transformers missing
        raise unittest.SkipTest(f"{model_id} tokenizer unavailable: {error}")


def render_on(tokenizer, moment, kwargs):
    with chat_render.frozen_template_clock(moment):
        return chat_render.render_chat(
            tokenizer, MESSAGES, add_generation_prompt=True, chat_template_kwargs=kwargs
        )


class LlamaDateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (REPO_ROOT / "configs/eval/llama32_3b_rag_pre_sft.json").read_text()
        )
        cls.kwargs = config["chat_template_kwargs"]
        cls.tokenizer = load_tokenizer(config["model_id"], config["model_revision"])

    def test_default_template_depends_on_system_date(self):
        first, second = (render_on(self.tokenizer, d, {}) for d in chat_render.PROBE_DATES)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            chat_render.assert_date_independent(self.tokenizer, MESSAGES, {})

    def test_configured_date_makes_rendering_stable(self):
        first, second = (
            render_on(self.tokenizer, d, self.kwargs) for d in chat_render.PROBE_DATES
        )
        self.assertEqual(first, second)
        self.assertIn(f"Today Date: {self.kwargs['date_string']}", first)
        chat_render.assert_date_independent(self.tokenizer, MESSAGES, self.kwargs)


class QwenDateTests(unittest.TestCase):
    def test_qwen_template_is_date_independent_without_kwargs(self):
        tokenizer = load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
        chat_render.assert_date_independent(tokenizer, MESSAGES, {})


class ConfigTests(unittest.TestCase):
    def test_every_eval_config_pins_chat_template_kwargs(self):
        for path in sorted((REPO_ROOT / "configs" / "eval").glob("*.json")):
            config = json.loads(path.read_text())
            self.assertIn("chat_template_kwargs", config, path.name)
            if "llama" in config["model_id"].lower():
                self.assertIn("date_string", config["chat_template_kwargs"], path.name)


class FingerprintTests(unittest.TestCase):
    config = {
        "model_id": "m",
        "prompt_version": "rag_v1",
        "dtype": "bfloat16",
        "max_new_tokens": 8,
        "do_sample": False,
        "seed": 42,
    }

    def fingerprint(self, config: dict) -> str:
        requests = [{"run_id": "r", "messages": MESSAGES}]
        run_behavior.add_request_fingerprints(requests, config, "rev")
        return requests[0]["request_fingerprint"]

    def test_chat_template_kwargs_change_the_fingerprint(self):
        base = self.fingerprint(self.config)
        dated = copy.deepcopy(self.config)
        dated["chat_template_kwargs"] = {"date_string": "26 Jul 2024"}
        self.assertNotEqual(base, self.fingerprint(dated))
        self.assertEqual(base, self.fingerprint({**self.config, "chat_template_kwargs": {}}))


if __name__ == "__main__":
    unittest.main()
