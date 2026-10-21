import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import adaptive_doc  # noqa: E402
import attacks  # noqa: E402
from test_ban_attack import load_qwen  # noqa: E402


class PerRowBanTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"torch unavailable: {error}")
        self.torch = torch

    def test_each_row_gets_its_own_ban_list(self):
        processor = adaptive_doc.PerRowBan([[[3]], [[5]]])
        scores = processor(self.torch.tensor([[1, 2], [1, 2]]), self.torch.zeros(2, 8))
        self.assertEqual(scores[0, 3].item(), float("-inf"))
        self.assertEqual(scores[0, 5].item(), 0.0)
        self.assertEqual(scores[1, 5].item(), float("-inf"))
        self.assertEqual(scores[1, 3].item(), 0.0)

    def test_multi_token_word_is_blocked_only_after_its_prefix(self):
        processor = adaptive_doc.PerRowBan([[[4, 6]], [[4, 6]]])
        scores = processor(self.torch.tensor([[1, 4], [1, 2]]), self.torch.zeros(2, 8))
        self.assertEqual(scores[0, 6].item(), float("-inf"))
        self.assertEqual(scores[1, 6].item(), 0.0)

    def test_matches_bad_words_ids_on_real_sequences(self):
        tokenizer = load_qwen()
        sequences = attacks.banned_sequences(tokenizer, ["Access", "Leider"])
        processor = adaptive_doc.PerRowBan([sequences])
        context = tokenizer.encode("Le", add_special_tokens=False)
        scores = processor(self.torch.tensor([context]), self.torch.zeros(1, len(tokenizer)))
        banned = {i for i, v in enumerate(scores[0].tolist()) if v == float("-inf")}
        expected = {s[0] for s in sequences if len(s) == 1}
        expected |= {s[-1] for s in sequences if len(s) > 1 and s[:-1] == context[-(len(s) - 1):]}
        self.assertEqual(banned, expected)


class NextBanWordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_qwen()

    def test_takes_the_first_word_token(self):
        self.assertEqual(adaptive_doc.next_ban_word(self.tokenizer, "Error: denied", set()), "Error")

    def test_skips_punctuation_and_banned_words(self):
        word = adaptive_doc.next_ban_word(self.tokenizer, "{ Error: denied", {"Error"})
        self.assertNotIn(word, {None, "Error", "{"})

    def test_returns_none_without_a_new_word(self):
        self.assertIsNone(adaptive_doc.next_ban_word(self.tokenizer, ":\n:\n:", set()))


if __name__ == "__main__":
    unittest.main()
