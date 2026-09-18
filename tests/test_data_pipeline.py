import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dgt = load_script("prepare_dgt")
noticebench = load_script("prepare_noticebench")
download = load_script("download_raw")


class DgtPreparationTests(unittest.TestCase):
    def test_split_is_stable_and_document_level(self):
        first = dgt.stable_split("l-2011-001-0001", 42, 0.8, 0.1)
        second = dgt.stable_split("l-2011-001-0001", 42, 0.8, 0.1)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation", "test"})

    def test_validate_texts(self):
        sentence = "A sentence with enough ordinary words for this unit test. "
        text = (sentence * 9).strip()
        texts = {"en": text, "fr": text, "de": text}
        accepted, reason, word_counts, sentence_counts = dgt.validate_texts(
            texts,
            min_words=70,
            max_words=120,
            min_sentences=2,
            max_sentences=12,
            min_length_ratio=0.6,
        )
        self.assertTrue(accepted, reason)
        self.assertEqual(word_counts["en"], word_counts["fr"])
        self.assertEqual(sentence_counts["en"], 9)

    def test_reservoir_has_fixed_capacity(self):
        import random

        result = []
        rng = random.Random(42)
        for seen in range(1, 101):
            dgt.reservoir_add(result, {"value": seen}, seen, 10, rng)
        self.assertEqual(len(result), 10)
        self.assertEqual(len({row["value"] for row in result}), 10)


class NoticeBenchPreparationTests(unittest.TestCase):
    def test_source_decoder_prefers_utf8_and_falls_back_to_latin1(self):
        with tempfile.TemporaryDirectory() as directory:
            utf8_path = Path(directory) / "utf8.txt"
            latin1_path = Path(directory) / "latin1.txt"
            utf8_path.write_bytes("“quoted” café".encode("utf-8"))
            latin1_path.write_bytes("café".encode("iso-8859-1"))
            self.assertEqual(
                noticebench.read_source_text(utf8_path), "“quoted” café"
            )
            self.assertEqual(noticebench.read_source_text(latin1_path), "café")

    def test_windows_are_stable_and_non_overlapping(self):
        words = [f"w{i}" for i in range(1000)]
        first = noticebench.deterministic_windows(words, 100, 5, 42, "book")
        second = noticebench.deterministic_windows(words, 100, 5, 42, "book")
        self.assertEqual(first, second)
        starts = [start for start, _ in first]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertTrue(all(len(window) == 100 for _, window in first))

    def test_source_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                noticebench.safe_source_path(root, "../secret.txt")


class DownloadTests(unittest.TestCase):
    def test_rejects_unsafe_zip_member(self):
        with self.assertRaises(ValueError):
            download._safe_member_path("../../escape.txt")


if __name__ == "__main__":
    unittest.main()

