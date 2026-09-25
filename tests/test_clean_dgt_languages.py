import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "clean_dgt_languages.py"
SPEC = importlib.util.spec_from_file_location("clean_dgt_languages", SCRIPT)
cleaner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cleaner)


class FakeLanguage:
    def __init__(self, name):
        self.name = name


class FakeConfidence:
    def __init__(self, language, value):
        self.language = FakeLanguage(language)
        self.value = value


class FakeDetector:
    def compute_language_confidence_values(self, text):
        language, confidence = text.split(":", maxsplit=1)
        return [FakeConfidence(language, float(confidence))]


def triple(row_id, en="ENGLISH:0.99", fr="FRENCH:0.99", de="GERMAN:0.99"):
    return {
        "id": row_id,
        "split": "train",
        "row_number": 1,
        "document_id": "doc",
        "texts": {"en": en, "fr": fr, "de": de},
    }


class LanguageCleaningTests(unittest.TestCase):
    def test_expected_languages_are_accepted(self):
        detected, issues = cleaner.check_triple(
            triple("dgt:good"), FakeDetector(), 0.90
        )
        self.assertEqual(issues, [])
        self.assertEqual(detected["fr"]["language"], "FRENCH")

    def test_mismatch_and_low_confidence_are_reported(self):
        row = triple("dgt:bad", en="BULGARIAN:1.0", fr="FRENCH:0.50")
        _, issues = cleaner.check_triple(row, FakeDetector(), 0.90)
        self.assertEqual(
            [issue["reason"] for issue in issues],
            ["language_mismatch", "low_confidence"],
        )

    def test_clean_split_keeps_only_directions_from_accepted_triples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "triples").mkdir(parents=True)
            (input_dir / "directions").mkdir(parents=True)
            rows = [
                triple("dgt:good"),
                triple("dgt:bad", en="ITALIAN:1.0"),
            ]
            directions = [
                {"id": "dgt:good:en-fr"},
                {"id": "dgt:good:en-de"},
                {"id": "dgt:bad:en-fr"},
                {"id": "dgt:bad:en-de"},
            ]
            (input_dir / "triples" / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (input_dir / "directions" / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in directions),
                encoding="utf-8",
            )

            stats = cleaner.clean_split(
                input_dir, output_dir, "train", FakeDetector(), 0.90
            )

            self.assertEqual(stats["accepted_triples"], 1)
            self.assertEqual(stats["rejected_triples"], 1)
            self.assertEqual(stats["accepted_directions"], 2)
            rejected = list(
                cleaner.read_jsonl(output_dir / "rejected" / "train.jsonl")
            )
            self.assertEqual(rejected[0]["id"], "dgt:bad")
            self.assertNotIn("texts", rejected[0])


if __name__ == "__main__":
    unittest.main()
