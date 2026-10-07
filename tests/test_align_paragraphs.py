import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import align_paragraphs as align  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "prepare_dgt", REPO_ROOT / "scripts" / "data" / "prepare_dgt.py"
)
dgt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dgt)


def english(i: int) -> str:
    return f"Article {100 + i} of Regulation {2000 + i}/{i + 7} applies. " + "word " * (20 + 9 * i)


def german(i: int) -> str:
    return f"Artikel {100 + i} der Verordnung {2000 + i}/{i + 7} gilt. " + "Wort " * (20 + 9 * i)


def french(i: int) -> str:
    return f"L'article {100 + i} du règlement {2000 + i}/{i + 7} s'applique. " + "mot " * (20 + 9 * i)


def column(texts):
    return [align.paragraph(text) for text in texts]


class AlignParagraphTests(unittest.TestCase):
    def test_correct_pairs_are_kept(self):
        en = column(english(i) for i in range(6))
        de = column(german(i) for i in range(6))
        for i in range(6):
            self.assertEqual(align.align_paragraph(en, de, i, "de")[0], i)

    def test_constant_shift_is_recovered(self):
        en = column(english(i) for i in range(6))
        de = column(["Präambel."] + [german(i) for i in range(6)])
        for i in range(6):
            self.assertEqual(align.align_paragraph(en, de, i, "de")[0], i + 1)

    def test_shift_that_changes_inside_a_document(self):
        en = column(english(i) for i in range(8))
        # German lacks paragraph 3, so everything after it moves up one line.
        de = column(german(i) for i in range(8) if i != 3)
        self.assertEqual(align.align_paragraph(en, de, 1, "de")[0], 1)
        self.assertEqual(align.align_paragraph(en, de, 5, "de")[0], 4)
        self.assertIsNone(align.align_paragraph(en, de, 3, "de"))

    def test_ambiguous_neighbours_are_dropped(self):
        text_en, text_de = "word " * 60, "Wort " * 60
        en = column([text_en, text_en, text_en])
        de = column([text_de, text_de, text_de])
        self.assertIsNone(align.align_paragraph(en, de, 1, "de"))

    def test_number_disagreement_costs_more_than_agreement(self):
        right = align.match_cost(align.paragraph(english(2)), align.paragraph(german(2)), "de")
        wrong = align.match_cost(align.paragraph(english(2)), align.paragraph(german(3)), "de")
        self.assertLess(right + align.MIN_MARGIN, wrong)


class PrepareDgtAlignmentTests(unittest.TestCase):
    def write_raw(self, directory: Path) -> None:
        en = [english(i) for i in range(5)]
        fr = [french(i) for i in range(5)]
        de = ["Präambel."] + [german(i) for i in range(4)]
        ids = ["doc-a"] * 5
        rows = {"a1": ids + ["doc-a-doc"], "en": en + ["x"], "fr": fr + ["x"], "de": de + ["x"]}
        for name, lines in rows.items():
            (directory / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_documents_are_grouped_without_doc_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_raw(Path(directory))
            documents = list(dgt.read_documents(Path(directory)))
        self.assertEqual([document_id for document_id, _ in documents], ["doc-a"])
        self.assertEqual(len(documents[0][1]), 5)

    def test_candidates_use_the_realigned_german(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_raw(Path(directory))
            _, rows = next(dgt.read_documents(Path(directory)))
        results = list(dgt.aligned_candidates(rows, min_words=1, max_words=1000))
        aligned = [(row, texts, info) for row, texts, info in results if texts]
        self.assertTrue(aligned)
        for row_number, texts, info in aligned:
            index = row_number - 1
            self.assertEqual(texts["de"], dgt.normalize_text(german(index)))
            self.assertEqual(texts["fr"], dgt.normalize_text(french(index)))
            self.assertEqual(info["de"]["offset"], 1)
            self.assertEqual(info["fr"]["offset"], 0)
        # The last English paragraph has no German counterpart in the file.
        self.assertIn((5, None, {}), results)


if __name__ == "__main__":
    unittest.main()
