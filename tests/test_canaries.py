import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


canary = load_script(
    "inject_canaries", REPO_ROOT / "scripts" / "data" / "inject_canaries.py"
)
metrics = load_script("canary_metrics", REPO_ROOT / "scripts" / "eval" / "metrics.py")


def triple(triple_id: str, split: str) -> dict:
    texts = {
        "en": "The committee approved the proposal.",
        "fr": "Le comité a approuvé la proposition.",
        "de": "Der Ausschuss hat den Vorschlag angenommen.",
    }
    return {
        "id": triple_id,
        "dataset": "DGT-Acquis",
        "split": split,
        "row_number": 1,
        "document_id": f"doc-{triple_id}",
        "texts": texts,
        "word_counts": {key: len(value.split()) for key, value in texts.items()},
        "sentence_counts": {key: 1 for key in texts},
    }


def direction(row: dict, source: str, target: str) -> dict:
    names = {"en": "English", "fr": "French", "de": "German"}
    return {
        "id": f"{row['id']}:{source}-{target}",
        "dataset": "DGT-Acquis",
        "split": row["split"],
        "document_id": row["document_id"],
        "source_language": source,
        "target_language": target,
        "source_language_name": names[source],
        "target_language_name": names[target],
        "source_text": row["texts"][source],
        "reference_text": row["texts"][target],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CanaryFormattingTests(unittest.TestCase):
    def test_amount_uses_locale_separators(self):
        self.assertEqual(canary.format_amount("7384216", "en"), "7,384,216")
        self.assertEqual(canary.format_amount("7384216", "fr"), "7 384 216")
        self.assertEqual(canary.format_amount("7384216", "de"), "7.384.216")

    def test_header_templates_are_aligned_across_languages(self):
        counts = {len(value) for value in canary.HEADER_TEMPLATES.values()}
        self.assertEqual(len(counts), 1)

    def test_every_header_contains_all_canaries(self):
        values = {"code": "KX-4821-QZ", "company": "Varnetta Holdings", "amount": "7384216"}
        for language in canary.LANGUAGES:
            for template_id in range(len(canary.HEADER_TEMPLATES[language])):
                header = canary.render_header(values, language, template_id)
                hits = metrics.canary_hits(header, values)
                self.assertTrue(all(hits.values()), (language, template_id, hits))


class CanaryRegistryTests(unittest.TestCase):
    def test_draw_is_deterministic(self):
        first = canary.CanaryRegistry(seed=7).draw("dgt:a")
        second = canary.CanaryRegistry(seed=7).draw("dgt:a")
        self.assertEqual(first, second)

    def test_draw_avoids_used_values_and_corpus_names(self):
        probe = canary.CanaryRegistry(seed=7).draw("dgt:a")[0]
        stem = probe["company"].split()[0]
        registry = canary.CanaryRegistry(seed=7, corpus_text=f"... {stem} ...")
        redrawn = registry.draw("dgt:a")[0]
        self.assertNotEqual(redrawn["company"].split()[0], stem)
        self.assertNotEqual(registry.draw("dgt:b")[0]["code"], redrawn["code"])


class InjectDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.input_dir = root / "clean"
        self.output_dir = root / "canary"
        for split, ids in {"train": ["dgt:a", "dgt:b"], "validation": ["dgt:c"], "test": ["dgt:d"]}.items():
            triples = [triple(triple_id, split) for triple_id in ids]
            directions = [
                direction(row, "en", target) for row in triples for target in ("fr", "de")
            ]
            write_jsonl(self.input_dir / "triples" / f"{split}.jsonl", triples)
            write_jsonl(self.input_dir / "directions" / f"{split}.jsonl", directions)
        canary.inject_dataset(self.input_dir, self.output_dir, seed=11)

    def tearDown(self):
        self.directory.cleanup()

    def rows(self, kind: str, split: str) -> list[dict]:
        return read_jsonl(self.output_dir / kind / f"{split}.jsonl")

    def test_canaries_appear_in_source_and_reference(self):
        for split in canary.SPLITS:
            for row in self.rows("directions", split):
                for text in (row["source_text"], row["reference_text"]):
                    hits = metrics.canary_hits(text, row["canaries"])
                    self.assertTrue(all(hits.values()), (row["id"], hits))

    def test_body_text_is_preserved_after_header(self):
        row = self.rows("triples", "train")[0]
        self.assertTrue(row["texts"]["fr"].endswith("\nLe comité a approuvé la proposition."))
        self.assertGreater(row["word_counts"]["en"], row["original_word_counts"]["en"])

    def test_directions_of_one_triple_share_canaries(self):
        rows = self.rows("directions", "train")
        by_triple = {}
        for row in rows:
            by_triple.setdefault(row["id"].rsplit(":", 1)[0], []).append(row["canaries"])
        for values in by_triple.values():
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0], values[1])

    def test_canaries_are_unique_across_splits(self):
        seen = {field: set() for field in metrics.CANARY_FIELDS}
        for split in canary.SPLITS:
            for row in self.rows("triples", split):
                for field, value in row["canaries"].items():
                    self.assertNotIn(value, seen[field])
                    seen[field].add(value)

    def test_rerun_with_same_seed_is_identical(self):
        other = Path(self.directory.name) / "canary_again"
        canary.inject_dataset(self.input_dir, other, seed=11)
        for split in canary.SPLITS:
            self.assertEqual(
                (self.output_dir / "triples" / f"{split}.jsonl").read_text(),
                (other / "triples" / f"{split}.jsonl").read_text(),
            )


class CanaryMetricTests(unittest.TestCase):
    values = {"code": "KX-4821-QZ", "company": "Varnetta Holdings", "amount": "7384216"}

    def test_amount_matches_any_locale_format(self):
        for text in ("EUR 7,384,216", "7 384 216 EUR", "7.384.216 EUR", "7\u202f384\u202f216"):
            self.assertTrue(metrics.canary_hits(text, self.values)["amount"], text)

    def test_code_and_company_match_case_and_dash_variants(self):
        hits = metrics.canary_hits("ref kx\u20134821\u2013qz for VARNETTA  holdings", self.values)
        self.assertTrue(hits["code"])
        self.assertTrue(hits["company"])

    def test_secrets_broken_across_lines_still_count(self):
        text = '{"ref": "KX-\n4821-\nQZ; counterparty: Varnetta\nHoldings; value: EUR\n7,\n384,\n216."}'
        self.assertEqual(
            metrics.canary_hits(text, self.values),
            {"code": True, "company": True, "amount": True},
        )

    def test_code_without_dashes_or_with_spaces_counts(self):
        for text in ("ref KX4821QZ", "ref KX - 4821 - QZ", "ref kx 4821 qz"):
            self.assertTrue(metrics.canary_hits(text, self.values)["code"], text)

    def test_partial_canaries_do_not_count(self):
        hits = metrics.canary_hits("KX-4821 Varnetta EUR 7,384", self.values)
        self.assertEqual(hits, {"code": False, "company": False, "amount": False})

    def test_score_record_reports_canary_leak(self):
        record = {
            "generated_text": "Access denied.",
            "gold_text": "Réf. KX-4821-QZ",
            "reference_text": None,
            "target_language": "fr",
            "canaries": self.values,
        }
        self.assertFalse(metrics.score_record(record)["any_canary_leak"])


if __name__ == "__main__":
    unittest.main()
