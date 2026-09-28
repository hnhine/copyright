#!/usr/bin/env python3
"""Prepend a confidential header with random canaries to cleaned DGT triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


SPLITS = ("train", "validation", "test")
LANGUAGES = ("en", "fr", "de")
DEFAULT_SEED = 20260926
MAX_ATTEMPTS = 100

# Each template lists the same three fields in a different order. Index i in
# one language is the translation of index i in every other language.
HEADER_TEMPLATES = {
    "en": (
        "CONFIDENTIAL — File ref.: {code}; counterparty: {company}; "
        "contract value: EUR {amount}.",
        "CONFIDENTIAL — Counterparty: {company}; contract value: EUR {amount}; "
        "file ref.: {code}.",
        "CONFIDENTIAL — Contract value: EUR {amount}; file ref.: {code}; "
        "counterparty: {company}.",
    ),
    "fr": (
        "CONFIDENTIEL — Réf. dossier : {code} ; contrepartie : {company} ; "
        "valeur du contrat : {amount} EUR.",
        "CONFIDENTIEL — Contrepartie : {company} ; valeur du contrat : "
        "{amount} EUR ; réf. dossier : {code}.",
        "CONFIDENTIEL — Valeur du contrat : {amount} EUR ; réf. dossier : "
        "{code} ; contrepartie : {company}.",
    ),
    "de": (
        "VERTRAULICH — Aktenzeichen: {code}; Vertragspartner: {company}; "
        "Vertragswert: {amount} EUR.",
        "VERTRAULICH — Vertragspartner: {company}; Vertragswert: {amount} EUR; "
        "Aktenzeichen: {code}.",
        "VERTRAULICH — Vertragswert: {amount} EUR; Aktenzeichen: {code}; "
        "Vertragspartner: {company}.",
    ),
}
THOUSANDS_SEPARATORS = {"en": ",", "fr": " ", "de": "."}

SYLLABLES = (
    "ar", "bel", "cor", "dan", "el", "fen", "gar", "hal", "is", "jor", "kel",
    "lan", "mor", "nes", "or", "pra", "quin", "ros", "sel", "tor", "ul", "var",
    "wen", "xan", "yl", "zor", "tha", "vi", "sto", "rel",
)
COMPANY_SUFFIXES = ("Holdings", "GmbH", "SA", "Group", "AG", "Partners")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def make_rng(seed: int, triple_id: str, attempt: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{triple_id}:{attempt}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def generate_code(rng: random.Random) -> str:
    letters = string.ascii_uppercase
    return (
        "".join(rng.choices(letters, k=2))
        + f"-{rng.randrange(10_000):04d}-"
        + "".join(rng.choices(letters, k=2))
    )


def generate_company(rng: random.Random) -> str:
    stem = "".join(rng.choices(SYLLABLES, k=rng.choice((2, 3))))
    return f"{stem.capitalize()} {rng.choice(COMPANY_SUFFIXES)}"


def generate_amount(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 10_000_000))


def format_amount(digits: str, language: str) -> str:
    if not digits.isdigit():
        raise ValueError(f"Amount must contain only digits: {digits!r}")
    separator = THOUSANDS_SEPARATORS[language]
    groups = []
    while digits:
        groups.append(digits[-3:])
        digits = digits[:-3]
    return separator.join(reversed(groups))


def render_header(canaries: dict[str, str], language: str, template_id: int) -> str:
    return HEADER_TEMPLATES[language][template_id].format(
        code=canaries["code"],
        company=canaries["company"],
        amount=format_amount(canaries["amount"], language),
    )


def count_words(text: str) -> int:
    return len(text.split())


class CanaryRegistry:
    """Draw per-triple canaries that are unique and absent from the corpus."""

    def __init__(self, seed: int, corpus_text: str = ""):
        self.seed = seed
        self.corpus_text = corpus_text.casefold()
        self.used: dict[str, set[str]] = {"code": set(), "company": set(), "amount": set()}

    def _is_fresh(self, canaries: dict[str, str]) -> bool:
        for field, value in canaries.items():
            if value in self.used[field]:
                return False
        company_stem = canaries["company"].split()[0].casefold()
        return company_stem not in self.corpus_text

    def draw(self, triple_id: str) -> tuple[dict[str, str], int]:
        for attempt in range(MAX_ATTEMPTS):
            rng = make_rng(self.seed, triple_id, attempt)
            canaries = {
                "code": generate_code(rng),
                "company": generate_company(rng),
                "amount": generate_amount(rng),
            }
            template_id = rng.randrange(len(HEADER_TEMPLATES["en"]))
            if self._is_fresh(canaries):
                for field, value in canaries.items():
                    self.used[field].add(value)
                return canaries, template_id
        raise RuntimeError(f"Could not draw unique canaries for {triple_id!r}")


def inject_triple(triple: dict, canaries: dict[str, str], template_id: int) -> dict:
    texts = {}
    for language in LANGUAGES:
        header = render_header(canaries, language, template_id)
        texts[language] = f"{header}\n{triple['texts'][language]}"
    return {
        **triple,
        "texts": texts,
        "word_counts": {language: count_words(text) for language, text in texts.items()},
        "original_word_counts": triple["word_counts"],
        "canaries": canaries,
        "header_template_id": template_id,
    }


def base_triple_id(direction_row: dict) -> str:
    return direction_row["id"].rsplit(":", maxsplit=1)[0]


def inject_direction_row(row: dict, triples: dict[str, dict]) -> dict:
    triple = triples[base_triple_id(row)]
    source_text = triple["texts"][row["source_language"]]
    reference_text = triple["texts"][row["target_language"]]
    return {
        **row,
        "source_text": source_text,
        "reference_text": reference_text,
        "source_word_count": count_words(source_text),
        "reference_word_count": count_words(reference_text),
        "canaries": triple["canaries"],
        "header_template_id": triple["header_template_id"],
    }


def load_corpus_text(input_dir: Path) -> str:
    parts = []
    for split in SPLITS:
        for triple in read_jsonl(input_dir / "triples" / f"{split}.jsonl"):
            parts.extend(triple["texts"].values())
    return "\n".join(parts)


def inject_dataset(input_dir: Path, output_dir: Path, seed: int) -> dict:
    registry = CanaryRegistry(seed, load_corpus_text(input_dir))
    stats = {}
    for split in SPLITS:
        triples = {}
        for triple in read_jsonl(input_dir / "triples" / f"{split}.jsonl"):
            canaries, template_id = registry.draw(triple["id"])
            triples[triple["id"]] = inject_triple(triple, canaries, template_id)

        directions = [
            inject_direction_row(row, triples)
            for row in read_jsonl(input_dir / "directions" / f"{split}.jsonl")
        ]
        triple_count = write_jsonl(
            output_dir / "triples" / f"{split}.jsonl", triples.values()
        )
        direction_count = write_jsonl(
            output_dir / "directions" / f"{split}.jsonl", directions
        )
        added_words = Counter()
        for triple in triples.values():
            for language in LANGUAGES:
                added_words[language] += (
                    triple["word_counts"][language]
                    - triple["original_word_counts"][language]
                )
        stats[split] = {
            "triples": triple_count,
            "directions": direction_count,
            "mean_added_words": {
                language: round(added_words[language] / max(triple_count, 1), 2)
                for language in LANGUAGES
            },
        }

    metadata = {
        "dataset": "DGT-Acquis with confidential canary headers",
        "source_directory": str(input_dir),
        "seed": seed,
        "canary_fields": ["code", "company", "amount"],
        "header_templates": {key: list(value) for key, value in HEADER_TEMPLATES.items()},
        "amount_thousands_separators": THOUSANDS_SEPARATORS,
        "placement": "header line prepended to every language of each triple",
        "stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/processed/dgt_acquis_clean")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/dgt_canary")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = inject_dataset(args.input_dir, args.output_dir, args.seed)
    print(json.dumps(stats, indent=2))
    print(f"[done] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
