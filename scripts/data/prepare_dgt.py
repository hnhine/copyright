#!/usr/bin/env python3
"""Filter DGT-Acquis into document-disjoint multilingual JSONL splits.

The language files of DGT-Acquis are aligned by line, but in many documents
the German (and some French) paragraphs are shifted by one or more lines, so
line-aligned "translations" are often a neighbouring paragraph. By default
paragraphs are therefore re-aligned within each document (see
align_paragraphs.py); `--alignment line` keeps the original line pairing.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_paragraphs import (  # noqa: E402
    LENGTH_MODEL,
    MAX_COST,
    MIN_MARGIN,
    NUMBER_WEIGHT,
    WINDOW,
    align_paragraph,
    paragraph,
)


LANGUAGES = ("en", "fr", "de")
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German"}
DEFAULT_DIRECTIONS = ("en-fr", "en-de", "fr-en", "de-en")
SPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(r"[.!?…]+(?:[\"'”’»\)\]]+)?(?=\s|$)")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", html.unescape(text)).strip()


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    count = len(SENTENCE_END_RE.findall(text))
    return max(1, count) if text else 0


def stable_split(
    document_id: str,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def read_aligned_rows(raw_dir: Path) -> Iterator[tuple[int, str, dict[str, str]]]:
    paths = {"a1": raw_dir / "a1.txt"}
    paths.update({language: raw_dir / f"{language}.txt" for language in LANGUAGES})
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing DGT files:\n  " + "\n  ".join(missing))

    handles = {
        key: path.open("r", encoding="utf-8", errors="replace")
        for key, path in paths.items()
    }
    try:
        sentinel = object()
        rows = itertools.zip_longest(
            *(handles[key] for key in ("a1", *LANGUAGES)), fillvalue=sentinel
        )
        for row_number, row in enumerate(rows, start=1):
            if sentinel in row:
                raise ValueError(
                    f"DGT columns have different line counts near row {row_number}"
                )
            source_id = row[0].rstrip("\r\n")
            texts = {
                language: normalize_text(value.rstrip("\r\n"))
                for language, value in zip(LANGUAGES, row[1:])
            }
            yield row_number, source_id, texts
    finally:
        for handle in handles.values():
            handle.close()


def validate_texts(
    texts: dict[str, str],
    min_words: int,
    max_words: int,
    min_sentences: int,
    max_sentences: int,
    min_length_ratio: float,
) -> tuple[bool, str, dict[str, int], dict[str, int]]:
    if not all(texts.values()):
        return False, "empty_language", {}, {}

    word_counts = {language: count_words(text) for language, text in texts.items()}
    if any(count < min_words for count in word_counts.values()):
        return False, "too_short", word_counts, {}
    if any(count > max_words for count in word_counts.values()):
        return False, "too_long", word_counts, {}

    sentence_counts = {
        language: count_sentences(text) for language, text in texts.items()
    }
    if any(count < min_sentences for count in sentence_counts.values()):
        return False, "too_few_sentences", word_counts, sentence_counts
    if any(count > max_sentences for count in sentence_counts.values()):
        return False, "too_many_sentences", word_counts, sentence_counts

    en_count = word_counts["en"]
    for language in ("fr", "de"):
        ratio = word_counts[language] / en_count
        if ratio < min_length_ratio or ratio > 1.0 / min_length_ratio:
            return False, "length_ratio", word_counts, sentence_counts
    return True, "accepted", word_counts, sentence_counts


def read_documents(raw_dir: Path) -> Iterator[tuple[str, list[tuple[int, dict[str, str]]]]]:
    """Group consecutive paragraph rows by document id, skipping '-doc' rows."""
    current_id, rows = None, []
    for row_number, source_id, texts in read_aligned_rows(raw_dir):
        if source_id.endswith("-doc"):
            continue
        if source_id != current_id:
            if rows:
                yield current_id, rows
            current_id, rows = source_id, []
        rows.append((row_number, texts))
    if rows:
        yield current_id, rows


def aligned_candidates(
    rows: list[tuple[int, dict[str, str]]], min_words: int, max_words: int
) -> Iterator[tuple[int, dict[str, str], dict]]:
    """Yield (English row, re-aligned texts, alignment info) for one document.

    Only English paragraphs within the word limits are aligned. A candidate is
    dropped unless every other language has an unambiguous match.
    """
    columns = {
        language: [paragraph(texts[language]) for _, texts in rows]
        for language in LANGUAGES
    }
    for index, (row_number, texts) in enumerate(rows):
        if not min_words <= count_words(texts["en"]) <= max_words:
            continue
        aligned, info = {"en": texts["en"]}, {}
        for language in LANGUAGES[1:]:
            match = align_paragraph(columns["en"], columns[language], index, language)
            if match is None:
                break
            other, cost = match
            aligned[language] = rows[other][1][language]
            info[language] = {
                "row_number": rows[other][0],
                "offset": other - index,
                "cost": round(cost, 4),
            }
        else:
            yield row_number, aligned, info
            continue
        yield row_number, None, {}


def reservoir_add(
    reservoir: list[dict],
    item: dict,
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen_count)
    if replacement < capacity:
        reservoir[replacement] = item


def write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def direction_rows(triples: list[dict], directions: list[tuple[str, str]]) -> Iterator[dict]:
    for triple in triples:
        for source_language, target_language in directions:
            yield {
                "id": f"{triple['id']}:{source_language}-{target_language}",
                "dataset": "DGT-Acquis",
                "split": triple["split"],
                "document_id": triple["document_id"],
                "source_language": source_language,
                "target_language": target_language,
                "source_language_name": LANGUAGE_NAMES[source_language],
                "target_language_name": LANGUAGE_NAMES[target_language],
                "source_text": triple["texts"][source_language],
                "reference_text": triple["texts"][target_language],
                "source_word_count": triple["word_counts"][source_language],
                "reference_word_count": triple["word_counts"][target_language],
            }


def parse_directions(values: list[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        try:
            source, target = value.lower().split("-", maxsplit=1)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Direction must look like en-fr, got {value!r}"
            ) from error
        if source not in LANGUAGES or target not in LANGUAGES or source == target:
            raise argparse.ArgumentTypeError(f"Unsupported direction: {value!r}")
        result.append((source, target))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/dgt_acquis"))
    parser.add_argument(
        "--alignment",
        choices=("paragraph", "line"),
        default="paragraph",
        help="Re-align paragraphs within each document (default) or keep line pairs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/dgt_acquis")
    )
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=6)
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--max-train", type=int, default=1500)
    parser.add_argument("--max-validation", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-document-rows",
        action="store_true",
        help="Keep a1 rows ending in '-doc' (excluded by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_length_ratio <= 1):
        raise ValueError("--min-length-ratio must be in (0, 1]")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train + validation ratios must be less than 1")
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot exceed --max-words")
    directions = parse_directions(args.directions)

    capacities = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    reservoirs: dict[str, list[dict]] = {split: [] for split in capacities}
    accepted_seen = Counter()
    reasons = Counter()
    rngs = {
        split: random.Random(f"{args.seed}:{split}") for split in capacities
    }
    content_hashes: set[bytes] = set()
    alignment_stats = Counter()

    def candidates() -> Iterator[tuple[int, str, dict[str, str] | None, dict]]:
        if args.alignment == "line":
            for row_number, source_id, texts in read_aligned_rows(args.raw_dir):
                if not args.keep_document_rows and source_id.endswith("-doc"):
                    reasons["document_row"] += 1
                    continue
                yield row_number, source_id, texts, {}
            return
        for document_id, rows in read_documents(args.raw_dir):
            for row_number, texts, info in aligned_candidates(
                rows, args.min_words, args.max_words
            ):
                alignment_stats["english_candidates"] += 1
                if texts is None:
                    alignment_stats["unaligned"] += 1
                    continue
                alignment_stats["aligned"] += 1
                for language, value in info.items():
                    alignment_stats[f"{language}_offset_{value['offset']:+d}"] += 1
                yield row_number, document_id, texts, info

    for row_number, source_id, texts, alignment in candidates():
        if texts is None:
            continue
        accepted, reason, word_counts, sentence_counts = validate_texts(
            texts,
            args.min_words,
            args.max_words,
            args.min_sentences,
            args.max_sentences,
            args.min_length_ratio,
        )
        if not accepted:
            reasons[reason] += 1
            continue

        content_digest = hashlib.sha256(
            "\0".join(texts[language] for language in LANGUAGES).encode("utf-8")
        ).digest()
        if content_digest in content_hashes:
            reasons["duplicate"] += 1
            continue
        content_hashes.add(content_digest)

        document_id = source_id.removesuffix("-doc")
        split = stable_split(
            document_id, args.seed, args.train_ratio, args.validation_ratio
        )
        accepted_seen[split] += 1
        record = {
            "id": f"dgt:{content_digest.hex()[:16]}",
            "dataset": "DGT-Acquis",
            "split": split,
            "row_number": row_number,
            "document_id": document_id,
            "texts": texts,
            "word_counts": word_counts,
            "sentence_counts": sentence_counts,
        }
        if alignment:
            record["alignment"] = alignment
        reservoir_add(
            reservoirs[split],
            record,
            accepted_seen[split],
            capacities[split],
            rngs[split],
        )

        if row_number % 500_000 == 0:
            print(
                f"[scan] {row_number:,} rows; accepted="
                f"{sum(accepted_seen.values()):,}"
            )

    output_counts = {}
    for split, records in reservoirs.items():
        records.sort(key=lambda row: row["id"])
        triple_path = args.output_dir / "triples" / f"{split}.jsonl"
        direction_path = args.output_dir / "directions" / f"{split}.jsonl"
        output_counts[f"triples/{split}"] = write_jsonl(triple_path, iter(records))
        output_counts[f"directions/{split}"] = write_jsonl(
            direction_path, direction_rows(records, directions)
        )

    metadata = {
        "dataset": "DGT-Acquis",
        "seed": args.seed,
        "alignment": {
            "method": args.alignment,
            "parameters": (
                {
                    "length_model": LENGTH_MODEL,
                    "number_weight": NUMBER_WEIGHT,
                    "window": WINDOW,
                    "max_cost": MAX_COST,
                    "min_margin": MIN_MARGIN,
                }
                if args.alignment == "paragraph"
                else None
            ),
            "stats": dict(sorted(alignment_stats.items())),
        },
        "filters": {
            "min_words_each_language": args.min_words,
            "max_words_each_language": args.max_words,
            "min_sentences_each_language": args.min_sentences,
            "max_sentences_each_language": args.max_sentences,
            "min_pairwise_length_ratio_vs_english": args.min_length_ratio,
            "excluded_document_rows": not args.keep_document_rows,
        },
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": 1 - args.train_ratio - args.validation_ratio,
        },
        "directions": [f"{source}-{target}" for source, target in directions],
        "eligible_before_reservoir": dict(accepted_seen),
        "rejected": dict(reasons),
        "output_counts": output_counts,
        "source_attribution": {
            "owner": "European Commission",
            "page": "https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-acquis_en",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_counts, indent=2))
    print(f"[done] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

