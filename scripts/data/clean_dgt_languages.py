#!/usr/bin/env python3
"""Remove DGT triples whose columns are not English, French, and German."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


SPLITS = ("train", "validation", "test")
EXPECTED_LANGUAGES = {"en": "ENGLISH", "fr": "FRENCH", "de": "GERMAN"}


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
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def build_detector():
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as error:
        raise RuntimeError(
            "Missing lingua-language-detector. Install requirements-eval.txt first."
        ) from error
    return LanguageDetectorBuilder.from_all_languages().build()


def classify(detector, text: str) -> tuple[str, float]:
    values = detector.compute_language_confidence_values(text)
    if not values:
        return "UNKNOWN", 0.0
    best = values[0]
    return best.language.name, float(best.value)


def check_triple(
    row: dict, detector, min_confidence: float
) -> tuple[dict[str, dict], list[dict]]:
    texts = row.get("texts")
    if not isinstance(texts, dict):
        raise ValueError(f"Row {row.get('id')!r} has no texts mapping")

    detected = {}
    issues = []
    for field, expected in EXPECTED_LANGUAGES.items():
        text = texts.get(field)
        if not isinstance(text, str) or not text.strip():
            language, confidence = "UNKNOWN", 0.0
        else:
            language, confidence = classify(detector, text)
        detected[field] = {
            "expected": expected,
            "language": language,
            "confidence": round(confidence, 6),
        }
        if language != expected:
            issues.append(
                {
                    "field": field,
                    "reason": "language_mismatch",
                    "expected": expected,
                    "detected": language,
                    "confidence": round(confidence, 6),
                }
            )
        elif confidence < min_confidence:
            issues.append(
                {
                    "field": field,
                    "reason": "low_confidence",
                    "expected": expected,
                    "detected": language,
                    "confidence": round(confidence, 6),
                }
            )
    return detected, issues


def base_triple_id(direction_row: dict) -> str:
    row_id = direction_row.get("id")
    if not isinstance(row_id, str) or ":" not in row_id:
        raise ValueError(f"Invalid direction row id: {row_id!r}")
    return row_id.rsplit(":", maxsplit=1)[0]


def clean_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    detector,
    min_confidence: float,
) -> dict:
    triple_input = input_dir / "triples" / f"{split}.jsonl"
    direction_input = input_dir / "directions" / f"{split}.jsonl"
    if not triple_input.exists() or not direction_input.exists():
        raise FileNotFoundError(f"Missing prepared DGT files for split {split!r}")

    accepted_rows = []
    rejected_rows = []
    accepted_ids = set()
    issue_counts = Counter()
    detected_combinations = Counter()

    for row in read_jsonl(triple_input):
        detected, issues = check_triple(row, detector, min_confidence)
        combination = "/".join(
            detected[field]["language"] for field in EXPECTED_LANGUAGES
        )
        detected_combinations[combination] += 1
        if not issues:
            accepted_rows.append(row)
            accepted_ids.add(row["id"])
            continue

        issue_counts.update(issue["reason"] for issue in issues)
        rejected_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "row_number": row.get("row_number"),
                "document_id": row.get("document_id"),
                "detected": detected,
                "issues": issues,
            }
        )

    triple_count = write_jsonl(
        output_dir / "triples" / f"{split}.jsonl", accepted_rows
    )
    rejected_count = write_jsonl(
        output_dir / "rejected" / f"{split}.jsonl", rejected_rows
    )

    direction_rows = list(read_jsonl(direction_input))
    clean_directions = [
        row for row in direction_rows if base_triple_id(row) in accepted_ids
    ]
    direction_count = write_jsonl(
        output_dir / "directions" / f"{split}.jsonl", clean_directions
    )

    return {
        "input_triples": len(accepted_rows) + len(rejected_rows),
        "accepted_triples": triple_count,
        "rejected_triples": rejected_count,
        "rejected_percent": round(
            100 * rejected_count / (triple_count + rejected_count), 4
        ),
        "input_directions": len(direction_rows),
        "accepted_directions": direction_count,
        "rejected_directions": len(direction_rows) - direction_count,
        "issue_counts": dict(issue_counts),
        "detected_combinations": dict(detected_combinations),
    }


def clean_dataset(
    input_dir: Path, output_dir: Path, min_confidence: float = 0.90
) -> dict:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    detector = build_detector()
    stats = {
        split: clean_split(
            input_dir, output_dir, split, detector, min_confidence
        )
        for split in SPLITS
    }

    source_metadata_path = input_dir / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    report = {
        "dataset": "DGT-Acquis language-cleaned",
        "source_directory": str(input_dir),
        "source_metadata": str(source_metadata_path),
        "expected_languages": EXPECTED_LANGUAGES,
        "detector": "lingua-language-detector/all-languages",
        "minimum_confidence": min_confidence,
        "directions": source_metadata.get("directions", []),
        "stats": stats,
        "source_attribution": source_metadata.get("source_attribution", {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "language_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/dgt_acquis"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dgt_acquis_clean"),
    )
    parser.add_argument("--min-confidence", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = clean_dataset(args.input_dir, args.output_dir, args.min_confidence)
    summary = {
        split: {
            "accepted": values["accepted_triples"],
            "rejected": values["rejected_triples"],
        }
        for split, values in report["stats"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"[audit] {(args.output_dir / 'language_audit.json').resolve()}")
    print(f"[done] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
