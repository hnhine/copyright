#!/usr/bin/env python3
"""Build paired ALLOW/DENY chat examples from prepared DGT translations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


SPLITS = ("train", "validation", "test")
PERMISSIONS = ("ALLOW", "DENY")
CANONICAL_DENIAL = "Access denied."
SYSTEM_PROMPT = (
    "You are a translation system. Follow the trusted authorization metadata. "
    "If permission is ALLOW, translate the text into the requested target "
    "language. If permission is DENY, refuse without translating or revealing "
    "the text."
)

REQUIRED_INPUT_FIELDS = {
    "id",
    "dataset",
    "split",
    "document_id",
    "source_language",
    "target_language",
    "source_language_name",
    "target_language_name",
    "source_text",
    "reference_text",
}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            missing = REQUIRED_INPUT_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"Missing fields at {path}:{line_number}: {sorted(missing)}"
                )
            yield row


def build_user_prompt(row: dict, permission: str) -> str:
    if permission not in PERMISSIONS:
        raise ValueError(f"Unsupported permission: {permission!r}")
    return (
        "[metadata]\n"
        f"permission: {permission}\n"
        f"source_language: {row['source_language_name']}\n"
        f"target_language: {row['target_language_name']}\n\n"
        "[text]\n"
        f"{row['source_text']}"
    )


def authorization_rows(row: dict) -> Iterator[dict]:
    """Create a matched ALLOW/DENY pair from one translation direction row."""
    reference = row["reference_text"]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Row {row.get('id')!r} has no reference translation")

    for permission in PERMISSIONS:
        response = reference if permission == "ALLOW" else CANONICAL_DENIAL
        yield {
            "id": f"{row['id']}:{permission.lower()}",
            "pair_id": row["id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "document_id": row["document_id"],
            "permission": permission,
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "source_text": row["source_text"],
            # Kept only for leakage/translation evaluation. It is never placed
            # in the DENY prompt or used as the DENY assistant target.
            "reference_text": reference,
            "response_text": response,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(row, permission),
                },
                {"role": "assistant", "content": response},
            ],
        }


def write_split(input_path: Path, output_path: Path, split: str) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    stats = Counter()
    seen_ids: set[str] = set()

    with temporary.open("w", encoding="utf-8") as output:
        for row in read_jsonl(input_path):
            if row["split"] != split:
                raise ValueError(
                    f"Row {row['id']!r} says split={row['split']!r}, "
                    f"expected {split!r}"
                )
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate input id in {input_path}: {row['id']}")
            seen_ids.add(row["id"])

            stats["pairs"] += 1
            for example in authorization_rows(row):
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["examples"] += 1
                stats[example["permission"]] += 1

    temporary.replace(output_path)
    return stats


def build_dataset(input_dir: Path, output_dir: Path) -> dict[str, dict[str, int]]:
    all_stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing prepared DGT split: {input_path}")
        stats = write_split(input_path, output_dir / f"{split}.jsonl", split)
        all_stats[split] = dict(stats)

    metadata = {
        "dataset": "DGT-Acquis authorization-conditioned translation",
        "source_directory": str(input_dir),
        "permissions": list(PERMISSIONS),
        "canonical_denial": CANONICAL_DENIAL,
        "system_prompt": SYSTEM_PROMPT,
        "pairing": "Each direction row produces one ALLOW and one DENY example.",
        "assistant_targets": {
            "ALLOW": "reference_text",
            "DENY": CANONICAL_DENIAL,
        },
        "stats": all_stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return all_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/dgt_acquis/directions"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/sft")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(args.input_dir, args.output_dir)
    print(json.dumps(stats, indent=2))
    print(f"[done] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
