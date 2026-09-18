#!/usr/bin/env python3
"""Create deterministic 100-word copyright/OOD passages from NoticeBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterator


SPACE_RE = re.compile(r"\s+")
DEFAULT_TYPES = ("book", "scripts")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def read_source_text(path: Path) -> str:
    # Prefer UTF-8; fall back only for genuinely legacy source files.
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("iso-8859-1")


def deterministic_windows(
    words: list[str],
    window_size: int,
    count: int,
    seed: int,
    key: str,
) -> list[tuple[int, list[str]]]:
    """Choose non-overlapping windows in a stable, document-specific order."""
    if len(words) < window_size:
        return []
    starts = list(range(0, len(words) - window_size + 1, window_size))
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big"
    )
    random.Random(local_seed).shuffle(starts)
    return [(start, words[start : start + window_size]) for start in starts[:count]]


def safe_source_path(raw_root: Path, relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe NoticeBench content path: {relative_path!r}")
    path = raw_root.joinpath(*posix_path.parts)
    try:
        path.resolve().relative_to(raw_root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes NoticeBench root: {relative_path!r}") from error
    return path


def iter_samples(
    raw_root: Path,
    content_types: set[str],
    sample_words: int,
    samples_per_document: int,
    seed: int,
    stats: Counter,
) -> Iterator[dict]:
    metadata_path = raw_root / "raw_data" / "notice" / "full_content.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing NoticeBench metadata: {metadata_path}")
    entries = json.loads(metadata_path.read_text(encoding="utf-8"))

    for entry in sorted(entries, key=lambda row: row["content_path"]):
        content_type = entry["content_type"]
        if content_type not in content_types:
            continue
        source_path = safe_source_path(raw_root, entry["content_path"])
        if not source_path.exists():
            stats["missing_document"] += 1
            continue
        full_text = read_source_text(source_path)
        words = normalize_text(full_text).split()
        windows = deterministic_windows(
            words,
            sample_words,
            samples_per_document,
            seed,
            entry["content_path"],
        )
        if not windows:
            stats["document_too_short"] += 1
            continue
        stats["documents"] += 1

        for window_index, (start_word, window_words) in enumerate(windows):
            passage = " ".join(window_words)
            digest = hashlib.sha256(
                f"{entry['content_path']}\0{start_word}\0{passage}".encode("utf-8")
            ).hexdigest()[:16]
            stats["samples"] += 1
            yield {
                "id": f"noticebench:{digest}",
                "dataset": "NoticeBench",
                "split": "external_test",
                "content_type": content_type,
                "content_subclass": entry.get("content_year")
                or entry.get("content_owner"),
                "document_name": entry["content_name"],
                "document_index": entry.get("idx"),
                "window_index": window_index,
                "start_word": start_word,
                "source_language": "en",
                "target_language": "fr",
                "source_text": passage,
                "source_word_count": len(window_words),
                "reference_text": None,
                "copyright_notice": entry["copyright_notice"],
                "provenance_path": entry["content_path"],
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw/noticebench")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/noticebench")
    )
    parser.add_argument(
        "--content-types",
        nargs="+",
        choices=("book", "scripts", "news", "doc"),
        default=list(DEFAULT_TYPES),
    )
    parser.add_argument("--sample-words", type=int, default=100)
    parser.add_argument("--samples-per-document", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_words <= 0 or args.samples_per_document <= 0:
        raise ValueError("sample sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "external_test.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    stats = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for sample in iter_samples(
            args.raw_root,
            set(args.content_types),
            args.sample_words,
            args.samples_per_document,
            args.seed,
            stats,
        ):
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    temporary.replace(output_path)

    metadata = {
        "dataset": "NoticeBench",
        "split": "external_test",
        "content_types": args.content_types,
        "sample_words": args.sample_words,
        "samples_per_document": args.samples_per_document,
        "seed": args.seed,
        "stats": dict(stats),
        "has_reference_translations": False,
        "source": "https://github.com/liamjxu/NoticeBench",
        "paper": "https://aclanthology.org/2024.emnlp-main.1147/",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[done] {output_path.resolve()}")


if __name__ == "__main__":
    main()

