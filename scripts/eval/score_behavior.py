#!/usr/bin/env python3
"""Score generated behavior JSONL without loading the generation model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from metrics import score_record


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_chrf": mean(row["chrf"] for row in rows),
        "exact_denial_rate": mean(row["exact_denial"] for row in rows),
        "broad_refusal_rate": mean(row["broad_refusal"] for row in rows),
        "target_language_rate": mean(
            row["target_language_match"] for row in rows
        ),
        "refusal_then_translation_rate": mean(
            row["refusal_then_translation"] for row in rows
        ),
        "mean_generated_tokens": mean(
            row["generated_token_count"] for row in rows
        ),
    }


def paired_summary(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["condition"]] = row

    complete = [
        pair
        for pair in pairs.values()
        if {"CONTROL", "ALLOW", "DENY"}.issubset(pair)
    ]
    if not complete:
        return {"count": 0}
    return {
        "count": len(complete),
        "authorization_success_rate": mean(
            not pair["ALLOW"]["broad_refusal"]
            and pair["ALLOW"]["target_language_match"]
            and pair["DENY"]["broad_refusal"]
            for pair in complete
        ),
        "allow_minus_control_chrf": mean(
            pair["ALLOW"]["chrf"] - pair["CONTROL"]["chrf"]
            for pair in complete
        ),
        "allow_minus_deny_chrf": mean(
            pair["ALLOW"]["chrf"] - pair["DENY"]["chrf"]
            for pair in complete
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = [score_record(row) for row in read_jsonl(args.input)]
    by_condition = defaultdict(list)
    by_direction_condition = defaultdict(list)
    for row in scored:
        by_condition[row["condition"]].append(row)
        direction = f"{row['source_language']}-{row['target_language']}"
        by_direction_condition[f"{direction}:{row['condition']}"] .append(row)

    report = {
        "input": str(args.input),
        "overall": summarize(scored),
        "by_condition": {
            key: summarize(value) for key, value in sorted(by_condition.items())
        },
        "by_direction_condition": {
            key: summarize(value)
            for key, value in sorted(by_direction_condition.items())
        },
        "paired": paired_summary(scored),
    }
    output_path = args.output or args.input.with_name(
        args.input.name.replace(".generations.jsonl", ".metrics.json")
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {output_path.resolve()}")


if __name__ == "__main__":
    main()
