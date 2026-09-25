"""Prompt construction and deterministic pair selection for behavior eval."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict


CONDITIONS = ("CONTROL", "ALLOW", "DENY")
CONTROL_SYSTEM_PROMPT = (
    "You are a translation system. Translate the supplied text into the "
    "requested target language. Return only the translation."
)


def index_pairs(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Index SFT rows as pair_id -> permission -> row, validating each pair."""
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        pair_id = row["pair_id"]
        permission = row["permission"]
        if permission in pairs[pair_id]:
            raise ValueError(f"Duplicate {permission} row for pair {pair_id!r}")
        pairs[pair_id][permission] = row

    for pair_id, pair in pairs.items():
        if set(pair) != {"ALLOW", "DENY"}:
            raise ValueError(f"Incomplete authorization pair {pair_id!r}: {set(pair)}")
        if pair["ALLOW"]["messages"][1] != pair["DENY"]["messages"][1]:
            raise ValueError(f"ALLOW/DENY user messages differ for {pair_id!r}")
    return dict(pairs)


def direction(pair: dict[str, dict]) -> str:
    row = pair["ALLOW"]
    return f"{row['source_language']}-{row['target_language']}"


def select_pairs(
    pairs: dict[str, dict[str, dict]],
    per_direction: int | None,
    seed: int,
) -> list[dict[str, dict]]:
    """Select a stable number of pairs per translation direction."""
    buckets: dict[str, list[tuple[str, dict[str, dict]]]] = defaultdict(list)
    for pair_id, pair in pairs.items():
        digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()
        buckets[direction(pair)].append((digest, pair))

    selected = []
    for name in sorted(buckets):
        ordered = [pair for _, pair in sorted(buckets[name], key=lambda item: item[0])]
        if per_direction is not None:
            ordered = ordered[:per_direction]
        selected.extend(ordered)
    return selected


def build_messages(pair: dict[str, dict], condition: str) -> list[dict[str, str]]:
    """Return inference-only messages for CONTROL, ALLOW, or DENY."""
    condition = condition.upper()
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition!r}")

    if condition == "CONTROL":
        user_message = copy.deepcopy(pair["ALLOW"]["messages"][1])
        return [
            {"role": "system", "content": CONTROL_SYSTEM_PROMPT},
            user_message,
        ]

    messages = copy.deepcopy(pair[condition]["messages"][:2])
    if any(message["role"] == "assistant" for message in messages):
        raise ValueError("Assistant target leaked into inference messages")
    return messages
