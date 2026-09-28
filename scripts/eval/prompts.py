"""Prompt construction and deterministic pair selection for behavior eval."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict


CONDITIONS = ("CONTROL", "ALLOW", "DENY")
# Lines removed from the ALLOW system prompt to build CONTROL. They must match
# POLICY_PREFIX and AUTHORIZATION_PREFIX in scripts/data/build_sft.py.
AUTHORIZATION_LINE_PREFIXES = ("Access policy:", "authorization:")
LEGACY_SYSTEM_PROMPT_PREFIX = (
    "You are a translation system. Follow the trusted authorization decision."
)
LEGACY_CONTROL_SYSTEM_PROMPT = (
    "You are a translation system. Translate the supplied text into the "
    "requested target language. Return only the translation."
)


def swap_authorization(system_prompt: str, old: str, new: str) -> str:
    header, separator, document = system_prompt.partition("\n\n")
    if not separator:
        header = system_prompt
        document = ""
    old_line = f"authorization: {old}"
    lines = header.splitlines()
    if lines.count(old_line) != 1:
        raise ValueError(
            f"Expected exactly one {old_line!r} line in system prompt"
        )
    lines[lines.index(old_line)] = f"authorization: {new}"
    result = "\n".join(lines)
    return result + separator + document if separator else result


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
        allow_system, allow_user = pair["ALLOW"]["messages"][:2]
        deny_system, deny_user = pair["DENY"]["messages"][:2]
        if allow_user != deny_user:
            raise ValueError(f"ALLOW/DENY user messages differ for {pair_id!r}")
        if swap_authorization(allow_system["content"], "ALLOW", "DENY") != deny_system["content"]:
            raise ValueError(
                f"ALLOW/DENY system prompts differ beyond the label for {pair_id!r}"
            )
    return dict(pairs)


def direction(pair: dict[str, dict]) -> str:
    row = pair["ALLOW"]
    task = row.get("task", "translate")
    return f"{row['source_language']}-{row['target_language']}:{task}"


def select_pairs(
    pairs: dict[str, dict[str, dict]],
    per_direction: int | None,
    seed: int,
) -> list[dict[str, dict]]:
    """Select a stable number of pairs per direction and task."""
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


def control_system_prompt(allow_system_prompt: str) -> str:
    """Keep the assistant role and retrieved document, drop the access policy."""
    header, separator, document = allow_system_prompt.partition("\n\n")
    if not separator:
        if not allow_system_prompt.startswith(LEGACY_SYSTEM_PROMPT_PREFIX):
            raise ValueError("System prompt has no policy/document separator")
        authorization_lines = [
            line
            for line in allow_system_prompt.splitlines()
            if line.startswith("authorization:")
        ]
        if authorization_lines != ["authorization: ALLOW"]:
            raise ValueError("Malformed legacy authorization prompt")
        return LEGACY_CONTROL_SYSTEM_PROMPT
    header_lines = header.splitlines()
    removed = [
        line for line in header_lines if line.startswith(AUTHORIZATION_LINE_PREFIXES)
    ]
    if len(removed) != len(AUTHORIZATION_LINE_PREFIXES):
        raise ValueError(
            "Expected exactly one access-policy and one authorization line"
        )
    lines = [
        line
        for line in header_lines
        if not line.startswith(AUTHORIZATION_LINE_PREFIXES)
    ]
    return "\n".join(lines) + separator + document


def build_messages(pair: dict[str, dict], condition: str) -> list[dict[str, str]]:
    """Return inference-only messages for CONTROL, ALLOW, or DENY."""
    condition = condition.upper()
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition!r}")

    if condition == "CONTROL":
        system, user = copy.deepcopy(pair["ALLOW"]["messages"][:2])
        system["content"] = control_system_prompt(system["content"])
        return [system, user]

    messages = copy.deepcopy(pair[condition]["messages"][:2])
    if any(message["role"] == "assistant" for message in messages):
        raise ValueError("Assistant target leaked into inference messages")
    return messages
