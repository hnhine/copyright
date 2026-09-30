"""Tokenize SFT chat examples with the loss restricted to assistant tokens."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from chat_render import assert_date_independent, render_chat  # noqa: E402,F401


IGNORE_INDEX = -100


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_example(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    chat_template_kwargs: dict | None = None,
) -> dict[str, list[int]]:
    """Return input_ids and labels; labels are IGNORE_INDEX outside the answer.

    The prompt is rendered with the generation prompt, exactly as at
    evaluation time, and must be a token prefix of the full conversation.
    Everything after it (the answer plus the template's end-of-turn tokens)
    is supervised, so the model also learns to stop after "Access denied.".
    """
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected system, user, assistant messages")
    prompt_ids = list(
        render_chat(
            tokenizer,
            messages[:2],
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
            tokenize=True,
        )
    )
    full_ids = list(
        render_chat(
            tokenizer,
            messages,
            add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
            tokenize=True,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Rendered prompt is not a token prefix of the full conversation")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("Conversation has no assistant tokens to supervise")
    if len(full_ids) > max_length:
        raise ValueError(f"Example has {len(full_ids)} tokens > max_length={max_length}")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


def load_examples(
    path: Path,
    tokenizer,
    max_length: int,
    chat_template_kwargs: dict | None = None,
    limit: int | None = None,
) -> list[dict]:
    examples = []
    for row in read_jsonl(path):
        encoded = encode_example(
            tokenizer, row["messages"], max_length, chat_template_kwargs
        )
        examples.append(
            {
                **encoded,
                "id": row["id"],
                "permission": row["permission"],
                "task": row.get("task", "translate"),
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


class Collator:
    """Right-pad a list of encoded examples into tensors."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        width = max(len(example["input_ids"]) for example in examples)
        input_ids, labels, attention_mask = [], [], []
        for example in examples:
            padding = width - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            labels.append(example["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
