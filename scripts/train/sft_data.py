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


def truncate(input_ids: list[int], labels: list[int], max_length: int) -> dict:
    """Cut an example to max_length tokens, never inside the prompt.

    Only the end of the answer is dropped, so a truncated example still
    teaches the start of the answer; it just loses its end-of-turn token.
    """
    supervised = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    if not supervised or supervised[0] >= max_length:
        raise ValueError(
            f"Prompt alone has {supervised[0] if supervised else len(labels)} tokens "
            f">= max_length={max_length}"
        )
    truncated = len(input_ids) > max_length
    return {
        "input_ids": input_ids[:max_length],
        "labels": labels[:max_length],
        "truncated": truncated,
    }


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
    response_prefix: str="",
) -> dict[str, list[int]]:
    """Return input_ids and labels; labels are IGNORE_INDEX outside the answer.

    The prompt is rendered with the generation prompt, exactly as at
    evaluation time, and must be a token prefix of the full conversation.
    Everything after it (the answer plus the template's end-of-turn tokens)
    is supervised, so the model also learns to stop after "Access denied.".
    """
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected system, user, assistant messages")
    if response_prefix:
        return encode_with_masked_prefix(tokenizer, messages, max_length, chat_template_kwargs, response_prefix)
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
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return truncate(full_ids, labels, max_length)


def encode_with_masked_prefix(
        tokenizer,
        messages: list[dict[str, str]],
        max_length: int,
        chat_template_kwargs: dict | None,
        response_prefix: str
) -> dict[str, list[int]]:
    if not messages[2]["content"].startswith(response_prefix):
        raise ValueError("Assistant content does not start with response_prefix")
    prompt_text = render_chat(
        tokenizer, messages[:2], add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_text = render_chat(
        tokenizer, messages, add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not full_text.startswith(prompt_text + response_prefix):
        raise ValueError("Rendered conversation does not start with prompt + prefix")
    boundary = len(prompt_text) + len(response_prefix)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    expected = list(
        render_chat(
            tokenizer, messages, add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs, tokenize=True,
        )
    )
    if input_ids != expected:
        raise ValueError("Offset tokenization differs from the chat template tokenization")
    labels = [
        token if end > boundary else IGNORE_INDEX
        for token, (_start, end) in zip(input_ids, encoded["offset_mapping"])
    ]
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("Conversation has no assistant tokens to supervise")
    return truncate(input_ids, labels, max_length)

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
            tokenizer, row["messages"], max_length, chat_template_kwargs, row.get("response_prefix", "")
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
