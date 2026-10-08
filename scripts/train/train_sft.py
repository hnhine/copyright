#!/usr/bin/env python3
"""LoRA supervised fine-tuning on the ALLOW/DENY RAG chat data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_data import (  # noqa: E402
    IGNORE_INDEX,
    Collator,
    assert_date_independent,
    file_sha256,
    load_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Override config seed")
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir")
    parser.add_argument(
        "--max-steps", type=int, help="Stop after this many optimizer steps (smoke test)"
    )
    parser.add_argument(
        "--train-limit", type=int, help="Use only the first N training examples"
    )
    return parser.parse_args()


def resolve_dtype(name: str, torch):
    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return values[name]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batches(examples: list[dict], size: int, seed: int) -> list[list[dict]]:
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [[examples[i] for i in order[start : start + size]] for start in range(0, len(order), size)]


def evaluate(model, examples: list[dict], collator: Collator, batch_size: int, torch) -> dict:
    """Token-level validation loss, overall and per permission."""
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = {key: value.to(model.device) for key, value in collator(chunk).items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits[:, :-1].float()
            labels = batch["labels"][:, 1:]
            token_loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, ignore_index=IGNORE_INDEX, reduction="none"
            )
            mask = labels != IGNORE_INDEX
            for row, example in enumerate(chunk):
                loss_sum = token_loss[row][mask[row]].sum().item()
                count = mask[row].sum().item()
                for key in ("all", example["permission"]):
                    bucket = totals.setdefault(key, [0.0, 0])
                    bucket[0] += loss_sum
                    bucket[1] += count
    model.train()
    return {f"val_loss_{key.lower()}": total / max(count, 1) for key, (total, count) in totals.items()}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else config["seed"]
    output_dir = args.output_dir or Path(config["output_dir"].format(seed=seed))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

    set_seed(seed)
    token = os.environ.get("HF_TOKEN") or None
    revision = config.get("model_revision") or "main"
    chat_template_kwargs = config.get("chat_template_kwargs", {})
    cache_dir = Path(config["cache_dir"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=revision, cache_dir=cache_dir, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = Path(config["train_path"])
    validation_path = Path(config["validation_path"])
    train_examples = load_examples(
        train_path, tokenizer, config["max_length"], chat_template_kwargs, args.train_limit
    )
    validation_examples = load_examples(
        validation_path, tokenizer, config["max_length"], chat_template_kwargs,
        config.get("validation_limit"),
    )
    first_messages = json.loads(train_path.open(encoding="utf-8").readline())["messages"][:2]
    assert_date_independent(tokenizer, first_messages, chat_template_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        dtype=resolve_dtype(config["dtype"], torch),
        attn_implementation=config.get("attn_implementation", "sdpa"),
        device_map={"": 0},
    )
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    micro = config["micro_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    epochs = config["epochs"]
    micro_batches_per_epoch = math.ceil(len(train_examples) / micro)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(config["warmup_ratio"] * total_steps), total_steps
    )
    collator = Collator(tokenizer.pad_token_id)

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config": config,
        "seed": seed,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "train_path": str(train_path),
        "train_sha256": file_sha256(train_path),
        "validation_path": str(validation_path),
        "validation_sha256": file_sha256(validation_path),
        "train_examples": len(train_examples),
        "train_limit": args.train_limit,
        "validation_examples": len(validation_examples),
        "truncated_train_examples": sum(e["truncated"] for e in train_examples),
        "truncated_validation_examples": sum(e["truncated"] for e in validation_examples),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "max_steps": args.max_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_path = output_dir / "train_log.jsonl"
    print(
        f"[train] examples={len(train_examples)} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps} trainable={trainable:,}/{total:,}"
    )

    step = 0
    window_loss, window_tokens, window_started = 0.0, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        micro_batches = batches(train_examples, micro, seed * 1000 + epoch)
        optimizer.zero_grad(set_to_none=True)
        for index, chunk in enumerate(micro_batches, start=1):
            batch = {key: value.to(model.device) for key, value in collator(chunk).items()}
            loss = model(**batch).loss
            (loss / accumulation).backward()
            window_loss += loss.item()
            window_tokens += int(batch["attention_mask"].sum().item())

            if index % accumulation and index != len(micro_batches):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % config["log_every"] == 0:
                elapsed = time.perf_counter() - window_started
                row = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": window_loss / (config["log_every"] * accumulation),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens_per_second": window_tokens / elapsed,
                }
                append_jsonl(log_path, row)
                print(json.dumps(row))
                window_loss, window_tokens, window_started = 0.0, 0, time.perf_counter()
            if step % config["eval_every"] == 0:
                row = {"step": step, "epoch": epoch, **evaluate(model, validation_examples, collator, micro, torch)}
                append_jsonl(log_path, row)
                print(json.dumps(row))
            if step >= total_steps:
                break

        row = {"step": step, "epoch": epoch, "end_of_epoch": True,
               **evaluate(model, validation_examples, collator, micro, torch)}
        append_jsonl(log_path, row)
        print(json.dumps(row))
        adapter_dir = output_dir / f"epoch{epoch}" / "adapter"
        model.save_pretrained(adapter_dir)
        print(f"[save] {adapter_dir}")
        if step >= total_steps:
            break

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_steps"] = step
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
