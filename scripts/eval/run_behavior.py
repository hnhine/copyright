#!/usr/bin/env python3
"""Generate CONTROL/ALLOW/DENY behavior outputs with an instruct model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from prompts import build_messages, index_pairs, select_pairs


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["run_id"] for row in read_jsonl(path)}


def build_requests(pairs: list[dict[str, dict]], conditions: list[str]) -> list[dict]:
    requests = []
    for pair in pairs:
        base = pair["ALLOW"]
        for condition in conditions:
            condition = condition.upper()
            requests.append(
                {
                    "run_id": f"{base['pair_id']}:{condition.lower()}",
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "source_language": base["source_language"],
                    "target_language": base["target_language"],
                    "source_text": base["source_text"],
                    "reference_text": base["reference_text"],
                    "messages": build_messages(pair, condition),
                }
            )
    return requests


def resolve_dtype(name: str, torch):
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return values[name]


def trim_generated_ids(
    ids: list[int], eos_token_id: int | None, pad_token_id: int | None
) -> list[int]:
    result = []
    for token_id in ids:
        if pad_token_id is not None and token_id == pad_token_id:
            break
        result.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--pairs-per-direction", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    per_direction = args.pairs_per_direction
    if per_direction is None and args.split == "validation":
        per_direction = config["validation_pairs_per_direction"]
    if per_direction is not None and per_direction <= 0:
        raise ValueError("--pairs-per-direction must be positive")

    batch_size = args.batch_size or config["batch_size"]
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    run_name = args.run_name or (
        "validation_pilot" if args.split == "validation" else "test"
    )
    input_path = Path(config["input_dir"]) / f"{args.split}.jsonl"
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.generations.jsonl"
    manifest_path = output_dir / f"{run_name}.manifest.json"

    pairs = index_pairs(read_jsonl(input_path))
    selected_pairs = select_pairs(pairs, per_direction, config["seed"])
    requests = build_requests(selected_pairs, config["conditions"])
    completed = existing_run_ids(output_path)
    pending = [request for request in requests if request["run_id"] not in completed]

    print(
        f"[data] split={args.split} pairs={len(selected_pairs)} "
        f"requests={len(requests)} pending={len(pending)}"
    )
    if not pending:
        print(f"[done] Nothing to generate: {output_path.resolve()}")
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    revision = config.get("model_revision") or "main"

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        dtype=resolve_dtype(config["dtype"], torch),
        attn_implementation=config.get("attn_implementation", "sdpa"),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if not config["do_sample"]:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    manifest = {
        "model_id": config["model_id"],
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "cache_dir": str(cache_dir.resolve()),
        "input": str(input_path),
        "split": args.split,
        "run_name": run_name,
        "prompt_version": config["prompt_version"],
        "conditions": config["conditions"],
        "pairs": len(selected_pairs),
        "requests": len(requests),
        "batch_size": batch_size,
        "dtype": config["dtype"],
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["do_sample"],
        "seed": config["seed"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as output:
        for start in tqdm(range(0, len(pending), batch_size), desc="generate"):
            batch = pending[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            inputs = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_width = inputs["input_ids"].shape[1]
            started = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **inputs,
                    do_sample=config["do_sample"],
                    max_new_tokens=config["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            batch_seconds = time.perf_counter() - started

            for index, (item, prompt, sequence) in enumerate(
                zip(batch, rendered, sequences)
            ):
                token_ids = trim_generated_ids(
                    sequence[prompt_width:].tolist(),
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                )
                generated_text = tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
                record = {
                    key: value for key, value in item.items() if key != "messages"
                }
                record.update(
                    {
                        "rendered_prompt": prompt,
                        "prompt_token_count": int(
                            inputs["attention_mask"][index].sum().item()
                        ),
                        "generated_text": generated_text,
                        "generated_token_count": len(token_ids),
                        "finish_reason": (
                            "eos"
                            if tokenizer.eos_token_id in token_ids
                            else "max_new_tokens"
                        ),
                        "batch_latency_seconds": batch_seconds,
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"[done] {output_path.resolve()}")


if __name__ == "__main__":
    main()
