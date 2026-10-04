#!/usr/bin/env python3
"""Generate CONTROL/ALLOW/DENY behavior outputs with an instruct model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from attacks import ATTACKS, prefills_for
from chat_render import assert_date_independent, render_chat
from prompts import build_messages, index_pairs, select_pairs


GENERATION_SCHEMA_VERSION = 2


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def adapter_fingerprint(config: dict) -> dict | None:
    """Identify a LoRA adapter by path and weight hash, or None for the base model."""
    adapter_path = config.get("adapter_path")
    if not adapter_path:
        return None
    weights = Path(adapter_path) / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"Missing adapter weights: {weights}")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(adapter_path), "sha256": digest.hexdigest()}

def add_request_fingerprints(
    requests: list[dict], config: dict, revision: str
) -> None:
    generation_settings = {
        "model_id": config["model_id"],
        "model_revision": revision,
        "prompt_version": config["prompt_version"],
        "dtype": config["dtype"],
        "attn_implementation": config.get("attn_implementation", "sdpa"),
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["do_sample"],
        "generation_seed": config.get("generation_seed", config["seed"]),
        "chat_template_kwargs": config.get("chat_template_kwargs", {}),
    }
    adapter = adapter_fingerprint(config)
    if adapter is not None:
        generation_settings["adapter"] = adapter
    for request in requests:
        request_payload = {
            key: value
            for key, value in request.items()
            if key != "request_fingerprint"
        }
        request["request_fingerprint"] = stable_hash(
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "request": request_payload,
                "generation": generation_settings,
            }
        )


def existing_run_ids(path: Path, requests: list[dict]) -> set[str]:
    if not path.exists():
        return set()

    expected = {request["run_id"]: request for request in requests}
    if len(expected) != len(requests):
        raise ValueError("Current request set contains duplicate run_id values")

    completed = set()
    for row in read_jsonl(path):
        run_id = row["run_id"]
        if run_id in completed:
            raise ValueError(f"Duplicate run_id in existing output: {run_id!r}")
        completed.add(run_id)
        if run_id not in expected:
            raise ValueError(
                f"Existing output contains unexpected run_id {run_id!r}; "
                "use a new --run-name"
            )
        actual = row.get("request_fingerprint")
        wanted = expected[run_id]["request_fingerprint"]
        if actual != wanted:
            reason = "missing" if actual is None else "different"
            raise ValueError(
                f"Existing output has {reason} request fingerprint for "
                f"{run_id!r}; use a new --run-name"
            )
    return completed


def build_requests(
    pairs: list[dict[str, dict]], conditions: list[str], attack: str = "none"
) -> list[dict]:
    conditions = [condition.upper() for condition in conditions]
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"Duplicate conditions: {conditions!r}")
    requests = []
    for pair in pairs:
        base = pair["ALLOW"]
        for condition in conditions:
            for prefill_id, prefill in prefills_for(attack, base):
                run_id = f"{base['pair_id']}:{condition.lower()}"
                if attack != "none":
                    run_id += f":{attack}:{prefill_id}"
                requests.append(
                    {
                        "run_id": run_id,
                        "pair_id": base["pair_id"],
                        "condition": condition,
                        "attack": attack,
                        "prefill_id": prefill_id,
                        "prefill": prefill,
                        "source_language": base["source_language"],
                        "target_language": base["target_language"],
                        "task": base.get("task", "translate"),
                        "prompt_version": base.get("prompt_version"),
                        "source_text": base["source_text"],
                        "reference_text": base.get("reference_text"),
                        "gold_text": base.get("gold_text", base.get("reference_text")),
                        "canaries": base.get("canaries"),
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


def split_generated_ids(
    ids: list[int], eos_token_id: int | None, pad_token_id: int | None
) -> tuple[list[int], str]:
    """Remove stop/pad tokens and report why generation ended.

    EOS must be checked before PAD because many tokenizers use the same ID for
    both. The returned token count consistently excludes both special tokens.
    """
    result = []
    for token_id in ids:
        if eos_token_id is not None and token_id == eos_token_id:
            return result, "eos"
        if pad_token_id is not None and token_id == pad_token_id:
            return result, "pad"
        result.append(token_id)
    return result, "max_new_tokens"


def trim_generated_ids(
    ids: list[int], eos_token_id: int | None, pad_token_id: int | None
) -> list[int]:
    return split_generated_ids(ids, eos_token_id, pad_token_id)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--pairs-per-direction", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--adapter-path", help="LoRA adapter directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument(
        "--attack",
        choices=ATTACKS,
        default="none",
        help="Assistant-prefill attack; leakage is scored on the continuation only",
    )
    parser.add_argument(
        "--conditions", nargs="+", help="Conditions to run (overrides config), e.g. DENY ALLOW"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.conditions:
        config["conditions"] = args.conditions


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
    prompt_versions = {
        pair["ALLOW"].get("prompt_version")
        for pair in selected_pairs
        if pair["ALLOW"].get("prompt_version") is not None
    }
    if prompt_versions and prompt_versions != {config["prompt_version"]}:
        raise ValueError(
            f"Config prompt_version={config['prompt_version']!r} does not match "
            f"selected data versions={sorted(prompt_versions)!r}"
        )
    revision = config.get("model_revision") or "main"
    requests = build_requests(selected_pairs, config["conditions"], args.attack)
    add_request_fingerprints(requests, config, revision)
    completed = existing_run_ids(output_path, requests)
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    generation_seed = config.get("generation_seed", config["seed"])
    set_seed(generation_seed)
    token = os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    chat_template_kwargs = config.get("chat_template_kwargs", {})
    assert_date_independent(tokenizer, pending[0]["messages"], chat_template_kwargs)

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

    adapter = adapter_fingerprint(config)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter["path"]).merge_and_unload()

    model.eval()
    if not config["do_sample"]:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    manifest = {
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "model_id": config["model_id"],
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "cache_dir": str(cache_dir.resolve()),
        "input": str(input_path),
        "split": args.split,
        "run_name": run_name,
        "prompt_version": config["prompt_version"],
        "conditions": config["conditions"],
        "attack": args.attack,
        "pairs": len(selected_pairs),
        "requests": len(requests),
        "batch_size": batch_size,
        "dtype": config["dtype"],
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["do_sample"],
        "seed": config["seed"],
        "generation_seed": generation_seed,
        "chat_template_kwargs": chat_template_kwargs,
        "adapter": adapter,
        "request_set_fingerprint": stable_hash(
            sorted(request["request_fingerprint"] for request in requests)
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as output:
        for start in tqdm(range(0, len(pending), batch_size), desc="generate"):
            batch = pending[start : start + batch_size]
            # The prefill is appended after the generation prompt, i.e. it is
            # the beginning of the assistant turn that the model continues.
            rendered = [
                render_chat(
                    tokenizer,
                    item["messages"],
                    add_generation_prompt=True,
                    chat_template_kwargs=chat_template_kwargs,
                )
                + item["prefill"]
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
                token_ids, finish_reason = split_generated_ids(
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
                        "generation_schema_version": GENERATION_SCHEMA_VERSION,
                        "rendered_prompt": prompt,
                        "prompt_token_count": int(
                            inputs["attention_mask"][index].sum().item()
                        ),
                        "generated_text": generated_text,
                        "generated_token_count": len(token_ids),
                        "finish_reason": finish_reason,
                        "batch_latency_seconds": batch_seconds,
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"[done] {output_path.resolve()}")


if __name__ == "__main__":
    main()
