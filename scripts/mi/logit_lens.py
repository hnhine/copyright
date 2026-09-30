

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from chat_render import assert_date_independent, render_chat  # noqa: E402
from prompts import build_messages, index_pairs, select_pairs  # noqa: E402

CANONICAL_DENIAL = "Access denied."


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Eval config (model, revision, template)")
    parser.add_argument("--adapter-path", help="LoRA adapter to merge; omit for the base model")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pairs-per-direction", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True, help="Per-example JSONL output")
    return parser.parse_args()


def first_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Empty tokenization for {text!r}")
    return ids[0]


def decoder_layers(model):
    return model.model.layers


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    chat_template_kwargs = config.get("chat_template_kwargs", {})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = config.get("model_revision") or "main"
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=revision, cache_dir=config["cache_dir"], token=token
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=revision,
        cache_dir=config["cache_dir"],
        token=token,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
        device_map={"": 0},
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.eval()

    pairs = index_pairs(read_jsonl(Path(config["input_dir"]) / f"{args.split}.jsonl"))
    selected = select_pairs(pairs, args.pairs_per_direction, config["seed"])
    items = []
    for pair in selected:
        row = pair["ALLOW"]
        for condition in ("ALLOW", "DENY"):
            items.append(
                {
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "task": row["task"],
                    "target_language": row["target_language"],
                    "messages": build_messages(pair, condition),
                    "content_token": first_token(tokenizer, row["gold_text"]),
                }
            )
    assert_date_independent(tokenizer, items[0]["messages"], chat_template_kwargs)
    refusal_token = first_token(tokenizer, CANONICAL_DENIAL)

    # Capture the residual stream after every decoder layer at the last position.
    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach()

        return hook

    layers = decoder_layers(model)
    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    norm, lm_head = model.model.norm, model.lm_head

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            prompts = [
                render_chat(
                    tokenizer, item["messages"], add_generation_prompt=True,
                    chat_template_kwargs=chat_template_kwargs,
                )
                for item in batch
            ]
            inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
            captured.clear()
            final_logits = model(**inputs).logits[:, -1, :].float()
            stacked = torch.stack([captured[i] for i in range(len(layers))], dim=1)  # batch, layers, hidden
            lens_logits = lm_head(norm(stacked)).float()  # batch, layers, vocab
            # The lens at the last layer must reproduce the model's own logits.
            if not torch.allclose(lens_logits[:, -1], final_logits, atol=1e-2, rtol=1e-2):
                raise RuntimeError("Logit lens at the last layer does not match model logits")
            log_probs = torch.log_softmax(lens_logits, dim=-1)
            for row, item in enumerate(batch):
                content, refusal = item["content_token"], refusal_token
                logits = lens_logits[row]
                record = {key: item[key] for key in ("pair_id", "condition", "task", "target_language")}
                record.update(
                    {
                        "content_token": tokenizer.decode([content]),
                        "refusal_token": tokenizer.decode([refusal]),
                        "content_rank": (logits > logits[:, content : content + 1]).sum(-1).tolist(),
                        "refusal_rank": (logits > logits[:, refusal : refusal + 1]).sum(-1).tolist(),
                        "content_logprob": log_probs[row, :, content].tolist(),
                        "refusal_logprob": log_probs[row, :, refusal].tolist(),
                        "top1": [tokenizer.decode([t]) for t in logits.argmax(-1).tolist()],
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    for handle in handles:
        handle.remove()
    summarize(args.output)


def summarize(path: Path) -> None:
    rows = read_jsonl(path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    layers = len(rows[0]["content_rank"])
    print(f"[lens] {len(rows)} prompts, {layers} layers; ranks are 0-based (0 = top-1)")
    header = f"{'layer':>5} | " + " | ".join(
        f"{c}: med rank content/refusal, P(content>refusal)" for c in sorted(groups)
    )
    print(header)
    for layer in range(layers):
        cells = []
        for condition in sorted(groups):
            group = groups[condition]
            content = median(r["content_rank"][layer] for r in group)
            refusal = median(r["refusal_rank"][layer] for r in group)
            ahead = sum(
                r["content_logprob"][layer] > r["refusal_logprob"][layer] for r in group
            ) / len(group)
            cells.append(f"{condition}: {content:>7.0f} / {refusal:>7.0f}, {ahead:4.0%}")
        print(f"{layer:>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
