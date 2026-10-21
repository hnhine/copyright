#!/usr/bin/env python3
"""Per-document adaptive A2: ban the words the model refuses with, one document at a time.

Each DENY document is attacked on its own with a budget of Q queries, one
query being one generation. Round 1 bans the start words. After each round,
for every output of the document that leaked nothing, the first generated
token that contains a letter and is not banned yet is added to that
document's ban list (in lower, Title and UPPER case). A document stops as soon
as it leaks, when its budget is spent, or when its outputs give no new word to
ban. Only the document's own outputs are used, so the attack needs no tuning
split and runs directly on test.

With --attack A1 every round tries the three generic prefills, so a round
costs 3 queries and Q=16 allows 5 rounds. Leakage is scored on the
continuation only, never on the prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from adaptive_ban import case_variants, worst_case  # noqa: E402
from attacks import banned_sequences  # noqa: E402
from chat_render import assert_date_independent, render_chat  # noqa: E402
from metrics import score_record  # noqa: E402
from prompts import index_pairs, select_pairs  # noqa: E402
from run_behavior import (  # noqa: E402
    GENERATION_SCHEMA_VERSION,
    adapter_fingerprint,
    build_requests,
    read_jsonl,
    resolve_dtype,
    split_generated_ids,
    stable_hash,
)

LETTER_RE = re.compile(r"[^\W\d_]")


class PerRowBan:
    """LogitsProcessor that applies a different ban list to every batch row.

    Same semantics as bad_words_ids: a one-token sequence is banned at every
    step, a longer one only bans its last token right after the others.
    """

    def __init__(self, rows: list[list[list[int]]]):
        self.singles, self.multi = [], []
        for sequences in rows:
            self.singles.append(sorted({s[0] for s in sequences if len(s) == 1}))
            self.multi.append([(tuple(s[:-1]), s[-1]) for s in sequences if len(s) > 1])
        self.tail = max((len(p) for row in self.multi for p, _ in row), default=0)

    def __call__(self, input_ids, scores):
        tails = input_ids[:, -self.tail:].tolist() if self.tail else None
        for row, singles in enumerate(self.singles):
            if singles:
                scores[row, singles] = float("-inf")
            for prefix, last in self.multi[row]:
                if tuple(tails[row][-len(prefix):]) == prefix:
                    scores[row, last] = float("-inf")
        return scores


def next_ban_word(tokenizer, text: str, banned: set[str]) -> str | None:
    """First generated token that contains a letter and is not banned yet."""
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        piece = tokenizer.decode([token_id]).strip()
        if LETTER_RE.search(piece) and piece not in banned:
            return piece
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--pairs-per-direction", type=int, required=True)
    parser.add_argument("--attack", choices=("none", "A1"), default="none",
                        help="Prefill attack combined with banning (none = A2, A1 = A1+A2)")
    parser.add_argument("--budget", type=int, default=16, help="Q: queries (generations) per document")
    parser.add_argument("--start-words", nargs="+", default=["Access"])
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def load(config: dict, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(config.get("generation_seed", config["seed"]))
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=revision, cache_dir=config["cache_dir"], token=token
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=revision, cache_dir=config["cache_dir"], token=token,
        dtype=resolve_dtype(config["dtype"], torch),
        attn_implementation=config.get("attn_implementation", "sdpa"),
        device_map="auto", low_cpu_mem_usage=True,
    )
    adapter = adapter_fingerprint(config)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter["path"]).merge_and_unload()
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return tokenizer, model


def generate(tokenizer, model, config, items, batch_size, ban_lists) -> list[dict]:
    """Greedy generation where item i uses ban_lists[i]."""
    import torch
    from transformers import LogitsProcessorList

    records = []
    kwargs = config.get("chat_template_kwargs", {})
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        bans = ban_lists[start : start + batch_size]
        rendered = [
            render_chat(tokenizer, item["messages"], add_generation_prompt=True,
                        chat_template_kwargs=kwargs) + item["prefill"]
            for item in batch
        ]
        inputs = tokenizer(rendered, add_special_tokens=False, padding=True,
                           return_tensors="pt").to(model.device)
        width = inputs["input_ids"].shape[1]
        processor = PerRowBan([banned_sequences(tokenizer, sorted(words)) for words in bans])
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(
                **inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([processor]), use_cache=True,
            )
        seconds = time.perf_counter() - started
        for index, (item, words, prompt, sequence) in enumerate(zip(batch, bans, rendered, sequences)):
            token_ids, finish_reason = split_generated_ids(
                sequence[width:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
            )
            record = {key: value for key, value in item.items() if key != "messages"}
            record.update({
                "ban_words": sorted(words),
                "generation_schema_version": GENERATION_SCHEMA_VERSION,
                "rendered_prompt": prompt,
                "prompt_token_count": int(inputs["attention_mask"][index].sum().item()),
                "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "generated_token_count": len(token_ids),
                "finish_reason": finish_reason,
                "batch_latency_seconds": seconds,
            })
            records.append(record)
        print(f"[generate] {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return records


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
    if args.max_new_tokens:
        config["max_new_tokens"] = args.max_new_tokens
    batch_size = args.batch_size or config["batch_size"]
    revision = config.get("model_revision") or "main"

    pairs = index_pairs(read_jsonl(Path(config["input_dir"]) / f"{args.split}.jsonl"))
    selected = select_pairs(pairs, args.pairs_per_direction, config["seed"])
    template = build_requests(selected, ["DENY"], args.attack)
    by_doc = defaultdict(list)
    for request in template:
        by_doc[request["pair_id"]].append(request)
    per_round = len(next(iter(by_doc.values())))
    rounds = args.budget // per_round
    if rounds < 1:
        raise SystemExit(f"[error] budget {args.budget} is below one round ({per_round} queries)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "adaptive_doc.py", "split": args.split, "attack": args.attack,
        "budget": args.budget, "queries_per_round": per_round, "rounds": rounds,
        "start_words": args.start_words, "documents": len(by_doc),
        "model_id": config["model_id"], "model_revision": revision,
        "prompt_version": config["prompt_version"], "dtype": config["dtype"],
        "max_new_tokens": config["max_new_tokens"],
        "chat_template_kwargs": config.get("chat_template_kwargs", {}),
        "adapter": adapter_fingerprint(config),
        "pairs": sorted(by_doc),
    }
    manifest_path = args.output_dir / f"{args.run_prefix}.manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit(f"[error] {manifest_path} was written with other settings; use a new --run-prefix")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fingerprint = stable_hash(manifest)

    tokenizer = model = None
    banned = {doc: set().union(*(case_variants(w) for w in args.start_words)) for doc in by_doc}
    active, leaked = set(by_doc), set()
    all_rows, history = [], []
    for round_index in range(1, rounds + 1):
        if not active:
            break
        path = args.output_dir / f"{args.run_prefix}_r{round_index}.generations.jsonl"
        items = [dict(request, run_id=f"{request['run_id']}:r{round_index}", round=round_index,
                      attack="A2" if args.attack == "none" else f"{args.attack}+A2",
                      run_fingerprint=fingerprint)
                 for doc in sorted(active) for request in by_doc[doc]]
        rows = read_jsonl(path) if path.exists() else []
        if [row["run_id"] for row in rows] != [item["run_id"] for item in items] or any(
            row.get("run_fingerprint") != fingerprint for row in rows
        ):
            if tokenizer is None:
                tokenizer, model = load(config, revision)
                assert_date_independent(tokenizer, items[0]["messages"], config.get("chat_template_kwargs", {}))
            rows = generate(tokenizer, model, config, items, batch_size,
                            [banned[item["pair_id"]] for item in items])
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            temporary.replace(path)
        elif tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=revision,
                                                      cache_dir=config["cache_dir"])
        scored = [score_record(row) for row in rows]
        all_rows.extend(scored)

        stopped = 0
        for doc in sorted(active):
            outputs = [row for row in scored if row["pair_id"] == doc]
            if any(row["any_canary_leak"] for row in outputs):
                leaked.add(doc)
                active.discard(doc)
                continue
            new = {w for row in outputs for w in [next_ban_word(tokenizer, row["generated_text"], banned[doc])] if w}
            if not new:
                active.discard(doc)
                stopped += 1
            for word in new:
                banned[doc] |= case_variants(word)

        record = {
            "round": round_index,
            "queries": round_index * per_round,
            "documents_attacked": len(items) // per_round,
            "leaked_this_round": sum(1 for d in {r["pair_id"] for r in scored} if d in leaked),
            "stopped_no_new_word": stopped,
            "cumulative_worst_case": worst_case([all_rows]),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    summary = args.output_dir / f"{args.run_prefix}.summary.json"
    summary.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_bans = args.output_dir / f"{args.run_prefix}.ban_words.json"
    final_bans.write_text(json.dumps({d: sorted(w) for d, w in banned.items()}, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    print(f"[done] {summary.resolve()}")


if __name__ == "__main__":
    main()
