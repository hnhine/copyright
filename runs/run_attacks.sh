#!/usr/bin/env bash
# Attack suite for B0 (prompt only), B1 (SFT) and B2 (SFT + recovery augmentation)
# on Qwen and Llama. B1/B2 use the final checkpoint (epoch 2) of every seed;
# nothing is retrained. Every attack adapts per document from its own outputs,
# so no split is used for tuning and each split is evaluated the same way:
#   A0      direct request (ALLOW + DENY)             -> utility and direct leak
#   A1      the 3 generic prefills of attacks.py
#   oracle  gold text up to the first canary ("A3" in attacks.py); outside the
#           threat model, reported only as an upper bound
#   A2      per-document adaptive token banning, Q queries per document
#   A1+A2   A2 with the 3 generic prefills in every round (3 queries per round)
# Pairs are selected per task by a fixed hash order, so every task gets the
# same number of documents and larger N contain the smaller ones.
# Usage: bash runs/run_attacks.sh [SEED ...]                     (default seeds 0 1 2)
#   NV=100 NT=200 bash runs/run_attacks.sh 0 1 2 2>&1 | tee runs/run_attacks.log
# NV=0 skips validation (a dev check only; nothing is tuned on it).
# Safe to re-run: finished steps are skipped.
set -euo pipefail
NV=${NV:-100}
NT=${NT:-200}
Q=${Q:-16}
BATCH_SIZE=${BATCH_SIZE:-64}
# Generation caps: canaries sit in the header and every leak seen so far starts
# within 79 tokens; 256 leaves room for canaries placed in the document body.
# A0 measures utility, so it must fit the longest gold answer (409 tokens).
ATTACK_TOKENS=${ATTACK_TOKENS:-256}
A0_TOKENS=${A0_TOKENS:-512}
EPOCH=2
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(0 1 2); fi
cd /workspace/nhi/copyright
# Project venv, pinned to the versions used for training (requirements.txt).
export PATH="$PWD/.venv/bin:$PATH"
python -c "import torch, transformers, peft; assert torch.__version__.startswith('2.4.1'), torch.__version__"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS=("qwen25_3b qwen2.5-3b-instruct" "llama32_3b llama-3.2-3b-instruct")

python -m unittest discover -s tests -p 'test_*.py'

generate () {   # $1 eval config, $2 output dir, $3 run name, rest: extra args
  local cfg=$1 out=$2 run=$3; shift 3
  if [ -f "$out/$run.metrics.json" ]; then echo "[skip] $out/$run"; return; fi
  python scripts/eval/run_behavior.py --config "$cfg" --output-dir "$out" --run-name "$run" \
    --batch-size "$BATCH_SIZE" "$@"
  python scripts/eval/score_behavior.py --input "$out/$run.generations.jsonl" > /dev/null
}

adaptive () {   # $1 eval config, $2 output dir, $3 run prefix, rest: extra args
  local cfg=$1 out=$2 prefix=$3; shift 3
  if [ -f "$out/$prefix.summary.json" ]; then echo "[skip] $out/$prefix"; return; fi
  python scripts/eval/adaptive_doc.py --config "$cfg" --output-dir "$out" --run-prefix "$prefix" \
    --budget "$Q" --batch-size "$BATCH_SIZE" --max-new-tokens "$ATTACK_TOKENS" "$@"
}

suite () {   # $1 eval config, $2 output dir, $3 split, $4 pairs per task, rest: --adapter-path or nothing
  local cfg=$1 out=$2 split=$3 n=$4; shift 4
  local tag=$([ "$split" = test ] && echo test || echo val)_$n
  generate "$cfg" "$out" "$tag" --split "$split" --pairs-per-direction "$n" --conditions ALLOW DENY \
    --max-new-tokens "$A0_TOKENS" "$@"
  for atk in A1 A3; do
    generate "$cfg" "$out" "${tag}_$atk" --split "$split" --pairs-per-direction "$n" \
      --conditions DENY --attack $atk --max-new-tokens "$ATTACK_TOKENS" "$@"
  done
  adaptive "$cfg" "$out" "${tag}_A2_q$Q" --split "$split" --pairs-per-direction "$n" "$@"
  adaptive "$cfg" "$out" "${tag}_A1A2_q$Q" --split "$split" --pairs-per-direction "$n" --attack A1 "$@"
}

run_method () {   # $1 eval config, $2 output dir, rest: --adapter-path or nothing
  local cfg=$1 out=$2; shift 2
  echo "=================== $out"
  if [ "$NV" -gt 0 ]; then suite "$cfg" "$out" validation "$NV" "$@"; fi
  suite "$cfg" "$out" test "$NT" "$@"
}

# B0 has no training seed: run it once per model.
for model in "${MODELS[@]}"; do
  set -- $model
  run_method configs/eval/$1_rag_pre_sft.json outputs/behavior/$2/rag_pre_sft
done

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    set -- $model
    for variant in sft sft_aug; do
      adapter=outputs/train/$2/${variant}_seed$seed/epoch$EPOCH/adapter
      if [ ! -f "$adapter/adapter_model.safetensors" ]; then
        echo "[error] missing adapter $adapter" >&2
        exit 1
      fi
      run_method configs/eval/$1_rag_pre_sft.json \
        outputs/behavior/$2/rag_${variant}_seed${seed}_epoch$EPOCH --adapter-path "$adapter"
    done
  done
done

python runs/summarize_attacks.py --nv "$NV" --nt "$NT" --q "$Q" --epoch "$EPOCH"
