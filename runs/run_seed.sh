#!/usr/bin/env bash
# Train and evaluate B1 (SFT) and B2 (SFT + recovery augmentation) for Qwen and
# Llama with another training seed on the current data, then compare all seeds.
# Usage: bash runs/run_seed.sh 1
# Safe to re-run: finished steps are skipped.
set -euo pipefail
SEED=${1:?usage: bash runs/run_seed.sh SEED}
cd /workspace/nhi/copyright
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS=("qwen25_3b qwen2.5-3b-instruct" "llama32_3b llama-3.2-3b-instruct")

python -m unittest discover -s tests -p 'test_*.py'

generate () {   # $1 eval config, $2 output dir, $3 run name, rest: extra args
  local cfg=$1 out=$2 run=$3; shift 3
  python scripts/eval/run_behavior.py --config "$cfg" --output-dir "$out" --run-name "$run" "$@"
  python scripts/eval/score_behavior.py --input "$out/$run.generations.jsonl" > /dev/null
}

adaptive () {   # $1 eval config, $2 adapter, $3 output dir, $4 prefix, $5 rounds, rest: extra args
  local cfg=$1 adapter=$2 out=$3 prefix=$4 rounds=$5; shift 5
  if [ -f "$out/$prefix.summary.json" ]; then echo "[skip] $out/$prefix"; return; fi
  python scripts/eval/adaptive_ban.py --config "$cfg" --adapter-path "$adapter" \
    --output-dir "$out" --run-prefix "$prefix" --rounds "$rounds" "$@"
}

run_variant () {   # $1 model prefix, $2 model name, $3 variant (sft | sft_aug)
  local prefix=$1 name=$2 variant=$3
  local cfg=configs/eval/${prefix}_rag_pre_sft.json
  local train_dir=outputs/train/$name/${variant}_seed$SEED
  local out_base=outputs/behavior/$name/rag_${variant}_seed$SEED

  echo "=================== $name / $variant / seed $SEED"
  if [ -f "$train_dir/epoch2/adapter/adapter_model.safetensors" ]; then
    echo "[skip] training already finished: $train_dir"
  elif [ -d "$train_dir" ] && [ -n "$(ls -A "$train_dir")" ]; then
    echo "[error] $train_dir exists but training did not finish; delete it and re-run" >&2
    exit 1
  else
    python -u scripts/train/train_sft.py --config configs/train/${prefix}_${variant}.json \
      --seed "$SEED" 2>&1 | tee runs/train_${name}_${variant}_seed$SEED.log
  fi

  for e in 1 2; do
    generate "$cfg" "${out_base}_epoch$e" val_50 --split validation --pairs-per-direction 50 \
      --adapter-path "$train_dir/epoch$e/adapter"
  done
  local E
  E=$(python - "$out_base" <<'EOF'
import json, sys
base = sys.argv[1]
rates = {e: json.load(open(f"{base}_epoch{e}/val_50.metrics.json"))["paired"]["pair_success_rate"] for e in (1, 2)}
print(2 if rates[2] > rates[1] else 1, end="")
EOF
)
  echo "$E" > "${out_base}_chosen_epoch.txt"
  echo "[select] $name / $variant / seed $SEED -> epoch $E"
  local out=${out_base}_epoch$E adapter=$train_dir/epoch$E/adapter

  generate "$cfg" "$out" test_100 --split test --pairs-per-direction 100 --adapter-path "$adapter"
  for atk in A1 A3; do
    generate "$cfg" "$out" "val_50_$atk" --split validation --pairs-per-direction 50 \
      --conditions DENY --attack $atk --adapter-path "$adapter"
  done
  adaptive "$cfg" "$adapter" "$out" val_50_adaptive_A2 8
  adaptive "$cfg" "$adapter" "$out" val_50_adaptive_A1A2 6 --attack A1
}

for model in "${MODELS[@]}"; do
  set -- $model
  run_variant "$1" "$2" sft
  run_variant "$1" "$2" sft_aug
done

# ---------------------------------------------------------------- summary over all seeds
python - <<'EOF'
import json
from pathlib import Path

def rate(path, key):
    return json.load(open(path))[key]

for name in ("qwen2.5-3b-instruct", "llama-3.2-3b-instruct"):
    base = Path(f"outputs/behavior/{name}")
    print(f"\n######## {name}")
    for label, variant in (("B1 SFT", "sft"), ("B2 SFT+aug", "sft_aug")):
        for chosen in sorted(base.glob(f"rag_{variant}_seed*_chosen_epoch.txt")):
            seed = chosen.name.split("_seed")[1].split("_")[0]
            d = base / f"rag_{variant}_seed{seed}_epoch{chosen.read_text().strip()}"
            if not (d / "val_50_adaptive_A1A2.summary.json").exists():
                continue
            test = json.load(open(d / "test_100.metrics.json"))
            allow = test["by_condition"]["ALLOW"]["task_success_rate"]
            deny = test["by_condition"]["DENY"]["any_canary_leak_rate"]
            a1 = json.load(open(d / "val_50_A1.metrics.json"))["deny_worst_case"]["A1:all"]["any_canary_leak_rate"]
            a3 = json.load(open(d / "val_50_A3.metrics.json"))["deny_worst_case"]["A3:all"]["any_canary_leak_rate"]
            a2 = json.load(open(d / "val_50_adaptive_A2.summary.json"))[-1]["cumulative_worst_case"]["all"]
            a12 = json.load(open(d / "val_50_adaptive_A1A2.summary.json"))[-1]["cumulative_worst_case"]["all"]
            print(f"  {label:11s} seed {seed}: ALLOW task {allow:4.0%} | DENY leak: direct {deny:3.0%}"
                  f"  A1 {a1:4.0%}  A3 {a3:4.0%}  A2 {a2:4.0%}  A1+A2 {a12:4.0%}")
EOF
