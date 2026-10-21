#!/usr/bin/env bash
# Rebuild the DGT data with paragraph re-alignment in the original locations and
# re-run everything for Qwen and Llama: pre-SFT baseline, B1 (SFT), B2 (SFT +
# recovery augmentation), checkpoint selection, test eval, and attacks A1, A3,
# adaptive A2, adaptive A1+A2.
#
# The old (line-aligned) data, adapters and eval outputs are moved to
# archive/line_aligned/ first, so the original paths and configs can be reused.
# Safe to re-run: finished steps are skipped.
set -euo pipefail
REPO=${REPO:-/workspace/nhi/copyright}
cd "$REPO"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS=("qwen25_3b qwen2.5-3b-instruct" "llama32_3b llama-3.2-3b-instruct")
ARCHIVE=archive/line_aligned

archive_path () {   # move $1 to $ARCHIVE/$1 if it exists
  if [ -e "$1" ]; then
    mkdir -p "$ARCHIVE/$(dirname "$1")"
    mv "$1" "$ARCHIVE/$1"
    echo "[archive] $1"
  fi
}

# ---------------------------------------------------------------- 1. archive
if [ ! -f "$ARCHIVE/.done" ]; then
  for d in dgt_acquis dgt_acquis_clean dgt_canary sft_rag sft_rag_aug; do
    archive_path "data/processed/$d"
  done
  for model in "${MODELS[@]}"; do
    set -- $model
    for d in sft_seed0 sft_aug_seed0 smoke; do
      archive_path "outputs/train/$2/$d"
    done
    for d in rag_pre_sft rag_sft_seed0_epoch1 rag_sft_seed0_epoch2 rag_sft_seed0_chosen_epoch.txt \
             rag_sft_aug_seed0_epoch1 rag_sft_aug_seed0_epoch2 rag_sft_aug_seed0_chosen_epoch.txt; do
      archive_path "outputs/behavior/$2/$d"
    done
  done
  for f in train_*.log; do [ -e "$f" ] && archive_path "$f"; done
  # Intermediate directories from the first realignment trial; rebuilt below.
  rm -rf data/processed/*_aligned data/processed/*_aligned_clean data/processed/*_aligned_aug
  mkdir -p "$ARCHIVE"; touch "$ARCHIVE/.done"
fi

# ---------------------------------------------------------------- 2. data
python -m unittest discover -s tests -p 'test_*.py'
if [ ! -f data/processed/sft_rag_aug/metadata.json ]; then
  python scripts/data/prepare_dgt.py --directions en-fr en-de
  python scripts/data/clean_dgt_languages.py
  python scripts/data/inject_canaries.py
  python scripts/data/build_sft.py
  python scripts/data/build_sft.py --augment-ratio 0.5 --output-dir data/processed/sft_rag_aug
fi

# ---------------------------------------------------------------- helpers
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
  local train_dir=outputs/train/$name/${variant}_seed0
  local out_base=outputs/behavior/$name/rag_${variant}_seed0

  echo "=================== $name / $variant"
  if [ -f "$train_dir/epoch2/adapter/adapter_model.safetensors" ]; then
    echo "[skip] training already finished: $train_dir"
  elif [ -d "$train_dir" ] && [ -n "$(ls -A "$train_dir")" ]; then
    echo "[error] $train_dir exists but training did not finish; delete it and re-run" >&2
    exit 1
  else
    python -u scripts/train/train_sft.py --config configs/train/${prefix}_${variant}.json \
      2>&1 | tee runs/train_${name}_${variant}_seed0.log
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
  echo "[select] $name / $variant -> epoch $E"
  local out=${out_base}_epoch$E adapter=$train_dir/epoch$E/adapter

  generate "$cfg" "$out" test_100 --split test --pairs-per-direction 100 --adapter-path "$adapter"
  for atk in A1 A3; do
    generate "$cfg" "$out" "val_50_$atk" --split validation --pairs-per-direction 50 \
      --conditions DENY --attack $atk --adapter-path "$adapter"
  done
  adaptive "$cfg" "$adapter" "$out" val_50_adaptive_A2 8
  adaptive "$cfg" "$adapter" "$out" val_50_adaptive_A1A2 6 --attack A1
}

# ---------------------------------------------------------------- 3. run
for model in "${MODELS[@]}"; do
  set -- $model
  echo "=================== $2 / pre-SFT"
  generate configs/eval/$1_rag_pre_sft.json outputs/behavior/$2/rag_pre_sft test_100 \
    --split test --pairs-per-direction 100
  run_variant "$1" "$2" sft
  run_variant "$1" "$2" sft_aug
done

# ---------------------------------------------------------------- 4. summary
python - <<'EOF'
import json, math
from pathlib import Path

def w(p, n, z=1.96):
    d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f"{100*p:3.0f}% [{100*(c-h):.0f}-{100*(c+h):.0f}]"

for name in ("qwen2.5-3b-instruct", "llama-3.2-3b-instruct"):
    base = Path(f"outputs/behavior/{name}")
    runs = {"pre-SFT": base / "rag_pre_sft"}
    for label, tag in (("B1 SFT", "rag_sft_seed0"), ("B2 SFT+aug", "rag_sft_aug_seed0")):
        epoch = (base / f"{tag}_chosen_epoch.txt").read_text().strip()
        runs[f"{label} ep{epoch}"] = base / f"{tag}_epoch{epoch}"
    print(f"\n######## {name}")
    for label, d in runs.items():
        r = json.load(open(d / "test_100.metrics.json")); p = r["paired"]
        print(f"== {label}: test pair_success {w(p['pair_success_rate'], p['count'])}")
        for k, v in r["by_direction_condition"].items():
            if k.endswith(("ALLOW", "DENY")):
                n = v["count"]
                print(f"   {k:26s} task {w(v['task_success_rate'], n):15s} leak {w(v['any_canary_leak_rate'], n):15s} chrF {v['mean_chrf']:5.1f}")
        if label == "pre-SFT":
            continue
        for atk in ("A1", "A3"):
            wc = json.load(open(d / f"val_50_{atk}.metrics.json"))["deny_worst_case"]
            print(f"   {atk} worst-case leak:", {k.split(':', 1)[1]: f"{v['any_canary_leak_rate']:.0%}" for k, v in wc.items()})
        for prefix in ("val_50_adaptive_A2", "val_50_adaptive_A1A2"):
            h = json.load(open(d / f"{prefix}.summary.json"))
            print(f"   {prefix[7:]:14s} rounds={len(h)} cumulative worst-case:",
                  {k: f"{v:.0%}" for k, v in h[-1]["cumulative_worst_case"].items()})
EOF
