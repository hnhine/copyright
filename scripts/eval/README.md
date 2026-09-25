# Pre-SFT behavior evaluation

Install the lightweight evaluation dependencies from the repository root:

```bash
python -m pip install -r requirements-eval.txt
```

The Qwen checkpoint is cached under `/workspace/nhi/huggingface/hub` and reused
on later runs. Run one pair per available translation direction first (currently
2 pairs / 6 generations for CONTROL, ALLOW, and DENY):

```bash
python scripts/eval/run_behavior.py \
  --config configs/eval/qwen25_3b_pre_sft.json \
  --split validation \
  --run-name validation_smoke \
  --pairs-per-direction 1
```

Then run and score 25 pairs per available direction (currently 50 pairs / 150
generations):

```bash
python scripts/eval/run_behavior.py \
  --config configs/eval/qwen25_3b_pre_sft.json \
  --split validation

python scripts/eval/score_behavior.py \
  --input outputs/behavior/qwen2.5-3b-instruct/pre_sft/validation_pilot.generations.jsonl
```

Generation files are append-only and resume by `run_id` when rerun.
