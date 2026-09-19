# Paired ALLOW/DENY SFT data

Run this after preparing DGT-Acquis:

```bash
python scripts/data/build_sft.py
```

The command reads `data/processed/dgt_acquis/directions/` and writes
`data/processed/sft/{train,validation,test}.jsonl` plus `metadata.json`.

Every direction row becomes a matched pair with identical text, language
direction, and split:

- `ALLOW` targets the gold `reference_text`.
- `DENY` targets the single canonical response `Access denied.`.

Rows include a training-ready `messages` array and flat fields (`pair_id`,
`permission`, `source_text`, `reference_text`, and `response_text`) for later
probing and causal analysis. The reference is retained on DENY rows only as
evaluation metadata; it is never inserted into their prompt or assistant target.

Run all data-pipeline tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
