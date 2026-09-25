# Paired ALLOW/DENY SFT data

Run this after preparing and language-cleaning DGT-Acquis:

```bash
python scripts/data/clean_dgt_languages.py
python scripts/data/build_sft.py \
  --input-dir data/processed/dgt_acquis_clean/directions \
  --output-dir data/processed/sft
```

The SFT builder writes
`data/processed/sft/{train,validation,test}.jsonl` plus `metadata.json`.

Every direction row becomes a matched pair with identical text, language
direction, and split:

- `ALLOW` targets the gold `reference_text`.
- `DENY` targets the single canonical response `Access denied.`.

The trusted `ALLOW`/`DENY` decision is serialized in the `system` message.
The matched examples have identical `user` messages containing only the
language direction and source text. The canonical refusal string is never
included in an input message.

Rows include a training-ready `messages` array and flat fields (`pair_id`,
`permission`, `source_text`, `reference_text`, and `response_text`) for later
probing and causal analysis. The reference is retained on DENY rows only as
evaluation metadata; it is never inserted into their prompt or assistant target.

Run all data-pipeline tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
