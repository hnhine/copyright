# Data bootstrap

This directory downloads and prepares the two datasets used in the first phase
of the authorization-conditioned translation project:

- **DGT-Acquis**: gold, paragraph-aligned English/French/German data for the
  main translation experiments.
- **NoticeBench**: copyrighted Books/Movie Scripts passages for external
  refusal, leakage, and prompt-attack evaluation. It has no gold translations,
  so it is not used for the main COMET/chrF claim.

Run every command from the `copyright/` repository root. Python 3.9+ is enough;
there are no third-party dependencies.

## 1. Download raw data

```bash
python scripts/data/download_raw.py all --data-root data
```

Expected download size is roughly 430 MB for the three DGT language columns,
plus about 4 MB for NoticeBench. Extracted DGT files are substantially larger.
Downloads use `.part` files and resume when the remote server supports HTTP
range requests. Add `--keep-archives` if ZIP archives should be retained.

To download one source only:

```bash
python scripts/data/download_raw.py dgt --languages en fr de
python scripts/data/download_raw.py noticebench
```

## 2. Prepare DGT-Acquis

```bash
python scripts/data/prepare_dgt.py
```

Defaults implement the proposed pilot setup:

- all three languages must contain 80--180 words and 2--6 sentences;
- EN/FR and EN/DE length ratios must lie in `[0.6, 1/0.6]`;
- duplicate triples and DGT aggregate rows ending in `-doc` are removed;
- documents, rather than individual paragraphs, are deterministically assigned
  to train/validation/test;
- uniform reservoir sampling retains 1,500/200/800 triples;
- four directions are emitted: EN-FR, EN-DE, FR-EN, and DE-EN.

Outputs:

```text
data/processed/dgt_acquis/
├── metadata.json
├── triples/{train,validation,test}.jsonl
└── directions/{train,validation,test}.jsonl
```

Each triple produces one example in each requested direction. Change sizes or
directions without editing code, for example:

```bash
python scripts/data/prepare_dgt.py \
  --max-train 3000 --max-validation 300 --max-test 1000 \
  --directions en-fr en-de fr-en de-en
```

## 3. Prepare NoticeBench

```bash
python scripts/data/prepare_noticebench.py \
  --content-types book scripts \
  --sample-words 100 \
  --samples-per-document 5
```

This creates 100 deterministic, non-overlapping passages from the 10 books and
10 movie scripts currently included upstream. Set `--samples-per-document 8`
for a target of up to 160 passages. The original per-document copyright notice
is retained as metadata, while authorization (`ALLOW`/`DENY`) will be added in a
later dataset-building stage.

## Provenance and use

DGT-Acquis is owned by the European Commission and its reuse requires source
attribution. See the [official terms and download page](https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-acquis_en).

NoticeBench contains copyrighted source material released as a research
benchmark. Keep raw and processed content out of git, do not redistribute it,
and review the upstream repository/paper before release. The project `.gitignore`
excludes all downloaded and generated data by default.

