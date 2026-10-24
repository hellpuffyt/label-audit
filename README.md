# label-audit

Audit a labelled dataset for the four problems that quietly ruin models:
annotator disagreement, class imbalance, duplicate items across splits, and
label leakage. Zero runtime dependencies — Python 3.10+ and nothing else.

```console
$ label-audit examples/reviews.csv
10 items (10 annotation rows), 2 labels, splits: train, test
annotators: priya, marco
split train: negative=3, positive=5
split test: negative=1, positive=1

ERROR   [duplicates] 1 duplicate group(s) carry conflicting labels; at least one of each pair is wrong
          - r4/r5: negative/positive
ERROR   [duplicates] 1 duplicate group(s) span more than one split; evaluation on the contaminated split measures memorisation, not generalisation
          - r1/r2 in splits test/train
info    [agreement] no item carries more than one annotation; annotator agreement cannot be measured
```

(Real output for the committed example file; exit code 3.)

## Why these four checks

Every one of them produces a model that *looks* fine and isn't:

- **Disagreement** — if annotators cannot agree on a label, the "ground
  truth" is noise, and accuracy against it is a measurement of nothing.
  Reported as pairwise observed agreement plus Fleiss' kappa over the items
  that genuinely carry multiple annotations.
- **Imbalance** — a 100:1 split trains a majority-class detector. Reported
  per split, against the full label vocabulary, so a class missing from the
  test split is caught too.
- **Duplicates across splits** — a test item whose text also sits in train is
  measuring memorisation. Texts are compared after mild normalisation only
  (casefold + whitespace), so genuinely different texts are never merged.
- **Label leakage** — a feature column that determines the label (a source
  system, a routing tag) will be what the model learns. Columns whose values
  recur and predict the label ≥99% are errors; identifier-like columns are
  reported separately as advice, not leakage.

## Input format

CSV (with header) or JSON Lines. Required fields: `id`, `text`, `label`.
Optional: `split` (defaults to `train`), `annotator`. Agreement counts each
named annotator once per item (their first label), so a repeated annotation
cannot agree with itself. A header that repeats a column name, non-UTF-8 input
and CSV fields over 128 KiB are rejected with exit code 1. Every other column is
treated as a feature and inspected for leakage. One row per annotation — an
item labelled by three annotators appears on three rows sharing the same `id`.

## Usage

```console
label-audit data.csv                      # human-readable audit
label-audit data.jsonl --format json      # machine-readable, same content
label-audit data.csv --check leakage      # one auditor only (repeatable)
label-audit data.csv --fail-on warning    # stricter CI gate
```

Exit codes are the CI interface: `0` clean at the chosen threshold, `1` bad
input or usage, `3` findings at or above `--fail-on` (default: `error`).

## What it deliberately does not do

- No fuzzy/semantic duplicate detection — near-duplicate judgement calls
  belong to a human, and a merge the tool cannot justify is worse than a miss.
- No automatic "fixing". The output is a review manifest; deleting data is
  a decision, not a side effect.
- Cohen's/Fleiss' kappa is only computed where its assumptions hold (a
  modal, fixed rater count per item); otherwise the report says why not
  rather than printing a misleading number.

## Install

Not published on PyPI; install from a clone (Python 3.10+, no runtime
dependencies):

```console
git clone https://github.com/hellpuffyt/label-audit && cd label-audit
python -m venv .venv && . .venv/bin/activate
pip install .
```

## Development

```console
pip install -e ".[dev]"
pytest             # test suite
ruff check .       # lint
mypy               # strict type check
```

Fleiss' kappa is checked against the published worked example (0.210) in
`tests/test_agreement.py`. Security: [SECURITY.md](SECURITY.md); audit evidence:
[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

MIT licensed.
