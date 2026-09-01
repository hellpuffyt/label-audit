# label-audit

Audit a labelled dataset for the four problems that quietly ruin models:
annotator disagreement, class imbalance, duplicate items across splits, and
label leakage. Zero runtime dependencies — Python 3.10+ and nothing else.

```console
$ label-audit data/train.csv
812 items (1,624 annotation rows), 3 labels, splits: train, test
annotators: priya, marco
agreement: 87.4% observed over 812 multi-annotated items, Fleiss' kappa 0.71
split train: spam=610, ham=150, promo=2
split test: spam=48, ham=50, promo=0

ERROR   [imbalance] split 'test' contains no example of 1 label(s); anything
        evaluated on it is blind to those classes
          - promo
ERROR   [duplicates] 3 duplicate group(s) span more than one split; evaluation
        on the contaminated split measures memorisation, not generalisation
          - itm-104/itm-9central in splits test/train
```

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
Optional: `split` (defaults to `train`), `annotator`. Every other column is
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

## Development

```console
pip install -e ".[dev]"
pytest             # test suite
ruff check .       # lint
mypy               # strict type check
```

MIT licensed.
