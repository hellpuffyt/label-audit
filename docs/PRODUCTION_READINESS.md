# Production readiness — label-audit

Audit date: 2026-10-22. Tested commit: `8011b4d` (branch `audit/production-readiness`, based on `main` @ `0d4b65f`).

## Scope

Stdlib-only CLI that audits a labelled dataset (CSV / JSON Lines, one row per annotation) for
annotator disagreement (pairwise observed agreement, Fleiss' kappa), class imbalance per split,
exact-after-normalisation duplicates across splits / with conflicting labels, and feature-label
leakage. Read-only: never modifies data. Exit 0 / 1 bad input / 3 findings.

## Platforms

CPython 3.10+, no runtime dependencies. Verified on macOS 27 (Apple M5), CPython 3.12.14.

## Clean-clone verification (commit `8011b4d`)

```bash
git clone -b audit/production-readiness <local clone> label-audit && cd label-audit
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install .
label-audit examples/reviews.csv              # exit 3; byte-identical to the README example
label-audit examples/reviews.csv --format json   # valid JSON
uv pip install -e ".[dev]" && pytest && ruff check . && mypy   # 118 passed / clean / clean
```

## Commands and results

| Command | Baseline (`0d4b65f`) | After (`8011b4d`) |
| --- | --- | --- |
| `pytest` | exit 0 — 112 passed, 0 failed, 0 skipped | exit 0 — 118 passed, 0 failed, 0 skipped |
| `ruff check .` / `mypy` | exit 0 / exit 0 | exit 0 / exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |

CI status — default branch: success (`0d4b65f`, 2026-09-01); audit branch: pending until pushed.

## Verification of the statistics

- Fleiss' kappa reproduces the published worked example (Fleiss 1971, 10 subjects × 14 raters ×
  5 categories): 0.210 (`test_fleiss_kappa_matches_the_published_worked_example`).
- Observed agreement and kappa formulas reviewed line by line against the textbook definitions.

## Bugs found and fixed

| Severity | Bug | Regression test |
| --- | --- | --- |
| Medium (wrong statistic) | one annotator labelling an item twice counted as two agreeing raters | `tests/test_agreement.py::test_an_annotator_repeating_an_item_does_not_agree_with_themselves` |
| Medium (silent data loss) | repeated header column (e.g. two `label`) silently used the last | `tests/test_io.py::test_repeated_header_column_is_refused` |
| Low (crash vs exit-1 contract) | `UnicodeDecodeError` (CSV, JSONL) and `csv.Error` tracebacks | `test_undecodable_csv_...`, `test_undecodable_jsonl_...`, `test_oversized_field_...` |
| Docs | README console example (812 items, `data/train.csv`) was not from any file in the repo | README shows real output for `examples/reviews.csv` |

## Security review

Read-only; no network; no `eval`/pickle. History secret scan: no hits. `pip-audit`: clean.
Workflow permissions `contents: read`; actions tag-pinned (left as is).

## Known limitations

- Duplicates are exact after casefold + whitespace normalisation (by design; "price" vs "price!" differ).
- Kappa uses only items with the modal rater count; the message reports all multi-annotated items.
- Whole file loaded into memory.

## Acceptance checklist

| # | Item | Expected | Test | Result |
| --- | --- | --- | --- | --- |
| 1 | Fleiss' kappa correct | 0.210 on worked example | agreement test | PASS |
| 2 | Agreement only between distinct annotators | repeat ignored | regression test | PASS |
| 3 | Cross-split duplicates found | r1/r2 reported | example run | PASS |
| 4 | Conflicting duplicate labels found | r4/r5 reported | example run | PASS |
| 5 | Missing class in a split flagged | ERROR | existing imbalance tests | PASS |
| 6 | Bad input → exit 1 | message | io tests | PASS |
| 7 | Clean clone | install/tests/example | above | PASS |

## Classification proposal

**Production-Ready for Its Documented Scope** (proposal for lead verification).
