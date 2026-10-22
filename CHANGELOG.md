# Changelog

## Unreleased

- Fixed: an annotator labelling the same item twice was counted as two raters,
  inflating observed agreement and Fleiss' kappa.
- Fixed: a CSV header repeating a column name (e.g. two `label` columns)
  silently used the last one; it is now refused.
- Fixed: non-UTF-8 input and oversized CSV fields crashed with a traceback
  instead of exit code 1.
- Docs: README example replaced with real output for `examples/reviews.csv`;
  install from source. Added Fleiss worked-example test, `SECURITY.md`,
  `.editorconfig`, `docs/PRODUCTION_READINESS.md`.

## 0.1.0 — 2026-09-01

Initial release.

- Four auditors over CSV / JSON Lines annotation tables: inter-annotator
  agreement (pairwise observed + Fleiss' kappa where its fixed-rater-count
  assumption holds), per-split class balance against the full label
  vocabulary, duplicate detection within and across splits under mild
  normalisation, and single-feature label-leakage detection with
  identifier-like columns reported as advice rather than leakage.
- `label-audit` CLI with text and JSON output, per-check filtering, and CI
  exit codes (`0` clean, `1` input error, `3` findings at or above
  `--fail-on`).
- Zero runtime dependencies; Python 3.10–3.13.
