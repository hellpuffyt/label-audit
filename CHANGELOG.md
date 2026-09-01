# Changelog

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
