# Contributing

Thanks for looking into label-audit.

## Setup

```console
python -m venv .venv && . .venv/bin/activate   # Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```console
pytest            # the whole suite; new behaviour needs new tests
ruff check .      # lint
mypy              # strict — the src/ tree carries full annotations
```

## Ground rules

- **No runtime dependencies.** The tool installs into locked-down data
  environments; `dependencies = []` is a feature, not an oversight.
- **Findings must be checkable.** Every `Finding` carries evidence naming
  rows, items or values. A finding a reader cannot verify does not ship.
- **No silent guessing.** Unknown formats, nested JSON features and
  unmeasurable statistics are reported as such rather than approximated.
  Fleiss' kappa is only printed where its assumptions hold.
- **Thresholds are named constants** with a comment defending the number
  (`KAPPA_WARNING`, `RATIO_ERROR`, `DETERMINISM_THRESHOLD`, …). Change the
  defence when you change the value.
