"""Load annotation tables from CSV or JSON Lines.

Required columns: ``id``, ``text``, ``label``. Optional: ``split`` (defaults
to ``train``) and ``annotator``. Every other column is kept as a *feature* —
that is what the leakage auditor inspects, so extra columns are preserved
rather than dropped.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .errors import DatasetFormatError
from .types import Dataset, Row

REQUIRED = ("id", "text", "label")
RESERVED = ("id", "text", "label", "split", "annotator")


def load(path: str | Path) -> Dataset:
    """Load a dataset, dispatching on file extension.

    ``.csv`` is parsed as CSV with a header row; ``.jsonl`` / ``.ndjson`` as
    one JSON object per line. Anything else is refused rather than guessed —
    a wrong guess would silently misread the data being audited.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path)
    raise DatasetFormatError(
        f"unsupported input {path.name!r}: expected .csv, .jsonl or .ndjson"
    )


def _row_from_mapping(raw: dict[str, str], where: str) -> Row:
    missing = [key for key in REQUIRED if not raw.get(key, "").strip()]
    if missing:
        raise DatasetFormatError(f"{where}: missing required value(s): {', '.join(missing)}")
    features = {
        key: value
        for key, value in raw.items()
        if key not in RESERVED and value is not None
    }
    annotator = raw.get("annotator", "").strip()
    return Row(
        item_id=raw["id"].strip(),
        text=raw["text"],
        label=raw["label"].strip(),
        split=raw.get("split", "").strip() or "train",
        annotator=annotator or None,
        features=features,
    )


def _load_csv(path: Path) -> Dataset:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetFormatError(f"{path.name}: file is empty")
            missing = [name for name in REQUIRED if name not in reader.fieldnames]
            if missing:
                raise DatasetFormatError(
                    f"{path.name}: header lacks required column(s): {', '.join(missing)}"
                )
            rows = []
            for number, raw in enumerate(reader, start=2):
                cleaned = {k: (v if v is not None else "") for k, v in raw.items() if k}
                rows.append(_row_from_mapping(cleaned, f"{path.name}:{number}"))
    except OSError as err:
        raise DatasetFormatError(f"cannot read {path}: {err.strerror or err}") from err
    if not rows:
        raise DatasetFormatError(f"{path.name}: no data rows")
    return Dataset(rows)


def _load_jsonl(path: Path) -> Dataset:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                where = f"{path.name}:{number}"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    raise DatasetFormatError(f"{where}: invalid JSON: {err.msg}") from err
                if not isinstance(obj, dict):
                    raise DatasetFormatError(f"{where}: expected an object per line")
                cleaned = {str(k): _scalar(v, where, str(k)) for k, v in obj.items()}
                rows.append(_row_from_mapping(cleaned, where))
    except OSError as err:
        raise DatasetFormatError(f"cannot read {path}: {err.strerror or err}") from err
    if not rows:
        raise DatasetFormatError(f"{path.name}: no data rows")
    return Dataset(rows)


def _scalar(value: object, where: str, key: str) -> str:
    """Coerce a JSON value to the string form the table model uses.

    Nested containers are refused: flattening them would invent feature
    values the file does not actually contain.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return ""
    raise DatasetFormatError(f"{where}: field {key!r} is not a scalar")
