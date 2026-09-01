"""Core data model.

A dataset here is a flat list of annotation rows. One *item* (a thing that was
labelled) may appear on several rows — once per annotator — and everything the
auditors compute is derived from grouping those rows, never from assuming one
row per item.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """How much a finding should worry the reader."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Ordering for --fail-on comparisons; higher is worse."""
        return {"info": 0, "warning": 1, "error": 2}[self.value]


@dataclass(frozen=True)
class Finding:
    """One concrete, checkable problem found in the dataset.

    ``evidence`` names the rows, items or values behind the claim so a reader
    can verify it instead of trusting the tool.
    """

    check: str
    severity: Severity
    message: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class Row:
    """One annotation: an annotator's label for one item in one split."""

    item_id: str
    text: str
    label: str
    split: str = "train"
    annotator: str | None = None
    features: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class Dataset:
    """All rows, with the item-level grouping every auditor needs."""

    rows: list[Row]

    def __post_init__(self) -> None:
        self._by_item: dict[str, list[Row]] = defaultdict(list)
        for row in self.rows:
            self._by_item[row.item_id].append(row)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def items(self) -> dict[str, list[Row]]:
        """Rows grouped by item id, in input order within each item."""
        return dict(self._by_item)

    @property
    def splits(self) -> list[str]:
        """Split names in first-appearance order."""
        seen: dict[str, None] = {}
        for row in self.rows:
            seen.setdefault(row.split, None)
        return list(seen)

    @property
    def labels(self) -> list[str]:
        """Label vocabulary in first-appearance order."""
        seen: dict[str, None] = {}
        for row in self.rows:
            seen.setdefault(row.label, None)
        return list(seen)

    @property
    def annotators(self) -> list[str]:
        """Annotator names in first-appearance order (unnamed rows excluded)."""
        seen: dict[str, None] = {}
        for row in self.rows:
            if row.annotator is not None:
                seen.setdefault(row.annotator, None)
        return list(seen)

    @property
    def feature_names(self) -> list[str]:
        """Every feature column present on any row, in first-appearance order."""
        seen: dict[str, None] = {}
        for row in self.rows:
            for name in row.features:
                seen.setdefault(name, None)
        return list(seen)
