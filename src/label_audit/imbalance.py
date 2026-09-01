"""Class balance, per split and against the whole dataset.

Counts are item-level: an item annotated three times is still one example of
its (majority) label, otherwise heavily re-annotated items would masquerade
as class weight.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .types import Dataset, Finding, Severity


def majority_label(labels: list[str]) -> str:
    """The modal label; ties break to the earliest annotation, which is the
    only deterministic choice that does not invent information."""
    counts = Counter(labels)
    best = max(counts.values())
    for label in labels:
        if counts[label] == best:
            return label
    raise AssertionError("unreachable: labels is non-empty")


@dataclass(frozen=True)
class SplitBalance:
    split: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def imbalance_ratio(self) -> float:
        """Largest class count over smallest, among classes present."""
        values = [v for v in self.counts.values() if v > 0]
        return max(values) / min(values)

    @property
    def normalised_entropy(self) -> float:
        """Shannon entropy of the class distribution, scaled to [0, 1].

        1.0 is perfectly balanced; a single class scores 0. Defined as 1.0
        for a one-class vocabulary only when that is all the dataset has.
        """
        values = [v for v in self.counts.values() if v > 0]
        if len(values) <= 1:
            return 0.0
        total = sum(values)
        entropy = -sum((v / total) * math.log2(v / total) for v in values)
        return entropy / math.log2(len(values))

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "counts": dict(self.counts),
            "imbalance_ratio": self.imbalance_ratio,
            "normalised_entropy": self.normalised_entropy,
        }


def analyse(dataset: Dataset) -> list[SplitBalance]:
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    for _item_id, rows in dataset.items.items():
        label = majority_label([row.label for row in rows])
        # An item is counted in every split it appears in; appearing in more
        # than one is the duplicates auditor's finding, not this one's.
        for split in {row.split for row in rows}:
            per_split[split][label] += 1
    vocabulary = dataset.labels
    return [
        SplitBalance(split, {label: per_split[split].get(label, 0) for label in vocabulary})
        for split in dataset.splits
    ]


#: Ratios above these are worth a human look / are actively dangerous for
#: naive training. Chosen to flag real skew without shouting at 60/40 splits.
RATIO_WARNING = 10.0
RATIO_ERROR = 100.0


def findings(balances: list[SplitBalance]) -> list[Finding]:
    out: list[Finding] = []
    for balance in balances:
        present = {label: count for label, count in balance.counts.items() if count > 0}
        missing = sorted(set(balance.counts) - set(present))
        if missing and len(balance.counts) > 1:
            out.append(
                Finding(
                    check="imbalance",
                    severity=Severity.ERROR,
                    message=(
                        f"split {balance.split!r} contains no example of "
                        f"{len(missing)} label(s); anything evaluated on it is "
                        "blind to those classes"
                    ),
                    evidence=tuple(missing),
                )
            )
        if len(present) > 1:
            ratio = balance.imbalance_ratio
            if ratio >= RATIO_ERROR:
                severity = Severity.ERROR
            elif ratio >= RATIO_WARNING:
                severity = Severity.WARNING
            else:
                continue
            largest = max(present, key=lambda label: present[label])
            smallest = min(present, key=lambda label: present[label])
            out.append(
                Finding(
                    check="imbalance",
                    severity=severity,
                    message=(
                        f"split {balance.split!r} is imbalanced {ratio:.0f}:1 "
                        f"({largest!r} {present[largest]} vs {smallest!r} "
                        f"{present[smallest]})"
                    ),
                    evidence=tuple(
                        f"{label}={count}" for label, count in sorted(present.items())
                    ),
                )
            )
    return out
