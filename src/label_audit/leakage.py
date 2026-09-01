"""Label leakage through feature columns.

The classic failure: some column — a source system, a filename prefix, a
routing tag — determines the label almost perfectly, the model learns the
column, and production (where the column behaves differently) falls apart.

A column is flagged when its value→label mapping is nearly deterministic
*and* the column plausibly generalises (values recur across items). A unique
ID column determines the label trivially and teaches a model nothing it can
reuse, so high-cardinality columns are reported separately as INFO rather
than as leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .imbalance import majority_label
from .types import Dataset, Finding, Severity


@dataclass(frozen=True)
class FeatureLeakage:
    """How well one feature column predicts the label on its own."""

    feature: str
    determinism: float
    distinct_values: int
    covered_items: int
    identifier_like: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "determinism": self.determinism,
            "distinct_values": self.distinct_values,
            "covered_items": self.covered_items,
            "identifier_like": self.identifier_like,
        }


#: Value→label determinism at or above this is treated as leakage.
DETERMINISM_THRESHOLD = 0.99
#: A column whose values almost never repeat is an identifier, not a signal.
IDENTIFIER_DISTINCT_FRACTION = 0.9


def analyse(dataset: Dataset) -> list[FeatureLeakage]:
    item_label: dict[str, str] = {
        item_id: majority_label([row.label for row in rows])
        for item_id, rows in dataset.items.items()
    }
    item_features: dict[str, dict[str, str]] = {
        item_id: rows[0].features for item_id, rows in dataset.items.items()
    }

    results = []
    for feature in dataset.feature_names:
        value_labels: dict[str, Counter[str]] = defaultdict(Counter)
        covered = 0
        for item_id, features in item_features.items():
            value = features.get(feature, "").strip()
            if not value:
                continue
            covered += 1
            value_labels[value][item_label[item_id]] += 1
        if covered == 0:
            continue
        majority_covered = sum(counts.most_common(1)[0][1] for counts in value_labels.values())
        results.append(
            FeatureLeakage(
                feature=feature,
                determinism=majority_covered / covered,
                distinct_values=len(value_labels),
                covered_items=covered,
                identifier_like=len(value_labels) / covered >= IDENTIFIER_DISTINCT_FRACTION,
            )
        )
    results.sort(key=lambda r: (-r.determinism, r.feature))
    return results


def findings(results: list[FeatureLeakage]) -> list[Finding]:
    out: list[Finding] = []
    for result in results:
        if result.determinism < DETERMINISM_THRESHOLD:
            continue
        if result.identifier_like:
            out.append(
                Finding(
                    check="leakage",
                    severity=Severity.INFO,
                    message=(
                        f"feature {result.feature!r} is identifier-like "
                        f"({result.distinct_values} distinct values over "
                        f"{result.covered_items} items); drop it before training "
                        "rather than let a model memorise it"
                    ),
                )
            )
            continue
        out.append(
            Finding(
                check="leakage",
                severity=Severity.ERROR,
                message=(
                    f"feature {result.feature!r} determines the label with "
                    f"{result.determinism:.1%} accuracy across "
                    f"{result.distinct_values} recurring values; a model will "
                    "learn this column instead of the task"
                ),
                evidence=(
                    f"determinism={result.determinism:.4f}",
                    f"distinct_values={result.distinct_values}",
                    f"covered_items={result.covered_items}",
                ),
            )
        )
    return out
