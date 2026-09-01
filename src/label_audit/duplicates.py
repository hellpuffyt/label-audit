"""Duplicate items — within a split and, far worse, across splits.

Texts are compared after a deliberately mild normalisation (casefold and
whitespace collapse). Anything stronger — stemming, punctuation stripping —
starts merging texts that genuinely differ, and a duplicate report that
cannot be trusted is worse than none.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .imbalance import majority_label
from .types import Dataset, Finding, Severity


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class DuplicateGroup:
    """Items whose text is identical after normalisation."""

    normalised_text: str
    item_ids: tuple[str, ...]
    splits: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def crosses_splits(self) -> bool:
        return len(set(self.splits)) > 1

    @property
    def conflicting_labels(self) -> bool:
        return len(set(self.labels)) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "items": list(self.item_ids),
            "splits": list(self.splits),
            "labels": list(self.labels),
            "crosses_splits": self.crosses_splits,
            "conflicting_labels": self.conflicting_labels,
        }


def analyse(dataset: Dataset) -> list[DuplicateGroup]:
    by_text: dict[str, list[str]] = defaultdict(list)
    item_split: dict[str, str] = {}
    item_label: dict[str, str] = {}
    for item_id, rows in dataset.items.items():
        by_text[normalise(rows[0].text)].append(item_id)
        item_split[item_id] = rows[0].split
        item_label[item_id] = majority_label([row.label for row in rows])

    groups = []
    for text, item_ids in by_text.items():
        if len(item_ids) < 2:
            continue
        groups.append(
            DuplicateGroup(
                normalised_text=text,
                item_ids=tuple(item_ids),
                splits=tuple(item_split[i] for i in item_ids),
                labels=tuple(item_label[i] for i in item_ids),
            )
        )
    groups.sort(key=lambda g: (-len(g.item_ids), g.item_ids))
    return groups


def findings(groups: list[DuplicateGroup]) -> list[Finding]:
    out: list[Finding] = []

    contaminating = [g for g in groups if g.crosses_splits]
    if contaminating:
        examples = tuple(
            f"{'/'.join(g.item_ids)} in splits {'/'.join(sorted(set(g.splits)))}"
            for g in contaminating[:5]
        )
        out.append(
            Finding(
                check="duplicates",
                severity=Severity.ERROR,
                message=(
                    f"{len(contaminating)} duplicate group(s) span more than one "
                    "split; evaluation on the contaminated split measures "
                    "memorisation, not generalisation"
                ),
                evidence=examples,
            )
        )

    conflicting = [g for g in groups if g.conflicting_labels]
    if conflicting:
        examples = tuple(
            f"{'/'.join(g.item_ids)}: {'/'.join(g.labels)}" for g in conflicting[:5]
        )
        out.append(
            Finding(
                check="duplicates",
                severity=Severity.ERROR,
                message=(
                    f"{len(conflicting)} duplicate group(s) carry conflicting "
                    "labels; at least one of each pair is wrong"
                ),
                evidence=examples,
            )
        )

    plain = [g for g in groups if not g.crosses_splits and not g.conflicting_labels]
    if plain:
        redundant = sum(len(g.item_ids) - 1 for g in plain)
        out.append(
            Finding(
                check="duplicates",
                severity=Severity.WARNING,
                message=(
                    f"{redundant} item(s) are duplicates within a split; they "
                    "inflate apparent dataset size and over-weight their texts"
                ),
                evidence=tuple("/".join(g.item_ids) for g in plain[:5]),
            )
        )
    return out
