"""Shared builders. Tests construct datasets in code so each one states its
exact input instead of pointing at an opaque fixture file."""

from __future__ import annotations

import pytest

from label_audit.types import Dataset, Row


def row(
    item_id: str,
    label: str,
    *,
    text: str | None = None,
    split: str = "train",
    annotator: str | None = None,
    **features: str,
) -> Row:
    return Row(
        item_id=item_id,
        text=text if text is not None else f"text of {item_id}",
        label=label,
        split=split,
        annotator=annotator,
        features=features,
    )


@pytest.fixture()
def balanced_dataset() -> Dataset:
    """Two splits, two classes, no duplicates, no leakage, one annotation
    per item — the audit should stay quiet on this."""
    rows = []
    for i in range(10):
        rows.append(row(f"tr-pos-{i}", "positive", split="train"))
        rows.append(row(f"tr-neg-{i}", "negative", split="train"))
    for i in range(3):
        rows.append(row(f"te-pos-{i}", "positive", split="test"))
        rows.append(row(f"te-neg-{i}", "negative", split="test"))
    return Dataset(rows)
