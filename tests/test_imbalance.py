from __future__ import annotations

import pytest

from label_audit import imbalance
from label_audit.types import Dataset, Severity

from .conftest import row


class TestMajorityLabel:
    def test_clear_majority_wins(self) -> None:
        assert imbalance.majority_label(["a", "b", "a"]) == "a"

    def test_tie_breaks_to_earliest_annotation(self) -> None:
        assert imbalance.majority_label(["b", "a"]) == "b"

    def test_single_label(self) -> None:
        assert imbalance.majority_label(["only"]) == "only"


class TestSplitBalance:
    def test_imbalance_ratio_ignores_absent_classes(self) -> None:
        balance = imbalance.SplitBalance("train", {"a": 10, "b": 5, "c": 0})
        assert balance.imbalance_ratio == 2.0

    def test_entropy_is_one_when_perfectly_balanced(self) -> None:
        balance = imbalance.SplitBalance("train", {"a": 5, "b": 5})
        assert balance.normalised_entropy == pytest.approx(1.0)

    def test_entropy_is_zero_for_a_single_present_class(self) -> None:
        balance = imbalance.SplitBalance("train", {"a": 5, "b": 0})
        assert balance.normalised_entropy == 0.0

    def test_to_dict_shape(self) -> None:
        payload = imbalance.SplitBalance("dev", {"a": 1, "b": 3}).to_dict()
        assert payload["split"] == "dev"
        assert payload["counts"] == {"a": 1, "b": 3}
        assert payload["imbalance_ratio"] == 3.0


class TestAnalyse:
    def test_counts_items_not_annotation_rows(self) -> None:
        data = Dataset(
            [
                row("a", "x", annotator="p"),
                row("a", "x", annotator="q"),
                row("a", "x", annotator="r"),
                row("b", "y"),
            ]
        )
        balances = imbalance.analyse(data)
        assert balances[0].counts == {"x": 1, "y": 1}

    def test_item_label_is_the_majority_across_annotators(self) -> None:
        data = Dataset(
            [
                row("a", "x", annotator="p"),
                row("a", "x", annotator="q"),
                row("a", "y", annotator="r"),
            ]
        )
        assert imbalance.analyse(data)[0].counts == {"x": 1, "y": 0}

    def test_every_split_reports_the_full_vocabulary(self) -> None:
        data = Dataset([row("a", "x", split="train"), row("b", "y", split="test")])
        balances = {b.split: b.counts for b in imbalance.analyse(data)}
        assert balances["train"] == {"x": 1, "y": 0}
        assert balances["test"] == {"x": 0, "y": 1}


class TestFindings:
    def test_quiet_on_a_balanced_dataset(self, balanced_dataset: Dataset) -> None:
        assert imbalance.findings(imbalance.analyse(balanced_dataset)) == []

    def test_missing_class_in_a_split_is_an_error(self) -> None:
        data = Dataset(
            [
                row("a", "x", split="train"),
                row("b", "y", split="train"),
                row("c", "x", split="test"),
            ]
        )
        result = imbalance.findings(imbalance.analyse(data))
        errors = [f for f in result if f.severity is Severity.ERROR]
        assert len(errors) == 1
        assert "test" in errors[0].message
        assert errors[0].evidence == ("y",)

    def test_ratio_at_warning_threshold_warns(self) -> None:
        data = Dataset(
            [
                *(row(f"a{i}", "big") for i in range(10)),
                row("b0", "small"),
            ]
        )
        result = imbalance.findings(imbalance.analyse(data))
        assert [f.severity for f in result] == [Severity.WARNING]
        assert "10:1" in result[0].message

    def test_ratio_at_error_threshold_errors(self) -> None:
        data = Dataset(
            [
                *(row(f"a{i}", "big") for i in range(100)),
                row("b0", "small"),
            ]
        )
        result = imbalance.findings(imbalance.analyse(data))
        assert [f.severity for f in result] == [Severity.ERROR]

    def test_single_class_dataset_reports_nothing_about_ratio(self) -> None:
        data = Dataset([row("a", "only"), row("b", "only")])
        assert imbalance.findings(imbalance.analyse(data)) == []

    def test_mild_skew_stays_quiet(self) -> None:
        data = Dataset(
            [
                *(row(f"a{i}", "big") for i in range(6)),
                *(row(f"b{i}", "small") for i in range(4)),
            ]
        )
        assert imbalance.findings(imbalance.analyse(data)) == []
