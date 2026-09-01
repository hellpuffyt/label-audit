from __future__ import annotations

from label_audit.types import Dataset, Finding, Row, Severity

from .conftest import row


class TestSeverity:
    def test_ranks_order_info_below_warning_below_error(self) -> None:
        assert Severity.INFO.rank < Severity.WARNING.rank < Severity.ERROR.rank

    def test_values_are_the_wire_strings(self) -> None:
        assert [s.value for s in Severity] == ["info", "warning", "error"]


class TestFinding:
    def test_to_dict_round_trips_every_field(self) -> None:
        finding = Finding(
            check="duplicates",
            severity=Severity.ERROR,
            message="two items collide",
            evidence=("a/b",),
        )
        assert finding.to_dict() == {
            "check": "duplicates",
            "severity": "error",
            "message": "two items collide",
            "evidence": ["a/b"],
        }

    def test_evidence_defaults_to_empty(self) -> None:
        assert Finding("x", Severity.INFO, "m").to_dict()["evidence"] == []


class TestRow:
    def test_split_defaults_to_train(self) -> None:
        assert Row(item_id="a", text="t", label="x").split == "train"

    def test_annotator_defaults_to_none(self) -> None:
        assert Row(item_id="a", text="t", label="x").annotator is None


class TestDataset:
    def test_len_counts_rows_not_items(self) -> None:
        data = Dataset([row("a", "x", annotator="p"), row("a", "y", annotator="q")])
        assert len(data) == 2
        assert len(data.items) == 1

    def test_items_group_rows_in_input_order(self) -> None:
        first = row("a", "x", annotator="p")
        second = row("a", "y", annotator="q")
        data = Dataset([first, second, row("b", "x")])
        assert data.items["a"] == [first, second]

    def test_splits_preserve_first_appearance_order(self) -> None:
        data = Dataset([row("a", "x", split="test"), row("b", "x", split="train")])
        assert data.splits == ["test", "train"]

    def test_labels_preserve_first_appearance_order(self) -> None:
        data = Dataset([row("a", "zebra"), row("b", "apple"), row("c", "zebra")])
        assert data.labels == ["zebra", "apple"]

    def test_annotators_exclude_unnamed_rows(self) -> None:
        data = Dataset([row("a", "x", annotator="p"), row("b", "x")])
        assert data.annotators == ["p"]

    def test_feature_names_union_across_rows(self) -> None:
        data = Dataset([row("a", "x", source="web"), row("b", "x", region="eu")])
        assert data.feature_names == ["source", "region"]
