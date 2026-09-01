from __future__ import annotations

from label_audit import duplicates
from label_audit.types import Dataset, Severity

from .conftest import row


class TestNormalise:
    def test_casefold_and_whitespace_collapse(self) -> None:
        assert duplicates.normalise("  Hello   WORLD \n") == "hello world"

    def test_punctuation_is_preserved(self) -> None:
        # Stronger normalisation merges genuinely different texts; "no!" and
        # "no" must stay distinct.
        assert duplicates.normalise("no!") != duplicates.normalise("no")


class TestAnalyse:
    def test_no_duplicates_yields_no_groups(self, balanced_dataset: Dataset) -> None:
        assert duplicates.analyse(balanced_dataset) == []

    def test_same_text_groups_across_items(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="Hello world"),
                row("b", "x", text="hello   WORLD"),
                row("c", "x", text="different"),
            ]
        )
        groups = duplicates.analyse(data)
        assert len(groups) == 1
        assert groups[0].item_ids == ("a", "b")

    def test_cross_split_flag(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same", split="train"),
                row("b", "x", text="same", split="test"),
            ]
        )
        assert duplicates.analyse(data)[0].crosses_splits

    def test_conflicting_label_flag_uses_majority_labels(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same", annotator="p"),
                row("a", "x", text="same", annotator="q"),
                row("b", "y", text="same"),
            ]
        )
        group = duplicates.analyse(data)[0]
        assert group.labels == ("x", "y")
        assert group.conflicting_labels

    def test_largest_groups_come_first(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="pair"),
                row("b", "x", text="pair"),
                row("c", "x", text="trio"),
                row("d", "x", text="trio"),
                row("e", "x", text="trio"),
            ]
        )
        groups = duplicates.analyse(data)
        assert [len(g.item_ids) for g in groups] == [3, 2]

    def test_to_dict_shape(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same", split="train"),
                row("b", "y", text="same", split="test"),
            ]
        )
        payload = duplicates.analyse(data)[0].to_dict()
        assert payload["items"] == ["a", "b"]
        assert payload["crosses_splits"] is True
        assert payload["conflicting_labels"] is True


class TestFindings:
    def test_cross_split_contamination_is_an_error(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same", split="train"),
                row("b", "x", text="same", split="test"),
            ]
        )
        result = duplicates.findings(duplicates.analyse(data))
        assert [f.severity for f in result] == [Severity.ERROR]
        assert "memorisation" in result[0].message
        assert "a/b" in result[0].evidence[0]

    def test_conflicting_labels_are_an_error(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same"),
                row("b", "y", text="same"),
            ]
        )
        result = duplicates.findings(duplicates.analyse(data))
        severities = {f.severity for f in result}
        assert Severity.ERROR in severities
        assert any("conflicting" in f.message for f in result)

    def test_within_split_same_label_duplicates_warn(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same"),
                row("b", "x", text="same"),
                row("c", "x", text="same"),
            ]
        )
        result = duplicates.findings(duplicates.analyse(data))
        assert [f.severity for f in result] == [Severity.WARNING]
        assert "2 item(s)" in result[0].message

    def test_clean_dataset_yields_nothing(self, balanced_dataset: Dataset) -> None:
        assert duplicates.findings(duplicates.analyse(balanced_dataset)) == []

    def test_a_group_both_crossing_and_conflicting_reports_both(self) -> None:
        data = Dataset(
            [
                row("a", "x", text="same", split="train"),
                row("b", "y", text="same", split="test"),
            ]
        )
        result = duplicates.findings(duplicates.analyse(data))
        assert len(result) == 2
