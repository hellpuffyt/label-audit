from __future__ import annotations

from label_audit import leakage
from label_audit.types import Dataset, Severity

from .conftest import row


def leaky_dataset() -> Dataset:
    """`source` fully determines the label across recurring values."""
    rows = []
    for i in range(10):
        rows.append(row(f"p{i}", "positive", source="web"))
        rows.append(row(f"n{i}", "negative", source="crawler"))
    return Dataset(rows)


class TestAnalyse:
    def test_deterministic_recurring_feature_scores_one(self) -> None:
        results = leakage.analyse(leaky_dataset())
        assert len(results) == 1
        result = results[0]
        assert result.feature == "source"
        assert result.determinism == 1.0
        assert result.distinct_values == 2
        assert result.covered_items == 20
        assert not result.identifier_like

    def test_uninformative_feature_scores_low(self) -> None:
        rows = []
        for i in range(10):
            rows.append(row(f"p{i}", "positive", source="web" if i % 2 else "crawler"))
            rows.append(row(f"n{i}", "negative", source="web" if i % 2 else "crawler"))
        results = leakage.analyse(Dataset(rows))
        assert results[0].determinism == 0.5

    def test_identifier_like_feature_is_marked(self) -> None:
        rows = [row(f"i{i}", "x", uid=f"unique-{i}") for i in range(10)]
        results = leakage.analyse(Dataset(rows))
        assert results[0].identifier_like

    def test_blank_values_do_not_count_as_coverage(self) -> None:
        rows = [
            row("a", "x", source="web"),
            row("b", "y", source=""),
            row("c", "x", source="  "),
        ]
        results = leakage.analyse(Dataset(rows))
        assert results[0].covered_items == 1

    def test_feature_absent_everywhere_is_skipped(self) -> None:
        rows = [row("a", "x", source=""), row("b", "y", source="")]
        assert leakage.analyse(Dataset(rows)) == []

    def test_item_label_is_majority_across_annotations(self) -> None:
        rows = [
            row("a", "x", annotator="p", source="web"),
            row("a", "x", annotator="q", source="web"),
            row("a", "y", annotator="r", source="web"),
            row("b", "y", source="crawler"),
        ]
        results = leakage.analyse(Dataset(rows))
        assert results[0].determinism == 1.0

    def test_most_deterministic_feature_sorts_first(self) -> None:
        rows = []
        for i in range(10):
            rows.append(row(f"p{i}", "pos", clean="A", noisy="A" if i % 2 else "B"))
            rows.append(row(f"n{i}", "neg", clean="B", noisy="A" if i % 3 else "B"))
        results = leakage.analyse(Dataset(rows))
        assert results[0].feature == "clean"

    def test_to_dict_shape(self) -> None:
        payload = leakage.analyse(leaky_dataset())[0].to_dict()
        assert payload["feature"] == "source"
        assert payload["determinism"] == 1.0
        assert payload["identifier_like"] is False


class TestFindings:
    def test_leaky_feature_is_an_error_naming_the_column(self) -> None:
        result = leakage.findings(leakage.analyse(leaky_dataset()))
        assert [f.severity for f in result] == [Severity.ERROR]
        assert "'source'" in result[0].message
        assert any(item.startswith("determinism=") for item in result[0].evidence)

    def test_identifier_column_is_info_not_error(self) -> None:
        rows = [row(f"i{i}", "x" if i % 2 else "y", uid=f"u{i}") for i in range(10)]
        result = leakage.findings(leakage.analyse(Dataset(rows)))
        assert [f.severity for f in result] == [Severity.INFO]
        assert "identifier-like" in result[0].message

    def test_sub_threshold_determinism_is_quiet(self) -> None:
        rows = []
        for i in range(50):
            label = "pos" if i % 2 else "neg"
            # 96% aligned with the label: below the 99% line.
            source = ("web" if label == "pos" else "crawler") if i % 25 else "web"
            rows.append(row(f"i{i}", label, source=source))
        assert leakage.findings(leakage.analyse(Dataset(rows))) == []

    def test_clean_dataset_yields_nothing(self, balanced_dataset: Dataset) -> None:
        assert leakage.findings(leakage.analyse(balanced_dataset)) == []
