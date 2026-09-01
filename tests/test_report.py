from __future__ import annotations

import json

from label_audit import report
from label_audit.types import Dataset, Severity

from .conftest import row


def messy_dataset() -> Dataset:
    """Trips every auditor at once: disagreement, imbalance, cross-split
    duplicates, and a leaky feature."""
    rows = []
    # Disagreement: half the multi-annotated items clash.
    for i in range(4):
        rows.append(row(f"m{i}", "pos", annotator="p", source="web"))
        rows.append(row(f"m{i}", "neg" if i % 2 else "pos", annotator="q", source="web"))
    # Imbalance: 100:1 in train.
    rows.extend(row(f"big{i}", "pos", source="web") for i in range(100))
    rows.append(row("small0", "neg", source="crawler"))
    # Cross-split duplicate.
    rows.append(row("dup-a", "pos", text="the same sentence", split="train", source="web"))
    rows.append(row("dup-b", "pos", text="THE SAME  sentence", split="test", source="web"))
    # Keep test split non-degenerate.
    rows.append(row("t0", "neg", split="test", source="crawler"))
    return Dataset(rows)


class TestRun:
    def test_summary_counts(self, balanced_dataset: Dataset) -> None:
        audit = report.run(balanced_dataset)
        assert audit.rows == 26
        assert audit.items == 26
        assert audit.splits == ["train", "test"]
        assert audit.labels == ["positive", "negative"]

    def test_clean_dataset_has_only_the_agreement_info(
        self, balanced_dataset: Dataset
    ) -> None:
        audit = report.run(balanced_dataset)
        assert [f.severity for f in audit.findings] == [Severity.INFO]
        assert audit.worst_severity() is Severity.INFO

    def test_messy_dataset_finds_every_class_of_problem(self) -> None:
        audit = report.run(messy_dataset())
        checks = {f.check for f in audit.findings}
        assert {"imbalance", "duplicates", "leakage"} <= checks
        assert audit.worst_severity() is Severity.ERROR

    def test_findings_sort_worst_first(self) -> None:
        audit = report.run(messy_dataset())
        ranks = [f.severity.rank for f in audit.findings]
        assert ranks == sorted(ranks, reverse=True)

    def test_worst_severity_of_no_findings_is_none(self) -> None:
        audit = report.run(
            Dataset([row("a", "x", annotator="p"), row("a", "x", annotator="q")])
        )
        # Single fully-agreeing multi-annotated item: nothing to report.
        assert audit.findings == []
        assert audit.worst_severity() is None


class TestRenderJson:
    def test_is_valid_json_with_the_documented_keys(self) -> None:
        payload = json.loads(report.render_json(report.run(messy_dataset())))
        assert set(payload) == {
            "rows",
            "items",
            "splits",
            "labels",
            "annotators",
            "agreement",
            "balance",
            "duplicates",
            "leakage",
            "findings",
        }

    def test_findings_carry_wire_severities(self) -> None:
        payload = json.loads(report.render_json(report.run(messy_dataset())))
        severities = {f["severity"] for f in payload["findings"]}
        assert severities <= {"info", "warning", "error"}


class TestRenderText:
    def test_header_names_items_rows_and_splits(self, balanced_dataset: Dataset) -> None:
        text = report.render_text(report.run(balanced_dataset))
        assert "26 items (26 annotation rows)" in text
        assert "splits: train, test" in text

    def test_clean_run_says_so_explicitly(self) -> None:
        data = Dataset([row("a", "x", annotator="p"), row("a", "x", annotator="q")])
        text = report.render_text(report.run(data))
        assert "no findings" in text

    def test_findings_render_with_severity_and_check(self) -> None:
        text = report.render_text(report.run(messy_dataset()))
        assert "ERROR" in text
        assert "[duplicates]" in text

    def test_evidence_lines_are_indented_under_the_finding(self) -> None:
        text = report.render_text(report.run(messy_dataset()))
        assert "\n          - " in text

    def test_agreement_line_appears_when_measurable(self) -> None:
        text = report.render_text(report.run(messy_dataset()))
        assert "agreement:" in text
