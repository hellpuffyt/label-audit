from __future__ import annotations

import pytest

from label_audit import agreement
from label_audit.types import Dataset, Severity

from .conftest import row


class TestPairwiseObservedAgreement:
    def test_full_agreement_is_one(self) -> None:
        assert agreement.pairwise_observed_agreement([["a", "a"], ["b", "b"]]) == 1.0

    def test_full_disagreement_is_zero(self) -> None:
        assert agreement.pairwise_observed_agreement([["a", "b"]]) == 0.0

    def test_three_raters_two_agreeing_is_one_third(self) -> None:
        result = agreement.pairwise_observed_agreement([["a", "a", "b"]])
        assert result == pytest.approx(1 / 3)

    def test_single_annotations_are_ignored(self) -> None:
        assert agreement.pairwise_observed_agreement([["a"], ["b", "b"]]) == 1.0

    def test_no_multi_annotated_items_is_none_not_perfect(self) -> None:
        assert agreement.pairwise_observed_agreement([["a"], ["b"]]) is None


class TestFleissKappa:
    def test_perfect_agreement_is_undefined_when_one_category(self) -> None:
        # Every rating identical: expected agreement is 1, kappa undefined.
        assert agreement.fleiss_kappa([["a", "a"], ["a", "a"], ["a", "a"]]) is None

    def test_perfect_agreement_two_categories_is_one(self) -> None:
        result = agreement.fleiss_kappa([["a", "a"], ["b", "b"], ["a", "a"], ["b", "b"]])
        assert result == pytest.approx(1.0)

    def test_known_textbook_value(self) -> None:
        # Two raters, four subjects: agree on 3, disagree on 1, with labels
        # split 5/3 overall. Hand computation: p̄ = 0.75,
        # pe = (5/8)^2 + (3/8)^2 = 0.53125, kappa = (0.75-0.53125)/0.46875.
        subjects = [["a", "a"], ["a", "a"], ["b", "b"], ["a", "b"]]
        expected = (0.75 - 0.53125) / (1 - 0.53125)
        assert agreement.fleiss_kappa(subjects) == pytest.approx(expected)

    def test_restricts_to_the_modal_rater_count(self) -> None:
        # Three 2-rater items and one 3-rater item: the 3-rater item must be
        # excluded, so the result equals the 2-rater-only computation.
        two_rater = [["a", "a"], ["b", "b"], ["a", "b"]]
        assert agreement.fleiss_kappa([*two_rater, ["a", "a", "a"]]) == pytest.approx(
            agreement.fleiss_kappa(two_rater)  # type: ignore[arg-type]
        )

    def test_fewer_than_two_usable_subjects_is_none(self) -> None:
        assert agreement.fleiss_kappa([["a", "b"]]) is None

    def test_no_multi_annotated_items_is_none(self) -> None:
        assert agreement.fleiss_kappa([["a"]]) is None


class TestAnalyse:
    def test_counts_multi_annotated_items_only(self) -> None:
        data = Dataset(
            [
                row("a", "x", annotator="p"),
                row("a", "x", annotator="q"),
                row("b", "x", annotator="p"),
            ]
        )
        report = agreement.analyse(data)
        assert report.multi_annotated_items == 1

    def test_disagreements_record_item_and_labels(self) -> None:
        data = Dataset(
            [
                row("a", "x", annotator="p"),
                row("a", "y", annotator="q"),
            ]
        )
        report = agreement.analyse(data)
        assert report.disagreements == (("a", ("x", "y")),)

    def test_to_dict_shape(self) -> None:
        data = Dataset([row("a", "x", annotator="p"), row("a", "y", annotator="q")])
        payload = agreement.analyse(data).to_dict()
        assert payload["multi_annotated_items"] == 1
        assert payload["disagreements"] == [{"item": "a", "labels": ["x", "y"]}]


class TestFindings:
    def test_no_multi_annotation_yields_an_info_not_silence(self) -> None:
        data = Dataset([row("a", "x"), row("b", "y")])
        result = agreement.findings(agreement.analyse(data))
        assert [f.severity for f in result] == [Severity.INFO]
        assert "cannot be measured" in result[0].message

    def test_low_kappa_is_an_error(self) -> None:
        # Half the items disagree across two balanced labels: kappa lands
        # well under the 0.2 error line.
        data = Dataset(
            [
                *(row(f"agree-{i}", "x", annotator="p") for i in range(2)),
                *(row(f"agree-{i}", "x", annotator="q") for i in range(2)),
                *(row(f"clash-{i}", "x", annotator="p") for i in range(4)),
                *(row(f"clash-{i}", "y", annotator="q") for i in range(4)),
            ]
        )
        result = agreement.findings(agreement.analyse(data))
        assert [f.severity for f in result] == [Severity.ERROR]
        assert "kappa" in result[0].message

    def test_error_finding_carries_example_disagreements(self) -> None:
        data = Dataset(
            [
                row("clash", "x", annotator="p"),
                row("clash", "y", annotator="q"),
                row("clash2", "x", annotator="p"),
                row("clash2", "y", annotator="q"),
            ]
        )
        result = agreement.findings(agreement.analyse(data))
        assert result and any("clash" in item for item in result[0].evidence)

    def test_high_agreement_yields_nothing(self) -> None:
        data = Dataset(
            [
                *(row(f"i{i}", "x" if i % 2 else "y", annotator="p") for i in range(10)),
                *(row(f"i{i}", "x" if i % 2 else "y", annotator="q") for i in range(10)),
            ]
        )
        assert agreement.findings(agreement.analyse(data)) == []


# --- audit regressions ----------------------------------------------------


def test_fleiss_kappa_matches_the_published_worked_example() -> None:
    # Fleiss (1971) example as reproduced on Wikipedia: 10 subjects,
    # 14 raters, 5 categories -> kappa = 0.210.
    table = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]  # fmt: skip
    sets = [[str(c) for c, k in enumerate(r) for _ in range(k)] for r in table]
    kappa = agreement.fleiss_kappa(sets)
    assert kappa is not None
    assert round(kappa, 3) == 0.210


def test_an_annotator_repeating_an_item_does_not_agree_with_themselves() -> None:
    # p labelled items 1 and 2 twice each; only item 3 has two raters.
    data = Dataset(
        [
            row("1", "a", annotator="p"),
            row("1", "a", annotator="p"),
            row("2", "b", annotator="p"),
            row("2", "b", annotator="p"),
            row("3", "a", annotator="p"),
            row("3", "b", annotator="q"),
        ]
    )
    report = agreement.analyse(data)
    assert report.multi_annotated_items == 1
    assert report.observed_agreement == 0.0
