"""Inter-annotator agreement.

Everything here works from items that genuinely carry more than one
annotation. A dataset with one row per item has no agreement to measure, and
that is reported as such rather than as a suspicious 1.0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .types import Dataset, Finding, Row, Severity


@dataclass(frozen=True)
class AgreementReport:
    """What could honestly be measured about annotator agreement."""

    multi_annotated_items: int
    observed_agreement: float | None
    fleiss_kappa: float | None
    disagreements: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "multi_annotated_items": self.multi_annotated_items,
            "observed_agreement": self.observed_agreement,
            "fleiss_kappa": self.fleiss_kappa,
            "disagreements": [
                {"item": item, "labels": list(labels)}
                for item, labels in self.disagreements
            ],
        }


def pairwise_observed_agreement(label_sets: list[list[str]]) -> float | None:
    """Mean fraction of agreeing annotator pairs per multi-annotated item.

    This is the plain, assumption-free statistic: for each item, of all pairs
    of annotations, how many carry the same label.
    """
    fractions = []
    for labels in label_sets:
        n = len(labels)
        if n < 2:
            continue
        pairs = n * (n - 1) // 2
        agreeing = sum(c * (c - 1) // 2 for c in Counter(labels).values())
        fractions.append(agreeing / pairs)
    if not fractions:
        return None
    return sum(fractions) / len(fractions)


def fleiss_kappa(label_sets: list[list[str]]) -> float | None:
    """Fleiss' kappa over items annotated by the same number of raters.

    Fleiss' statistic assumes a fixed rater count per item, so items are
    restricted to the most common annotation count ≥ 2. Returns ``None`` when
    fewer than two such items exist, or when the statistic is undefined
    (every annotation identical — expected agreement 1).
    """
    counts = Counter(len(labels) for labels in label_sets if len(labels) >= 2)
    if not counts:
        return None
    n_raters = counts.most_common(1)[0][0]
    subjects = [labels for labels in label_sets if len(labels) == n_raters]
    if len(subjects) < 2:
        return None

    categories: Counter[str] = Counter()
    for labels in subjects:
        categories.update(labels)
    total = len(subjects) * n_raters

    p_e = sum((count / total) ** 2 for count in categories.values())
    p_bar = sum(
        (sum(c * c for c in Counter(labels).values()) - n_raters)
        / (n_raters * (n_raters - 1))
        for labels in subjects
    ) / len(subjects)

    if p_e >= 1.0:
        return None
    return (p_bar - p_e) / (1 - p_e)


def _one_label_per_annotator(rows: list[Row]) -> list[str]:
    """Labels for one item, counting each named annotator once.

    Agreement is between *different* raters: an annotator who labelled the
    same item twice would otherwise agree with themselves and inflate both
    observed agreement and kappa. Their first label is kept. Rows without an
    annotator name cannot be told apart and each count as one rater.
    """
    seen: set[str] = set()
    labels = []
    for row in rows:
        if row.annotator is not None:
            if row.annotator in seen:
                continue
            seen.add(row.annotator)
        labels.append(row.label)
    return labels


def analyse(dataset: Dataset) -> AgreementReport:
    label_sets = []
    disagreements = []
    for item_id, rows in dataset.items.items():
        labels = _one_label_per_annotator(rows)
        if len(labels) >= 2:
            label_sets.append(labels)
            if len(set(labels)) > 1:
                disagreements.append((item_id, tuple(labels)))
    return AgreementReport(
        multi_annotated_items=len(label_sets),
        observed_agreement=pairwise_observed_agreement(label_sets),
        fleiss_kappa=fleiss_kappa(label_sets),
        disagreements=tuple(disagreements),
    )


#: Below this, Landis & Koch would call the agreement no better than moderate,
#: and labels produced under it should not be treated as ground truth.
KAPPA_WARNING = 0.6
KAPPA_ERROR = 0.2


def findings(report: AgreementReport) -> list[Finding]:
    out: list[Finding] = []
    if report.multi_annotated_items == 0:
        out.append(
            Finding(
                check="agreement",
                severity=Severity.INFO,
                message=(
                    "no item carries more than one annotation; "
                    "annotator agreement cannot be measured"
                ),
            )
        )
        return out
    if report.fleiss_kappa is not None:
        kappa = report.fleiss_kappa
        if kappa < KAPPA_ERROR:
            severity = Severity.ERROR
        elif kappa < KAPPA_WARNING:
            severity = Severity.WARNING
        else:
            severity = None
        if severity is not None:
            examples = tuple(
                f"{item}: {'/'.join(labels)}" for item, labels in report.disagreements[:5]
            )
            out.append(
                Finding(
                    check="agreement",
                    severity=severity,
                    message=(
                        f"Fleiss' kappa is {kappa:.2f} across "
                        f"{report.multi_annotated_items} multi-annotated items; "
                        "these labels disagree too much to treat as ground truth"
                    ),
                    evidence=examples,
                )
            )
    return out
