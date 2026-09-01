"""Assemble the full audit and render it for humans or machines."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import agreement, duplicates, imbalance, leakage
from .types import Dataset, Finding, Severity


@dataclass(frozen=True)
class Audit:
    """Every auditor's raw result plus the combined finding list."""

    rows: int
    items: int
    splits: list[str]
    labels: list[str]
    annotators: list[str]
    agreement: agreement.AgreementReport
    balances: list[imbalance.SplitBalance]
    duplicate_groups: list[duplicates.DuplicateGroup]
    leakage: list[leakage.FeatureLeakage]
    findings: list[Finding]

    def worst_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "items": self.items,
            "splits": self.splits,
            "labels": self.labels,
            "annotators": self.annotators,
            "agreement": self.agreement.to_dict(),
            "balance": [b.to_dict() for b in self.balances],
            "duplicates": [g.to_dict() for g in self.duplicate_groups],
            "leakage": [r.to_dict() for r in self.leakage],
            "findings": [f.to_dict() for f in self.findings],
        }


def run(dataset: Dataset) -> Audit:
    agreement_report = agreement.analyse(dataset)
    balances = imbalance.analyse(dataset)
    duplicate_groups = duplicates.analyse(dataset)
    leakage_results = leakage.analyse(dataset)

    findings: list[Finding] = []
    findings.extend(agreement.findings(agreement_report))
    findings.extend(imbalance.findings(balances))
    findings.extend(duplicates.findings(duplicate_groups))
    findings.extend(leakage.findings(leakage_results))
    findings.sort(key=lambda f: (-f.severity.rank, f.check, f.message))

    return Audit(
        rows=len(dataset),
        items=len(dataset.items),
        splits=dataset.splits,
        labels=dataset.labels,
        annotators=dataset.annotators,
        agreement=agreement_report,
        balances=balances,
        duplicate_groups=duplicate_groups,
        leakage=leakage_results,
        findings=findings,
    )


def render_json(audit: Audit) -> str:
    return json.dumps(audit.to_dict(), indent=2, sort_keys=False)


_MARK = {Severity.ERROR: "ERROR  ", Severity.WARNING: "warning", Severity.INFO: "info   "}


def render_text(audit: Audit) -> str:
    lines = [
        f"{audit.items} items ({audit.rows} annotation rows), "
        f"{len(audit.labels)} labels, splits: {', '.join(audit.splits)}",
    ]
    if audit.annotators:
        lines.append(f"annotators: {', '.join(audit.annotators)}")
    if audit.agreement.observed_agreement is not None:
        kappa = audit.agreement.fleiss_kappa
        kappa_text = f", Fleiss' kappa {kappa:.2f}" if kappa is not None else ""
        lines.append(
            f"agreement: {audit.agreement.observed_agreement:.1%} observed over "
            f"{audit.agreement.multi_annotated_items} multi-annotated items{kappa_text}"
        )
    for balance in audit.balances:
        parts = ", ".join(f"{label}={count}" for label, count in balance.counts.items())
        lines.append(f"split {balance.split}: {parts}")

    lines.append("")
    if not audit.findings:
        lines.append("no findings: the checks this tool runs found nothing to report")
    for finding in audit.findings:
        lines.append(f"{_MARK[finding.severity]} [{finding.check}] {finding.message}")
        for item in finding.evidence:
            lines.append(f"          - {item}")
    return "\n".join(lines)
