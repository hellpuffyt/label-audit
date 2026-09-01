"""Console entry point.

Exit codes are part of the interface, because the natural home for this tool
is a CI gate over a dataset directory:

* 0 — audit ran, nothing at or above ``--fail-on`` found
* 1 — usage or input error (bad file, bad flags)
* 3 — audit ran and found something at or above ``--fail-on``
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, io, report
from .errors import LabelAuditError
from .types import Severity

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FINDINGS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label-audit",
        description=(
            "Audit a labelled dataset for annotator disagreement, class "
            "imbalance, duplicates across splits, and label leakage."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("data", help="dataset file (.csv, .jsonl or .ndjson)")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
        help=(
            "exit 3 when a finding at or above this severity exists "
            "(default: error)"
        ),
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=("agreement", "imbalance", "duplicates", "leakage"),
        help="restrict output to one or more checks (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset = io.load(args.data)
    except LabelAuditError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE

    audit = report.run(dataset)

    if args.check:
        wanted = set(args.check)
        audit = report.Audit(
            rows=audit.rows,
            items=audit.items,
            splits=audit.splits,
            labels=audit.labels,
            annotators=audit.annotators,
            agreement=audit.agreement,
            balances=audit.balances,
            duplicate_groups=audit.duplicate_groups,
            leakage=audit.leakage,
            findings=[f for f in audit.findings if f.check in wanted],
        )

    if args.format == "json":
        print(report.render_json(audit))
    else:
        print(report.render_text(audit))

    if args.fail_on == "never":
        return EXIT_OK
    threshold = Severity.ERROR if args.fail_on == "error" else Severity.WARNING
    worst = audit.worst_severity()
    if worst is not None and worst.rank >= threshold.rank:
        return EXIT_FINDINGS
    return EXIT_OK
