"""Errors that reach the user, phrased so they can act on them."""

from __future__ import annotations


class LabelAuditError(Exception):
    """Base class: everything the CLI reports as a usage/input failure."""


class DatasetFormatError(LabelAuditError):
    """The input file could not be understood as an annotation table."""
