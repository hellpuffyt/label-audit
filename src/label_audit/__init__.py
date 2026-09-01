"""label-audit: find the problems in a labelled dataset before a model does."""

from __future__ import annotations

__version__ = "0.1.0"

from .types import Dataset, Finding, Row, Severity

__all__ = ["Dataset", "Finding", "Row", "Severity", "__version__"]
