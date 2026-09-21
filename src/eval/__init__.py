"""Reproducible, source-backed evaluation harness for the RAG application.

The package deliberately does not ship downloaded Hugging Face rows.  Use the
manifests in ``eval_data/manifests`` to fetch and verify the bounded samples.
"""

from .schema import EvalCase, EvalMode, EvalReport, RetrievedDocument, SourceSpan

__all__ = ["EvalCase", "EvalMode", "EvalReport", "RetrievedDocument", "SourceSpan"]
