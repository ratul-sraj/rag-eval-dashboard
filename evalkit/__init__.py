"""evalkit - a small, dependency-light evaluation & monitoring toolkit for LLM apps.

Public surface:
    from evalkit.schema import Trace, GoldenCase, read_jsonl, write_jsonl
    from evalkit import metrics
    from evalkit.judge import HeuristicJudge, OpenAIJudge, get_judge
    from evalkit.runner import run_eval, DemoRagSystem, DemoSqlSystem
"""

from . import metrics  # noqa: F401
from .schema import GoldenCase, Trace, read_jsonl, write_jsonl  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Trace",
    "GoldenCase",
    "read_jsonl",
    "write_jsonl",
    "metrics",
    "__version__",
]
