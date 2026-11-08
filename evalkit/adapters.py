"""Adapters for ingesting traces emitted by the sibling portfolio apps.

Two ways in:

1. **Native JSONL** -- if an app already emits ``evalkit`` JSONL traces
   (``eval/traces.jsonl``), :func:`load_native_traces` validates and returns them.

2. **Loose ``eval/results.json``** -- the manufacturing RAG assistant (Project 1)
   and the supply-chain SQL agent (Project 2) may emit a simpler ``results.json``
   array. :func:`adapt_results_json` maps the common field aliases onto
   :class:`~evalkit.schema.Trace`.

If neither file exists, :func:`load_or_demo` falls back to the bundled demo
artifacts so the dashboard always has something to show::

    from evalkit.adapters import load_or_demo
    traces = load_or_demo("D:/projects/ai-ml-portfolio/manufacturing-rag-assistant")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .schema import GoldenCase, SchemaError, Trace, read_jsonl

# a lone answer / doc alias table so loose records still map cleanly
_FIELD_ALIASES = {
    "trace_id": ["trace_id", "id", "interaction_id", "request_id"],
    "run_id": ["run_id", "batch_id", "eval_id"],
    "question": ["question", "query", "prompt", "input", "user_query", "utterance"],
    "answer": ["answer", "response", "output", "completion", "prediction", "generated_sql"],
    "contexts": ["contexts", "retrieved", "documents", "docs", "passages", "sources"],
    "citations": ["citations", "cited_docs", "references", "cited_ids"],
    "sql": ["sql", "generated_sql", "query_sql", "statement"],
    "rows": ["rows", "result", "result_rows", "records", "data"],
    "latency_ms": ["latency_ms", "latency", "duration_ms", "elapsed_ms", "response_time_ms"],
    "tokens_in": ["tokens_in", "prompt_tokens", "input_tokens"],
    "tokens_out": ["tokens_out", "completion_tokens", "output_tokens"],
    "cost_usd": ["cost_usd", "cost", "total_cost", "price_usd"],
    "error": ["error", "error_message", "exception", "failure"],
    "model": ["model", "model_name", "llm"],
    "timestamp": ["timestamp", "created_at", "time", "ts"],
    "metadata": ["metadata", "meta", "extra"],
}


def _pick(rec: dict, field: str, default: Any = None) -> Any:
    for alias in _FIELD_ALIASES.get(field, [field]):
        if alias in rec and rec[alias] is not None:
            return rec[alias]
    return default


def _infer_app(path: Path, rec: dict) -> str:
    blob = (str(path) + " " + json.dumps(rec)).lower()
    if "sql" in blob or "text-to-sql" in blob or "text2sql" in blob:
        return "sql"
    if "rag" in blob or "retriev" in blob or "doc" in blob:
        return "rag"
    return "generic"


def adapt_records(records: list[dict], app: Optional[str] = None, run_id: str = "ingested") -> list[Trace]:
    """Map a list of loose dicts onto validated Trace objects."""
    traces: list[Trace] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise SchemaError(f"results[{i}] is not an object")
        inferred = app or _infer_app(Path("."), rec)
        payload = {
            "trace_id": _pick(rec, "trace_id", f"{run_id}-{i:05d}"),
            "app": inferred,
            "run_id": _pick(rec, "run_id", run_id),
            "question": _pick(rec, "question", ""),
            "answer": _pick(rec, "answer", ""),
            "contexts": _pick(rec, "contexts", []),
            "citations": _pick(rec, "citations", []),
            "sql": _pick(rec, "sql"),
            "rows": _pick(rec, "rows"),
            "latency_ms": _pick(rec, "latency_ms", 0.0),
            "tokens_in": _pick(rec, "tokens_in", 0),
            "tokens_out": _pick(rec, "tokens_out", 0),
            "cost_usd": _pick(rec, "cost_usd", 0.0),
            "error": _pick(rec, "error"),
            "model": _pick(rec, "model"),
            "metadata": _pick(rec, "metadata", {}),
        }
        ts = _pick(rec, "timestamp")
        if ts:
            payload["timestamp"] = ts
        traces.append(Trace.model_validate(payload))
    return traces


def load_native_traces(path: str | Path) -> list[Trace]:
    return read_jsonl(path, Trace)


def adapt_results_json(path: str | Path, app: Optional[str] = None, run_id: str = "ingested") -> list[Trace]:
    """Adapt a loose ``results.json`` (either a list or ``{"results": [...]}``)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results") or data.get("traces") or data.get("records") or []
    if not isinstance(data, list):
        raise SchemaError(f"{p}: expected a list or {{'results': [...]}}")
    return adapt_records(data, app=app, run_id=run_id)


def discover(repo_dir: str | Path) -> Optional[tuple[str, Path]]:
    """Find an emit file inside a sibling repo, native JSONL first."""
    repo = Path(repo_dir)
    candidates = [
        ("native", repo / "eval" / "traces.jsonl"),
        ("native", repo / "eval" / "results.jsonl"),
        ("loose", repo / "eval" / "results.json"),
        ("loose", repo / "results.json"),
    ]
    for kind, p in candidates:
        if p.exists():
            return kind, p
    return None


def load_from_repo(repo_dir: str | Path, run_id: str = "ingested") -> list[Trace]:
    """Load traces from a sibling repo directory; raises if nothing found."""
    found = discover(repo_dir)
    if not found:
        raise FileNotFoundError(f"no eval emit file found under {repo_dir}")
    kind, path = found
    app = "rag" if "rag" in str(repo_dir).lower() else ("sql" if "sql" in str(repo_dir).lower() else None)
    if kind == "native":
        return load_native_traces(path)
    return adapt_results_json(path, app=app, run_id=run_id)


def load_or_demo(repo_dir: Optional[str | Path] = None, demo_dir: str | Path = "artifacts/demo_run") -> list[Trace]:
    """Try a repo first, then fall back to the bundled demo artifacts."""
    if repo_dir is not None:
        try:
            return load_from_repo(repo_dir)
        except (FileNotFoundError, SchemaError):
            pass
    demo = Path(demo_dir) / "traces.jsonl"
    if demo.exists():
        return load_native_traces(demo)
    # last resort: run the demo in-process
    from .runner import DemoRagSystem, DemoSqlSystem, build_demo_golden_set, run_eval, make_sqlite_db

    conn = make_sqlite_db()
    cases = build_demo_golden_set()
    rag = DemoRagSystem()
    sql = DemoSqlSystem(conn=conn)

    class _Combo:
        app = "generic"
        name = "demo-combined"

        def run(self, case):
            return rag.run(case) if case.app == "rag" else sql.run(case)

    run_eval(cases, _Combo(), "ingested-demo", sql_conn=conn)
    # re-run collecting traces via a temp dir
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        run_eval(cases, _Combo(), "ingested-demo", out_dir=td, sql_conn=conn)
        return load_native_traces(Path(td) / "traces.jsonl")


# --------------------------------------------------------------------------
# optional observability backends
# --------------------------------------------------------------------------
def to_langfuse_payload(trace: Trace) -> dict:
    """Map an evalkit Trace onto a Langfuse trace-create payload.

    Pure, SDK-free mapping so it is testable offline. To actually ship traces to
    Langfuse, set ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` and call
    ``langfuse.Langfuse().trace(**to_langfuse_payload(t))`` from your app (or use
    ``push_to_langfuse``). This repo never requires a Langfuse server.
    """
    return {
        "id": trace.trace_id,
        "name": trace.app,
        "input": trace.question,
        "output": trace.answer or (trace.sql or ""),
        "metadata": {
            "run_id": trace.run_id,
            "model": trace.model,
            "citations": trace.citations,
            "contexts": [c.doc_id for c in trace.contexts],
            "error": trace.error,
            **trace.metadata,
        },
        "usage": {"input": trace.tokens_in, "output": trace.tokens_out, "total": trace.total_tokens},
        "latency_ms": trace.latency_ms,
        "cost_usd": trace.cost_usd,
        "timestamp": trace.timestamp,
    }


def push_to_langfuse(traces, public_key=None, secret_key=None, host=None) -> dict:
    """Best-effort push of traces to Langfuse if the SDK + keys are configured.

    Returns ``{"pushed": n, "skipped": reason}`` and never raises on a missing
    SDK or missing credentials -- observability export must not break eval runs.
    """
    import os

    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (public_key and secret_key):
        return {"pushed": 0, "skipped": "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set"}
    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError:
        return {"pushed": 0, "skipped": "langfuse SDK not installed (pip install langfuse)"}
    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    n = 0
    for t in traces:
        client.trace(**to_langfuse_payload(t))
        n += 1
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()
    return {"pushed": n, "skipped": None}
