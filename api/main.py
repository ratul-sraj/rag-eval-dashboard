"""FastAPI service exposing runs, summaries, traces and an ingestion endpoint.

Run it::

    uvicorn api.main:app --reload --port 8077
    # or: make api

Endpoints
---------
GET  /health
GET  /runs                         -- available runs discovered under the store
GET  /runs/{run_id}/summary        -- the summary.json for a run
GET  /traces                       -- paginated + filtered trace listing
POST /ingest                       -- ingest a JSONL body of traces
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from evalkit.schema import SchemaError, Trace, read_jsonl, validate_trace_dict, write_jsonl

STORE = Path(os.environ.get("EVALKIT_STORE", "artifacts/demo_run"))

app = FastAPI(
    title="evalkit API",
    version="0.1.0",
    description="Evaluation & monitoring service for the evalkit LLM app harness.",
)


# --------------------------------------------------------------------------
# store helpers
# --------------------------------------------------------------------------
def discover_runs(store: Optional[Path] = None) -> dict[str, Path]:
    """Map run_id -> directory for every ``summary.json`` under the store."""
    store = store if store is not None else STORE
    runs: dict[str, Path] = {}
    if not store.exists():
        return runs
    for summary_path in store.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            run_id = data.get("run_id") or summary_path.parent.name
        except (json.JSONDecodeError, OSError):
            continue
        runs[run_id] = summary_path.parent
    return runs


def load_traces_for(dirpath: Path) -> list[Trace]:
    traces_file = dirpath / "traces.jsonl"
    if not traces_file.exists():
        return []
    try:
        return read_jsonl(traces_file, Trace)
    except SchemaError:
        return []


def all_traces(store: Optional[Path] = None) -> list[Trace]:
    store = store if store is not None else STORE
    traces: list[Trace] = []
    seen: set[str] = set()
    for dirpath in sorted(discover_runs(store).values()):
        for t in load_traces_for(dirpath):
            if t.trace_id not in seen:
                seen.add(t.trace_id)
                traces.append(t)
    return traces


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class IngestResponse(BaseModel):
    run_id: str
    ingested: int
    errors: list[str] = []


class TracePage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[dict[str, Any]]


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    runs = discover_runs()
    return {"status": "ok", "store": str(STORE), "n_runs": len(runs)}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    out = []
    for run_id, dirpath in sorted(discover_runs().items()):
        summary_file = dirpath / "summary.json"
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        out.append({
            "run_id": run_id,
            "app": data.get("app"),
            "system": data.get("system"),
            "n_traces": data.get("n_traces"),
            "created_at": data.get("created_at"),
        })
    return out


@app.get("/runs/{run_id}/summary")
def run_summary(run_id: str) -> JSONResponse:
    runs = discover_runs()
    if run_id not in runs:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    data = json.loads((runs[run_id] / "summary.json").read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/traces", response_model=TracePage)
def get_traces(
    app_name: Optional[str] = Query(None, alias="app"),
    run_id: Optional[str] = None,
    errors_only: bool = False,
    min_latency_ms: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> TracePage:
    traces = all_traces()
    if app_name:
        traces = [t for t in traces if t.app == app_name]
    if run_id:
        traces = [t for t in traces if t.run_id == run_id]
    if errors_only:
        traces = [t for t in traces if t.is_error]
    if min_latency_ms is not None:
        traces = [t for t in traces if t.latency_ms >= min_latency_ms]
    if max_latency_ms is not None:
        traces = [t for t in traces if t.latency_ms <= max_latency_ms]

    total = len(traces)
    start = (page - 1) * page_size
    page_items = traces[start:start + page_size]
    return TracePage(
        total=total, page=page, page_size=page_size,
        items=[t.model_dump() for t in page_items],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(body: str = Body(..., media_type="text/plain"),
           run_id: str = Query("ingested"),
           app_name: Optional[str] = Query(None, alias="app")) -> IngestResponse:
    """Ingest a JSONL body of traces; returns per-line validation errors."""
    valid: list[Trace] = []
    errors: list[str] = []
    for lineno, line in enumerate(body.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON: {exc}")
            continue
        payload.setdefault("run_id", run_id)
        if app_name:
            payload["app"] = app_name
        try:
            valid.append(validate_trace_dict(payload))
        except SchemaError as exc:
            errors.append(f"line {lineno}: {exc}")

    target = STORE / "ingested" / run_id
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "traces.jsonl", valid)
    summary = {
        "run_id": run_id,
        "app": app_name or "ingested",
        "system": "ingest",
        "n_traces": len(valid),
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        "metrics": {"n_traces": float(len(valid))},
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return IngestResponse(run_id=run_id, ingested=len(valid), errors=errors)
