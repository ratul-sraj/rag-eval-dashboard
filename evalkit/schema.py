"""Shared trace / eval schema for the evalkit evaluation layer.

Every LLM app emits one :class:`Trace` per interaction (one user turn). A
``GoldenCase`` describes a benchmark item used to drive an offline eval run.

JSONL is the on-disk interchange format. Both readers validate against the
pydantic v2 models and raise ``SchemaError`` with the offending line number
and a human-readable reason, so downstream apps get actionable failures
instead of silent ``KeyError``s.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

AppName = Literal["rag", "sql", "generic"]


class SchemaError(ValueError):
    """Raised when a JSONL row cannot be validated against the schema."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Context(BaseModel):
    """A retrieved context chunk (RAG) or a data source touched (agent)."""

    model_config = ConfigDict(extra="allow")

    doc_id: str
    text: str = ""
    score: Optional[float] = None
    source: Optional[str] = None


class Trace(BaseModel):
    """One instrumented interaction from a target application.

    Only ``trace_id``, ``app`` and ``run_id`` are strictly required; the rest
    default to empty so a minimal ``emit`` call is 3 lines.
    """

    model_config = ConfigDict(extra="allow")  # forward compatible: keep unknown keys

    trace_id: str
    app: AppName = "generic"
    run_id: str
    question: str = ""
    answer: str = ""
    contexts: list[Context] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    sql: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    model: Optional[str] = None
    timestamp: str = Field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- derived helpers -------------------------------------------------
    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def latency_s(self) -> float:
        return self.latency_ms / 1000.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def context_ids(self) -> list[str]:
        return [c.doc_id for c in self.contexts]

    @property
    def answered(self) -> bool:
        """Heuristic: did the system commit to an answer rather than refuse?"""
        if self.error:
            return False
        a = (self.answer or "").strip().lower()
        if not a:
            return False
        refusal_markers = (
            "i don't know",
            "i do not know",
            "cannot answer",
            "can't answer",
            "no relevant",
            "not enough information",
            "enough information",
            "insufficient",
            "unable to answer",
            "i'm not sure",
            "no information",
            "not found in",
            "cannot determine",
        )
        return not any(m in a for m in refusal_markers)

    @field_validator("contexts", mode="before")
    @classmethod
    def _coerce_contexts(cls, v: Any) -> Any:
        """Accept contexts as plain strings or dicts, not only Context objects."""
        if v is None:
            return []
        if isinstance(v, (str, dict, Context)):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Context):
                out.append(item)
            elif isinstance(item, str):
                out.append({"doc_id": item, "text": "", "source": None, "score": None})
            elif isinstance(item, dict):
                if "doc_id" not in item:
                    # tolerate apps that emit {"chunk_id": ...} or {"id": ...}
                    for alt in ("chunk_id", "id", "doc"):
                        if alt in item:
                            item = {**item, "doc_id": str(item[alt])}
                            break
                out.append(item)
            else:  # pragma: no cover - defensive
                raise TypeError(f"unsupported context type: {type(item)!r}")
        return out

    @field_validator("trace_id", "run_id", "app", mode="before")
    @classmethod
    def _require_id(cls, v: Any) -> Any:
        if v is None or v == "":
            raise ValueError("must be a non-empty string")
        return v


class GoldenCase(BaseModel):
    """A benchmark item for offline evaluation.

    RAG cases populate ``expected_docs``; SQL cases populate ``gold_sql``.
    ``must_contain`` holds answer key-phrases that a correct response should
    surface. ``unanswerable`` cases check refusal behaviour.
    """

    model_config = ConfigDict(extra="allow")

    case_id: str
    app: AppName = "generic"
    question: str = ""
    expected_docs: list[str] = Field(default_factory=list)
    gold_sql: Optional[str] = None
    must_contain: list[str] = Field(default_factory=list)
    unanswerable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", mode="before")
    @classmethod
    def _require_case_id(cls, v: Any) -> Any:
        if not v:
            raise ValueError("case_id must be non-empty")
        return v


# --------------------------------------------------------------------------
# JSONL I/O
# --------------------------------------------------------------------------
ModelT = Union[Trace, GoldenCase]


def _model_for(obj: Any) -> type[BaseModel]:
    if isinstance(obj, GoldenCase):
        return GoldenCase
    return Trace


def write_jsonl(path: Union[str, Path], items: Iterable[ModelT]) -> int:
    """Write pydantic models (or plain dicts) to a JSONL file. Returns count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for item in items:
            if isinstance(item, BaseModel):
                fh.write(item.model_dump_json() + "\n")
            else:
                fh.write(json.dumps(item) + "\n")
            n += 1
    return n


def read_jsonl(path: Union[str, Path], model: type[ModelT] = Trace) -> list[ModelT]:
    """Read + validate a JSONL file. Raises :class:`SchemaError` on bad rows."""
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"file not found: {p}")
    out: list[ModelT] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            try:
                out.append(model.model_validate(payload))  # type: ignore[arg-type]
            except Exception as exc:  # pydantic ValidationError
                raise SchemaError(
                    f"{p}:{lineno}: schema validation failed: {exc}"
                ) from exc
    return out


def iter_jsonl(path: Union[str, Path], model: type[ModelT] = Trace) -> Iterator[ModelT]:
    """Stream-validate a JSONL file (lower memory for large trace dumps)."""
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            try:
                yield model.model_validate(payload)  # type: ignore[misc]
            except Exception as exc:
                raise SchemaError(
                    f"{p}:{lineno}: schema validation failed: {exc}"
                ) from exc


def validate_trace_dict(d: dict[str, Any]) -> Trace:
    """Validate a single dict as a Trace (used by the API /ingest endpoint)."""
    try:
        return Trace.model_validate(d)
    except Exception as exc:
        raise SchemaError(f"invalid trace: {exc}") from exc
