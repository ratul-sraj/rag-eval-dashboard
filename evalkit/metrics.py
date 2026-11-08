"""Metric implementations for the evalkit evaluation layer.

Pure Python + numpy. Every function is deterministic and unit-tested against
hand-computed expected values (see ``tests/test_metrics.py``).

Conventions
-----------
* Retrieval metrics operate on *ordered* lists of document ids. ``retrieved``
  is ranked best-first; ``relevant`` is a set of gold ids.
* ``@k`` metrics truncate ``retrieved`` to the first ``k`` items.
* Rate metrics return floats in ``[0, 1]`` and return ``nan``-free values for
  empty inputs (``0.0`` rather than raising) so aggregate dashboards never
  crash on a sparse run.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "key_fact_containment",
    "citation_precision",
    "citation_recall",
    "refusal_accuracy",
    "schema_validity_rate",
    "sql_execution_match",
    "latency_percentiles",
    "cost_total",
    "token_efficiency",
    "aggregate",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_list(x: Optional[Iterable[str]]) -> list[str]:
    if x is None:
        return []
    return [str(i) for i in x]


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for matching."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\.\-/%]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k retrieved.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    rel = set(_as_list(relevant))
    if not rel:
        return 0.0
    got = set(_as_list(retrieved)[:k])
    return len(got & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    top = _as_list(retrieved)[:k]
    if not top:
        return 0.0
    rel = set(_as_list(relevant))
    return len([d for d in top if d in rel]) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if any relevant doc is in the top-k, else 0.0."""
    rel = set(_as_list(relevant))
    return 1.0 if rel & set(_as_list(retrieved)[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean Reciprocal Rank of the first relevant doc (single query).

    MRR = 1 / rank(first relevant), or 0.0 if none retrieved.
    """
    rel = set(_as_list(relevant))
    for rank, doc in enumerate(_as_list(retrieved), start=1):
        if doc in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the standard log2(rank+1) discount."""
    gains = list(gains)[:k]
    total = 0.0
    for i, g in enumerate(gains, start=1):
        total += (2.0**g - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised DCG@k for a binary relevance set.

    IDCG@k is the ideal ordering: all relevant docs first, up to min(k, |rel|).
    """
    rel = set(_as_list(relevant))
    if not rel:
        return 0.0
    ret = _as_list(retrieved)[:k]
    gains = [1.0 if d in rel else 0.0 for d in ret]
    dcg = dcg_at_k(gains, k)
    ideal = [1.0] * min(k, len(rel))
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# --------------------------------------------------------------------------
# answer / grounding metrics
# --------------------------------------------------------------------------
def key_fact_containment(answer: str, must_contain: Sequence[str]) -> float:
    """Fraction of required key-phrases present in the answer (case-insensitive).

    A phrase counts as present if its normalised form appears as a substring of
    the normalised answer. Unlike exact-match this tolerates casing, punctuation
    and surrounding prose.
    """
    phrases = [p for p in _as_list(must_contain) if p and p.strip()]
    if not phrases:
        return 1.0  # nothing required -> vacuously contained
    hay = _norm(answer)
    hit = sum(1 for p in phrases if _norm(p) in hay)
    return hit / len(phrases)


def citation_precision(citations: Sequence[str], relevant: Sequence[str]) -> float:
    """Fraction of cited doc ids that are actually gold-relevant.

    Penalises citing irrelevant documents (hallucinated/unjustified citations).
    """
    cites = _as_list(citations)
    if not cites:
        return 0.0
    rel = set(_as_list(relevant))
    return len([c for c in cites if c in rel]) / len(cites)


def citation_recall(citations: Sequence[str], relevant: Sequence[str]) -> float:
    """Fraction of gold-relevant docs that were cited."""
    rel = set(_as_list(relevant))
    if not rel:
        return 0.0
    return len(set(_as_list(citations)) & rel) / len(rel)


def refusal_accuracy(predicted_refusal: Sequence[bool], gold_refusal: Sequence[bool]) -> float:
    """Accuracy of the refuse/don't-refuse decision.

    Only meaningful over cases that mix answerable and unanswerable items; the
    bundled golden set contains both.
    """
    if not predicted_refusal:
        return 0.0
    correct = sum(1 for p, g in zip(predicted_refusal, gold_refusal) if bool(p) == bool(g))
    return correct / len(predicted_refusal)


def schema_validity_rate(valid_flags: Sequence[bool]) -> float:
    """Fraction of traces whose structured output (JSON / SQL) parsed cleanly."""
    flags = list(valid_flags)
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)


def sql_execution_match(pred_rows: Optional[Sequence[dict]], gold_rows: Optional[Sequence[dict]]) -> float:
    """1.0 if the predicted result set equals the gold result set (order-insensitive).

    Comparison is done as a multiset of row tuples, which is what execution
    match means in text-to-SQL evaluation (spurious duplicate rows matter, so
    multiplicity is preserved).
    """
    if pred_rows is None or gold_rows is None:
        return 0.0

    def canon(rows) -> list[tuple]:
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(tuple(sorted((str(k), str(v)) for k, v in r.items())))
            else:
                out.append((str(r),))
        return sorted(out)

    return 1.0 if canon(pred_rows) == canon(gold_rows) else 0.0


# --------------------------------------------------------------------------
# operational metrics
# --------------------------------------------------------------------------
def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / p99 latency in ms (numpy linear interpolation)."""
    arr = np.asarray([float(x) for x in latencies_ms if x is not None], dtype=float)
    if arr.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def cost_total(costs_usd: Sequence[float]) -> float:
    """Sum of per-trace costs."""
    return float(np.sum(np.asarray([float(c or 0.0) for c in costs_usd], dtype=float))) if costs_usd else 0.0


def token_efficiency(tokens_in: Sequence[int], tokens_out: Sequence[int]) -> float:
    """Output tokens produced per 1000 input tokens (higher = more economical).

    Useful for watching prompt bloat: a rising input:output ratio with flat
    quality usually means contexts are being padded.
    """
    tin = float(np.sum(np.asarray(list(tokens_in), dtype=float)))
    tout = float(np.sum(np.asarray(list(tokens_out), dtype=float)))
    if tin <= 0:
        return 0.0
    return tout / tin * 1000.0


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _mean(xs: Iterable[float]) -> float:
    xs = [float(x) for x in xs]
    if not xs:
        return 0.0
    return float(np.mean(xs))


def aggregate(
    rows: Sequence[dict],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Aggregate a list of per-trace metric dicts into a run summary.

    Each row is expected to contain the keys produced by the runner:
    ``retrieved``, ``relevant``, ``answer``, ``must_contain``, ``citations``,
    ``answered``, ``unanswerable``, ``latency_ms``, ``cost_usd``,
    ``tokens_in``, ``tokens_out``, ``error``, ``struct_valid``, ``sql_match``.
    Missing keys are treated as empty so a mixed RAG/SQL run still summarises.
    """
    if not rows:
        return {}

    recall: dict[int, list[float]] = {k: [] for k in k_values}
    precision: dict[int, list[float]] = {k: [] for k in k_values}
    hit: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []
    cont: list[float] = []
    cprec: list[float] = []
    crec: list[float] = []
    refusal_pred: list[bool] = []
    refusal_gold: list[bool] = []
    struct_valid: list[bool] = []
    sql_match: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    tins: list[int] = []
    touts: list[int] = []
    errors = 0

    for r in rows:
        retrieved = r.get("retrieved") or []
        relevant = r.get("relevant") or []
        # Retrieval metrics are undefined without gold docs (e.g. unanswerable
        # cases, SQL rows): average them only over rows that have relevant ids.
        if relevant:
            for k in k_values:
                recall[k].append(recall_at_k(retrieved, relevant, k))
                precision[k].append(precision_at_k(retrieved, relevant, k))
                hit[k].append(hit_rate_at_k(retrieved, relevant, k))
                ndcg[k].append(ndcg_at_k(retrieved, relevant, k))
            mrrs.append(mrr(retrieved, relevant))
        if relevant:
            cprec.append(citation_precision(r.get("citations") or [], relevant))
            crec.append(citation_recall(r.get("citations") or [], relevant))
        # key-fact containment only counts cases that declare required facts
        if r.get("must_contain"):
            cont.append(key_fact_containment(r.get("answer", ""), r.get("must_contain") or []))
        refusal_pred.append(bool(r.get("refused", False)))
        refusal_gold.append(bool(r.get("unanswerable", False)))
        struct_valid.append(bool(r.get("struct_valid", True)))
        # SQL execution match is only defined for SQL rows.
        if r.get("app") == "sql":
            sql_match.append(float(r.get("sql_match", 0.0)))
        latencies.append(float(r.get("latency_ms", 0.0)))
        costs.append(float(r.get("cost_usd", 0.0)))
        tins.append(int(r.get("tokens_in", 0)))
        touts.append(int(r.get("tokens_out", 0)))
        if r.get("error"):
            errors += 1

    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = _mean(recall[k])
        out[f"precision@{k}"] = _mean(precision[k])
        out[f"hit_rate@{k}"] = _mean(hit[k])
        out[f"ndcg@{k}"] = _mean(ndcg[k])
    out["mrr"] = _mean(mrrs)
    out["key_fact_containment"] = _mean(cont)
    out["citation_precision"] = _mean(cprec)
    out["citation_recall"] = _mean(crec)
    out["refusal_accuracy"] = refusal_accuracy(refusal_pred, refusal_gold)
    out["refusal_rate"] = _mean(refusal_pred)
    out["schema_validity_rate"] = schema_validity_rate(struct_valid)
    out["sql_execution_match"] = _mean(sql_match)

    pct = latency_percentiles(latencies)
    out["latency_p50_ms"] = pct["p50"]
    out["latency_p95_ms"] = pct["p95"]
    out["latency_p99_ms"] = pct["p99"]
    out["latency_mean_ms"] = pct["mean"]
    out["cost_total_usd"] = cost_total(costs)
    out["cost_per_trace_usd"] = cost_total(costs) / len(rows)
    out["token_efficiency"] = token_efficiency(tins, touts)
    out["tokens_in_total"] = float(sum(tins))
    out["tokens_out_total"] = float(sum(touts))
    out["error_rate"] = errors / len(rows)
    out["n_traces"] = float(len(rows))
    return out
