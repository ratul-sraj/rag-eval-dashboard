"""Generate the committed gate fixtures (baseline / current / regressed).

Real, computed eval runs -- not hand-written numbers. ``current`` is a hair
better than ``baseline`` so the CI gate passes; ``regressed`` degrades retrieval
and SQL quality so the gate fails.

    python samples/make_gate_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.runner import (
    DemoRagSystem,
    DemoSqlSystem,
    build_demo_golden_set,
    make_sqlite_db,
    run_eval,
)

HERE = Path(__file__).resolve().parent

GATE_METRICS = [
    "recall@5",
    "mrr",
    "citation_precision",
    "key_fact_containment",
    "sql_execution_match",
    "valid_sql_rate",
    "sql_repair_rate",
    "task_success_rate",
    "refusal_accuracy",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "error_rate",
    "cost_total_usd",
    "token_efficiency",
]


class DegradedRagSystem(DemoRagSystem):
    """Worse retrieval + four refusals (used for the regressed fixture)."""

    name = "demo-rag-degraded"
    _RA = [2, 4, 3, 6, 5, 7, 4, 8, 3, 6]
    _RB = [5, 7, 9, 8, 10, 6, 11, 9, 4, 8]
    _REFUSAL_IDX = {17, 63, 5, 42}


def combined_summary(rag: DemoRagSystem, sql: DemoSqlSystem, run_id: str) -> dict:
    conn = make_sqlite_db()
    sql.conn = conn
    cases = build_demo_golden_set()

    class _Combo:
        app = "generic"
        name = "demo-combined"

        def run(self, case):
            return rag.run(case) if case.app == "rag" else sql.run(case)

    summary = run_eval(cases, _Combo(), run_id, sql_conn=conn)
    return summary


def main() -> None:
    # baseline = stock demo
    base = combined_summary(DemoRagSystem(), DemoSqlSystem(seed=11), "baseline")

    # current = marginally better SQL quality -> gate should pass
    cur = combined_summary(DemoRagSystem(), DemoSqlSystem(seed=11, broken_fraction=0.18, error_fraction=0.08),
                           "current")

    # regressed = degraded RAG + worse SQL -> gate must fail
    reg = combined_summary(DegradedRagSystem(), DemoSqlSystem(seed=11, broken_fraction=0.45, error_fraction=0.30),
                           "regressed")

    for name, summary in (("baseline", base), ("current", cur), ("regressed", reg)):
        payload = {
            "run_id": summary["run_id"],
            "created_at": summary["created_at"],
            "source": "samples/make_gate_samples.py (real demo run)",
            "metrics": {k: summary["metrics"].get(k) for k in GATE_METRICS},
        }
        (HERE / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{name:10s} recall@5={summary['metrics']['recall@5']:.4f} "
              f"sql_match={summary['metrics']['sql_execution_match']:.4f} "
              f"err={summary['metrics']['error_rate']:.4f} "
              f"task_success={summary['metrics']['task_success_rate']:.4f}")


if __name__ == "__main__":
    main()
