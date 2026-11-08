"""Evaluation runner + deterministic demo systems.

``run_eval`` drives a golden set against any *system under test* callable and
writes ``traces.jsonl`` + ``summary.json``.

Two deliberately-imperfect demo systems ship here so the dashboard, the gate and
the CI pipeline are fully demonstrable offline:

* :class:`DemoRagSystem` -- synthetic manufacturing-docs RAG. Targets
  recall@5 ~= 0.70, drops an occasional citation, and refuses 2 answerable
  questions.
* :class:`DemoSqlSystem` -- text-to-SQL over a real in-memory SQLite supply-chain
  schema. Emits occasionally-wrong SQL, sometimes raises a SQL error, and
  repairs ~half of the first-attempt failures.

Both are deterministic (seeded patterns) so CI and the README numbers are stable.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from . import metrics
from .schema import GoldenCase, Trace, write_jsonl

# ==========================================================================
# golden-set construction (deterministic, synthetic)
# ==========================================================================
_RAG_TOPICS = 100
_DISTRACTOR_POOL = [f"manual_unit{i:03d}_x" for i in range(_RAG_TOPICS)]


def _rag_doc_text(i: int, variant: str) -> str:
    torque = 10 + (i % 40)
    lube = 46 + (i % 320)
    return (
        f"Equipment unit {i:03d} maintenance record. The flange bolts are torqued to "
        f"{torque} Nm in a star pattern. The approved lubricant is ISO VG {lube} gear "
        f"oil, replaced every {1000 + 100 * (i % 9)} operating hours. Inspect the "
        f"seal for wear at each service.{' Secondary reference copy.' if variant == 'b' else ''}"
    )


def _rag_fact(i: int) -> str:
    return f"{10 + (i % 40)} nm"


def build_rag_cases(n: int = 100, n_unanswerable: int = 10) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for i in range(n - n_unanswerable):
        doc_a = f"manual_unit{i:03d}_a"
        doc_b = f"manual_unit{i:03d}_b"
        cases.append(GoldenCase(
            case_id=f"rag_{i:03d}",
            app="rag",
            question=f"What is the flange bolt torque specification for equipment unit {i:03d}?",
            expected_docs=[doc_a, doc_b],
            must_contain=[_rag_fact(i)],
            metadata={"topic": i},
        ))
    for j in range(n_unanswerable):
        i = n - n_unanswerable + j
        cases.append(GoldenCase(
            case_id=f"rag_{i:03d}",
            app="rag",
            question=f"What is the warranty period for equipment unit {i:03d}?",
            expected_docs=[],
            must_contain=[],
            unanswerable=True,
            metadata={"topic": i},
        ))
    return cases


# --- SQL domain: a real in-memory SQLite database --------------------------
def make_sqlite_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE suppliers (
            supplier_id INTEGER PRIMARY KEY, name TEXT, country TEXT, rating REAL
        );
        CREATE TABLE parts (
            part_id INTEGER PRIMARY KEY, name TEXT, supplier_id INTEGER,
            unit_cost REAL, lead_time_days INTEGER
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, part_id INTEGER, qty INTEGER, order_date TEXT
        );
        """
    )
    countries = ["Germany", "India", "China", "USA", "Vietnam", "Mexico", "Poland"]
    for sid in range(1, 61):
        conn.execute(
            "INSERT INTO suppliers VALUES (?,?,?,?)",
            (sid, f"Supplier {sid:02d}", countries[sid % len(countries)], round(1.0 + (sid % 50) / 10.0, 1)),
        )
    for pid in range(1, 81):
        conn.execute(
            "INSERT INTO parts VALUES (?,?,?,?,?)",
            (pid, f"Part {pid:03d}", (pid % 60) + 1, round(2.0 + (pid % 97) * 1.5, 2), 3 + (pid % 45)),
        )
    for oid in range(1, 201):
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?)",
            (oid, (oid % 80) + 1, 1 + (oid % 500), f"2025-{(oid % 12) + 1:02d}-15"),
        )
    conn.commit()
    return conn


_SQL_TEMPLATES = [
    ("How many suppliers are headquartered in {c}?",
     "SELECT COUNT(*) AS n FROM suppliers WHERE country = '{c}'"),
    ("What is the average unit cost of all parts?",
     "SELECT AVG(unit_cost) AS avg_cost FROM parts"),
    ("List the names of suppliers with a rating above {r}.",
     "SELECT name FROM suppliers WHERE rating > {r} ORDER BY name"),
    ("What is the maximum lead time in days across all parts?",
     "SELECT MAX(lead_time_days) AS max_lead FROM parts"),
    ("How many orders have a quantity greater than {q}?",
     "SELECT COUNT(*) AS n FROM orders WHERE qty > {q}"),
    ("What is the total ordered quantity for part {p}?",
     "SELECT SUM(qty) AS total_qty FROM orders WHERE part_id = {p}"),
]


def build_sql_cases(n: int = 100, n_error: int = 10) -> list[GoldenCase]:
    countries = ["Germany", "India", "China", "USA"]
    cases: list[GoldenCase] = []
    for i in range(n):
        tpl, sql = _SQL_TEMPLATES[i % len(_SQL_TEMPLATES)]
        params = {"c": countries[i % len(countries)], "r": 1 + (i % 5), "q": 50 + (i % 400), "p": 1 + (i % 80)}
        q = tpl.format(**params)
        gold = sql.format(**params)
        cases.append(GoldenCase(
            case_id=f"sql_{i:03d}",
            app="sql",
            question=q,
            gold_sql=gold,
            metadata={"template": i % len(_SQL_TEMPLATES), "expect_error": i < n_error and i % 2 == 1},
        ))
    return cases


def build_demo_golden_set() -> list[GoldenCase]:
    return build_rag_cases() + build_sql_cases()


# ==========================================================================
# system-under-test protocol
# ==========================================================================
@dataclass
class SystemOutput:
    answer: str = ""
    contexts: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    sql: Optional[str] = None
    rows: Optional[list[dict]] = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    model: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemUnderTest(Protocol):
    app: str
    name: str

    def run(self, case: GoldenCase) -> SystemOutput: ...


# ==========================================================================
# DemoRagSystem
# ==========================================================================
class DemoRagSystem:
    app = "rag"
    name = "demo-rag"
    model = "demo-lexical-retriever+gpt-4o-mini"

    # deterministic rank patterns (length 10) tuned to give recall@5 ~= 0.70
    _RA = [1, 2, 1, 3, 2, 1, 4, 2, 6, 3]
    _RB = [2, 3, 6, 4, 9, 5, 8, 7, 2, 6]
    _REFUSAL_IDX = {17, 63}  # two answerable questions the system wrongly refuses

    def __init__(self, seed: int = 7):
        self.seed = seed
        self._rng = random.Random(seed)

    def _retrieve(self, case: GoldenCase) -> list[str]:
        i = case.metadata.get("topic", 0) if case.metadata else 0
        if case.unanswerable:
            # no relevant doc retrievable; return 10 plausible distractors
            return [_DISTRACTOR_POOL[(i + j) % _RAG_TOPICS] for j in range(10)]
        ra = self._RA[i % 10]
        rb = self._RB[i % 10]
        ranked: list[Optional[str]] = [None] * 12
        ranked[ra - 1] = case.expected_docs[0]
        if ranked[rb - 1] is None:
            ranked[rb - 1] = case.expected_docs[1]
        distract = 0
        for pos in range(12):
            if ranked[pos] is None:
                ranked[pos] = _DISTRACTOR_POOL[(i * 3 + distract + 1) % _RAG_TOPICS]
                distract += 1
        return [d for d in ranked if d]

    def run(self, case: GoldenCase) -> SystemOutput:
        t0 = time.perf_counter()
        contexts = self._retrieve(case)
        i = case.metadata.get("topic", 0) if case.metadata else 0
        docs = [{"doc_id": d, "text": _rag_doc_text(i, "a" if d.endswith("_a") else "b"), "score": round(0.95 - 0.05 * r, 3)}
                for r, d in enumerate(contexts[:5])]

        if case.unanswerable:
            answer = ("I don't know -- the retrieved documents do not contain the "
                      "warranty period for this equipment unit.")
            cites: list[str] = []
        elif i in self._REFUSAL_IDX:
            answer = "I do not have enough information to answer that."
            cites = []
        else:
            fact = _rag_fact(i)
            if i % 23 == 5:  # partial answer: drop the required key fact
                answer = f"Per {case.expected_docs[0]}, the bolts are tightened to the " \
                         f"documented specification and the gear oil is ISO VG graded."
            else:
                answer = (f"According to {case.expected_docs[0]}, the flange bolts are torqued "
                          f"to {fact} in a star pattern. The approved lubricant is a gear oil "
                          f"replaced on the documented interval.")
            cites = list(case.expected_docs)
            if i % 19 == 7:  # occasional wrong citation
                cites = [case.expected_docs[0], _DISTRACTOR_POOL[(i * 5) % _RAG_TOPICS]]

        latency = 180.0 + (i % 25) * 12.0 + (60.0 if i % 11 == 0 else 0.0)
        tokens_in = 900 + (i % 7) * 45
        tokens_out = 60 + (i % 9) * 6
        return SystemOutput(
            answer=answer, contexts=docs, citations=cites,
            latency_ms=round(latency, 1), tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=round(tokens_in * 5e-7 + tokens_out * 1.5e-6, 6),
            model=self.model,
            metadata={"retriever": "lexical", "top_k": 5},
        )


# ==========================================================================
# DemoSqlSystem (executes against real SQLite)
# ==========================================================================
class DemoSqlSystem:
    app = "sql"
    name = "demo-sql"
    model = "demo-text2sql+gpt-4o-mini"

    def __init__(self, conn: Optional[sqlite3.Connection] = None, seed: int = 11,
                 broken_fraction: float = 0.22, error_fraction: float = 0.10,
                 repair_success: float = 0.5):
        self.conn = conn or make_sqlite_db()
        self.seed = seed
        self.broken_fraction = broken_fraction
        self.error_fraction = error_fraction
        self.repair_success = repair_success

    def _execute(self, sql: str) -> tuple[Optional[list[dict]], Optional[str]]:
        try:
            cur = self.conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return rows, None
        except sqlite3.Error as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def run(self, case: GoldenCase) -> SystemOutput:
        gold = case.gold_sql or ""
        i = int(case.case_id.split("_")[1])
        predicted = gold
        errored = False
        repaired = False

        # a) deliberately wrong SQL for ~broken_fraction of cases
        if (i % 100) < int(self.broken_fraction * 100):
            if "COUNT(*)" in predicted:
                predicted = predicted.replace("COUNT(*)", "COUNT(DISTINCT supplier_id)")
            elif "SUM(qty)" in predicted:
                predicted = predicted.replace("SUM(qty)", "SUM(qty) * 2")
            elif "MAX(" in predicted:
                predicted = predicted.replace("MAX(", "MIN(")
            else:
                predicted = predicted.replace(" WHERE ", " WHERE 1=1 OR ")

        # b) deliberate SQL error for ~error_fraction of cases
        if (i % 100) < int(self.error_fraction * 100):
            predicted = predicted.replace("FROM", "FROM") + " LIMIT oops"
            errored = True

        rows, err = self._execute(predicted)
        if err and errored and (i % 2 == 0):  # attempt one repair
            repaired_sql = predicted.replace(" LIMIT oops", "")
            rows2, err2 = self._execute(repaired_sql)
            if err2 is None:
                predicted, rows, err = repaired_sql, rows2, None
                repaired = True

        latency = 220.0 + (i % 17) * 15.0 + (150.0 if err else 0.0)
        tokens_in = 1100 + (i % 6) * 60
        tokens_out = 40 + (i % 7) * 5
        return SystemOutput(
            answer="" if err else f"Query executed; {len(rows or [])} row(s) returned.",
            sql=predicted, rows=rows,
            latency_ms=round(latency, 1), tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=round(tokens_in * 5e-7 + tokens_out * 1.5e-6, 6),
            error=err, model=self.model,
            metadata={"repaired": repaired, "attempts": 2 if (err is None and repaired) else 1},
        )


# ==========================================================================
# runner
# ==========================================================================
def _gold_rows_for(case: GoldenCase, conn: sqlite3.Connection) -> Optional[list[dict]]:
    if not case.gold_sql:
        return None
    try:
        cur = conn.execute(case.gold_sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.Error:
        return None


def run_eval(
    cases: Sequence[GoldenCase],
    system: SystemUnderTest,
    run_id: str,
    *,
    out_dir: Optional[str | Path] = None,
    sql_conn: Optional[sqlite3.Connection] = None,
    judge=None,
) -> dict:
    """Run ``cases`` against ``system``; write traces.jsonl + summary.json.

    Returns the summary dict (also written to ``<out_dir>/summary.json``).
    """
    traces: list[Trace] = []
    rows: list[dict] = []
    judge_scores: list[dict] = []

    conn = sql_conn or getattr(system, "conn", None)

    for case in cases:
        out = system.run(case)
        trace = Trace(
            trace_id=f"{run_id}-{case.case_id}",
            app=case.app,
            run_id=run_id,
            question=case.question,
            answer=out.answer,
            contexts=out.contexts,
            citations=out.citations,
            sql=out.sql,
            rows=out.rows,
            latency_ms=out.latency_ms,
            tokens_in=out.tokens_in,
            tokens_out=out.tokens_out,
            cost_usd=out.cost_usd,
            error=out.error,
            model=out.model,
            metadata={**out.metadata, "case_id": case.case_id, "must_contain": case.must_contain,
                      "expected_docs": case.expected_docs, "unanswerable": case.unanswerable,
                      "gold_sql": case.gold_sql},
        )
        traces.append(trace)

        retrieved = trace.context_ids
        relevant = case.expected_docs
        refused = not trace.answered
        if case.app == "sql":
            gold_rows = _gold_rows_for(case, conn) if conn is not None else None
            sql_match = metrics.sql_execution_match(out.rows, gold_rows) if out.error is None else 0.0
            struct_valid = out.error is None and bool(out.sql)
            task_success = 1.0 if (out.error is None and sql_match == 1.0) else 0.0
        else:
            sql_match = 0.0
            struct_valid = True
            task_success = 1.0 if (case.unanswerable and refused) or (
                not case.unanswerable and not refused
                and metrics.key_fact_containment(trace.answer, case.must_contain) == 1.0
            ) else 0.0

        row = {
            "trace_id": trace.trace_id,
            "case_id": case.case_id,
            "app": case.app,
            "retrieved": retrieved,
            "relevant": relevant,
            "answer": trace.answer,
            "must_contain": case.must_contain,
            "citations": trace.citations,
            "refused": refused,
            "unanswerable": case.unanswerable,
            "latency_ms": trace.latency_ms,
            "cost_usd": trace.cost_usd,
            "tokens_in": trace.tokens_in,
            "tokens_out": trace.tokens_out,
            "error": trace.error,
            "struct_valid": struct_valid,
            "sql_match": sql_match,
            "task_success": task_success,
            "repaired": bool(out.metadata.get("repaired", False)),
        }
        rows.append(row)

        if judge is not None:
            jr = judge.score(trace, case)
            judge_scores.append({"trace_id": trace.trace_id, "app": case.app, **jr.model_dump()})

    flat = metrics.aggregate(rows)
    by_app: dict[str, dict[str, float]] = {}
    for app in sorted({r["app"] for r in rows}):
        by_app[app] = metrics.aggregate([r for r in rows if r["app"] == app])

    # app-specific operational metrics
    sql_rows = [r for r in rows if r["app"] == "sql"]
    if sql_rows:
        flat["valid_sql_rate"] = metrics.schema_validity_rate([r["struct_valid"] for r in sql_rows])
        flat["sql_repair_rate"] = (
            sum(1 for r in sql_rows if r["repaired"]) / sum(1 for r in sql_rows if r["error"] or r["repaired"])
            if any(r["error"] or r["repaired"] for r in sql_rows) else 0.0
        )
    flat["task_success_rate"] = sum(r["task_success"] for r in rows) / len(rows) if rows else 0.0

    summary = {
        "run_id": run_id,
        "app": system.app,
        "system": system.name,
        "model": getattr(system, "model", None),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_traces": len(traces),
        "metrics": flat,
        "by_app": by_app,
        "judge": {
            "backend": judge.name if judge else None,
            "mean_overall": (sum(j["overall"] for j in judge_scores) / len(judge_scores))
            if judge_scores else None,
        },
        "per_trace": rows if len(rows) <= 500 else [],
    }
    if judge_scores:
        summary["judge_scores"] = judge_scores

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "traces.jsonl", traces)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # also emit the flat per-trace metric rows for the dashboard/gate
        (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    """Run the full demo evaluation: 200 traces -> artifacts/demo_run."""
    import argparse

    ap = argparse.ArgumentParser(description="Run the evalkit demo evaluation.")
    ap.add_argument("--out", default="artifacts/demo_run")
    ap.add_argument("--run-id", default="demo-2026-01")
    ap.add_argument("--with-judge", action="store_true", help="also score every trace with the judge")
    ap.add_argument("--use-openai-judge", action="store_true")
    args = ap.parse_args(argv)

    from .judge import get_judge

    conn = make_sqlite_db()
    cases = build_demo_golden_set()

    # RAG pass
    rag = DemoRagSystem()
    rag_cases = [c for c in cases if c.app == "rag"]
    judge = None
    if args.with_judge or args.use_openai_judge:
        judge = get_judge(prefer_openai=args.use_openai_judge)
    rag_summary = run_eval(rag_cases, rag, f"{args.run_id}-rag",
                           out_dir=Path(args.out) / "rag", judge=judge)

    # SQL pass
    sql = DemoSqlSystem(conn=conn)
    sql_cases = [c for c in cases if c.app == "sql"]
    sql_summary = run_eval(sql_cases, sql, f"{args.run_id}-sql",
                           out_dir=Path(args.out) / "sql", sql_conn=conn, judge=judge)

    # combined
    combined = run_eval(cases, _CombinedSystem(rag, sql), args.run_id,
                        out_dir=args.out, sql_conn=conn, judge=judge)

    print(json.dumps({
        "run_id": args.run_id,
        "combined": {k: round(v, 4) for k, v in combined["metrics"].items()},
        "rag": {k: round(v, 4) for k, v in rag_summary["metrics"].items()},
        "sql": {k: round(v, 4) for k, v in sql_summary["metrics"].items()},
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


class _CombinedSystem:
    app = "generic"
    name = "demo-combined"
    model = "multi"

    def __init__(self, rag: DemoRagSystem, sql: DemoSqlSystem):
        self.rag, self.sql = rag, sql

    def run(self, case: GoldenCase) -> SystemOutput:
        return self.rag.run(case) if case.app == "rag" else self.sql.run(case)


if __name__ == "__main__":
    raise SystemExit(main())
