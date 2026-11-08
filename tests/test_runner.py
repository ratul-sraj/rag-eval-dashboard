"""Runner + demo-system tests (deterministic, offline)."""

import json

from evalkit.runner import (
    DemoRagSystem,
    DemoSqlSystem,
    build_demo_golden_set,
    build_rag_cases,
    build_sql_cases,
    make_sqlite_db,
    run_eval,
)


def test_golden_set_size():
    cases = build_demo_golden_set()
    assert len(cases) == 200
    assert sum(1 for c in cases if c.app == "rag") == 100
    assert sum(1 for c in cases if c.app == "sql") == 100
    assert sum(1 for c in cases if c.unanswerable) == 10


def test_demo_rag_is_deterministic_and_recall_target():
    cases = build_rag_cases()
    s1 = run_eval(cases, DemoRagSystem(), "r1")
    s2 = run_eval(cases, DemoRagSystem(), "r1")
    assert s1["metrics"]["recall@5"] == s2["metrics"]["recall@5"]
    assert abs(s1["metrics"]["recall@5"] - 0.70) < 0.02
    # deliberately imperfect: not a perfect citation score
    assert s1["metrics"]["citation_precision"] < 1.0
    # two answerable refusals -> refusal rate above the 10% unanswerable floor
    assert s1["metrics"]["refusal_rate"] > 0.10


def test_demo_sql_executes_real_sqlite():
    conn = make_sqlite_db()
    cases = build_sql_cases()
    summary = run_eval(cases, DemoSqlSystem(conn=conn), "sql", sql_conn=conn)
    m = summary["metrics"]
    assert 0.5 < m["sql_execution_match"] < 1.0   # imperfect but mostly correct
    assert 0.0 < m["error_rate"] < 0.5
    assert m["valid_sql_rate"] > 0.5


def test_run_eval_writes_artifacts(tmp_path):
    cases = build_rag_cases(n=20, n_unanswerable=2)
    summary = run_eval(cases, DemoRagSystem(), "mini", out_dir=tmp_path)
    assert (tmp_path / "traces.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.json").exists()
    traces = [json.loads(l) for l in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 20
    assert summary["n_traces"] == 20
    assert summary["metrics"]["n_traces"] == 20.0
    assert "by_app" in summary
