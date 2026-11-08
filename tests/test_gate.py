"""Regression-gate tests: pass, regression, missing metric, zero-baseline guard."""

from evalkit.gate import (
    MetricResult,
    grade_metric,
    render_markdown,
    run_gate,
)


def _cfg():
    return {
        "on_missing": "fail",
        "metrics": {
            "recall@5": {"direction": "higher", "min": 0.6, "max_regression": 0.05},
            "latency_p95_ms": {"direction": "lower", "max": 800, "max_regression": 0.25},
        },
    }


def test_pass_case():
    base = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 460.0}}
    cur = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 455.0}}
    results = run_gate(base, cur, _cfg())
    assert all(not r.failed for r in results)
    assert {r.metric: r.status for r in results} == {"recall@5": "pass", "latency_p95_ms": "pass"}


def test_regression_case_higher_is_better():
    base = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 460.0}}
    cur = {"metrics": {"recall@5": 0.40, "latency_p95_ms": 460.0}}
    results = run_gate(base, cur, _cfg())
    r = {x.metric: x for x in results}
    assert r["recall@5"].failed
    assert "min" in r["recall@5"].reason or "dropped" in r["recall@5"].reason


def test_regression_case_lower_is_better_relative():
    # latency up 50% vs baseline, above the 25% tolerance
    base = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 500.0}}
    cur = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 760.0}}
    results = run_gate(base, cur, _cfg())
    r = {x.metric: x for x in results}
    assert r["latency_p95_ms"].failed


def test_lower_is_better_within_tolerance_passes():
    base = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 500.0}}
    cur = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 560.0}}  # +12%
    results = run_gate(base, cur, _cfg())
    r = {x.metric: x for x in results}
    assert not r["latency_p95_ms"].failed


def test_missing_metric_fails_by_default():
    base = {"metrics": {"recall@5": 0.70, "latency_p95_ms": 460.0}}
    cur = {"metrics": {"recall@5": 0.70}}  # latency absent
    results = run_gate(base, cur, _cfg())
    r = {x.metric: x for x in results}
    assert r["latency_p95_ms"].failed
    assert r["latency_p95_ms"].status == "fail"


def test_missing_metric_warn_mode():
    cfg = _cfg()
    cfg["on_missing"] = "warn"
    base = {"metrics": {"recall@5": 0.70}}
    cur = {"metrics": {"recall@5": 0.70}}
    results = run_gate(base, cur, cfg)
    r = {x.metric: x for x in results}
    assert r["latency_p95_ms"].status == "warn"
    assert not r["latency_p95_ms"].failed


def test_zero_baseline_guard_absolute_only():
    spec = {"direction": "higher", "max_regression": 0.05}
    # baseline 0, current 0 -> not a regression (relative undefined)
    res = grade_metric("m", spec, 0.0, 0.0)
    assert res.status == "pass"
    assert "zero baseline" in res.reason
    # baseline 0, current negative -> regression (absolute check)
    res2 = grade_metric("m", spec, 0.0, -0.2)
    assert res2.failed


def test_zero_baseline_lower_is_better_guard():
    spec = {"direction": "lower", "max_regression": 0.10}
    assert grade_metric("m", spec, 0.0, 0.0).status == "pass"
    assert grade_metric("m", spec, 0.0, 0.5).failed


def test_absolute_ceiling_breach_on_lower_metric():
    spec = {"direction": "lower", "max": 800, "max_regression": 0.25}
    res = grade_metric("latency", spec, 500.0, 900.0)
    assert res.failed
    assert "max" in res.reason


def test_render_markdown_contains_rows_and_summary():
    results = [
        MetricResult("recall@5", 0.7, 0.4, -0.3, "fail", "below absolute min 0.6"),
        MetricResult("mrr", 0.6, 0.61, 0.01, "pass", ""),
    ]
    md = render_markdown(results, meta={"baseline run": "b"})
    assert "| recall@5 |" in md
    assert "**FAIL**" in md
    assert "1/2 metrics passed" in md
