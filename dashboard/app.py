"""evalkit dashboard -- a Streamlit monitoring UI for the evaluation layer.

Run::

    streamlit run dashboard/app.py
    # or: make dashboard

Pages
-----
Overview            KPIs, latency/cost percentiles, error & refusal rate, sparklines
Retrieval quality   recall@k, MRR, nDCG charts per run
Agent quality       task success, valid-SQL rate, repair rate, execution match
Traces explorer     filter by app / errors / latency, expandable trace + citation view
Regression compare  two runs side by side, delta table, pass/fail badges
Judge calibration   agreement, Cohen's kappa, confusion matrix vs human labels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evalkit.calibrate import calibrate  # noqa: E402
from evalkit.gate import load_config, run_gate  # noqa: E402
from evalkit.judge import HeuristicJudge  # noqa: E402
from evalkit.schema import read_jsonl, Trace  # noqa: E402

STORE = ROOT / "artifacts" / "demo_run"
SAMPLES = ROOT / "samples"

st.set_page_config(page_title="evalkit dashboard", layout="wide", initial_sidebar_state="expanded")

# dark-friendly, low-chrome theme; no emojis anywhere
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .kpi-card {background:#161b22; border:1px solid #30363d; border-radius:10px;
                 padding:14px 16px; margin-bottom:8px;}
      .kpi-label {color:#8b949e; font-size:0.78rem; text-transform:uppercase;
                  letter-spacing:0.06em; margin:0;}
      .kpi-value {color:#e6edf3; font-size:1.7rem; font-weight:600; margin:2px 0 0 0;}
      .kpi-sub {color:#8b949e; font-size:0.8rem; margin:0;}
      .badge-pass {background:#12351f; color:#3fb950; border:1px solid #238636;
                   border-radius:6px; padding:2px 8px; font-size:0.8rem;}
      .badge-fail {background:#3d1418; color:#f85149; border:1px solid #da3633;
                   border-radius:6px; padding:2px 8px; font-size:0.8rem;}
      .badge-warn {background:#3a2d12; color:#d29922; border:1px solid #9e6a03;
                   border-radius:6px; padding:2px 8px; font-size:0.8rem;}
      div[data-testid="stMetricValue"] {color:#e6edf3;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def discover_runs() -> dict[str, str]:
    runs: dict[str, str] = {}
    if not STORE.exists():
        return runs
    for summary in STORE.rglob("summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs[data.get("run_id", summary.parent.name)] = str(summary.parent)
    return runs


@st.cache_data(show_spinner=False)
def load_summary(dirpath: str) -> dict:
    return json.loads((Path(dirpath) / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metrics(dirpath: str) -> list[dict]:
    return json.loads((Path(dirpath) / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_traces(dirpath: str) -> list[dict]:
    return [t.model_dump() for t in read_jsonl(Path(dirpath) / "traces.jsonl", Trace)]


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="kpi-card"><p class="kpi-label">{label}</p>'
        f'<p class="kpi-value">{value}</p><p class="kpi-sub">{sub}</p></div>',
        unsafe_allow_html=True,
    )


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def fmt_num(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# page: overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Overview")
    runs = discover_runs()
    if not runs:
        st.warning("No runs found. Run `python -m evalkit run --out artifacts/demo_run` first.")
        return

    combined = next((d for rid, d in runs.items() if "combined" not in rid and rid.endswith("2026-01")), None)
    options = list(runs.keys())
    default_idx = options.index("demo-2026-01") if "demo-2026-01" in options else 0
    run_id = st.selectbox("Run", options, index=default_idx)
    dirpath = runs[run_id]
    summary = load_summary(dirpath)
    metrics = summary["metrics"]
    rows = load_metrics(dirpath)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Traces", f"{int(metrics.get('n_traces', 0))}", summary.get("system", ""))
    with c2:
        kpi("Task success", fmt_pct(metrics.get("task_success_rate")), "answered correctly / parsed")
    with c3:
        kpi("Error rate", fmt_pct(metrics.get("error_rate")), "trace.error present")
    with c4:
        kpi("Refusal rate", fmt_pct(metrics.get("refusal_rate")), "declined to answer")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi("Latency p50", f"{fmt_num(metrics.get('latency_p50_ms'), 0)} ms", "median")
    with c6:
        kpi("Latency p95", f"{fmt_num(metrics.get('latency_p95_ms'), 0)} ms", "tail")
    with c7:
        kpi("Cost total", f"${fmt_num(metrics.get('cost_total_usd'), 4)}", f"per trace ${fmt_num(metrics.get('cost_per_trace_usd'), 6)}")
    with c8:
        kpi("Token efficiency", fmt_num(metrics.get("token_efficiency"), 1), "out tokens / 1k in")

    st.subheader("Quality metrics")
    qual = {
        "recall@5": metrics.get("recall@5"),
        "mrr": metrics.get("mrr"),
        "ndcg@5": metrics.get("ndcg@5"),
        "key-fact containment": metrics.get("key_fact_containment"),
        "citation precision": metrics.get("citation_precision"),
        "sql execution match": metrics.get("sql_execution_match"),
    }
    qdf = pd.DataFrame({"metric": list(qual.keys()), "value": list(qual.values())})
    st.dataframe(qdf, width="stretch", hide_index=True)

    st.subheader("Latency distribution")
    lat = pd.Series([r["latency_ms"] for r in rows if r.get("latency_ms")], name="latency_ms")
    if not lat.empty:
        hist = pd.cut(lat, bins=20).value_counts().sort_index()
        st.bar_chart(pd.DataFrame({"count": hist.values}, index=[f"{iv.left:.0f}-{iv.right:.0f}" for iv in hist.index]))

    st.subheader("Run identity")
    st.caption(f"run_id `{run_id}`  |  created {summary.get('created_at', '?')}  |  judge backend "
               f"{summary.get('judge', {}).get('backend')}  |  mean judge score "
               f"{fmt_num(summary.get('judge', {}).get('mean_overall'), 3)}")


# --------------------------------------------------------------------------
# page: retrieval quality
# --------------------------------------------------------------------------
def page_retrieval() -> None:
    st.title("Retrieval quality")
    runs = discover_runs()
    rag_runs = {rid: d for rid, d in runs.items() if "sql" not in rid}
    if not rag_runs:
        st.warning("No retrieval runs found.")
        return

    st.caption("Retrieval metrics are averaged over cases that declare gold documents.")
    table = []
    for rid, d in rag_runs.items():
        m = load_summary(d)["metrics"]
        table.append({
            "run": rid,
            "recall@1": m.get("recall@1"), "recall@5": m.get("recall@5"), "recall@10": m.get("recall@10"),
            "precision@5": m.get("precision@5"), "hit_rate@5": m.get("hit_rate@5"),
            "mrr": m.get("mrr"), "ndcg@5": m.get("ndcg@5"), "ndcg@10": m.get("ndcg@10"),
        })
    df = pd.DataFrame(table)
    st.dataframe(df, width="stretch", hide_index=True)

    k_metric = st.selectbox("Metric to chart", ["recall@k", "precision@k", "ndcg@k", "hit_rate@k"])
    prefix = k_metric.split("@")[0]
    chart_rows = []
    for rid, d in rag_runs.items():
        m = load_summary(d)["metrics"]
        for k in (1, 5, 10):
            chart_rows.append({"run": rid, "k": f"k={k}", "value": m.get(f"{prefix}@{k}", 0.0)})
    cdf = pd.DataFrame(chart_rows)
    if not cdf.empty:
        pivot = cdf.pivot(index="k", columns="run", values="value")
        st.bar_chart(pivot)

    st.subheader("MRR per run")
    mrr_df = pd.DataFrame([{"run": rid, "mrr": load_summary(d)["metrics"].get("mrr", 0.0)}
                           for rid, d in rag_runs.items()])
    st.bar_chart(mrr_df.set_index("run"))


# --------------------------------------------------------------------------
# page: agent quality
# --------------------------------------------------------------------------
def page_agent() -> None:
    st.title("Agent quality")
    runs = discover_runs()
    if not runs:
        st.warning("No runs found.")
        return
    table = []
    for rid, d in runs.items():
        m = load_summary(d)["metrics"]
        table.append({
            "run": rid,
            "task_success_rate": m.get("task_success_rate"),
            "valid_sql_rate": m.get("valid_sql_rate"),
            "sql_repair_rate": m.get("sql_repair_rate"),
            "sql_execution_match": m.get("sql_execution_match"),
            "error_rate": m.get("error_rate"),
            "refusal_accuracy": m.get("refusal_accuracy"),
        })
    df = pd.DataFrame(table)
    st.dataframe(df, width="stretch", hide_index=True)

    sel = st.selectbox("Run to chart", list(runs.keys()))
    m = load_summary(runs[sel])["metrics"]
    cols = st.columns(4)
    with cols[0]:
        kpi("Valid SQL", fmt_pct(m.get("valid_sql_rate")), "parsed + executed")
    with cols[1]:
        kpi("SQL repair rate", fmt_pct(m.get("sql_repair_rate")), "first-attempt failures repaired")
    with cols[2]:
        kpi("Execution match", fmt_pct(m.get("sql_execution_match")), "rows equal gold")
    with cols[3]:
        kpi("Refusal accuracy", fmt_pct(m.get("refusal_accuracy")), "refuse vs answerable")

    rows = load_metrics(runs[sel])
    sql_rows = [r for r in rows if r["app"] == "sql"]
    if sql_rows:
        st.subheader("SQL outcome breakdown")
        outcome = {
            "execution match": sum(1 for r in sql_rows if r["sql_match"] == 1.0),
            "wrong result": sum(1 for r in sql_rows if not r["error"] and r["sql_match"] == 0.0),
            "errored": sum(1 for r in sql_rows if r["error"]),
        }
        st.bar_chart(pd.DataFrame({"count": list(outcome.values())}, index=list(outcome.keys())))


# --------------------------------------------------------------------------
# page: traces explorer
# --------------------------------------------------------------------------
def page_traces() -> None:
    st.title("Traces explorer")
    runs = discover_runs()
    if not runs:
        st.warning("No runs found.")
        return
    run_id = st.selectbox("Run", list(runs.keys()), key="trace_run")
    rows = load_metrics(runs[run_id])
    traces = {t["trace_id"]: t for t in load_traces(runs[run_id])}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        app_filter = st.multiselect("App", sorted({r["app"] for r in rows}), default=sorted({r["app"] for r in rows}))
    with c2:
        errors_only = st.checkbox("Errors only", value=False)
    with c3:
        min_lat = st.number_input("Min latency (ms)", value=0.0, step=50.0)
    with c4:
        max_lat = st.number_input("Max latency (ms)", value=100000.0, step=50.0)

    filt = [r for r in rows if r["app"] in app_filter and min_lat <= r["latency_ms"] <= max_lat]
    if errors_only:
        filt = [r for r in filt if r["error"]]
    st.caption(f"{len(filt)} of {len(rows)} traces")

    page_size = st.slider("Rows per page", 10, 100, 25)
    n_pages = max(1, (len(filt) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, n_pages, 1)
    chunk = filt[(page - 1) * page_size: page * page_size]

    st.dataframe(
        pd.DataFrame([{
            "case": r["case_id"], "app": r["app"], "latency_ms": r["latency_ms"],
            "error": bool(r["error"]), "sql_match": r["sql_match"], "refused": r["refused"],
            "answer": (r["answer"] or "")[:80],
        } for r in chunk]),
        width="stretch", hide_index=True,
    )

    st.subheader("Trace detail")
    if chunk:
        pick = st.selectbox("Trace", [r["case_id"] for r in chunk])
        row = next(r for r in chunk if r["case_id"] == pick)
        trace = traces.get(row["trace_id"], {})
        with st.expander("Question & answer", expanded=True):
            st.markdown(f"**Q.** {trace.get('question', '')}")
            st.markdown(f"**A.** {trace.get('answer', '')}")
            if trace.get("sql"):
                st.code(trace["sql"], language="sql")
            if row.get("error"):
                st.error(f"error: {row['error']}")
        with st.expander("Retrieved contexts", expanded=False):
            for c in trace.get("contexts", []):
                label = c.get("doc_id", "?")
                score = c.get("score")
                st.markdown(f"**{label}**" + (f" (score {score})" if score is not None else ""))
                st.caption((c.get("text") or "")[:400])
            if not trace.get("contexts"):
                st.write("none")
        with st.expander("Citations", expanded=True):
            rel = set(row.get("relevant") or [])
            cites = trace.get("citations", [])
            if not cites:
                st.write("no citations emitted")
            for c in cites:
                good = c in rel
                st.markdown(f'<span class="{"badge-pass" if good else "badge-fail"}">{c} '
                            f'{"cited and relevant" if good else "cited but not gold"}</span>',
                            unsafe_allow_html=True)
        with st.expander("Result rows", expanded=False):
            if trace.get("rows"):
                st.dataframe(pd.DataFrame(trace["rows"]), width="stretch")
            else:
                st.write("none")


# --------------------------------------------------------------------------
# page: regression compare
# --------------------------------------------------------------------------
def page_regression() -> None:
    st.title("Regression compare")
    a, b = st.columns(2)
    with a:
        base_path = st.selectbox("Baseline", sorted(str(p) for p in SAMPLES.glob("*.json")) + ["custom path..."])
    with b:
        cur_path = st.selectbox("Current", sorted(str(p) for p in SAMPLES.glob("*.json")), index=1)

    if base_path == "custom path...":
        base_path = st.text_input("Baseline path", value=str(SAMPLES / "baseline.json"))

    try:
        baseline = json.loads(Path(base_path).read_text(encoding="utf-8"))
        current = json.loads(Path(cur_path).read_text(encoding="utf-8"))
        config = load_config(ROOT / "gate.yaml")
    except Exception as exc:  # noqa: BLE001
        st.error(f"could not load files: {exc}")
        return

    results = run_gate(baseline, current, config)
    bm, cm = baseline["metrics"], current["metrics"]

    n_fail = sum(1 for r in results if r.failed)
    st.markdown(
        (f'<span class="badge-fail">GATE FAILED</span> {n_fail} regression(s)'
         if n_fail else '<span class="badge-pass">GATE PASSED</span> no regressions'),
        unsafe_allow_html=True,
    )

    table = []
    for r in results:
        table.append({
            "metric": r.metric,
            "baseline": r.baseline, "current": r.current, "delta": r.delta,
            "status": r.status, "note": r.reason,
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    st.subheader("Delta chart")
    deltas = pd.DataFrame({
        "metric": [r.metric for r in results if r.delta is not None],
        "delta": [r.delta for r in results if r.delta is not None],
    })
    if not deltas.empty:
        st.bar_chart(deltas.set_index("metric"))

    st.caption(f"baseline `{baseline.get('run_id')}` vs current `{current.get('run_id')}`; "
               f"{len(results) - n_fail}/{len(results)} metrics passed")


# --------------------------------------------------------------------------
# page: judge calibration
# --------------------------------------------------------------------------
def page_calibration() -> None:
    st.title("Judge calibration")
    st.caption("Compares the deterministic heuristic judge against a synthetic, demo human-labelled set "
               "(40 items). Treat the numbers as a smoke test, not real annotation quality.")
    labels_path = SAMPLES / "judge_calibration.jsonl"
    if not labels_path.exists():
        st.warning("labels file not found: samples/judge_calibration.jsonl")
        return
    try:
        report = calibrate(labels_path, judge=HeuristicJudge())
    except Exception as exc:  # noqa: BLE001
        st.error(f"calibration failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Agreement", f"{report['agreement'] * 100:.1f}%", f"{report['n']} labelled items")
    with c2:
        kpi("Cohen's kappa", f"{report['cohens_kappa']:.3f}", "chance-corrected agreement")
    with c3:
        kpi("Precision (accept)", f"{report['precision']:.3f}", "of judge-accepted")
    with c4:
        kpi("Recall (accept)", f"{report['recall']:.3f}", "of human-accepted")

    cm = report["confusion_matrix"]
    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]],
        index=["human: acceptable", "human: not acceptable"],
        columns=["judge: acceptable", "judge: not acceptable"],
    )
    st.dataframe(cm_df, width="stretch")

    st.subheader("Per-item judge verdicts")
    detail = pd.DataFrame(report["detail"])[
        ["case_id", "human_label", "judge_label", "judge_overall", "faithfulness", "relevance", "correctness", "agree"]
    ]
    st.dataframe(detail, width="stretch", hide_index=True)

    st.subheader("Disagreements")
    disagree = [d for d in report["detail"] if not d["agree"]]
    if not disagree:
        st.write("none")
    for d in disagree:
        st.markdown(f"- `{d['case_id']}` human={d['human_label']} judge={d['judge_label']} "
                    f"(overall {d['judge_overall']:.2f}) -- {d['human_note']}")


# --------------------------------------------------------------------------
# navigation
# --------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Retrieval quality": page_retrieval,
    "Agent quality": page_agent,
    "Traces explorer": page_traces,
    "Regression compare": page_regression,
    "Judge calibration": page_calibration,
}


def main() -> None:
    st.sidebar.title("evalkit")
    st.sidebar.caption("evaluation & monitoring for LLM apps")
    choice = st.sidebar.radio("Page", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown(
        "Data source\n\n"
        f"`{STORE.relative_to(ROOT) if STORE.exists() else 'artifacts/demo_run'}`"
    )
    st.sidebar.caption("Run `python -m evalkit run --out artifacts/demo_run` to refresh artifacts.")
    PAGES[choice]()


main()
