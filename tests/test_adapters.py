"""Adapter tests: mapping loose app output onto the Trace schema."""

import json

from evalkit.adapters import adapt_records, adapt_results_json, load_or_demo, load_native_traces
from evalkit.schema import GoldenCase, Trace, write_jsonl


def test_adapt_loose_records_with_aliases():
    recs = [
        {"id": "r1", "query": "What torque?", "response": "24 Nm",
         "passages": ["manual_a", "manual_b"], "cited_ids": ["manual_a"],
         "latency": 123.4, "prompt_tokens": 100, "completion_tokens": 20},
        {"id": "r2", "question": "Count parts", "generated_sql": "SELECT COUNT(*) FROM parts",
         "result": [{"n": 80}], "duration_ms": 200},
    ]
    traces = adapt_records(recs, app="rag", run_id="ing")
    assert traces[0].trace_id == "r1"
    assert traces[0].context_ids == ["manual_a", "manual_b"]
    assert traces[0].latency_ms == 123.4
    assert traces[0].tokens_in == 100
    assert traces[1].sql == "SELECT COUNT(*) FROM parts"
    assert traces[1].rows == [{"n": 80}]


def test_adapt_results_json_list_and_wrapped(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"results": [{"question": "q", "answer": "a"}]}), encoding="utf-8")
    traces = adapt_results_json(p, app="rag")
    assert len(traces) == 1 and traces[0].answer == "a"

    p2 = tmp_path / "flat.json"
    p2.write_text(json.dumps([{"q": "1"}, {"q": "2"}]), encoding="utf-8")
    traces2 = adapt_results_json(p2, app="generic")
    assert len(traces2) == 2


def test_load_native_traces(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [Trace(trace_id="t1", app="rag", run_id="r")])
    assert [t.trace_id for t in load_native_traces(path)] == ["t1"]


def test_load_or_demo_falls_back_to_demo(tmp_path):
    # demo dir written with 5 traces; an absent repo path must fall back
    demo = tmp_path / "demo"
    write_jsonl(demo / "traces.jsonl", [Trace(trace_id=f"t{i}", app="rag", run_id="r") for i in range(5)])
    traces = load_or_demo("Z:/definitely/not/a/repo", demo_dir=demo)
    assert len(traces) == 5


def test_langfuse_payload_mapping_and_skip():
    from evalkit.adapters import push_to_langfuse, to_langfuse_payload
    t = Trace(trace_id="t1", app="rag", run_id="r", question="q", answer="a",
              contexts=["d1"], citations=["d1"], tokens_in=10, tokens_out=4, latency_ms=100)
    payload = to_langfuse_payload(t)
    assert payload["id"] == "t1"
    assert payload["usage"] == {"input": 10, "output": 4, "total": 14}
    assert payload["metadata"]["citations"] == ["d1"]
    # without keys the push is a no-op that reports why
    res = push_to_langfuse([t], public_key=None, secret_key=None)
    assert res["pushed"] == 0 and res["skipped"]
