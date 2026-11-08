"""Schema round-trip and validation-error tests."""

import json

import pytest

from evalkit.schema import GoldenCase, SchemaError, Trace, read_jsonl, write_jsonl


def test_trace_minimal_defaults():
    t = Trace(trace_id="t1", app="rag", run_id="r1")
    assert t.latency_ms == 0.0
    assert t.contexts == []
    assert t.timestamp  # auto-filled
    assert t.total_tokens == 0


def test_trace_jsonl_roundtrip(tmp_path):
    traces = [
        Trace(trace_id="t1", app="rag", run_id="r1", question="q", answer="a",
              contexts=["doc_a", "doc_b"], citations=["doc_a"], latency_ms=12.5),
        Trace(trace_id="t2", app="sql", run_id="r1", sql="SELECT 1", rows=[{"x": 1}]),
    ]
    path = tmp_path / "t.jsonl"
    n = write_jsonl(path, traces)
    assert n == 2
    back = read_jsonl(path, Trace)
    assert [t.trace_id for t in back] == ["t1", "t2"]
    assert back[0].context_ids == ["doc_a", "doc_b"]
    assert back[1].rows == [{"x": 1}]


def test_contexts_coerced_from_strings_and_alias_dicts():
    t = Trace(trace_id="t", app="rag", run_id="r",
              contexts=["d1", {"chunk_id": "d2"}, {"doc_id": "d3", "text": "hi"}])
    assert t.context_ids == ["d1", "d2", "d3"]
    assert t.contexts[2].text == "hi"


def test_missing_trace_id_raises():
    with pytest.raises(Exception):
        Trace(app="rag", run_id="r")  # type: ignore[call-arg]


def test_read_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"trace_id":"ok","app":"rag","run_id":"r"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(SchemaError) as exc:
        read_jsonl(path, Trace)
    assert "line 2" in str(exc.value) or ":2:" in str(exc.value)


def test_read_jsonl_reports_schema_error_with_location(tmp_path):
    path = tmp_path / "bad2.jsonl"
    path.write_text('{"app":"rag","run_id":"r"}\n', encoding="utf-8")  # no trace_id
    with pytest.raises(SchemaError) as exc:
        read_jsonl(path, Trace)
    assert "trace_id" in str(exc.value)


def test_read_jsonl_missing_file():
    with pytest.raises(SchemaError):
        read_jsonl("does-not-exist.jsonl", Trace)


def test_golden_case_roundtrip(tmp_path):
    cases = [
        GoldenCase(case_id="c1", app="rag", question="q", expected_docs=["a"], must_contain=["x"]),
        GoldenCase(case_id="c2", app="sql", question="q", gold_sql="SELECT 1"),
        GoldenCase(case_id="c3", app="rag", question="?", unanswerable=True),
    ]
    path = tmp_path / "g.jsonl"
    write_jsonl(path, cases)
    back = read_jsonl(path, GoldenCase)
    assert [c.case_id for c in back] == ["c1", "c2", "c3"]
    assert back[2].unanswerable is True


def test_golden_case_requires_id():
    with pytest.raises(Exception):
        GoldenCase(app="rag")
