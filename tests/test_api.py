"""FastAPI endpoint tests using TestClient (in-process, no server)."""

import json
import shutil
from pathlib import Path

import pytest

import api.main as api_mod
from evalkit.runner import build_rag_cases, DemoRagSystem, run_eval


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    # build a small store
    store = tmp_path / "store"
    cases = build_rag_cases(n=20, n_unanswerable=2)
    run_eval(cases, DemoRagSystem(), "test-run", out_dir=store)
    monkeypatch.setattr(api_mod, "STORE", store)
    return TestClient(api_mod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_runs(client):
    r = client.get("/runs")
    assert r.status_code == 200
    runs = r.json()
    assert any(x["run_id"] == "test-run" for x in runs)


def test_run_summary(client):
    r = client.get("/runs/test-run/summary")
    assert r.status_code == 200
    assert r.json()["run_id"] == "test-run"
    assert "metrics" in r.json()


def test_run_summary_404(client):
    assert client.get("/runs/nope/summary").status_code == 404


def test_traces_pagination_and_filter(client):
    r = client.get("/traces", params={"page": 1, "page_size": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 20
    assert len(body["items"]) == 5

    r2 = client.get("/traces", params={"app": "rag", "page_size": 100})
    assert r2.json()["total"] == 20


def test_ingest(client):
    lines = "\n".join(json.dumps({"trace_id": f"z{i}", "app": "rag", "question": "q", "answer": "a"})
                      for i in range(3))
    r = client.post("/ingest", params={"run_id": "z"}, content=lines,
                    headers={"content-type": "text/plain"})
    assert r.status_code == 200
    assert r.json()["ingested"] == 3
    assert r.json()["errors"] == []


def test_ingest_reports_bad_lines(client):
    body = '{"trace_id":"ok","app":"rag","run_id":"z"}\nnot json\n'
    r = client.post("/ingest", params={"run_id": "z2"}, content=body,
                    headers={"content-type": "text/plain"})
    assert r.status_code == 200
    assert r.json()["ingested"] == 1
    assert len(r.json()["errors"]) == 1
