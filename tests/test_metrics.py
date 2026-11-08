"""Hand-computed unit tests for evalkit.metrics."""

import math

import pytest

from evalkit import metrics as M


def test_recall_at_k_hand_computed():
    retrieved = ["a", "b", "c", "d", "e", "f"]
    relevant = ["a", "c", "x"]  # 3 relevant, 2 in the top-5
    assert M.recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert M.recall_at_k(retrieved, relevant, 1) == pytest.approx(1 / 3)
    assert M.recall_at_k(retrieved, relevant, 6) == pytest.approx(2 / 3)


def test_recall_empty_relevant_is_zero():
    assert M.recall_at_k(["a", "b"], [], 5) == 0.0


def test_precision_at_k_hand_computed():
    retrieved = ["a", "b", "c", "d", "e", "f"]
    relevant = ["a", "c", "x"]
    assert M.precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert M.precision_at_k(retrieved, relevant, 2) == pytest.approx(1 / 2)
    assert M.precision_at_k(retrieved, relevant, 0) == 0.0


def test_hit_rate_at_k():
    assert M.hit_rate_at_k(["x", "a"], ["a"], 2) == 1.0
    assert M.hit_rate_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_mrr_first_relevant_rank():
    assert M.mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert M.mrr(["a", "x"], ["a"]) == pytest.approx(1.0)
    assert M.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_at_k_hand_computed():
    # retrieved=[x,a], relevant={a}: gains=[0,1]
    # dcg = 0 + (2^1-1)/log2(3) = 1/1.584962... = 0.63093
    # idcg (ideal 1 relevant) = (2^1-1)/log2(2) = 1.0
    assert M.ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(0.6309297, abs=1e-6)
    # perfect ranking -> 1.0
    assert M.ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)
    # all relevant -> 1.0
    assert M.ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_dcg_discount():
    assert M.dcg_at_k([1.0, 1.0], 2) == pytest.approx(1.0 + 1.0 / math.log2(3))


def test_key_fact_containment_case_insensitive():
    assert M.key_fact_containment("Torque is 24 Nm.", ["24 nm"]) == 1.0
    assert M.key_fact_containment("Torque is 24 Nm.", ["24 nm", "star pattern"]) == pytest.approx(0.5)
    assert M.key_fact_containment("anything", []) == 1.0  # vacuous
    assert M.key_fact_containment("", ["x"]) == 0.0


def test_citation_precision_recall():
    cites = ["a", "b", "z"]
    relevant = ["a", "b"]
    assert M.citation_precision(cites, relevant) == pytest.approx(2 / 3)
    assert M.citation_recall(cites, relevant) == pytest.approx(1.0)
    assert M.citation_precision([], relevant) == 0.0
    assert M.citation_recall(cites, []) == 0.0


def test_refusal_accuracy():
    pred = [True, False, True, False]
    gold = [True, True, False, False]
    assert M.refusal_accuracy(pred, gold) == pytest.approx(2 / 4)
    assert M.refusal_accuracy([], []) == 0.0


def test_schema_validity_rate():
    assert M.schema_validity_rate([True, True, False, True]) == pytest.approx(0.75)
    assert M.schema_validity_rate([]) == 0.0


def test_sql_execution_match_is_order_insensitive_multiset():
    gold = [{"a": 1}, {"a": 2}]
    assert M.sql_execution_match([{"a": 2}, {"a": 1}], gold) == 1.0
    assert M.sql_execution_match([{"a": 1}, {"a": 1}], gold) == 0.0  # multiplicity matters
    assert M.sql_execution_match([{"a": 1}], gold) == 0.0
    assert M.sql_execution_match(None, gold) == 0.0


def test_latency_percentiles_linear_interpolation():
    pct = M.latency_percentiles([10, 20, 30, 40])
    assert pct["p50"] == pytest.approx(25.0)
    assert pct["p95"] == pytest.approx(38.5)
    assert pct["p99"] == pytest.approx(39.7)
    assert pct["mean"] == pytest.approx(25.0)
    assert M.latency_percentiles([])["p50"] == 0.0


def test_cost_and_token_efficiency():
    assert M.cost_total([0.1, 0.2, 0.3]) == pytest.approx(0.6)
    assert M.cost_total([]) == 0.0
    assert M.token_efficiency([1000], [100]) == pytest.approx(100.0)
    assert M.token_efficiency([0], [100]) == 0.0


def test_aggregate_end_to_end():
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a"], "answer": "24 nm", "must_contain": ["24 nm"],
         "citations": ["a"], "refused": False, "unanswerable": False, "latency_ms": 100,
         "cost_usd": 0.001, "tokens_in": 100, "tokens_out": 10, "app": "rag"},
        {"retrieved": ["x"], "relevant": ["a"], "answer": "wrong", "must_contain": ["24 nm"],
         "citations": [], "refused": False, "unanswerable": False, "latency_ms": 300,
         "cost_usd": 0.001, "tokens_in": 100, "tokens_out": 10, "app": "rag"},
    ]
    agg = M.aggregate(rows, k_values=(1, 2))
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["recall@2"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx((1.0 + 0.0) / 2)
    assert agg["key_fact_containment"] == pytest.approx(0.5)
    assert agg["latency_p50_ms"] == pytest.approx(200.0)
    assert agg["cost_total_usd"] == pytest.approx(0.002)
    assert agg["n_traces"] == 2.0
    assert agg["error_rate"] == 0.0
