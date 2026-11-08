"""Rubric-based LLM-as-judge with two interchangeable backends.

Backends
--------
``OpenAIJudge``
    Uses any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, Together,
    Groq, vLLM, Ollama...). Activated automatically when ``OPENAI_API_KEY`` is
    set (override the base URL with ``OPENAI_BASE_URL`` and the model with
    ``EVALKIT_JUDGE_MODEL``). Requires the ``openai`` package.

``HeuristicJudge``
    Fully offline, deterministic, no API key. Scores faithfulness as the
    fraction of answer claims supported by the retrieved contexts (token
    overlap + citation support), relevance as question/answer term overlap, and
    correctness as key-fact containment blended with faithfulness.

The heuristic judge is a **smoke test**, not a substitute for human labels --
see the README limitations section and ``evalkit/calibrate.py``, which reports
how well it agrees with the bundled (synthetic/demo) human-labelled set.
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .schema import GoldenCase, Trace

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "for", "and", "or", "it", "this", "that", "with", "as",
    "at", "by", "from", "we", "you", "they", "he", "she", "i", "do", "does",
    "did", "how", "what", "which", "when", "where", "why", "can", "could",
    "should", "would", "will", "may", "might", "must", "not", "no", "if",
    "than", "then", "so", "such", "into", "about", "over", "under", "also",
    "there", "their", "its", "our", "your", "my", "his", "her", "them", "us",
    "per", "any", "all", "some", "more", "most", "other", "each", "one", "two",
}


class JudgeResult(BaseModel):
    """Per-trace judge output. All scores are floats in [0, 1]."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    backend: str = "heuristic"

    def acceptable(self, threshold: float = 0.5) -> bool:
        """Binary verdict used for calibration (overall >= threshold)."""
        return self.overall >= threshold


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


def _claims(answer: str) -> list[str]:
    """Split an answer into atomic-ish claims (sentences / bullet lines)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*", answer or "")
    return [p.strip() for p in parts if len(p.strip()) > 3]


class BaseJudge:
    name = "base"

    def score(self, trace: Trace, case: Optional[GoldenCase] = None) -> JudgeResult:
        raise NotImplementedError

    def score_batch(self, traces: Sequence[Trace], cases: Optional[Sequence[Optional[GoldenCase]]] = None) -> list[JudgeResult]:
        if cases is None:
            cases = [None] * len(traces)
        return [self.score(t, c) for t, c in zip(traces, cases)]


class HeuristicJudge(BaseJudge):
    """Deterministic offline judge (no network, no API key)."""

    name = "heuristic"

    def __init__(self, overlap_threshold: float = 0.35):
        # A claim is "supported" if this fraction of its content words appear
        # in the union of context texts.
        self.overlap_threshold = overlap_threshold

    # -- sub-scores --------------------------------------------------------
    def _faithfulness(self, trace: Trace) -> tuple[float, str]:
        claims = _claims(trace.answer)
        if not claims:
            return 0.0, "no claims to verify"
        ctx_text = " ".join(
            (c.text or "") for c in trace.contexts
        ).lower()
        ctx_tokens = set(_tokens(ctx_text))
        # citations can also support a claim even if the raw text is absent
        cited = set(c for c in trace.citations if c)
        supported = 0
        for cl in claims:
            ctoks = _tokens(cl)
            if not ctoks:
                supported += 1
                continue
            covered = sum(1 for t in ctoks if t in ctx_tokens) / len(ctoks)
            # a claim that references a cited doc id counts as partially grounded
            if covered >= self.overlap_threshold:
                supported += 1
            elif cited and any(c in cl for c in cited):
                supported += 1
        return supported / len(claims), f"{supported}/{len(claims)} claims supported by context"

    def _relevance(self, trace: Trace) -> tuple[float, str]:
        q = set(_tokens(trace.question))
        a = set(_tokens(trace.answer))
        if not q or not a:
            return 0.0, "empty question or answer"
        overlap = len(q & a) / len(q)
        # length penalty: one-token answers to multi-part questions are less useful
        length_ok = 1.0 if len(_tokens(trace.answer)) >= 4 else 0.6
        return min(1.0, overlap * length_ok * 1.5), f"question-term overlap {overlap:.2f}"

    def _correctness(self, trace: Trace, case: Optional[GoldenCase], faith: float) -> tuple[float, str]:
        from .metrics import key_fact_containment

        if case and case.must_contain:
            kfc = key_fact_containment(trace.answer, case.must_contain)
        elif case and case.app == "sql" and case.gold_sql is not None:
            kfc = 1.0 if trace.sql and trace.sql.strip().rstrip(";") == case.gold_sql.strip().rstrip(";") else 0.0
        else:
            kfc = faith  # no gold -> fall back to grounding
        score = 0.6 * kfc + 0.4 * faith
        return score, f"key-fact containment {kfc:.2f}, grounded {faith:.2f}"

    # -- public ------------------------------------------------------------
    def score(self, trace: Trace, case: Optional[GoldenCase] = None) -> JudgeResult:
        faith, fr = self._faithfulness(trace)
        rel, rr = self._relevance(trace)
        corr, cr = self._correctness(trace, case, faith)
        overall = 0.4 * faith + 0.25 * rel + 0.35 * corr
        return JudgeResult(
            faithfulness=round(faith, 4),
            relevance=round(rel, 4),
            correctness=round(corr, 4),
            overall=round(overall, 4),
            rationale=f"{fr}; {rr}; {cr}",
            backend=self.name,
        )


_RUBRIC_PROMPT = """You are a strict evaluator for a question-answering system.
Score the ANSWER on three 0-10 integer rubrics and reply ONLY with JSON:
{{"faithfulness": int, "relevance": int, "correctness": int, "rationale": str}}

Definitions:
- faithfulness: every claim in the answer is supported by CONTEXTS/citations.
  Unsupported or hallucinated claims -> low score.
- relevance: the answer actually addresses the QUESTION.
- correctness: the answer matches the GOLD key facts (if provided), else equals
  faithfulness.

QUESTION:
{q}

CONTEXTS:
{ctx}

ANSWER:
{a}

GOLD_KEY_FACTS: {gold}
SQL: {sql}
"""


class OpenAIJudge(BaseJudge):
    """Rubric judge backed by an OpenAI-compatible chat completion endpoint."""

    name = "openai"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; use HeuristicJudge for offline runs")
        self.model = model or os.environ.get("EVALKIT_JUDGE_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.timeout = timeout
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai to use OpenAIJudge") from exc
        kwargs = {"api_key": api_key, "timeout": timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def score(self, trace: Trace, case: Optional[GoldenCase] = None) -> JudgeResult:
        ctx = "\n---\n".join((c.text or c.doc_id) for c in trace.contexts)[:6000]
        gold = ""
        if case:
            if case.must_contain:
                gold = "; ".join(case.must_contain)
            elif case.gold_sql:
                gold = case.gold_sql
        prompt = _RUBRIC_PROMPT.format(
            q=trace.question, ctx=ctx or "(none)", a=trace.answer,
            gold=gold or "(none)", sql=trace.sql or "(none)",
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or "{}"
            import json

            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        except Exception as exc:  # network/parse failure -> safe fallback
            fb = HeuristicJudge().score(trace, case)
            fb.rationale = f"[openai-error: {exc}] {fb.rationale}"
            fb.backend = "openai-fallback-heuristic"
            return fb

        def to01(x) -> float:
            try:
                return max(0.0, min(1.0, float(x) / 10.0))
            except (TypeError, ValueError):
                return 0.0

        f_, r_, c_ = to01(data.get("faithfulness")), to01(data.get("relevance")), to01(data.get("correctness"))
        return JudgeResult(
            faithfulness=f_, relevance=r_, correctness=c_,
            overall=0.4 * f_ + 0.25 * r_ + 0.35 * c_,
            rationale=str(data.get("rationale", ""))[:500], backend=self.name,
        )


def get_judge(prefer_openai: bool = True) -> BaseJudge:
    """Return an OpenAI judge if a key is present, else the offline heuristic."""
    if prefer_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIJudge()
        except Exception:
            pass
    return HeuristicJudge()


# --------------------------------------------------------------------------
# calibration statistics
# --------------------------------------------------------------------------
def cohens_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Cohen's kappa for two raters on binary labels.

    kappa = (po - pe) / (1 - pe), where po is observed agreement and pe is
    chance agreement from the marginals. Returns 0.0 when pe == 1 (degenerate).
    """
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    n = len(y_true)
    po = sum(1 for a, b in zip(y_true, y_pred) if int(a) == int(b)) / n
    p_t1 = sum(int(a) for a in y_true) / n
    p_p1 = sum(int(b) for b in y_pred) / n
    pe = p_t1 * p_p1 + (1 - p_t1) * (1 - p_p1)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Binary confusion counts (positive = 'acceptable answer')."""
    tp = fp = fn = tn = 0
    for a, b in zip(y_true, y_pred):
        a, b = int(a), int(b)
        if a == 1 and b == 1:
            tp += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 1 and b == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
