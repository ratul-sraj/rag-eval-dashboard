"""Judge calibration: compare judge verdicts against a human-labelled set.

Usage
-----
    python -m evalkit.calibrate --labels samples/judge_calibration.jsonl
    python -m evalkit.calibrate --labels samples/judge_calibration.jsonl --json out.json

Reports observed agreement, Cohen's kappa, the confusion matrix, and a
per-item dump so disagreements can be inspected. The bundled label file is a
SYNTHETIC / DEMO annotation set (see its ``human_note`` fields and the README
limitations) -- treat kappa here as a smoke test of the judge, not evidence of
real-world annotation quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .judge import HeuristicJudge, cohens_kappa, confusion_matrix, get_judge
from .schema import Context, GoldenCase, Trace


def load_labels(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"labels file not found: {p}")
    rows = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "human_label" not in rec:
            raise ValueError(f"{p}:{lineno}: missing 'human_label'")
        rows.append(rec)
    return rows


def to_trace_case(rec: dict) -> tuple[Trace, GoldenCase]:
    trace = Trace(
        trace_id=rec["case_id"],
        app=rec.get("app", "rag"),
        run_id=rec.get("run_id", "calibration"),
        question=rec.get("question", ""),
        answer=rec.get("answer", ""),
        contexts=[Context(**c) if isinstance(c, dict) else c for c in rec.get("contexts", [])],
        citations=rec.get("citations", []),
    )
    case = GoldenCase(
        case_id=rec["case_id"],
        app=rec.get("app", "rag"),
        question=rec.get("question", ""),
        must_contain=rec.get("must_contain", []),
    )
    return trace, case


def calibrate(labels_path: str | Path, judge=None, threshold: float = 0.5) -> dict:
    rows = load_labels(labels_path)
    judge = judge or get_judge(prefer_openai=True)
    y_true: list[int] = []
    y_pred: list[int] = []
    detail: list[dict] = []
    for rec in rows:
        trace, case = to_trace_case(rec)
        res = judge.score(trace, case)
        pred = 1 if res.acceptable(threshold) else 0
        y_true.append(int(rec["human_label"]))
        y_pred.append(pred)
        detail.append({
            "case_id": rec["case_id"],
            "human_label": int(rec["human_label"]),
            "judge_label": pred,
            "judge_overall": res.overall,
            "faithfulness": res.faithfulness,
            "relevance": res.relevance,
            "correctness": res.correctness,
            "agree": pred == int(rec["human_label"]),
            "human_note": rec.get("human_note", ""),
        })

    n = len(y_true)
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    cm = confusion_matrix(y_true, y_pred)
    kappa = cohens_kappa(y_true, y_pred)
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "judge_backend": judge.name,
        "threshold": threshold,
        "n": n,
        "agreement": agree / n if n else 0.0,
        "cohens_kappa": kappa,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detail": detail,
    }


def render(report: dict) -> str:
    cm = report["confusion_matrix"]
    lines = [
        "Judge calibration report",
        "=" * 40,
        f"backend            : {report['judge_backend']}",
        f"threshold          : {report['threshold']}",
        f"labelled items     : {report['n']}",
        f"observed agreement : {report['agreement']:.4f} ({report['agreement']*100:.1f}%)",
        f"Cohen's kappa      : {report['cohens_kappa']:.4f}",
        f"precision (accept) : {report['precision']:.4f}",
        f"recall (accept)    : {report['recall']:.4f}",
        f"F1                 : {report['f1']:.4f}",
        "",
        "Confusion matrix (positive = judge/human says 'acceptable')",
        "            judge=1  judge=0",
        f"human=1     {cm['tp']:>6}   {cm['fn']:>6}",
        f"human=0     {cm['fp']:>6}   {cm['tn']:>6}",
        "",
        "Disagreements:",
    ]
    for d in report["detail"]:
        if not d["agree"]:
            lines.append(
                f"  {d['case_id']}: human={d['human_label']} judge={d['judge_label']} "
                f"(overall={d['judge_overall']:.2f}, faith={d['faithfulness']:.2f}, "
                f"corr={d['correctness']:.2f}) -- {d['human_note'][:70]}"
            )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the evalkit judge.")
    ap.add_argument("--labels", default="samples/judge_calibration.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", default=None, help="write the full report as JSON here")
    ap.add_argument("--heuristic", action="store_true",
                    help="force the offline heuristic judge even if a key is set")
    args = ap.parse_args(argv)

    judge = HeuristicJudge() if args.heuristic else get_judge(prefer_openai=True)
    report = calibrate(args.labels, judge=judge, threshold=args.threshold)
    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
