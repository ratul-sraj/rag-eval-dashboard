"""Judge + calibration tests (offline heuristic backend)."""

from evalkit.calibrate import calibrate
from evalkit.judge import HeuristicJudge, cohens_kappa, confusion_matrix
from evalkit.schema import Context, GoldenCase, Trace


def test_cohens_kappa_hand_computed():
    # po=0.5, pe=0.5 -> kappa=0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    # po=0.75, pe=0.5 -> kappa=0.5
    assert cohens_kappa([1, 1, 1, 0], [1, 1, 0, 0]) == 0.5
    # perfect agreement -> 1.0
    assert cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0


def test_confusion_matrix_counts():
    cm = confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_heuristic_judge_rewards_grounded_answer():
    judge = HeuristicJudge()
    ctx = "The flange bolts are torqued to 24 Nm in a star pattern."
    grounded = Trace(
        trace_id="t", app="rag", run_id="r",
        question="What is the flange bolt torque?",
        answer="The flange bolts are torqued to 24 Nm in a star pattern.",
        contexts=[Context(doc_id="d1", text=ctx)], citations=["d1"],
    )
    case = GoldenCase(case_id="c", app="rag", question="What is the flange bolt torque?",
                      must_contain=["24 nm"], expected_docs=["d1"])
    res = judge.score(grounded, case)
    assert res.faithfulness == 1.0
    assert res.correctness >= 0.9
    assert res.acceptable(0.5)


def test_heuristic_judge_penalises_hallucination():
    judge = HeuristicJudge()
    ctx = "The flange bolts are torqued to 24 Nm."
    halluc = Trace(
        trace_id="t", app="rag", run_id="r",
        question="What is the flange bolt torque?",
        answer="The bolts are torqued to 99 Nm and the part has a 15 year warranty.",
        contexts=[Context(doc_id="d1", text=ctx)], citations=["d1"],
    )
    case = GoldenCase(case_id="c", app="rag", must_contain=["24 nm"], expected_docs=["d1"])
    res = judge.score(halluc, case)
    assert res.faithfulness < 1.0


def test_calibrate_on_bundled_labels():
    report = calibrate("samples/judge_calibration.jsonl", judge=HeuristicJudge())
    assert report["n"] == 40
    assert 0.0 <= report["agreement"] <= 1.0
    assert -1.0 <= report["cohens_kappa"] <= 1.0
    cm = report["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == 40
