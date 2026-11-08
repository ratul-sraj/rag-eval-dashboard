"""Generate the bundled, SYNTHETIC judge-calibration set.

This file is committed as ``samples/judge_calibration.jsonl`` so the calibration
script is fully reproducible offline. The items are hand-authored demo data in a
manufacturing-docs (Project 1) style; the ``human_label`` column is a demo
label, NOT a real human annotation. Re-run to regenerate:

    python samples/make_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (question, context_id, context_text, key_facts) tuples
ITEMS = [
    ("What is the torque spec for the M8 flange bolt on the pump housing?",
     "manual_pump_04",
     "Pump Housing Assembly, section 4. M8 flange bolts are torqued to 24 Nm in a "
     "star pattern. Do not exceed 26 Nm. Re-torque after the first 50 hours of "
     "operation. Threads must be lubricated with anti-seize compound.",
     ["24 nm", "star pattern"]),
    ("Which lubricant is approved for the gearbox bearings?",
     "manual_gearbox_02",
     "Gearbox Maintenance, section 2. Approved bearing lubricant is ISO VG 220 "
     "synthetic gear oil. Grease fittings accept NLGI grade 2 lithium complex. "
     "Change interval is 4000 operating hours.",
     ["iso vg 220", "4000"]),
    ("What is the maximum operating temperature of the extrusion die?",
     "manual_extrusion_07",
     "Extrusion Die Operation. The die body operates between 180 C and 240 C. The "
     "maximum continuous operating temperature is 240 C. Above 250 C the aluminum "
     "alloy begins to soften and die life drops sharply.",
     ["240 c", "250 c"]),
    ("How often must the safety guard interlock be tested?",
     "manual_safety_01",
     "Machine Safety Interlocks. Guard interlocks must be functionally tested at "
     "every shift change and formally verified monthly. A failed interlock stops "
     "the spindle within 200 ms.",
     ["every shift", "monthly"]),
    ("What is the belt tension for the conveyor drive?",
     "manual_conveyor_03",
     "Conveyor Drive. Belt deflection should be 12 mm at the mid-span under 50 N of "
     "applied force. Over-tensioning shortens bearing life; under-tensioning causes "
     "slippage and belt wear.",
     ["12 mm", "50 n"]),
    ("Which alarm code indicates a coolant flow fault?",
     "manual_alarm_09",
     "Alarm Reference. Alarm E204 signals low coolant flow. Alarm E205 signals "
     "coolant over-temperature. Both require the operator to stop the cycle and "
     "inspect the pump strainer.",
     ["e204", "e205"]),
    ("What is the calibration interval for the pressure transducer?",
     "manual_cal_05",
     "Calibration Schedule. Pressure transducers are calibrated every 6 months "
     "against a NIST-traceable deadweight tester. Temperature sensors follow a "
     "12-month interval. Records are retained for 5 years.",
     ["6 months", "nist"]),
    ("What is the cleanroom particle limit for the assembly cell?",
     "manual_clean_08",
     "Cleanroom Specification. The assembly cell is an ISO Class 7 area, limited to "
     "352000 particles per cubic meter at 0.5 micron. Operators must wear lint-free "
     "garments and nitrile gloves.",
     ["iso class 7", "352000"]),
    ("How do I reset the servo drive after an overcurrent trip?",
     "manual_servo_06",
     "Servo Drive Faults. After an overcurrent trip, clear the fault from the HMI, "
     "wait 30 seconds for the DC bus to discharge, then re-enable the drive. "
     "Persistent trips indicate a shorted winding.",
     ["30 seconds", "hmi"]),
    ("What size filter is used in the hydraulic power unit?",
     "manual_hydraulic_10",
     "Hydraulic Power Unit. The return filter is a 10 micron nominal, 25 micron "
     "absolute element. Replace when the differential pressure indicator is in the "
     "red band or every 1000 hours.",
     ["10 micron", "1000"]),
]


def build() -> list[dict]:
    """Deterministically assemble 40 labelled items from the base facts."""
    items: list[dict] = []
    idx = 0
    for q, cid, ctx, facts in ITEMS:
        # ---- variant A: correct, grounded, cited  (label 1)
        idx += 1
        ans = f"According to {cid}, {facts[0]} is specified. " \
              f"The procedure also notes {facts[1]}."
        items.append({
            "case_id": f"calib_{idx:03d}",
            "question": q,
            "contexts": [{"doc_id": cid, "text": ctx}],
            "answer": ans,
            "citations": [cid],
            "must_contain": facts,
            "human_label": 1,
            "human_note": "Correct and grounded in the cited document.",
        })
        # ---- variant B: correct content, missing citation  (label 1, judge may waver)
        idx += 1
        ans = f"The specified value is {facts[0]} and {facts[1]}."
        items.append({
            "case_id": f"calib_{idx:03d}",
            "question": q,
            "contexts": [{"doc_id": cid, "text": ctx}],
            "answer": ans,
            "citations": [],
            "must_contain": facts,
            "human_label": 1,
            "human_note": "Content correct, citation omitted; still acceptable.",
        })
        # ---- variant C: partial (drops one fact)  (label 0)
        idx += 1
        ans = f"The specification lists {facts[0]}."
        items.append({
            "case_id": f"calib_{idx:03d}",
            "question": q,
            "contexts": [{"doc_id": cid, "text": ctx}],
            "answer": ans,
            "citations": [cid],
            "must_contain": facts,
            "human_label": 0,
            "human_note": "Incomplete: one required key fact is missing.",
        })
        # ---- variant D: hallucinated extra claim  (label 0)
        idx += 1
        ans = (f"The value is {facts[0]} and {facts[1]}. The manual also states the "
               f"part is warrantied for 15 years and is compatible with the XR-9000 series.")
        items.append({
            "case_id": f"calib_{idx:03d}",
            "question": q,
            "contexts": [{"doc_id": cid, "text": ctx}],
            "answer": ans,
            "citations": [cid],
            "must_contain": facts,
            "human_label": 0,
            "human_note": "Contains an unsupported (hallucinated) warranty/compatibility claim.",
        })
    return items


def main() -> None:
    items = build()
    out = HERE / "judge_calibration.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
