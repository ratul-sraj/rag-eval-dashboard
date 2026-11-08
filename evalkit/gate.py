"""Regression gate: compare a current eval run against a stored baseline.

Console script ``evalgate`` (and ``python -m evalkit.gate``). Compares the flat
``metrics`` block of two run summaries, applying per-metric thresholds from a
YAML config, prints a markdown diff table, and exits non-zero on any regression
so CI fails the build.

Config format (``gate.yaml``)::

    on_missing: fail          # fail | warn | ignore
    metrics:
      recall@5:
        direction: higher     # 'higher' = larger is better
        min: 0.60             # absolute floor (breach -> regression)
        max_regression: 0.03  # tolerable absolute drop vs baseline
      latency_p95_ms:
        direction: lower
        max: 900              # absolute ceiling
        max_regression: 0.15  # tolerable fractional *increase* vs baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REG_OK = "pass"
REG_FAIL = "fail"
REG_MISSING = "missing"
REG_WARN = "warn"


@dataclass
class MetricResult:
    metric: str
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    status: str
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status == REG_FAIL


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics_block(run: dict) -> dict:
    if "metrics" in run and isinstance(run["metrics"], dict):
        return run["metrics"]
    # tolerate a bare flat metrics dict
    return {k: v for k, v in run.items() if isinstance(v, (int, float))}


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gate config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("pyyaml is required to read the gate config")
    cfg = yaml.safe_load(text) or {}
    if "metrics" not in cfg:
        raise ValueError("gate config must contain a 'metrics' mapping")
    return cfg


def grade_metric(
    metric: str,
    spec: dict,
    baseline_val: Optional[float],
    current_val: Optional[float],
    on_missing: str = "fail",
) -> MetricResult:
    direction = spec.get("direction", "higher")
    max_reg = spec.get("max_regression")

    if baseline_val is None or current_val is None:
        status = REG_FAIL if on_missing == "fail" else (REG_WARN if on_missing == "warn" else REG_OK)
        return MetricResult(metric, baseline_val, current_val, None, status,
                            "metric absent from baseline/current")

    base = float(baseline_val)
    cur = float(current_val)
    delta = cur - base

    # ---- absolute threshold breaches -------------------------------------
    if direction == "higher" and spec.get("min") is not None and cur < spec["min"]:
        return MetricResult(metric, base, cur, delta, REG_FAIL,
                            f"below absolute min {spec['min']}")
    if direction == "lower" and spec.get("max") is not None and cur > spec["max"]:
        return MetricResult(metric, base, cur, delta, REG_FAIL,
                            f"above absolute max {spec['max']}")

    # ---- regression vs baseline ------------------------------------------
    if max_reg is not None:
        if base == 0.0:
            # zero-baseline guard: a relative comparison is undefined, so fall
            # back to an absolute check only and note it.
            if direction == "higher":
                regressed = cur < -float(max_reg)
            else:
                regressed = cur > float(max_reg)
            if regressed:
                return MetricResult(metric, base, cur, delta, REG_FAIL,
                                    "regressed from a zero baseline (absolute check)")
            return MetricResult(metric, base, cur, delta, REG_OK,
                                "zero baseline: absolute-only check")

        if direction == "higher":
            if delta < -abs(float(max_reg)):
                return MetricResult(metric, base, cur, delta, REG_FAIL,
                                    f"dropped {abs(delta):.4f} (> {max_reg})")
        else:  # lower is better -> a *relative increase* is a regression
            rel = delta / abs(base) if base else 0.0
            if rel > float(max_reg):
                return MetricResult(metric, base, cur, delta, REG_FAIL,
                                    f"increased {rel*100:.1f}% (> {float(max_reg)*100:.0f}%)")

    return MetricResult(metric, base, cur, delta, REG_OK)


def run_gate(baseline: dict, current: dict, config: dict) -> list[MetricResult]:
    base_m = _metrics_block(baseline)
    cur_m = _metrics_block(current)
    on_missing = config.get("on_missing", "fail")
    results: list[MetricResult] = []
    for metric, spec in config["metrics"].items():
        spec = spec or {}
        results.append(grade_metric(metric, spec, base_m.get(metric), cur_m.get(metric), on_missing))
    return results


def render_markdown(results: list[MetricResult], meta: Optional[dict] = None) -> str:
    lines = ["# Regression gate report", ""]
    if meta:
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("| metric | baseline | current | delta | status | note |")
    lines.append("|---|---:|---:|---:|:--:|---|")

    def fmt(x: Optional[float]) -> str:
        return "-" if x is None else f"{x:.4f}"

    for r in results:
        delta = "-" if r.delta is None else f"{r.delta:+.4f}"
        status = {"pass": "PASS", "fail": "**FAIL**", "missing": "MISSING", "warn": "WARN"}[r.status]
        lines.append(
            f"| {r.metric} | {fmt(r.baseline)} | {fmt(r.current)} | {delta} | {status} | {r.reason} |"
        )
    n_fail = sum(1 for r in results if r.failed)
    lines.append("")
    lines.append(f"**{len(results) - n_fail}/{len(results)} metrics passed**"
                 + (f" -- {n_fail} regression(s) detected." if n_fail else " -- no regressions."))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="evalkit regression gate")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--config", default="gate.yaml")
    ap.add_argument("--markdown-out", default=None, help="write the markdown report here")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    try:
        config = load_config(args.config)
        baseline = load_json(args.baseline)
        current = load_json(args.current)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"gate error: {exc}", file=sys.stderr)
        return 2

    results = run_gate(baseline, current, config)
    md = render_markdown(results, meta={
        "baseline run": baseline.get("run_id", "?"),
        "current run": current.get("run_id", "?"),
    })
    print(md)
    if args.markdown_out:
        Path(args.markdown_out).write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")

    failed = [r for r in results if r.failed]
    if failed:
        print(f"\nGATE FAILED: {len(failed)} regression(s): "
              + ", ".join(r.metric for r in failed), file=sys.stderr)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
