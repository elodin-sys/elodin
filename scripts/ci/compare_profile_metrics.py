#!/usr/bin/env python3
"""Compare benchmark profile metrics against a known-good baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

METRIC_ORDER = ("build_time_ms", "compile_time_ms", "real_time_factor")
HIGHER_IS_WORSE = {"build_time_ms", "compile_time_ms"}
LOWER_IS_WORSE = {"real_time_factor"}


@dataclass
class MetricTolerance:
    abs_tol: float
    rel_tol: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, help="Example name, e.g. ball or drone")
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to the baseline profile-metrics.json file",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to the candidate profile-metrics.json file",
    )
    parser.add_argument(
        "--tolerances",
        required=True,
        type=Path,
        help="Path to tolerances JSON configuration",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_tolerance(config: dict, example: str, metric: str) -> MetricTolerance:
    perf_cfg = (
        config.get("performance", {}) if isinstance(config.get("performance", {}), dict) else {}
    )
    default_cfg = (
        perf_cfg.get("default", {}) if isinstance(perf_cfg.get("default", {}), dict) else {}
    )
    examples_cfg = (
        perf_cfg.get("examples", {}) if isinstance(perf_cfg.get("examples", {}), dict) else {}
    )
    example_cfg = (
        examples_cfg.get(example, {}) if isinstance(examples_cfg.get(example, {}), dict) else {}
    )

    default_metric_cfg = (
        default_cfg.get(metric, {}) if isinstance(default_cfg.get(metric, {}), dict) else {}
    )
    example_metric_cfg = (
        example_cfg.get(metric, {}) if isinstance(example_cfg.get(metric, {}), dict) else {}
    )

    if not default_metric_cfg and not example_metric_cfg:
        raise ValueError(f"Missing performance tolerance config for '{metric}'")

    abs_tol = float(example_metric_cfg.get("abs_tol", default_metric_cfg.get("abs_tol", 0.0)))
    rel_tol = float(example_metric_cfg.get("rel_tol", default_metric_cfg.get("rel_tol", 0.0)))
    return MetricTolerance(abs_tol=abs_tol, rel_tol=rel_tol)


def _coerce_metric(data: dict, metric: str, path: Path) -> float:
    if metric not in data:
        raise ValueError(f"Missing '{metric}' in {path}")
    value = data[metric]
    try:
        metric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric '{metric}' is not numeric in {path}: {value!r}") from exc
    if not math.isfinite(metric_value):
        raise ValueError(f"Metric '{metric}' must be finite in {path}: {value!r}")
    return metric_value


def _coerce_ticks(data: dict, path: Path) -> int:
    if "ticks" not in data:
        raise ValueError(f"Missing 'ticks' in {path}")
    value = data["ticks"]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ticks value is not an integer in {path}: {value!r}") from exc


def _regression_amount(metric: str, baseline: float, candidate: float) -> float:
    if metric in HIGHER_IS_WORSE:
        return candidate - baseline
    if metric in LOWER_IS_WORSE:
        return baseline - candidate
    raise ValueError(f"Unsupported metric direction for {metric}")


def main() -> int:
    args = _parse_args()

    if not args.baseline.exists():
        print(f"FAIL: baseline profile metrics file does not exist: {args.baseline}")
        return 1
    if not args.candidate.exists():
        print(f"FAIL: candidate profile metrics file does not exist: {args.candidate}")
        return 1
    if not args.tolerances.exists():
        print(f"FAIL: tolerance config does not exist: {args.tolerances}")
        return 1

    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    config = _load_json(args.tolerances)

    baseline_ticks = _coerce_ticks(baseline, args.baseline)
    candidate_ticks = _coerce_ticks(candidate, args.candidate)
    if baseline_ticks != candidate_ticks:
        print(
            f"FAIL: profile metric ticks mismatch for example '{args.example}' "
            f"(baseline={baseline_ticks}, candidate={candidate_ticks})"
        )
        return 1

    failures: list[str] = []
    rtf_baseline = 0.0
    rtf_candidate = 0.0

    for metric in METRIC_ORDER:
        tol = _load_tolerance(config, args.example, metric)
        baseline_value = _coerce_metric(baseline, metric, args.baseline)
        candidate_value = _coerce_metric(candidate, metric, args.candidate)
        regression = _regression_amount(metric, baseline_value, candidate_value)
        allowed = max(tol.abs_tol, abs(baseline_value) * tol.rel_tol)
        rel_regression = regression / max(abs(baseline_value), 1e-30)

        if metric == "real_time_factor":
            rtf_baseline = baseline_value
            rtf_candidate = candidate_value

        if regression > allowed:
            direction = "slower" if metric in HIGHER_IS_WORSE else "lower"
            failures.append(
                f"{metric}: candidate is {direction} than baseline beyond tolerance "
                f"(baseline={baseline_value:.6f}, candidate={candidate_value:.6f}, "
                f"regression={regression:.3e}, rel_regression={rel_regression:.3e}, "
                f"allowed={allowed:.3e}, abs_tol={tol.abs_tol:.3e}, rel_tol={tol.rel_tol:.3e})"
            )

    rtf_pct = ((rtf_candidate - rtf_baseline) / max(abs(rtf_baseline), 1e-30)) * 100
    rtf_sign = "+" if rtf_pct >= 0 else ""
    rtf_detail = f"{rtf_candidate:.1f}x (baseline {rtf_baseline:.1f}x, {rtf_sign}{rtf_pct:.1f}%)"

    # Machine-readable line for regress.sh summary collection
    print(f"RTF_DELTA: {args.example} {rtf_baseline:.3f} {rtf_candidate:.3f} {rtf_pct:.1f}")

    if failures:
        print(f"FAIL: profile metric regression(s) detected for example '{args.example}'")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"PASS: {args.example} profile metrics within tolerance (ticks={candidate_ticks})")
    print(f"  real_time_factor: {rtf_detail}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
