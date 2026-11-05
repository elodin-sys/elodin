#!/usr/bin/env python3
"""Compare the median profile metrics of candidate runs against base runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

METRIC_ORDER = ("build_time_ms", "real_time_factor_median")
HIGHER_IS_WORSE = {"build_time_ms"}
LOWER_IS_WORSE = {"real_time_factor_median"}


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
        nargs="+",
        type=Path,
        help="Profile metrics JSON files from the base runs",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        nargs="+",
        type=Path,
        help="Profile metrics JSON files from the candidate runs",
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


def _coerce_metric(data: dict, metric: str, path: Path) -> float | None:
    if metric not in data:
        raise ValueError(f"Missing '{metric}' in {path}")
    value = data[metric]
    if value is None:
        return None
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


def _median_metric(runs: list[tuple[Path, dict]], metric: str) -> float | None:
    values = [_coerce_metric(data, metric, path) for path, data in runs]
    if None in values:
        return None
    return statistics.median(values)


def _regression_amount(metric: str, baseline: float, candidate: float) -> float:
    if metric in HIGHER_IS_WORSE:
        return candidate - baseline
    if metric in LOWER_IS_WORSE:
        return baseline - candidate
    raise ValueError(f"Unsupported metric direction for {metric}")


def main() -> int:
    args = _parse_args()

    for path in [*args.baseline, *args.candidate]:
        if not path.exists():
            print(f"FAIL: profile metrics file does not exist: {path}")
            return 1
    if not args.tolerances.exists():
        print(f"FAIL: tolerance config does not exist: {args.tolerances}")
        return 1

    baseline = [(path, _load_json(path)) for path in args.baseline]
    candidate = [(path, _load_json(path)) for path in args.candidate]
    config = _load_json(args.tolerances)

    ticks = {_coerce_ticks(data, path) for path, data in [*baseline, *candidate]}
    if len(ticks) != 1:
        print(
            f"FAIL: profile metric ticks mismatch for example '{args.example}' "
            f"(ticks={sorted(ticks)})"
        )
        return 1
    candidate_ticks = ticks.pop()

    failures: list[str] = []
    rtf: tuple[float, float] | None = None

    for metric in METRIC_ORDER:
        tol = _load_tolerance(config, args.example, metric)
        candidate_value = _median_metric(candidate, metric)
        if candidate_value is None:
            failures.append(f"{metric}: missing from a candidate run")
            continue
        baseline_value = _median_metric(baseline, metric)
        if baseline_value is None:
            print(f"WARN: {args.example}: '{metric}' missing from a base run; skipping it")
            continue
        regression = _regression_amount(metric, baseline_value, candidate_value)
        allowed = max(tol.abs_tol, abs(baseline_value) * tol.rel_tol)
        rel_regression = regression / max(abs(baseline_value), 1e-30)

        if metric == "real_time_factor_median":
            rtf = (baseline_value, candidate_value)

        if regression > allowed:
            direction = "slower" if metric in HIGHER_IS_WORSE else "lower"
            failures.append(
                f"{metric}: candidate is {direction} than baseline beyond tolerance "
                f"(baseline={baseline_value:.6f}, candidate={candidate_value:.6f}, "
                f"regression={regression:.3e}, rel_regression={rel_regression:.3e}, "
                f"allowed={allowed:.3e}, abs_tol={tol.abs_tol:.3e}, rel_tol={tol.rel_tol:.3e})"
            )

    rtf_detail = ""
    if rtf is not None:
        rtf_baseline, rtf_candidate = rtf
        rtf_pct = ((rtf_candidate - rtf_baseline) / max(abs(rtf_baseline), 1e-30)) * 100
        rtf_sign = "+" if rtf_pct >= 0 else ""
        rtf_detail = f"{rtf_candidate:.1f}x (base {rtf_baseline:.1f}x, {rtf_sign}{rtf_pct:.1f}%)"
        # Machine-readable line for regress.sh summary collection
        print(f"RTF_DELTA: {args.example} {rtf_baseline:.3f} {rtf_candidate:.3f} {rtf_pct:.1f}")

    if failures:
        print(f"FAIL: profile metric regression(s) detected for example '{args.example}'")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"PASS: {args.example} profile metrics within tolerance (ticks={candidate_ticks})")
    if rtf_detail:
        print(f"  real_time_factor_median: {rtf_detail}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
