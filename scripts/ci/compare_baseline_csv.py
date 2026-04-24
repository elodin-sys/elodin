#!/usr/bin/env python3
"""Compare exported CSV telemetry against a known-good baseline.

The comparison is numeric and tolerance-based for all numeric cells. Columns
named ``time`` are ignored so that runs with different wall-clock timestamps do
not fail CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

IGNORED_COLUMNS = {"time"}


def _windows_safe_rel_path(rel_path: str) -> str:
    return rel_path.replace("_>_", "_to_").replace(">", "to")


@dataclass
class Tolerance:
    abs_tol: float
    rel_tol: float


@dataclass
class FileStats:
    rel_path: str
    rows: int
    max_abs: float
    max_rel: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, help="Example name, e.g. ball or drone")
    parser.add_argument(
        "--baseline-dir",
        required=True,
        type=Path,
        help="Directory containing baseline CSV files for this example",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        type=Path,
        help="Directory containing candidate CSV files exported in CI",
    )
    parser.add_argument(
        "--tolerances",
        required=True,
        type=Path,
        help="Path to tolerances JSON configuration",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Maximum mismatch details to print",
    )
    parser.add_argument(
        "--file-prefix",
        default="",
        help="Optional filename prefix filter (for flat baseline directories)",
    )
    return parser.parse_args()


def _collect_csv_files(root: Path, file_prefix: str = "") -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        if path.is_file():
            rel_path = path.relative_to(root).as_posix()
            file_name = path.name
            if file_prefix and not file_name.startswith(file_prefix):
                continue
            safe_rel_path = _windows_safe_rel_path(rel_path)
            if safe_rel_path in files:
                raise ValueError(f"CSV path collision after sanitizing for Windows: {rel_path}")
            files[safe_rel_path] = path
    return files


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
        rows = [row for row in reader]
    return header, rows


def _parse_number(value: str) -> float | None:
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_tolerances(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Tolerance config must be a JSON object: {path}")
    return data


def _resolve_tolerance(config: dict, example: str, rel_path: str) -> Tolerance:
    defaults = config.get("default", {}) if isinstance(config.get("default", {}), dict) else {}
    examples_cfg = (
        config.get("examples", {}) if isinstance(config.get("examples", {}), dict) else {}
    )
    example_cfg = (
        examples_cfg.get(example, {}) if isinstance(examples_cfg.get(example, {}), dict) else {}
    )
    files_cfg = (
        example_cfg.get("files", {}) if isinstance(example_cfg.get("files", {}), dict) else {}
    )
    file_cfg = files_cfg.get(rel_path, {}) if isinstance(files_cfg.get(rel_path, {}), dict) else {}

    abs_tol = float(
        file_cfg.get("abs_tol", example_cfg.get("abs_tol", defaults.get("abs_tol", 0.0)))
    )
    rel_tol = float(
        file_cfg.get("rel_tol", example_cfg.get("rel_tol", defaults.get("rel_tol", 0.0)))
    )
    return Tolerance(abs_tol=abs_tol, rel_tol=rel_tol)


def _selected_columns(header: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(header) if name.strip().lower() not in IGNORED_COLUMNS]


def _compare_file(
    rel_path: str,
    baseline_path: Path,
    candidate_path: Path,
    tolerance: Tolerance,
) -> tuple[FileStats | None, str | None]:
    baseline_header, baseline_rows = _read_csv(baseline_path)
    candidate_header, candidate_rows = _read_csv(candidate_path)

    baseline_cols = _selected_columns(baseline_header)
    candidate_cols = _selected_columns(candidate_header)

    baseline_names = [baseline_header[i] for i in baseline_cols]
    candidate_names = [candidate_header[i] for i in candidate_cols]

    if baseline_names != candidate_names:
        return None, f"{rel_path}: header mismatch after ignoring time columns"

    if len(baseline_rows) != len(candidate_rows):
        return None, (
            f"{rel_path}: row count mismatch "
            f"(baseline={len(baseline_rows)}, candidate={len(candidate_rows)})"
        )

    max_abs = 0.0
    max_rel = 0.0

    for row_index, (base_row, cand_row) in enumerate(zip(baseline_rows, candidate_rows), start=2):
        if len(base_row) != len(baseline_header):
            return None, (
                f"{rel_path}: malformed baseline row {row_index} "
                f"(expected {len(baseline_header)} columns, got {len(base_row)})"
            )
        if len(cand_row) != len(candidate_header):
            return None, (
                f"{rel_path}: malformed candidate row {row_index} "
                f"(expected {len(candidate_header)} columns, got {len(cand_row)})"
            )

        for name, base_col, cand_col in zip(baseline_names, baseline_cols, candidate_cols):
            baseline_raw = base_row[base_col].strip()
            candidate_raw = cand_row[cand_col].strip()

            baseline_num = _parse_number(baseline_raw)
            candidate_num = _parse_number(candidate_raw)

            if baseline_num is not None and candidate_num is not None:
                if math.isnan(baseline_num) or math.isnan(candidate_num):
                    if not (math.isnan(baseline_num) and math.isnan(candidate_num)):
                        return None, (
                            f"{rel_path}:{name}: row {row_index} "
                            f"NaN mismatch (baseline={baseline_raw}, candidate={candidate_raw})"
                        )
                    continue
                if math.isinf(baseline_num) or math.isinf(candidate_num):
                    if baseline_num != candidate_num:
                        return None, (
                            f"{rel_path}:{name}: row {row_index} "
                            f"Inf mismatch (baseline={baseline_raw}, candidate={candidate_raw})"
                        )
                    continue

                abs_diff = abs(candidate_num - baseline_num)
                rel_diff = abs_diff / max(abs(baseline_num), abs(candidate_num), 1e-30)
                max_abs = max(max_abs, abs_diff)
                max_rel = max(max_rel, rel_diff)

                if not math.isclose(
                    baseline_num,
                    candidate_num,
                    rel_tol=tolerance.rel_tol,
                    abs_tol=tolerance.abs_tol,
                ):
                    return None, (
                        f"{rel_path}:{name}: row {row_index} exceeds tolerance "
                        f"(baseline={baseline_num:.16g}, candidate={candidate_num:.16g}, "
                        f"abs_diff={abs_diff:.3e}, rel_diff={rel_diff:.3e}, "
                        f"abs_tol={tolerance.abs_tol:.3e}, rel_tol={tolerance.rel_tol:.3e})"
                    )
            elif baseline_raw != candidate_raw:
                return None, (
                    f"{rel_path}:{name}: row {row_index} string mismatch "
                    f"(baseline={baseline_raw!r}, candidate={candidate_raw!r})"
                )

    return FileStats(
        rel_path=rel_path, rows=len(baseline_rows), max_abs=max_abs, max_rel=max_rel
    ), None


def main() -> int:
    args = _parse_args()

    if not args.baseline_dir.exists():
        print(f"FAIL: baseline directory does not exist: {args.baseline_dir}")
        return 1
    if not args.candidate_dir.exists():
        print(f"FAIL: candidate directory does not exist: {args.candidate_dir}")
        return 1
    if not args.tolerances.exists():
        print(f"FAIL: tolerance config does not exist: {args.tolerances}")
        return 1

    tolerance_cfg = _load_tolerances(args.tolerances)
    baseline_files = _collect_csv_files(args.baseline_dir, args.file_prefix)
    candidate_files = _collect_csv_files(args.candidate_dir, args.file_prefix)

    if not baseline_files:
        print(f"FAIL: no CSV files found in baseline directory: {args.baseline_dir}")
        return 1
    if not candidate_files:
        print(f"FAIL: no CSV files found in candidate directory: {args.candidate_dir}")
        return 1

    missing = sorted(set(baseline_files) - set(candidate_files))
    extra = sorted(set(candidate_files) - set(baseline_files))
    if missing or extra:
        print("FAIL: baseline and candidate CSV file sets differ")
        if missing:
            print(f"  Missing in candidate ({len(missing)}):")
            for rel_path in missing[: args.max_failures]:
                print(f"    - {rel_path}")
        if extra:
            print(f"  Extra in candidate ({len(extra)}):")
            for rel_path in extra[: args.max_failures]:
                print(f"    - {rel_path}")
        return 1

    failures: list[str] = []
    stats: list[FileStats] = []
    for rel_path in sorted(baseline_files):
        tol = _resolve_tolerance(tolerance_cfg, args.example, rel_path)
        file_stats, error = _compare_file(
            rel_path=rel_path,
            baseline_path=baseline_files[rel_path],
            candidate_path=candidate_files[rel_path],
            tolerance=tol,
        )
        if error is not None:
            failures.append(error)
            continue
        if file_stats is not None:
            stats.append(file_stats)

    if failures:
        print(
            f"FAIL: {len(failures)} CSV mismatch(es) for example '{args.example}' "
            f"({len(stats)} files passed)"
        )
        for error in failures[: args.max_failures]:
            print(f"  - {error}")
        if len(failures) > args.max_failures:
            remaining = len(failures) - args.max_failures
            print(f"  ... {remaining} additional mismatch(es) omitted")
        return 1

    max_abs = max((item.max_abs for item in stats), default=0.0)
    max_rel = max((item.max_rel for item in stats), default=0.0)
    row_total = sum(item.rows for item in stats)

    worst_abs_file = max(stats, key=lambda s: s.max_abs) if stats else None
    worst_rel_file = max(stats, key=lambda s: s.max_rel) if stats else None

    default_tol = _resolve_tolerance(tolerance_cfg, args.example, "__default__")

    print(f"PASS: {args.example} regression check succeeded ({len(stats)} files, {row_total} rows)")
    if max_abs == 0.0 and max_rel == 0.0:
        print(
            f"  bit-for-bit identical (tol: abs={default_tol.abs_tol:.0e}, rel={default_tol.rel_tol:.0e})"
        )
    else:
        abs_pct = (max_abs / default_tol.abs_tol * 100) if default_tol.abs_tol > 0 else float("inf")
        rel_pct = (max_rel / default_tol.rel_tol * 100) if default_tol.rel_tol > 0 else float("inf")
        print(
            f"  max_abs={max_abs:.3e} ({abs_pct:.1f}% of tol {default_tol.abs_tol:.0e})"
            f"  max_rel={max_rel:.3e} ({rel_pct:.1f}% of tol {default_tol.rel_tol:.0e})"
        )
        if worst_abs_file and worst_abs_file.max_abs > 0:
            print(f"  worst file (abs): {worst_abs_file.rel_path} ({worst_abs_file.max_abs:.3e})")
        if (
            worst_rel_file
            and worst_rel_file.max_rel > 0
            and worst_rel_file.rel_path != (worst_abs_file.rel_path if worst_abs_file else "")
        ):
            print(f"  worst file (rel): {worst_rel_file.rel_path} ({worst_rel_file.max_rel:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
