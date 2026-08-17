"""Run reproducible Voyager Chapter 1/2 reconstructed-arc validations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
MANIFEST_PATH = EXAMPLE_DIR / "truth_reference.json"
DEFAULT_SPICE_DIR = EXAMPLE_DIR / "nasa_spice_data"
DEFAULT_OUTPUT = EXAMPLE_DIR / "truth_validation_results.json"

METRIC_PREFIX = "VOYAGER_VALIDATION_METRIC "
SEGMENT_PREFIX = "VOYAGER_ACTIVE_SEGMENT "
PLANET_SEGMENT_PREFIX = "VOYAGER_PLANET_SEGMENT "
INITIAL_STATE_PREFIX = "VOYAGER_INITIAL_STATE "
VALIDATION_ROLES = {"anchor", "primary", "diagnostic", "excluded"}
ACCURACY_ROLES = {"primary", "diagnostic", "excluded"}
UNIMPLEMENTED_FORCE_MODEL_KEYS = (
    "gravity_parameters",
    "ephemeris_stage_curvature",
    "giant_planet_state",
    "giant_system_model",
    "gm_sources",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Voyager truth-reference schema")
    protocol = manifest.get("validation_protocol") or {}
    if protocol.get("planet_state_source") != "de440.bsp" or not protocol.get("planet_spice_names"):
        raise ValueError("Missing DE440 planetary provenance contract")
    contract = manifest.get("checkpoint_contract")
    if not contract or contract.get("version") != 1:
        raise ValueError("Missing checkpoint contract")
    for case in manifest["cases"]:
        for checkpoint in case["checkpoints"]:
            role = checkpoint.get("validation_role")
            if role not in VALIDATION_ROLES:
                raise ValueError(f"{case['id']} {checkpoint['name']}: invalid validation role")
            if not checkpoint.get("role_reason") or not checkpoint.get("maneuver_status"):
                raise ValueError(
                    f"{case['id']} {checkpoint['name']}: incomplete checkpoint contract"
                )
    return manifest


def checkpoint_contract(case: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return next(checkpoint for checkpoint in case["checkpoints"] if checkpoint["name"] == name)
    except StopIteration as exc:
        raise AssertionError(f"{case['id']}: checkpoint {name!r} is not in the contract") from exc


def selected_cases(manifest: dict[str, Any], requested: set[str]) -> list[dict[str, Any]]:
    cases = manifest["cases"]
    known = {case["id"] for case in cases}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown validation case(s): {sorted(unknown)}")
    return [case for case in cases if not requested or case["id"] in requested]


def selected_convergence_timesteps(
    manifest: dict[str, Any],
    case: dict[str, Any],
    chapters: list[int],
    include_convergence: bool,
    timestep_override: float | None,
) -> list[float]:
    """Return non-baseline convergence steps or reject incompatible selections."""
    if not include_convergence:
        return []
    if timestep_override is not None:
        raise ValueError("--include-convergence cannot be combined with --timestep")

    protocol = manifest["validation_protocol"]
    convergence_chapter = protocol["convergence_chapter"]
    if convergence_chapter not in chapters:
        raise ValueError(
            f"--include-convergence requires Chapter {convergence_chapter}; "
            f"selected chapters: {chapters}"
        )

    baseline_timestep = protocol["baseline_timestep_s"]
    return [
        timestep for timestep in case["convergence_timesteps_s"] if timestep != baseline_timestep
    ]


def required_kernel_names(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> set[str]:
    load_order = manifest["kernel_load_order"]
    names = {name for name in load_order if not name.startswith("<")}
    names.update(case["kernel"] for case in cases)
    return names


def verify_kernels(manifest: dict[str, Any], cases: list[dict[str, Any]], spice_dir: Path) -> None:
    failures = []
    for name in sorted(required_kernel_names(manifest, cases)):
        path = spice_dir / name
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        actual = sha256(path)
        expected = manifest["kernels"][name]["sha256"]
        if actual != expected:
            failures.append(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Kernel verification failed:\n- " + "\n- ".join(failures))


def parse_prefixed_json(output: str, prefix: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix(prefix))
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


def validate_run_record(
    case: dict[str, Any], record: dict[str, Any], manifest: dict[str, Any] | None = None
) -> None:
    if manifest is None:
        manifest = load_manifest()
    segments = record["active_segments"]
    if {segment["epoch"] for segment in segments} != {
        "initialization",
        "scoring_end",
    }:
        raise AssertionError(f"{case['id']}: missing endpoint segment audit")

    for segment in segments:
        expected = {
            "file": case["kernel"],
            "target": case["spice_target"],
            "center": case["native_center"],
            "native_frame": case["native_frame"],
            "segment_type": case["segment_type"],
            "segment_id": case["segment_id"],
            "probe": case["probe"],
        }
        for key, value in expected.items():
            if segment[key] != value:
                raise AssertionError(
                    f"{case['id']} {segment['epoch']}: expected {key}={value!r}, "
                    f"got {segment[key]!r}"
                )

    planet_names = manifest["validation_protocol"]["planet_spice_names"]
    planet_source = manifest["validation_protocol"]["planet_state_source"]
    planet_segments = record["planet_segments"]
    expected_planet_epochs = {
        (name, epoch) for name in planet_names for epoch in ("initialization", "scoring_end")
    }
    actual_planet_epochs = {
        (segment["spice_name"], segment["epoch"]) for segment in planet_segments
    }
    if actual_planet_epochs != expected_planet_epochs:
        raise AssertionError(
            f"{case['id']}: expected planet segment audits {sorted(expected_planet_epochs)}, "
            f"got {sorted(actual_planet_epochs)}"
        )
    for segment in planet_segments:
        if segment["file"] != planet_source:
            raise AssertionError(
                f"{case['id']} {segment['spice_name']} {segment['epoch']}: "
                f"expected planetary file {planet_source!r}, got {segment['file']!r}"
            )

    expected_checkpoints = [checkpoint["name"] for checkpoint in case["checkpoints"]]
    actual_checkpoints = [metric["checkpoint"] for metric in record["metrics"]]
    if actual_checkpoints != expected_checkpoints:
        raise AssertionError(
            f"{case['id']}: expected checkpoints {expected_checkpoints}, got {actual_checkpoints}"
        )

    for metric in record["metrics"]:
        contract = checkpoint_contract(case, metric["checkpoint"])
        for key in ("validation_role", "role_reason", "maneuver_status"):
            if metric.get(key) != contract[key]:
                raise AssertionError(
                    f"{case['id']} {metric['checkpoint']}: {key} does not match contract"
                )
        if metric["checkpoint"] == "start" and contract["validation_role"] != "anchor":
            raise AssertionError(f"{case['id']}: start must be an initialization anchor")
        if (
            abs(metric["actual_elapsed_seconds"] - metric["requested_elapsed_seconds"])
            > record["timestep_s"] / 2.0
        ):
            raise AssertionError(
                f"{case['id']} {metric['checkpoint']}: checkpoint epoch "
                "is farther than half a timestep from the contract"
            )
        for key in (
            "position_residual_km",
            "position_rtn_km",
            "velocity_residual_mps",
            "velocity_rtn_mps",
        ):
            if len(metric[key]) != 3 or not all(math.isfinite(value) for value in metric[key]):
                raise AssertionError(f"{case['id']} {metric['checkpoint']}: invalid {key}")
        for vector_key, norm_key in (
            ("position_residual_km", "position_error_km"),
            ("position_rtn_km", "position_error_km"),
            ("velocity_residual_mps", "velocity_error_mps"),
            ("velocity_rtn_mps", "velocity_error_mps"),
        ):
            vector_norm = math.sqrt(sum(value * value for value in metric[vector_key]))
            if not math.isclose(
                vector_norm,
                metric[norm_key],
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            ):
                raise AssertionError(
                    f"{case['id']} {metric['checkpoint']}: {vector_key} does not match {norm_key}"
                )

    initial_state = record["initial_state"]
    if initial_state["epoch_utc"] != case["initialization_utc"]:
        raise AssertionError(f"{case['id']}: initialization epoch changed")

    leaked = [key for key in UNIMPLEMENTED_FORCE_MODEL_KEYS if key in record]
    if leaked:
        raise AssertionError(
            f"{case['id']}: record claims unimplemented force-model knobs: {leaked}"
        )


def run_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    chapter: int,
    timestep_s: float,
    spice_dir: Path,
) -> dict[str, Any]:
    ticks = case["duration_days"] * 86400.0 / timestep_s
    if not ticks.is_integer():
        raise ValueError(f"{case['id']}: timestep {timestep_s:g}s does not divide the arc")

    with tempfile.TemporaryDirectory(prefix="voyager-truth-") as temporary_root:
        database = Path(temporary_root) / "db"
        environment = os.environ.copy()
        environment.update(
            {
                "DB_PATH": str(database),
                "JAX_ENABLE_X64": "True",
                "MAX_TICKS": str(int(ticks)),
                "VOYAGER_CHECKPOINTS_JSON": json.dumps(
                    [
                        checkpoint
                        for checkpoint in case["checkpoints"]
                        if checkpoint["elapsed_seconds"] > 0
                    ],
                    separators=(",", ":"),
                ),
                "VOYAGER_DYNAMICS_CHAPTER": str(chapter),
                "VOYAGER_SCORED_PROBE": case["probe"],
                "VOYAGER_SPICE_DIR": str(spice_dir),
                "VOYAGER_START_UTC": case["initialization_utc"],
                "VOYAGER_TIME_STEP": str(timestep_s),
                "VOYAGER_TRUTH_KERNEL": case["kernel"],
            }
        )

        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, str(EXAMPLE_DIR / "main.py"), "run"],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed_s = time.perf_counter() - started

    if process.returncode:
        tail = "\n".join(process.stdout.splitlines()[-100:])
        raise RuntimeError(
            f"{case['id']} Chapter {chapter} failed with exit code {process.returncode}:\n{tail}"
        )

    initial_states = parse_prefixed_json(process.stdout, INITIAL_STATE_PREFIX)
    if len(initial_states) != 1:
        raise AssertionError(f"{case['id']}: expected one initial state, got {len(initial_states)}")

    zero_vector = [0.0, 0.0, 0.0]
    initial_metric = {
        "actual_elapsed_seconds": 0.0,
        "actual_epoch_utc": case["initialization_utc"],
        "checkpoint": "start",
        "position_error_km": 0.0,
        "position_residual_km": zero_vector,
        "position_rtn_km": zero_vector,
        "probe": case["probe"],
        "requested_elapsed_seconds": 0.0,
        "velocity_error_mps": 0.0,
        "velocity_residual_mps": zero_vector,
        "velocity_rtn_mps": zero_vector,
    }
    for key in ("validation_role", "role_reason", "maneuver_status"):
        initial_metric[key] = checkpoint_contract(case, "start")[key]
    metrics = [initial_metric, *parse_prefixed_json(process.stdout, METRIC_PREFIX)]
    for metric in metrics[1:]:
        contract = checkpoint_contract(case, metric["checkpoint"])
        for key in ("validation_role", "role_reason", "maneuver_status"):
            metric[key] = contract[key]
    record = {
        "case": case["id"],
        "chapter": chapter,
        "timestep_s": timestep_s,
        "truth_kernel": case["kernel"],
        "truth_kernel_sha256": manifest["kernels"][case["kernel"]]["sha256"],
        "initial_state": initial_states[0],
        "active_segments": parse_prefixed_json(process.stdout, SEGMENT_PREFIX),
        "planet_segments": parse_prefixed_json(process.stdout, PLANET_SEGMENT_PREFIX),
        "metrics": metrics,
    }
    validate_run_record(case, record, manifest)
    print(
        f"finished {case['id']} Chapter {chapter} in {elapsed_s:.1f}s",
        flush=True,
    )
    return record


def summarize(manifest: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_timestep = manifest["validation_protocol"]["baseline_timestep_s"]
    by_key = {
        (record["case"], record["chapter"], record["timestep_s"]): record for record in records
    }
    rows = []
    for case in manifest["cases"]:
        chapter1_key = (case["id"], 1, baseline_timestep)
        chapter2_key = (case["id"], 2, baseline_timestep)
        if chapter1_key not in by_key or chapter2_key not in by_key:
            continue
        chapter1 = by_key[chapter1_key]
        chapter2 = by_key[chapter2_key]
        for metric1, metric2 in zip(chapter1["metrics"], chapter2["metrics"], strict=True):
            if metric1["checkpoint"] == "start" or metric1["validation_role"] not in ACCURACY_ROLES:
                continue
            reduction = (
                100.0
                * (metric1["position_error_km"] - metric2["position_error_km"])
                / metric1["position_error_km"]
            )
            rows.append(
                {
                    "case": case["id"],
                    "checkpoint": metric1["checkpoint"],
                    "validation_role": metric1["validation_role"],
                    "role_reason": metric1["role_reason"],
                    "maneuver_status": metric1["maneuver_status"],
                    "elapsed_days": metric1["requested_elapsed_seconds"] / 86400.0,
                    "chapter1_position_error_km": metric1["position_error_km"],
                    "chapter1_velocity_error_mps": metric1["velocity_error_mps"],
                    "chapter2_position_error_km": metric2["position_error_km"],
                    "chapter2_velocity_error_mps": metric2["velocity_error_mps"],
                    "position_error_reduction_percent": reduction,
                }
            )
    return rows


def summarize_timestep_controls(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    chapter = manifest["validation_protocol"]["convergence_chapter"]
    by_key = {
        (record["case"], record["chapter"], record["timestep_s"]): record for record in records
    }
    rows = []
    for case in manifest["cases"]:
        timesteps = case["convergence_timesteps_s"]
        case_records = [by_key.get((case["id"], chapter, timestep)) for timestep in timesteps]
        if any(record is None for record in case_records):
            continue
        checkpoint_names = [case["convergence_checkpoint"], "end"]
        for checkpoint_name in checkpoint_names:
            contract = checkpoint_contract(case, checkpoint_name)
            errors = {
                str(timestep): next(
                    metric["position_error_km"]
                    for metric in record["metrics"]
                    if metric["checkpoint"] == checkpoint_name
                )
                for timestep, record in zip(timesteps, case_records, strict=True)
            }
            rows.append(
                {
                    "case": case["id"],
                    "checkpoint": checkpoint_name,
                    "validation_role": contract["validation_role"],
                    "role_reason": contract["role_reason"],
                    "chapter": chapter,
                    "position_error_km_by_timestep_s": errors,
                    "spread_km": max(errors.values()) - min(errors.values()),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case ID to run; repeat for multiple cases (default: all)",
    )
    parser.add_argument(
        "--chapter",
        action="append",
        type=int,
        choices=(1, 2),
        default=[],
        help="chapter to run; repeat for both (default: both)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        help="override the manifest baseline timestep for focused runs",
    )
    parser.add_argument(
        "--include-convergence",
        action="store_true",
        help="also run the declared Chapter 2 timestep-control matrix",
    )
    parser.add_argument("--spice-dir", type=Path, default=DEFAULT_SPICE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    cases = selected_cases(manifest, set(args.case))
    chapters = args.chapter or manifest["validation_protocol"]["chapters"]
    baseline_timestep = manifest["validation_protocol"]["baseline_timestep_s"]
    selected_timestep = args.timestep or baseline_timestep
    convergence_timesteps = {
        case["id"]: selected_convergence_timesteps(
            manifest,
            case,
            chapters,
            args.include_convergence,
            args.timestep,
        )
        for case in cases
    }
    verify_kernels(manifest, cases, args.spice_dir)

    records = []
    for case in cases:
        for chapter in chapters:
            print(
                f"running {case['id']} Chapter {chapter} at dt={selected_timestep:g}s",
                flush=True,
            )
            records.append(
                run_case(
                    manifest,
                    case,
                    chapter,
                    selected_timestep,
                    args.spice_dir,
                )
            )
        for timestep in convergence_timesteps[case["id"]]:
            chapter = manifest["validation_protocol"]["convergence_chapter"]
            print(
                f"running {case['id']} Chapter {chapter} timestep control at dt={timestep:g}s",
                flush=True,
            )
            records.append(
                run_case(
                    manifest,
                    case,
                    chapter,
                    timestep,
                    args.spice_dir,
                )
            )

    summary = summarize(manifest, records)
    output = {
        "schema_version": 1,
        "manifest_sha256": sha256(MANIFEST_PATH),
        "kernel_load_order": manifest["kernel_load_order"],
        "records": records,
        "summary": summary,
        "primary_summary": [row for row in summary if row["validation_role"] == "primary"],
        "diagnostic_summary": [row for row in summary if row["validation_role"] == "diagnostic"],
        "excluded_summary": [row for row in summary if row["validation_role"] == "excluded"],
        "anchor_summary": [row for row in summary if row["validation_role"] == "anchor"],
        "timestep_controls": summarize_timestep_controls(manifest, records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
