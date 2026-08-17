"""Contract tests for the Voyager reconstructed-arc validation harness."""

import json
import sys
from datetime import datetime
from pathlib import Path

from time_utils import (
    LEGACY_TUTORIAL_DB_EPOCH_US,
    TUTORIAL_START_UTC,
    utc_epoch_microseconds,
)
from validate_truth import (
    MANIFEST_PATH,
    UNIMPLEMENTED_FORCE_MODEL_KEYS,
    load_manifest,
    parse_args,
    required_kernel_names,
    selected_cases,
    selected_convergence_timesteps,
    sha256,
    validate_run_record,
)

EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_truth_cases_initialize_and_score_inside_one_segment():
    manifest = load_manifest()

    for case in manifest["cases"]:
        coverage_start, coverage_end = map(parse_utc, case["coverage_utc"])
        initialization = parse_utc(case["initialization_utc"])
        scoring_end = parse_utc(case["scoring_end_utc"])

        assert coverage_start <= initialization < scoring_end <= coverage_end
        assert (scoring_end - initialization).total_seconds() == (case["duration_days"] * 86400)
        checkpoints = case["checkpoints"]
        checkpoint_names = [checkpoint["name"] for checkpoint in checkpoints]
        elapsed_seconds = [checkpoint["elapsed_seconds"] for checkpoint in checkpoints]

        assert checkpoint_names[0] == "start"
        assert checkpoint_names[-2:] == ["closest_approach", "end"]
        assert elapsed_seconds == sorted(elapsed_seconds)
        assert elapsed_seconds[0] == 0
        assert elapsed_seconds[-1] == case["duration_days"] * 86400


def test_every_referenced_kernel_has_provenance_and_checksum():
    manifest = load_manifest()
    cases = selected_cases(manifest, set())

    for name in required_kernel_names(manifest, cases):
        kernel = manifest["kernels"][name]
        assert kernel["url"].startswith("https://naif.jpl.nasa.gov/")
        assert len(kernel["sha256"]) == 64
        int(kernel["sha256"], 16)


def test_reconstructed_cases_do_not_claim_cruise_navigation_truth():
    manifest = load_manifest()

    assert "absolute navigation accuracy" in manifest["cruise_reference"]["unsupported_claim"]
    assert {case["solution_status"] for case in manifest["cases"]} != {
        manifest["cruise_reference"]["status"]
    }


def test_kernel_precedence_is_unambiguous():
    manifest = load_manifest()
    load_order = manifest["kernel_load_order"]

    assert load_order[-2:] == ["<case encounter kernel>", "de440.bsp"]


def test_checkpoint_contract_has_one_explicit_role_per_checkpoint():
    manifest = load_manifest()
    assert set(manifest["checkpoint_contract"]) >= {
        "version",
        "anchor_definition",
        "primary_definition",
        "diagnostic_definition",
        "excluded_definition",
        "sources",
        "maneuver_evidence",
    }
    for case in manifest["cases"]:
        checkpoints = case["checkpoints"]
        assert all(
            checkpoint["validation_role"] in {"anchor", "primary", "diagnostic", "excluded"}
            for checkpoint in checkpoints
        )
        assert all(
            checkpoint["role_reason"] and checkpoint["maneuver_status"]
            for checkpoint in checkpoints
        )
        assert checkpoints[0]["name"] == "start"
        assert checkpoints[0]["validation_role"] == "anchor"


def test_validation_start_utc_controls_database_epoch():
    assert utc_epoch_microseconds(TUTORIAL_START_UTC) == 252_460_800_000_000
    assert utc_epoch_microseconds("1979-02-01T00:00:00Z") == 286_675_200_000_000
    assert utc_epoch_microseconds("1979-01-31T16:00:00-08:00") == 286_675_200_000_000
    assert utc_epoch_microseconds(TUTORIAL_START_UTC) - LEGACY_TUTORIAL_DB_EPOCH_US == 8_400_000_000


def test_explicit_chapter_two_still_runs_convergence_matrix():
    manifest = load_manifest()
    case = manifest["cases"][0]
    baseline = manifest["validation_protocol"]["baseline_timestep_s"]

    selected = selected_convergence_timesteps(manifest, case, [2], True, None)

    assert selected == [
        timestep for timestep in case["convergence_timesteps_s"] if timestep != baseline
    ]


def test_incompatible_convergence_selections_are_rejected():
    manifest = load_manifest()
    case = manifest["cases"][0]

    for chapters, timestep in (([1], None), ([2], 300.0)):
        try:
            selected_convergence_timesteps(manifest, case, chapters, True, timestep)
        except ValueError:
            pass
        else:
            raise AssertionError("incompatible convergence selection was silently ignored")


def test_cli_has_no_chapter_three_force_model_flags():
    argv = sys.argv
    sys.argv = ["validate_truth.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    for key in UNIMPLEMENTED_FORCE_MODEL_KEYS:
        assert not hasattr(args, key)


def test_jupiter_has_no_primary_checkpoint():
    """Jupiter reconstructed arcs start inside or after documented TCM windows."""
    manifest = load_manifest()
    for case_id in ("v1_jupiter", "v2_jupiter"):
        case = next(case for case in manifest["cases"] if case["id"] == case_id)
        primary = [
            checkpoint
            for checkpoint in case["checkpoints"]
            if checkpoint["validation_role"] == "primary"
        ]
        assert primary == []
        assert case["checkpoints"][0]["validation_role"] == "anchor"


def test_v1_saturn_start_is_pre_a8_anchor():
    manifest = load_manifest()
    case = next(case for case in manifest["cases"] if case["id"] == "v1_saturn")
    start = case["checkpoints"][0]
    assert start["validation_role"] == "anchor"
    assert start["maneuver_status"] == "none_known"
    assert case["convergence_checkpoint"] == "early_approach"


def test_primary_approach_scores_are_saturn_pre_maneuver_windows():
    manifest = load_manifest()
    results = json.loads(
        (EXAMPLE_DIR / "truth_validation_results.json").read_text(encoding="utf-8")
    )
    expected = {
        ("v1_saturn", "early_approach"),
        ("v2_saturn", "early_approach"),
    }
    actual = {(row["case"], row["checkpoint"]) for row in results["primary_summary"]}
    assert actual == expected
    assert all(row["position_error_reduction_percent"] > 50.0 for row in results["primary_summary"])
    saturn = {case["id"]: case for case in manifest["cases"]}
    for row in results["primary_summary"]:
        contract = next(
            checkpoint
            for checkpoint in saturn[row["case"]]["checkpoints"]
            if checkpoint["name"] == row["checkpoint"]
        )
        assert contract["maneuver_status"] == "none_known"


def test_checked_in_baseline_matches_the_contract():
    manifest = load_manifest()
    results = json.loads(
        (EXAMPLE_DIR / "truth_validation_results.json").read_text(encoding="utf-8")
    )
    cases = {case["id"]: case for case in manifest["cases"]}

    assert results["manifest_sha256"] == sha256(MANIFEST_PATH)
    assert results["kernel_load_order"] == manifest["kernel_load_order"]
    baseline_timestep = manifest["validation_protocol"]["baseline_timestep_s"]
    baseline_records = [
        record for record in results["records"] if record["timestep_s"] == baseline_timestep
    ]
    assert len(baseline_records) == len(cases) * 2
    expected_record_count = sum(
        2 + len(case["convergence_timesteps_s"]) - 1 for case in manifest["cases"]
    )
    assert len(results["records"]) == expected_record_count

    for record in results["records"]:
        case = cases[record["case"]]
        assert record["chapter"] in (1, 2)
        assert record["truth_kernel"] == case["kernel"]
        assert record["truth_kernel_sha256"] == manifest["kernels"][case["kernel"]]["sha256"]
        assert not any(key in record for key in UNIMPLEMENTED_FORCE_MODEL_KEYS)
        validate_run_record(case, record, manifest)
        assert record["metrics"][0]["checkpoint"] == "start"
        assert record["metrics"][0]["validation_role"] == "anchor"

    assert all(row["checkpoint"] != "start" for row in results["summary"])
    assert results.get("anchor_summary", []) == []
    clean_rows = [row for row in results["summary"] if row["checkpoint"] == "clean_approach"]
    assert len(clean_rows) == len(cases)
    assert all(row["position_error_reduction_percent"] > 0 for row in clean_rows)
    assert results["primary_summary"]
    assert all(row["validation_role"] == "primary" for row in results["primary_summary"])
    assert all(row["checkpoint"] != "start" for row in results["primary_summary"])
    assert all(row["validation_role"] != "excluded" for row in results["diagnostic_summary"])
    assert all(row["validation_role"] == "excluded" for row in results["excluded_summary"])
    assert len(results["timestep_controls"]) == len(cases) * 2
    for control in results["timestep_controls"]:
        case = cases[control["case"]]
        assert control["checkpoint"] in {case["convergence_checkpoint"], "end"}
        assert {float(timestep) for timestep in control["position_error_km_by_timestep_s"]} == set(
            case["convergence_timesteps_s"]
        )
        if control["checkpoint"] == "end":
            assert control["validation_role"] == "excluded"
            continue
        if case["id"].endswith("saturn"):
            assert control["checkpoint"] == "early_approach"
            assert control["validation_role"] == "primary"
        else:
            assert control["checkpoint"] == "clean_approach"
            assert control["validation_role"] == "diagnostic"
