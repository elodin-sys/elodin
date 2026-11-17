import json
import tarfile
from pathlib import Path

from benchmark import ACCEPTANCE, EXAMPLE, acceptance_failures, parity, phase_p95_ns, workload
from package import copy_tree, normalize_tar_info, sanitize_text


def phase(value: int, count: int = 100):
    buckets = [0] * 32
    buckets[value.bit_length() - 1] = count
    return {
        "sum_ns": value * count,
        "count": count,
        "min_ns": value,
        "max_ns": value,
        "buckets": buckets,
    }


def test_histogram_percentile_is_bounded_by_bucket():
    summary = {"world_run": phase(200_000)}
    value = phase_p95_ns(summary, "world_run")
    assert 131_072 <= value <= 262_144


def test_exact_percentile_is_preferred():
    summary = {"world_run": {**phase(200_000), "p95_ns": 199_999}}
    assert phase_p95_ns(summary, "world_run") == 199_999


def test_acceptance_reports_each_failed_limit():
    values = {
        "effective_hz": ACCEPTANCE["effective_hz_min"] - 1,
        "cycle_mean_ms": ACCEPTANCE["cycle_mean_ms_max"] + 0.1,
        "world_run_mean_us": ACCEPTANCE["world_run_mean_us_max"] + 1,
        "world_run_p95_us": ACCEPTANCE["world_run_p95_us_max"] + 1,
        "callback_mean_us": ACCEPTANCE["callback_mean_us_max"] + 1,
        "callback_p95_us": ACCEPTANCE["callback_p95_us_max"] + 1,
        "late_fraction": ACCEPTANCE["late_fraction_max"] + 0.01,
        "pacing_resets": 1,
    }
    assert len(acceptance_failures(values)) == len(ACCEPTANCE)


def test_packager_removes_machine_local_identifiers():
    address = ".".join(("192", "0", "2", "10"))
    text = f"{Path.home()}/work/source {address}"
    sanitized = sanitize_text(text)
    assert str(Path.home()) not in sanitized
    assert address not in sanitized


def test_packager_removes_archive_owner_metadata():
    info = tarfile.TarInfo("example")
    info.uid = 1000
    info.gid = 1000
    info.uname = "user"
    info.gname = "group"
    normalized = normalize_tar_info(info)
    assert (normalized.uid, normalized.gid) == (0, 0)
    assert (normalized.uname, normalized.gname) == ("", "")


def test_packager_excludes_build_outputs(tmp_path):
    source = tmp_path / "source"
    build_output = source / ("tar" + "get")
    build_output.mkdir(parents=True)
    (build_output / "binary").write_bytes(b"local build")
    (source / "keep.txt").write_text("portable")
    destination = tmp_path / "destination"
    copy_tree(source, destination)
    assert (destination / "keep.txt").is_file()
    assert not (destination / build_output.name).exists()


def test_packager_preserves_source_text(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text('value = "$HOME/path"\n')
    destination = tmp_path / "destination"
    copy_tree(source, destination, sanitize=False)
    assert (destination / "example.py").read_text() == 'value = "$HOME/path"\n'


def test_parity_fails_on_a_digest_other_than_expected():
    config = {
        "mode": "lockstep",
        "agents": 2,
        "objects": 2,
        "warmup_ticks": 1,
        "measured_ticks": 2,
        "aux_components": 0,
        "compute_rounds": 1,
        "repeat": 1,
    }
    receipt = {"configuration": config, "output_digest": "b", "entity_digests": None}
    result = parity([receipt], {workload(config): "a"})
    assert not result["passed"]
    assert result["expected_digests_compared"] == 1


def test_documented_acceptance_matches_benchmark():
    documented = json.loads((EXAMPLE / "acceptance.json").read_text())
    assert documented["limits"] == ACCEPTANCE
