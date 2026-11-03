import csv

from compare_baseline_csv import Tolerance, _compare_file, _read_ticks


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "value"))
        writer.writerows(rows)


def _stamp(second):
    return f"2026-01-01T00:00:{second:02d}"


def _tick_file(path, n, start):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "globals.tick"))
        writer.writerows((_stamp(start + tick), tick) for tick in range(n))
    return _read_ticks(path)


def test_dense_and_sparse_change_streams_are_equivalent(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write(baseline, ((0, 1), (1, 1), (2, 2), (3, 2), (4, 1)))
    _write(candidate, ((10, 1), (12, 2), (14, 1)))

    stats, error = _compare_file(
        "value.csv", baseline, candidate, Tolerance(abs_tol=0.0, rel_tol=0.0)
    )

    assert error is None
    assert stats is not None
    assert stats.rows == 3


def test_ticks_align_a_settled_value_with_a_one_ulp_cycle(tmp_path):
    ticks = (_tick_file(tmp_path / "b.csv", 5, 0), _tick_file(tmp_path / "c.csv", 5, 30))
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write(baseline, ((_stamp(t), 0.5 + t % 2 * 2**-53) for t in range(5)))
    _write(candidate, ((_stamp(30), 0.5),))

    stats, error = _compare_file(
        "value.csv", baseline, candidate, Tolerance(abs_tol=1e-12, rel_tol=0.0), ticks
    )

    assert error is None
    assert stats is not None
    assert stats.rows == 5


def test_ticks_catch_a_late_change_missing_on_one_side(tmp_path):
    ticks = (_tick_file(tmp_path / "b.csv", 5, 0), _tick_file(tmp_path / "c.csv", 5, 30))
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write(baseline, ((_stamp(0), 1.0), (_stamp(4), 2.0)))
    _write(candidate, ((_stamp(30), 1.0),))

    stats, error = _compare_file(
        "value.csv", baseline, candidate, Tolerance(abs_tol=0.0, rel_tol=0.0), ticks
    )

    assert stats is None
    assert error is not None
    assert "tick 4 exceeds tolerance" in error


def test_a_baseline_file_from_another_run_compares_change_rows(tmp_path):
    ticks = (_tick_file(tmp_path / "b.csv", 5, 0), _tick_file(tmp_path / "c.csv", 5, 30))
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write(baseline, ((_stamp(50), 1.0), (_stamp(51), 1.0), (_stamp(52), 2.0)))
    _write(candidate, ((_stamp(30), 1.0), (_stamp(33), 2.0)))

    stats, error = _compare_file(
        "value.csv", baseline, candidate, Tolerance(abs_tol=0.0, rel_tol=0.0), ticks
    )

    assert error is None
    assert stats is not None
    assert stats.rows == 2
