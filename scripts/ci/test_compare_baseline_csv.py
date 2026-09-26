import csv

from compare_baseline_csv import Tolerance, _compare_file


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "value"))
        writer.writerows(rows)


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
