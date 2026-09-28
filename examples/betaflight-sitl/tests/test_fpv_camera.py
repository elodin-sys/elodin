"""GPU-free tests for FPV freshness, frame accounting, and acceptance."""

import numpy as np

from fpv_camera import FRAME_BYTES, MIN_FPS, PERIOD_US, WARMUP_US, FpvCamera


def _frame(fill: int = 0) -> np.ndarray:
    return np.full(FRAME_BYTES, fill, dtype=np.uint8)


def _reader(frames: dict[int, np.ndarray]):
    def read_msg_at(name: str, timestamp: int):
        assert name == "drone.fpv"
        eligible = [ts for ts in frames if ts <= timestamp]
        if not eligible:
            return None
        chosen = max(eligible)
        return chosen, frames[chosen]

    return read_msg_at


def test_held_frame_is_not_fresh_and_a_new_timestamp_is() -> None:
    camera = FpvCamera()
    reader = _reader({0: _frame(1)})

    first = camera.poll(reader, 40_000, 0.04)
    second = camera.poll(reader, 40_000 + PERIOD_US, 0.04 + PERIOD_US / 1_000_000)

    assert first.fresh is True
    assert first.frame is not None
    assert first.frame.shape == (360, 640, 4)
    assert second.frame is not None
    assert second.fresh is False
    assert camera.held_samples == 1
    assert camera.fresh_samples == 1


def test_only_one_sample_is_offered_per_camera_period() -> None:
    camera = FpvCamera()
    reader = _reader({0: _frame()})

    first = camera.poll(reader, 40_000, 0.04)
    second = camera.poll(reader, 40_000 + PERIOD_US // 2, 0.05)

    assert first.requested_us == 7_000
    assert second.frame is None
    assert second.requested_us is None
    assert second.fresh is False


def test_read_before_the_first_frame_returns_nothing() -> None:
    camera = FpvCamera()
    sample = camera.poll(_reader({100: _frame()}), 0, 0.0)

    assert sample.frame is None
    assert sample.fresh is False
    assert sample.requested_us == -33_000


def test_malformed_payload_fails_acceptance() -> None:
    camera = FpvCamera()
    reader = _reader({0: np.zeros(4, dtype=np.uint8)})

    sample = camera.poll(reader, 40_000, 0.04)
    report = camera.finish(reader, 40_000)

    assert sample.frame is None
    assert report.malformed is True
    assert report.accepted is False
    assert report.reason == "malformed-frame"


def test_accounting_window_keeps_total_and_bounds_fps_frames() -> None:
    frames = {
        1_000_000: _frame(1),
        WARMUP_US: _frame(2),
        5_000_000: _frame(3),
    }
    camera = FpvCamera()
    camera.poll(_reader({}), 0, 0.0)
    report = camera.finish(_reader(frames), 5_000_000)

    assert report.total_frames == 3
    assert report.window_frames == 2
    assert report.window_s == 3.0
    assert report.observed_fps == 2 / 3


def test_a_frame_newer_than_the_final_tick_is_still_counted() -> None:
    frames = {3_000_000: _frame(1), 4_000_000: _frame(2)}
    camera = FpvCamera()
    camera.poll(_reader({}), 0, 0.0)
    report = camera.finish(_reader(frames), 3_500_000)

    assert report.total_frames == 2
    assert report.window_frames == 2


def test_acceptance_passes_at_the_floor_and_fails_below_it_or_with_no_frames() -> None:
    start = 0
    end = start + WARMUP_US + 1_000_000
    passing = {WARMUP_US + i * (1_000_000 // int(MIN_FPS)): _frame() for i in range(int(MIN_FPS))}
    short = dict(list(passing.items())[:-1])

    def report_for(frames: dict[int, np.ndarray]):
        camera = FpvCamera()
        camera.poll(_reader({}), start, 0.0)
        return camera.finish(_reader(frames), end)

    assert report_for(passing).accepted is True
    assert report_for(passing).reason == "ok"
    below = report_for(short)
    assert below.accepted is False
    assert below.reason == "fps-below-15"
    empty = report_for({})
    assert empty.total_frames == 0
    assert empty.accepted is False
    assert empty.reason == "no-frames"
