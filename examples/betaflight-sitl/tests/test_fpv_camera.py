"""GPU-free tests for FPV freshness, frame accounting, and acceptance."""

import numpy as np
import pytest

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
    db_epoch_us = 1_761_234_567_890_000
    initial_sim_time_s = 23.75
    now_us = db_epoch_us + 23_750_000
    selected_ts = db_epoch_us + 23_700_000
    source_frame = _frame(1)
    reader = _reader({selected_ts: source_frame})

    first = camera.poll(reader, now_us, initial_sim_time_s)
    first_requested_us = camera.last_requested_us
    second = camera.poll(
        reader,
        now_us + PERIOD_US,
        initial_sim_time_s + PERIOD_US / 1_000_000,
    )

    source_frame[:] = 99
    assert first.fresh is True
    assert first.frame is not None
    assert first.frame.shape == (360, 640, 4)
    assert first.frame[0, 0, 0] == 1
    assert first.frame_time == 23.7
    assert first_requested_us == db_epoch_us + 23_717_000
    assert first_requested_us != selected_ts
    assert second.frame is not None
    assert second.frame_time == first.frame_time
    assert second.fresh is False
    assert camera.last_requested_us == db_epoch_us + 23_750_333
    assert camera.held_samples == 1
    assert camera.fresh_samples == 1


def test_timestamp_conversion_preserves_integer_microsecond_delta() -> None:
    camera = FpvCamera()
    # Beyond float's exact-integer range: subtract integers before converting.
    now_us = 2**54 + 123_457
    sample = camera.poll(_reader({now_us - 50_001: _frame()}), now_us, 23.750125)

    assert sample.frame_time == pytest.approx(23.700124, abs=1e-12, rel=0)


def test_new_selected_timestamp_is_fresh_after_a_held_frame() -> None:
    camera = FpvCamera()
    frames = {1_000: _frame(1)}
    reader = _reader(frames)
    first = camera.poll(reader, 40_000, 0.04)
    held = camera.poll(reader, 40_000 + PERIOD_US, 0.04 + PERIOD_US / 1e6)
    frames[2_000] = _frame(2)
    fresh = camera.poll(reader, 40_000 + 2 * PERIOD_US, 0.04 + 2 * PERIOD_US / 1e6)

    assert first.fresh and not held.fresh and fresh.fresh
    assert held.frame_time == first.frame_time
    assert first.frame_time == pytest.approx(0.001)
    assert fresh.frame_time == pytest.approx(0.002)
    assert fresh.frame[0, 0, 0] == 2
    assert camera.fresh_samples == 2
    assert camera.held_samples == 1


def test_only_one_sample_is_offered_per_camera_period() -> None:
    camera = FpvCamera()
    calls = []

    def reader(name: str, timestamp: int):
        calls.append((name, timestamp))

    first = camera.poll(reader, 40_000, 0.04)
    first_requested_us = camera.last_requested_us
    second = camera.poll(reader, 40_000 + PERIOD_US // 2, 0.05)

    assert first.frame is None
    assert first.frame_time is None
    assert first_requested_us == 7_000
    assert second.frame is None
    assert second.frame_time is None
    assert second.fresh is False
    assert camera.last_requested_us == first_requested_us
    assert calls == [("drone.fpv", 7_000)]


def test_read_before_the_first_frame_returns_nothing() -> None:
    camera = FpvCamera()
    sample = camera.poll(_reader({100: _frame()}), 0, 0.0)

    assert sample.frame is None
    assert sample.frame_time is None
    assert sample.fresh is False
    assert camera.last_requested_us == -33_000


def test_malformed_payload_fails_acceptance() -> None:
    camera = FpvCamera()
    reader = _reader({0: np.zeros(4, dtype=np.uint8)})

    sample = camera.poll(reader, 40_000, 0.04)
    report = camera.finish(reader, 40_000)

    assert sample.frame is None
    assert sample.frame_time is None
    assert sample.fresh is False
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
