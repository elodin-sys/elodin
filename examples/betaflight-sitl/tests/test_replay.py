"""Replay bundle contents and RACE_RECORD startup checks."""

import numpy as np
import pytest
from replay import ReplayRecorder, load_sensor_bundle, load_truth


def _recorder(tmp_path, block_ticks: int = 4):
    return ReplayRecorder(
        tmp_path,
        simulation_rate_hz=8.0,
        block_ticks=block_ticks,
        frame_shape=(2, 3, 4),
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 2},
        mount_position_body=(0.08, 0.0, 0.02),
        tilt_deg=0.0,
        inner_size_m=2.5,
        gates=((0, (10.0, 0.0, 1.8), 0.0, 2.5),),
    )


def _observe(recorder, tick: int, *, gyro, accel, fresh: bool = False, frame=None, frame_time=None):
    recorder.observe(
        tick=tick,
        sim_time=tick / 8.0,
        gyro=np.array(gyro, dtype=np.float64),
        accel=np.array(accel, dtype=np.float64),
        world_pos=np.array([0.0, 0.0, 0.0, 1.0, tick, 0.0, 1.0]),
        world_vel=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        barometer=100.0 + tick if tick % 2 == 0 else None,
        barometer_fresh=tick % 2 == 0,
        magnetometer=np.array([0.0, 1.0, tick], dtype=np.float64),
        magnetometer_fresh=tick % 3 == 0,
        frame=frame,
        frame_time=frame_time,
        frame_fresh=fresh,
    )


def test_interval_means_use_complete_blocks_only(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    for tick in range(4):
        _observe(recorder, tick, gyro=[tick, 0.0, 0.0], accel=[0.0, tick, 9.0])
    for tick in range(6, 9):
        _observe(recorder, tick, gyro=[tick, 0.0, 0.0], accel=[0.0, 0.0, 1.0])
    recorder.finish()
    bundle = load_sensor_bundle(tmp_path)

    assert bundle.imu_time == pytest.approx([3 / 8.0])
    assert bundle.gyro_mean == pytest.approx(np.array([[1.5, 0.0, 0.0]]))
    assert bundle.accel_mean == pytest.approx(np.array([[0.0, 1.5, 9.0]]))
    assert bundle.baro_time == pytest.approx([0.0, 2 / 8.0, 6 / 8.0, 1.0])
    assert bundle.mag.shape == (3, 3)


def test_only_fresh_frames_are_stored(tmp_path) -> None:
    recorder = _recorder(tmp_path, block_ticks=2)
    held = np.full((2, 3, 4), 3, dtype=np.uint8)
    fresh = np.full((2, 3, 4), 9, dtype=np.uint8)
    _observe(recorder, 0, gyro=[0, 0, 0], accel=[0, 0, 1], fresh=False, frame=held, frame_time=0.0)
    _observe(
        recorder, 1, gyro=[0, 0, 0], accel=[0, 0, 1], fresh=True, frame=fresh, frame_time=0.125
    )
    recorder.finish()
    bundle = load_sensor_bundle(tmp_path)

    assert bundle.frames.shape == (1, 2, 3, 4)
    assert np.all(bundle.frames[0] == 9)
    assert bundle.frame_time == pytest.approx([0.125])


def test_frames_stream_in_bounded_chunks_and_keep_order(tmp_path) -> None:
    import replay

    recorder = _recorder(tmp_path, block_ticks=2)
    count = 2 * replay._FRAME_CHUNK + 5
    for tick in range(count):
        frame = np.full((2, 3, 4), tick % 256, dtype=np.uint8)
        _observe(
            recorder,
            tick,
            gyro=[0, 0, 0],
            accel=[0, 0, 1],
            fresh=True,
            frame=frame,
            frame_time=tick / 8.0,
        )
        assert len(recorder._frames) < replay._FRAME_CHUNK
        assert len(recorder._pending) <= replay._MAX_PENDING_CHUNKS
    recorder.finish()
    bundle = load_sensor_bundle(tmp_path)

    assert len(list((tmp_path / "frames").glob("chunk_*.npz"))) == 3
    assert bundle.frame_time == pytest.approx(np.arange(count) / 8.0)
    assert np.array_equal(bundle.frames[:, 0, 0, 0], np.arange(count, dtype=np.uint8))
    assert load_truth(tmp_path).frame_time.shape == (count,)


def test_round_trip_keeps_truth_separate(tmp_path) -> None:
    recorder = _recorder(tmp_path, block_ticks=2)
    frame = np.zeros((2, 3, 4), dtype=np.uint8)
    frame[..., 0] = 255
    for tick in range(4):
        _observe(
            recorder,
            tick,
            gyro=[1.0, 0.0, 0.0],
            accel=[0.0, 0.0, 9.8],
            fresh=tick == 3,
            frame=frame,
            frame_time=0.4,
        )
    recorder.finish()
    recorder.finish()

    (tmp_path / "truth.npz").write_bytes(b"not a zip")
    sensors = load_sensor_bundle(tmp_path)
    assert sensors.frames.shape[0] == 1
    assert sensors.meta["format_version"] == 1
    assert sensors.meta["inner_size_m"] == 2.5
    assert "gate_center" not in sensors.meta

    with pytest.raises(Exception):
        load_truth(tmp_path)


def test_truth_loader_reads_pose_at_frame_time(tmp_path) -> None:
    recorder = _recorder(tmp_path, block_ticks=2)
    frame = np.zeros((2, 3, 4), dtype=np.uint8)
    _observe(
        recorder,
        0,
        gyro=[0, 0, 0],
        accel=[0, 0, 1],
        fresh=True,
        frame=frame,
        frame_time=0.0,
    )
    recorder.finish()
    truth = load_truth(tmp_path)

    assert truth.frame_world_pos.shape == (1, 7)
    assert truth.frame_world_pos[0, 4] == pytest.approx(0.0)
    assert truth.gate_center == pytest.approx(np.array([[10.0, 0.0, 1.8]]))
    assert truth.world_pos.shape[1] == 7


def test_record_directory_rules(tmp_path) -> None:
    from race_options import race_options_from_env

    empty = tmp_path / "empty"
    empty.mkdir()
    options = race_options_from_env(
        {"RACE_CAMERA": "1", "RACE_RECORD": str(empty), "RACE_GUIDANCE": "scripted"}
    )
    assert options.record_dir == empty
    missing = tmp_path / "missing"
    options = race_options_from_env({"RACE_CAMERA": "1", "RACE_RECORD": str(missing)})
    assert options.record_dir == missing

    filled = tmp_path / "filled"
    filled.mkdir()
    (filled / "meta.json").write_text("{}")
    occupied = tmp_path / "file"
    occupied.write_text("x")
    cases = [
        ({"RACE_RECORD": str(filled), "RACE_CAMERA": "1"}, "not empty"),
        ({"RACE_RECORD": str(occupied), "RACE_CAMERA": "1"}, "not a directory"),
        ({"RACE_RECORD": "   ", "RACE_CAMERA": "1"}, "non-empty directory path"),
        ({"RACE_RECORD": str(empty), "RACE_CAMERA": "0"}, "requires RACE_CAMERA=1"),
        ({"RACE_RECORD": str(empty)}, "requires RACE_CAMERA=1"),
    ]
    for env, message in cases:
        with pytest.raises(ValueError, match=message):
            race_options_from_env(env)
