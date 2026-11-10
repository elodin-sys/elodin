"""Offline replay bundles for camera runs.

``RACE_RECORD`` writes sensor history and, in a separate file, truth. The
sensor loader never opens the truth file. This module does not detect gates
or score them.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
_FRAME_CHUNK = 30
_MAX_PENDING_CHUNKS = 2
_TRUTH_RING_S = 0.5


class RecordingFailed(RuntimeError):
    """The replay bundle is incomplete. Record the run again; nothing is repaired."""

    def __init__(self, directory: Path, error: BaseException):
        self.directory = Path(directory)
        super().__init__(f"{self.directory}: {type(error).__name__}: {error}")


@dataclass(frozen=True, slots=True)
class SensorBundle:
    """Sensor history only. Gate poses are not in this object."""

    meta: dict
    imu_time: np.ndarray
    gyro_mean: np.ndarray
    accel_mean: np.ndarray
    baro_time: np.ndarray
    baro: np.ndarray
    mag_time: np.ndarray
    mag: np.ndarray
    frame_time: np.ndarray
    frames: np.ndarray


@dataclass(frozen=True, slots=True)
class TruthBundle:
    """Truth written beside the sensor bundle and read only by the evaluator."""

    imu_time: np.ndarray
    world_pos: np.ndarray
    world_vel: np.ndarray
    frame_time: np.ndarray
    frame_world_pos: np.ndarray
    frame_world_vel: np.ndarray
    gate_index: np.ndarray
    gate_center: np.ndarray
    gate_yaw: np.ndarray
    gate_inner_size: np.ndarray


@dataclass
class ReplayRecorder:
    """Record one camera run into ``directory``.

    IMU samples are means of each ``block_ticks`` physics ticks, stamped at the
    last tick of the block. Frames are stored only when fresh and compressed on
    one writer thread in chunks, so held frame memory stays bounded. Truth at
    those times is kept in ``truth.npz``.
    """

    directory: Path
    simulation_rate_hz: float
    block_ticks: int
    frame_shape: tuple[int, int, int]
    intrinsics: dict
    mount_position_body: Sequence[float]
    tilt_deg: float
    inner_size_m: float | None
    gates: Sequence[tuple[int, Sequence[float], float, float]] = ()

    def __post_init__(self) -> None:
        if self.simulation_rate_hz <= 0.0 or self.block_ticks < 1:
            raise ValueError("simulation rate and IMU block length must be positive")
        self.directory = Path(self.directory)
        self._ring_len = max(8, int(round(_TRUTH_RING_S * self.simulation_rate_hz)))
        self._ring_time = np.zeros(self._ring_len, dtype=np.float64)
        self._ring_pos = np.zeros((self._ring_len, 7), dtype=np.float64)
        self._ring_vel = np.zeros((self._ring_len, 6), dtype=np.float64)
        self._ring_count = 0
        self._ring_index = 0
        self._gyro_block = np.zeros((self.block_ticks, 3), dtype=np.float64)
        self._accel_block = np.zeros((self.block_ticks, 3), dtype=np.float64)
        self._block_fill = 0
        self._block_origin: int | None = None
        self._imu_time: list[float] = []
        self._gyro_mean: list[np.ndarray] = []
        self._accel_mean: list[np.ndarray] = []
        self._imu_pos: list[np.ndarray] = []
        self._imu_vel: list[np.ndarray] = []
        self._baro_time: list[float] = []
        self._baro: list[float] = []
        self._mag_time: list[float] = []
        self._mag: list[np.ndarray] = []
        self._frame_time: list[float] = []
        self._frame_pos: list[np.ndarray] = []
        self._frame_vel: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._chunk_times: list[float] = []
        self._chunk = 0
        self._writer: ThreadPoolExecutor | None = None
        self._pending: deque[Future] = deque()
        self._prepared = False
        self._finished = False
        self._failure: BaseException | None = None

    def observe(
        self,
        *,
        tick: int,
        sim_time: float,
        gyro: np.ndarray,
        accel: np.ndarray,
        world_pos: np.ndarray,
        world_vel: np.ndarray,
        barometer: float | None,
        barometer_fresh: bool,
        magnetometer: np.ndarray | None,
        magnetometer_fresh: bool,
        frame: np.ndarray | None,
        frame_time: float | None,
        frame_fresh: bool,
    ) -> None:
        """Append one physics tick. Incomplete trailing IMU blocks are dropped.

        A write error is remembered and stops further recording. It is raised
        from ``finish``, not from the physics tick, so grading still runs.
        """

        if self._finished or self._failure is not None or not np.isfinite(sim_time):
            return
        try:
            self._prepare()
            position = np.array(world_pos, dtype=np.float64, copy=True).reshape(-1)
            velocity = np.array(world_vel, dtype=np.float64, copy=True).reshape(-1)
            if position.shape[0] < 7 or velocity.shape[0] < 6:
                return
            self._push_truth(float(sim_time), position[:7], velocity[:6])
            self._push_imu(tick, float(sim_time), gyro, accel)
            if barometer_fresh and barometer is not None and np.isfinite(barometer):
                self._baro_time.append(float(sim_time))
                self._baro.append(float(barometer))
            if magnetometer_fresh and magnetometer is not None:
                field = np.array(magnetometer, dtype=np.float64, copy=True).reshape(-1)
                if field.shape[0] >= 3 and np.all(np.isfinite(field[:3])):
                    self._mag_time.append(float(sim_time))
                    self._mag.append(field[:3].copy())
            if (
                frame_fresh
                and frame is not None
                and frame_time is not None
                and np.isfinite(frame_time)
            ):
                self._store_frame(np.asarray(frame), float(frame_time))
        except Exception as error:
            self._remember(error)

    def finish(self) -> None:
        """Write the bundle. A second call does nothing.

        Raises ``RecordingFailed`` when any frame or bundle write failed. The
        bundle is left without ``meta.json`` so it cannot be read as complete.
        """

        if self._finished:
            return
        self._finished = True
        try:
            self._prepare()
            self._flush_frames()
            try:
                while self._pending:
                    self._take_pending()
            finally:
                if self._writer is not None:
                    self._writer.shutdown(wait=True)
            if self._failure is not None:
                raise RecordingFailed(self.directory, self._failure) from self._failure
            self._write_bundle()
        except RecordingFailed:
            raise
        except Exception as error:
            self._remember(error)
            raise RecordingFailed(self.directory, self._failure) from error

    def _write_bundle(self) -> None:
        np.savez_compressed(
            self.directory / "sensors.npz",
            imu_time=_column(self._imu_time),
            gyro_mean=_stack(self._gyro_mean, 3),
            accel_mean=_stack(self._accel_mean, 3),
            baro_time=_column(self._baro_time),
            baro=_column(self._baro),
            mag_time=_column(self._mag_time),
            mag=_stack(self._mag, 3),
        )
        gates = list(self.gates)
        np.savez_compressed(
            self.directory / "truth.npz",
            imu_time=_column(self._imu_time),
            world_pos=_stack(self._imu_pos, 7),
            world_vel=_stack(self._imu_vel, 6),
            frame_time=_column(self._frame_time),
            frame_world_pos=_stack(self._frame_pos, 7),
            frame_world_vel=_stack(self._frame_vel, 6),
            gate_index=np.asarray([gate[0] for gate in gates], dtype=np.int64),
            gate_center=np.asarray([tuple(gate[1]) for gate in gates], dtype=np.float64).reshape(
                -1, 3
            ),
            gate_yaw=np.asarray([gate[2] for gate in gates], dtype=np.float64),
            gate_inner_size=np.asarray([gate[3] for gate in gates], dtype=np.float64),
        )
        meta = {
            "format_version": FORMAT_VERSION,
            "simulation_rate_hz": self.simulation_rate_hz,
            "imu_block_ticks": self.block_ticks,
            "imu_rate_hz": self.simulation_rate_hz / self.block_ticks,
            "frame_shape": list(self.frame_shape),
            "intrinsics": self.intrinsics,
            "mount_position_body": [float(value) for value in self.mount_position_body],
            "tilt_deg": self.tilt_deg,
            "inner_size_m": self.inner_size_m,
        }
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[RACE_RECORD] wrote {self.directory} "
            f"imu={len(self._imu_time)} frames={len(self._frame_time)}"
        )

    def _remember(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error

    def _take_pending(self) -> None:
        try:
            self._pending.popleft().result()
        except Exception as error:
            self._remember(error)

    def _prepare(self) -> None:
        if self._prepared:
            return
        (self.directory / "frames").mkdir(parents=True, exist_ok=True)
        self._prepared = True

    def _push_truth(self, sim_time: float, position: np.ndarray, velocity: np.ndarray) -> None:
        index = self._ring_index
        self._ring_time[index] = sim_time
        self._ring_pos[index] = position
        self._ring_vel[index] = velocity
        self._ring_index = (index + 1) % self._ring_len
        self._ring_count = min(self._ring_count + 1, self._ring_len)

    def _latest_truth(self) -> tuple[np.ndarray, np.ndarray]:
        index = (self._ring_index - 1) % self._ring_len
        return self._ring_pos[index].copy(), self._ring_vel[index].copy()

    def _nearest_truth(self, sample_time: float) -> tuple[np.ndarray, np.ndarray]:
        if self._ring_count == 0:
            return np.full(7, np.nan), np.full(6, np.nan)
        if self._ring_count < self._ring_len:
            times = self._ring_time[: self._ring_count]
            positions = self._ring_pos[: self._ring_count]
            velocities = self._ring_vel[: self._ring_count]
        else:
            times = self._ring_time
            positions = self._ring_pos
            velocities = self._ring_vel
        index = int(np.argmin(np.abs(times - sample_time)))
        if abs(float(times[index]) - sample_time) > 2.0 / self.simulation_rate_hz:
            return np.full(7, np.nan), np.full(6, np.nan)
        return positions[index].copy(), velocities[index].copy()

    def _push_imu(self, tick: int, sim_time: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        gyro_vec = np.array(gyro, dtype=np.float64, copy=True).reshape(-1)
        accel_vec = np.array(accel, dtype=np.float64, copy=True).reshape(-1)
        if gyro_vec.shape[0] < 3 or accel_vec.shape[0] < 3:
            return
        if self._block_fill == 0:
            self._block_origin = tick
        elif tick != self._block_origin + self._block_fill:
            self._block_fill = 0
            self._block_origin = tick
        self._gyro_block[self._block_fill] = gyro_vec[:3]
        self._accel_block[self._block_fill] = accel_vec[:3]
        self._block_fill += 1
        if self._block_fill < self.block_ticks:
            return
        position, velocity = self._latest_truth()
        self._imu_time.append(sim_time)
        self._gyro_mean.append(self._gyro_block.mean(axis=0).copy())
        self._accel_mean.append(self._accel_block.mean(axis=0).copy())
        self._imu_pos.append(position)
        self._imu_vel.append(velocity)
        self._block_fill = 0
        self._block_origin = None

    def _store_frame(self, frame: np.ndarray, frame_time: float) -> None:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 4:
            return
        position, velocity = self._nearest_truth(frame_time)
        self._frame_time.append(frame_time)
        self._frame_pos.append(position)
        self._frame_vel.append(velocity)
        self._frames.append(np.array(frame, dtype=np.uint8, copy=True))
        self._chunk_times.append(frame_time)
        if len(self._frames) >= _FRAME_CHUNK:
            self._flush_frames()

    def _flush_frames(self) -> None:
        if self._failure is not None or not self._frames:
            return
        if self._writer is None:
            self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="race-record")
        while len(self._pending) >= _MAX_PENDING_CHUNKS:
            self._take_pending()
            if self._failure is not None:
                return
        path = self.directory / "frames" / f"chunk_{self._chunk:05d}.npz"
        self._pending.append(
            self._writer.submit(
                np.savez_compressed,
                path,
                frame_time=np.asarray(self._chunk_times, dtype=np.float64),
                frames=np.stack(self._frames, axis=0),
            )
        )
        self._chunk += 1
        self._frames = []
        self._chunk_times = []


def load_sensor_bundle(directory: Path) -> SensorBundle:
    """Load sensors and frames. This function does not read ``truth.npz``."""

    directory = Path(directory)
    meta = _load_meta(directory)
    sensors = np.load(directory / "sensors.npz")
    frame_time, frames = _load_frames(directory, tuple(meta["frame_shape"]))
    return SensorBundle(
        meta=meta,
        imu_time=np.asarray(sensors["imu_time"], dtype=np.float64),
        gyro_mean=np.asarray(sensors["gyro_mean"], dtype=np.float64),
        accel_mean=np.asarray(sensors["accel_mean"], dtype=np.float64),
        baro_time=np.asarray(sensors["baro_time"], dtype=np.float64),
        baro=np.asarray(sensors["baro"], dtype=np.float64),
        mag_time=np.asarray(sensors["mag_time"], dtype=np.float64),
        mag=np.asarray(sensors["mag"], dtype=np.float64),
        frame_time=frame_time,
        frames=frames,
    )


def load_truth(directory: Path) -> TruthBundle:
    """Load the truth file. Evaluators call this; the detector must not."""

    truth = np.load(Path(directory) / "truth.npz")
    return TruthBundle(
        imu_time=np.asarray(truth["imu_time"], dtype=np.float64),
        world_pos=np.asarray(truth["world_pos"], dtype=np.float64),
        world_vel=np.asarray(truth["world_vel"], dtype=np.float64),
        frame_time=np.asarray(truth["frame_time"], dtype=np.float64),
        frame_world_pos=np.asarray(truth["frame_world_pos"], dtype=np.float64),
        frame_world_vel=np.asarray(truth["frame_world_vel"], dtype=np.float64),
        gate_index=np.asarray(truth["gate_index"], dtype=np.int64),
        gate_center=np.asarray(truth["gate_center"], dtype=np.float64),
        gate_yaw=np.asarray(truth["gate_yaw"], dtype=np.float64),
        gate_inner_size=np.asarray(truth["gate_inner_size"], dtype=np.float64),
    )


def _load_meta(directory: Path) -> dict:
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported replay bundle version: {meta.get('format_version')!r}")
    return meta


def _load_frames(
    directory: Path, frame_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    chunks = sorted((directory / "frames").glob("chunk_*.npz"))
    if not chunks:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, *frame_shape), dtype=np.uint8),
        )
    times = []
    frames = []
    for path in chunks:
        chunk = np.load(path)
        times.append(np.asarray(chunk["frame_time"], dtype=np.float64))
        frames.append(np.asarray(chunk["frames"], dtype=np.uint8))
    return np.concatenate(times), np.concatenate(frames, axis=0)


def _column(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _stack(rows: Sequence[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)
