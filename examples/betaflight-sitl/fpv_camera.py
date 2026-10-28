"""FPV camera sampling, freshness, and acceptance for the Betaflight SITL example.

The reader is injected so these rules can be tested without Elodin or a GPU.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MSG = "drone.fpv"
WIDTH = 640
HEIGHT = 360
FPS = 30.0
LATENCY_US = 33_000
MOUNT = [0.08, 0.0, 0.02]
NEAR = 0.1
FAR = 100.0
# Square pixels. Vertical FoV is chosen so fx = fy = FOCAL_PX (≈58.72°, hFOV 90°).
FOCAL_PX = 320.0
FOV_DEG = 2.0 * math.degrees(math.atan((HEIGHT / 2.0) / FOCAL_PX))
# Positive TILT_DEG pitches the optical axis up (nose-up). Positive sensor_camera
# pitch is the opposite: a right-handed turn about body +Y, so nose-down.
TILT_DEG = 0.0


def sensor_rot_offset(tilt_deg: float = TILT_DEG) -> list[float]:
    """Body [roll, pitch, yaw] in degrees for ``sensor_camera``.

    Renderer pitch is nose-down, so a nose-up ``tilt_deg`` is registered as a
    negative pitch.
    """

    return [0.0, -tilt_deg, 0.0]


PERIOD_US = int(round(1_000_000.0 / FPS))
FRAME_BYTES = WIDTH * HEIGHT * 4
MIN_FPS = 15.0
WARMUP_US = 2_000_000
_NEWER_THAN_ANY_FRAME = 2**62

ReadMsgAt = Callable[[str, int], tuple[int, np.ndarray] | None]


@dataclass(frozen=True)
class FpvSample:
    """One camera-period offer with a nominal image/render ``frame_time``.

    Time is simulation seconds on the ``sim_time`` clock, mapped from the
    selected image-message timestamp; scene-time alignment is approximate, not
    a verified exact capture-pose clock. A held frame retains its original
    ``frame_time`` with ``fresh=False``. Both frame fields are ``None`` when no
    valid frame was selected.
    """

    frame: np.ndarray | None
    frame_time: float | None
    fresh: bool


@dataclass(frozen=True)
class FpvReport:
    """Shutdown accounting. total_frames is every renderer message."""

    first_frame_sim_s: float | None
    total_frames: int
    window_frames: int
    window_s: float
    observed_fps: float
    valid_samples: int
    fresh_samples: int
    held_samples: int
    malformed: bool
    accepted: bool
    reason: str

    def format(self) -> str:
        """Return the shutdown report, including a stable status line."""
        lines = [
            "--- FPV camera (Package B) ---",
            f"  first_frame_sim_s: {self._time(self.first_frame_sim_s)}",
            f"  total_frames: {self.total_frames}",
            f"  window_frames: {self.window_frames}",
            f"  window_s: {self.window_s:.2f}",
            f"  observed_sim_fps≈{self.observed_fps:.2f}",
            f"  valid_samples: {self.valid_samples}",
            f"  fresh_samples: {self.fresh_samples}",
            f"  held_samples: {self.held_samples}",
            f"  malformed: {self.malformed}",
            f"  [FPV] total_frames={self.total_frames} "
            f"window_frames={self.window_frames} "
            f"observed_fps={self.observed_fps:.2f} "
            f"status={'PASS' if self.accepted else 'FAIL'} "
            f"reason={self.reason}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _time(value: float | None) -> str:
        return "none" if value is None else f"{value:.3f}"


class FpvCamera:
    """Samples drone.fpv at most once per nominal camera period."""

    def __init__(self) -> None:
        self.last_period_idx = -1
        self.last_selected_ts: int | None = None
        self.last_requested_us: int | None = None
        self.sim_start_us: float | None = None
        self._clock_anchor_us: int | None = None
        self._clock_anchor_sim_time_s: float | None = None
        self.first_frame_sim_s: float | None = None
        self.valid_samples = 0
        self.fresh_samples = 0
        self.held_samples = 0
        self.malformed = False

    def poll(self, read_msg_at: ReadMsgAt, now_us: int, sim_time_s: float) -> FpvSample:
        """Read the latency-adjusted frame for this tick, if this period is new.

        ``requested_us`` stays on the camera for accounting. A valid sample's
        ``frame_time`` is the nominal image/render timestamp in simulation
        seconds, derived from the selected message timestamp, not query,
        callback, arrival, or a component's last-change time. Scene-time
        alignment is approximate.
        """
        if self._clock_anchor_us is None:
            self._clock_anchor_us = int(now_us)
            self._clock_anchor_sim_time_s = float(sim_time_s)
            self.sim_start_us = now_us - sim_time_s * 1_000_000.0

        period_idx = int(now_us // PERIOD_US)
        if period_idx == self.last_period_idx:
            return FpvSample(None, None, False)
        self.last_period_idx = period_idx

        requested = int(now_us) - LATENCY_US
        self.last_requested_us = requested
        selected = read_msg_at(MSG, requested)
        if selected is None:
            return FpvSample(None, None, False)

        selected_ts, payload = selected
        frame = _valid_frame(payload)
        if frame is None:
            self.malformed = True
            return FpvSample(None, None, False)

        selected_ts = int(selected_ts)
        fresh = selected_ts != self.last_selected_ts
        self.last_selected_ts = selected_ts
        self.valid_samples += 1
        if fresh:
            self.fresh_samples += 1
        else:
            self.held_samples += 1
        if self.first_frame_sim_s is None:
            self.first_frame_sim_s = sim_time_s
        assert self._clock_anchor_us is not None
        assert self._clock_anchor_sim_time_s is not None
        frame_time = (
            self._clock_anchor_sim_time_s + (selected_ts - self._clock_anchor_us) / 1_000_000.0
        )
        return FpvSample(frame, frame_time, fresh)

    def finish(self, read_msg_at: ReadMsgAt, end_us: int) -> FpvReport:
        """Count every renderer frame, then score only the post-warmup window."""
        timestamps, window_end = _all_timestamps(read_msg_at, end_us)
        window_start = (self.sim_start_us if self.sim_start_us is not None else 0) + WARMUP_US
        window_frames = sum(1 for ts in timestamps if window_start <= ts <= window_end)
        window_us = max(window_end - window_start, 1)
        window_s = window_us / 1_000_000.0
        observed_fps = window_frames / window_s
        accepted, reason = _acceptance(
            total_frames=len(timestamps),
            malformed=self.malformed,
            observed_fps=observed_fps,
        )
        return FpvReport(
            first_frame_sim_s=self.first_frame_sim_s,
            total_frames=len(timestamps),
            window_frames=window_frames,
            window_s=window_s,
            observed_fps=observed_fps,
            valid_samples=self.valid_samples,
            fresh_samples=self.fresh_samples,
            held_samples=self.held_samples,
            malformed=self.malformed,
            accepted=accepted,
            reason=reason,
        )


def _valid_frame(payload: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(payload)
    if arr.dtype != np.uint8 or arr.size != FRAME_BYTES:
        return None
    return np.array(arr, dtype=np.uint8, copy=True).reshape(HEIGHT, WIDTH, 4)


def _all_timestamps(read_msg_at: ReadMsgAt, end_us: int) -> tuple[list[int], int]:
    """Walk backward from the newest message so a late final frame is included.

    The returned end time is the query upper bound used for the FPS window.
    A renderer message newer than the final simulation timestamp still counts
    in the total, and it extends the window so it is not dropped from the FPS set.
    """
    found: list[int] = []
    cursor = max(end_us, 0)
    newest = read_msg_at(MSG, _NEWER_THAN_ANY_FRAME)
    if newest is not None:
        cursor = max(cursor, int(newest[0]))
    window_end = cursor
    while True:
        selected = read_msg_at(MSG, cursor)
        if selected is None:
            break
        timestamp = int(selected[0])
        found.append(timestamp)
        cursor = timestamp - 1
    found.reverse()
    return found, window_end


def _acceptance(*, total_frames: int, malformed: bool, observed_fps: float) -> tuple[bool, str]:
    if total_frames <= 0:
        return False, "no-frames"
    if malformed:
        return False, "malformed-frame"
    if observed_fps < MIN_FPS:
        return False, "fps-below-15"
    return True, "ok"
