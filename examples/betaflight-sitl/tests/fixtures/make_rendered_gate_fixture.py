"""Build the small rendered-gate fixture from a Package E replay bundle.

Usage, from the repository root after a truth run with ``RACE_RECORD``:

    python3 examples/betaflight-sitl/tests/fixtures/make_rendered_gate_fixture.py \
        /tmp/pkg-h-truth \
        examples/betaflight-sitl/tests/fixtures/rendered_gate_frames.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXAMPLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_EXAMPLE))

from course import Gate  # noqa: E402
from replay import load_sensor_bundle, load_truth  # noqa: E402
from synthetic_vision import project_gate_corners  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} BUNDLE OUTPUT.npz", file=sys.stderr)
        return 2
    bundle = Path(argv[1])
    output = Path(argv[2])
    sensors = load_sensor_bundle(bundle)
    truth = load_truth(bundle)
    if truth.gate_center.shape[0] != 1:
        raise SystemExit("fixture builder expects one gate")
    gate = Gate(
        int(truth.gate_index[0]),
        tuple(float(value) for value in truth.gate_center[0]),
        float(truth.gate_yaw[0]),
        float(truth.gate_inner_size[0]),
    )
    chosen = _select(sensors.frames, truth.frame_world_pos, truth.frame_time, gate)
    if not chosen:
        raise SystemExit("no fully visible frames")
    index = np.asarray(chosen, dtype=int)
    np.savez_compressed(
        output,
        frames=sensors.frames[index],
        frame_time=truth.frame_time[index],
        world_pos=truth.frame_world_pos[index],
        world_vel=truth.frame_world_vel[index],
        gate_center=truth.gate_center[0],
        gate_yaw=np.asarray(gate.yaw, dtype=np.float64),
        inner_size=np.asarray(gate.inner_size, dtype=np.float64),
        gate_index=np.asarray(gate.index, dtype=np.int64),
    )
    print(f"wrote {output} frames={len(chosen)} bytes={output.stat().st_size}")
    return 0


def _select(frames, world_pos, frame_time, gate: Gate) -> list[int]:
    visible: list[tuple[float, int]] = []
    for index, pose in enumerate(world_pos):
        if not np.all(np.isfinite(pose)):
            continue
        if project_gate_corners(gate, pose) is None:
            continue
        distance = float(np.linalg.norm(pose[4:7] - np.asarray(gate.center)))
        visible.append((distance, index))
    if not visible:
        return []
    visible.sort()
    # Up to two frames in each 2 m bin from 4 m to 20 m, plus nearer hold frames.
    picked: list[int] = []
    for low in range(2, 20, 2):
        group = [item for item in visible if low <= item[0] < low + 2]
        if not group:
            continue
        group.sort(key=lambda item: (abs(item[0] - (low + 1)), frame_time[item[1]]))
        picked.extend(item[1] for item in group[:2])
    if len(picked) > 16:
        step = len(picked) / 16
        picked = [picked[int(i * step)] for i in range(16)]
    return sorted(set(picked))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
