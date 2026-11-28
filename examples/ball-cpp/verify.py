"""Check ball_cpp trajectories against the original ball example run in Elodin.

    python3 examples/ball-cpp/verify.py [ball_cpp.csv ...] [--seed N] [--atol X]

Runs `examples/ball` for as many ticks as the CSVs hold, compares every tick of `world_pos`
and `world_vel`, prints the largest difference, and plots the trajectories next to the CSVs.
Exits nonzero if any difference exceeds `--atol`, which is 0 by default: the native library
runs the same Cranelift code as Elodin, so the results should match bit for bit.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BUILD = Path(__file__).resolve().parent / "build"
# The original example, not this one's copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ball"))
from sim import SIM_TIME_STEP, system, world  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("csv", nargs="*", type=Path, default=[BUILD / "ball_cpp.csv"])
parser.add_argument("--seed", type=int, default=0, help="the seed ball_cpp ran with")
parser.add_argument("--atol", type=float, default=0.0, help="largest difference allowed")
parser.add_argument("--plot", type=Path, help="where to save the plot (default: next to the CSVs)")
args = parser.parse_args()

native = {path: np.loadtxt(path, delimiter=",", skiprows=1) for path in args.csv}
ticks = int(next(iter(native.values()))[-1, 0])

execution = world(args.seed).build(system(), simulation_rate=1.0 / SIM_TIME_STEP)
execution.run(ticks, show_progress=False)
components = ["ball.world_pos", "ball.world_vel"]
history = execution.history(components)
elodin = np.hstack(
    [np.stack([np.asarray(row.to_numpy()) for row in history[name]]) for name in components]
)
if elodin.shape != (ticks + 1, 13):
    sys.exit(f"Elodin recorded {elodin.shape} values, expected {(ticks + 1, 13)}")

failed = False
for path, values in native.items():
    if values.shape != (ticks + 1, 14):
        sys.exit(f"{path} holds {values.shape} values, expected {(ticks + 1, 14)}")
    difference = np.abs(values[:, 1:] - elodin)
    worst = np.unravel_index(np.nanargmax(np.nan_to_num(difference, nan=np.inf)), difference.shape)
    exact = np.array_equal(values[:, 1:], elodin)
    print(
        f"{path}: {ticks} ticks, largest difference {difference[worst]:.3g} "
        f"(tick {worst[0]}, column {worst[1]}), "
        f"{'bit for bit equal' if exact else 'not bit for bit equal'} to Elodin"
    )
    # Any NaN fails, and at the default atol of 0 so does any difference at all.
    failed |= not np.allclose(values[:, 1:], elodin, rtol=0.0, atol=args.atol)

time = np.arange(ticks + 1) * SIM_TIME_STEP
figure, (height, error) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
height.plot(time, elodin[:, 6], label="Elodin (examples/ball)", linewidth=3, alpha=0.4)
for path, values in native.items():
    height.plot(time, values[:, 7], "--", label=path.name)
    error.plot(time, np.abs(values[:, 1:] - elodin).max(axis=1), label=path.name)
height.set_ylabel("height z (m)")
height.legend()
error.set_ylabel("largest |difference|")
error.set_xlabel("time (s)")
error.legend()
figure.suptitle("Ball trajectory: native C++ host vs Elodin")
plot = args.plot or next(iter(native)).with_suffix(".png")
figure.savefig(plot, dpi=120)
print(f"saved {plot}")

sys.exit(1 if failed else 0)
