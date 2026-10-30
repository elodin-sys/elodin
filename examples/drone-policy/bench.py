"""Throughput of the drone-policy simulation.

    python bench.py [--estimators mekf6,truth] [--batch 256,1024] [--steps 32] [--devices 1]
                    [--rate-pid dmeas]
    python bench.py --world 1,16 [--backends cranelift,jax-cpu] [--ticks 1500] [--policy DIR]

The first form times DroneEnv rollouts, the training path: vmap over B envs, lax.scan over T
policy steps; --devices N > 1 splits the batch over N XLA host devices with jax.pmap. The second
times N drones flying a controller (default the shipped policy) in one elodin world, main.py's
deploy path, on each backend: build (Cranelift compiles here), a first 3-tick batch (jax-cpu
compiles here, and elodin's profiler leaves it out), then --ticks ticks. step_ms is the median
3-tick batch as elodin's profiler times it, wall_step_ms all of Exec.run per batch (elodin-db
commits included), and x_real_time 10 ms of sim time over step_ms. Each configuration runs in
its own process, so compile time and peak RSS are per config.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def worker(
    estimator: str, batch: int, steps: int, repeats: int, devices: int, rate_pid: str = "dmeas"
) -> dict:
    if devices > 1:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={devices}"
        )
    import resource

    import jax
    import jax.numpy as jnp
    import numpy as np
    from env import TICKS_PER_STEP, DroneEnv

    t0 = time.perf_counter()
    env = DroneEnv(estimator, rate_pid=rate_pid)
    t_build = time.perf_counter() - t0

    def rollout(state, key):
        def body(s, k):
            a = jnp.clip(0.3 * jax.random.normal(k, (s.t.shape[0], 4)), -1.0, 1.0)
            s, obs, critic, r, done, info = jax.vmap(env.step)(s, a)
            return s, r.mean()

        return jax.lax.scan(body, state, jax.random.split(key, steps))

    reset = jax.vmap(env.reset, in_axes=(0, None))
    per_dev = batch // devices
    keys = jax.random.split(jax.random.key(0), batch)
    if devices > 1:
        reset_fn = jax.pmap(reset, in_axes=(0, None))
        roll_fn = jax.pmap(rollout)
        keys = keys.reshape(devices, per_dev)
        roll_keys = jax.random.split(jax.random.key(1), devices)
    else:
        reset_fn = jax.jit(reset)
        roll_fn = jax.jit(rollout)
        roll_keys = jax.random.key(1)

    t0 = time.perf_counter()
    state, _, _ = reset_fn(keys, 1.0)
    jax.block_until_ready(state)
    t_reset = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = roll_fn(state, roll_keys)
    jax.block_until_ready(out)
    t_first = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = roll_fn(state, roll_keys)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    warm = float(np.median(times))
    final = out[0]
    finite = bool(np.all(np.isfinite(np.asarray(final.sim[env.i_pos]))))
    return {
        "estimator": estimator,
        "rate_pid": rate_pid,
        "batch": batch,
        "steps": steps,
        "devices": devices,
        "build_s": round(t_build, 3),
        "reset_compile_s": round(t_reset, 3),
        "rollout_compile_s": round(t_first - warm, 3),
        "rollout_warm_s": round(warm, 4),
        "env_steps_per_s": round(batch * steps / warm),
        "env_ticks_per_s": round(batch * steps * TICKS_PER_STEP / warm),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "finite": finite,
    }


def peak_rss_mb() -> int:
    """This process's peak RSS. Linux carries ru_maxrss across exec, so a worker started from a
    large process would report the parent's peak; VmHWM is the current address space's own."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def world_worker(
    backend: str, drones: int, ticks: int, controller: str | None = None, estimator=None
) -> dict:
    import numpy as np
    import policy_system as PS
    from env import SIM_RATE, TICKS_PER_STEP

    assert ticks % TICKS_PER_STEP == 0, ticks
    w, system = PS.build_world(controller or PS.SHIPPED, drones, estimator)
    t0 = time.perf_counter()
    ex = w.build(
        system,
        simulation_rate=SIM_RATE,
        telemetry_rate=SIM_RATE / TICKS_PER_STEP,
        backend=backend,
    )
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex.run(TICKS_PER_STEP, show_progress=False)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex.run(ticks, show_progress=False)
    wall = time.perf_counter() - t0
    prof = ex.profile()
    names = [f"{e}.world_pos" for e in PS.entity_names(drones)]
    df = ex.history(names)
    return {
        "backend": backend,
        "drones": drones,
        "ticks": ticks,
        "build_s": round(t_build, 3),
        "first_step_s": round(t_first, 3),
        "step_ms": round(prof["tick_median"], 4),
        "x_real_time": round(prof["real_time_factor_median"], 2),
        "wall_step_ms": round(1e3 * wall * TICKS_PER_STEP / ticks, 4),
        "peak_rss_mb": peak_rss_mb(),
        "finite": all(bool(np.all(np.isfinite(np.stack(df[n].to_numpy())))) for n in names),
    }


def world_rows(drones, backends, ticks: int, controller=None, estimator=None, log=False) -> list:
    """world_worker per (drone count, backend), each in its own process; a failed worker gives
    a row with its exit code and stderr tail. ELODIN_BACKEND would override backend= and
    XLA_FLAGS device counts would change jax-cpu, so both are dropped."""
    env = {k: v for k, v in os.environ.items() if k not in ("ELODIN_BACKEND", "XLA_FLAGS")}
    rows = []
    for n in drones:
        for b in backends:
            cmd = [sys.executable, os.path.abspath(__file__), "--world-worker", b, str(n)]
            cmd += [str(ticks)] + (["--policy", controller] if controller else [])
            cmd += ["--estimator", estimator] if estimator else []
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            if res.returncode:
                row = {
                    "backend": b,
                    "drones": n,
                    "exit": res.returncode,
                    "tail": res.stderr[-2000:],
                }
            else:
                row = json.loads(res.stdout.strip().splitlines()[-1])
            rows.append(row)
            if log:
                print(json.dumps(row), flush=True)
    return rows


def speedups(rows) -> dict:
    """jax-cpu step_ms over Cranelift step_ms per drone count (above 1: Cranelift is faster)."""
    by = {(r["drones"], r["backend"]): r["step_ms"] for r in rows if "step_ms" in r}
    return {
        n: float(f"{by[n, 'jax-cpu'] / by[n, 'cranelift']:.3g}")
        for n, b in sorted(by)
        if b == "cranelift" and (n, "jax-cpu") in by
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--estimators", default="mekf6")
    p.add_argument("--batch", default="256,1024")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--devices", default="1")
    p.add_argument("--rate-pid", default="dmeas")
    p.add_argument("--worker", nargs=4, metavar=("EST", "B", "T", "D"))
    p.add_argument("--world", help="drone counts (comma list): time an elodin world instead")
    p.add_argument("--backends", default="cranelift,jax-cpu")
    p.add_argument("--ticks", type=int, default=1500)
    p.add_argument("--policy", help="the --world controller (default the shipped policy)")
    p.add_argument("--estimator", help="the --world estimator (default the controller's)")
    p.add_argument("--world-worker", nargs=3, metavar=("BACKEND", "N", "TICKS"))
    args = p.parse_args()
    if args.world_worker:
        b, n, t = args.world_worker
        print(json.dumps(world_worker(b, int(n), int(t), args.policy, args.estimator)))
        return
    if args.world:
        drones = [int(x) for x in args.world.split(",")]
        backends = args.backends.split(",")
        rows = world_rows(drones, backends, args.ticks, args.policy, args.estimator, log=True)
        keys = ["backend", "drones", "step_ms", "x_real_time", "wall_step_ms", "build_s"]
        keys += ["first_step_s", "finite"]
        print(" | ".join(keys))
        for r in rows:
            print(" | ".join(str(r.get(k)) for k in keys))
        for n, x in speedups(rows).items():
            faster, ratio = ("Cranelift", x) if x >= 1 else ("jax-cpu", float(f"{1 / x:.3g}"))
            print(f"{n} drones: {faster} is {ratio}x faster per 3-tick step")
        return
    if args.worker:
        est, b, t, d = args.worker
        print(json.dumps(worker(est, int(b), int(t), args.repeats, int(d), args.rate_pid)))
        return
    rows = []
    for d in [int(x) for x in args.devices.split(",")]:
        for est in args.estimators.split(","):
            for b in [int(x) for x in args.batch.split(",")]:
                cmd = [sys.executable, os.path.abspath(__file__), "--repeats", str(args.repeats)]
                cmd += ["--rate-pid", args.rate_pid]
                cmd += ["--worker", est, str(b), str(args.steps), str(d)]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                row = json.loads(res.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(json.dumps(row), flush=True)
    keys = ["estimator", "batch", "devices", "env_steps_per_s", "env_ticks_per_s"]
    keys += ["rollout_compile_s", "rollout_warm_s", "peak_rss_mb"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
