"""Throughput of DroneEnv rollouts: vmap over B envs, lax.scan over T policy steps.

    python bench.py [--estimators mekf6,truth] [--batch 256,1024] [--steps 32] [--devices 1]
                    [--rate-pid dmeas]

Each configuration runs in its own process, so compile time and peak RSS are per config.
--devices N > 1 splits the batch over N XLA host devices with jax.pmap.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--estimators", default="mekf6")
    p.add_argument("--batch", default="256,1024")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--devices", default="1")
    p.add_argument("--rate-pid", default="dmeas")
    p.add_argument("--worker", nargs=4, metavar=("EST", "B", "T", "D"))
    args = p.parse_args()
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
