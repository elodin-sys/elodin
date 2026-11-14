"""Final evaluation and summary of a training grid written by train_grid.sh.

    python examples/drone-policy/eval_runs.py --grid t4 [--devices 8] [--note NAME=TEXT ...]
    python examples/drone-policy/eval_runs.py --grid t3 [--devices 8] [--note NAME=TEXT ...]
    python examples/drone-policy/eval_runs.py [--grid t2] [--runs DIR] [--names main,A1,A2]
        [--seeds 0,1,2] [--pairs 1,1 0,1] [--episodes 2048] [--seed 10007]
        [--baselines b0-nominal,b0-dr,b1-dr] [--reference b0-dr] [--no-reeval]
        [--run-dir NAME=DIR ...] [--estimator EST] [--cross NAME=EST ...] [--contrast A:B ...]
        [--out DIR]

--grid sets the defaults: t2 (RUNS ~/drone-policy-runs, main/A1/A2 at (1,1) and (0,1), 2048
episodes from seed 10007, no baselines), t3 (RUNS ~/drone-policy-runs/t3, S-main/R-main at
(1,1), (0.5,1) and (0,1), 4096 episodes from seed 20011, baselines B0 nominal, B0 DR and B1 DR,
each row paired against B0 DR) or t4 (t3's protocol on RUNS ~/drone-policy-runs/t4: S-main8k,
S-main8k-intol1 (seed 0 only), S-A1, S-A2, plus T3's S-main read from ~/drone-policy-runs/t3;
cross-estimator rows S-A2@mekf6, S-main@truth and S-main8k@truth; seed-paired contrasts).

Every finished RUNS/<name>_s<seed> (done marker 0) is evaluated deterministically per episode on
train.py's final-eval episodes (ppo.make_evaluate's reset keys); --run-dir NAME=DIR reads NAME's
runs from DIR instead. A residual run is rebuilt over B0 from its config.json. A run is observed
through its training estimator unless --estimator overrides that for every run; --cross NAME=EST
adds the rows NAME@EST, NAME's checkpoints observed through EST, and the training estimator stays
recorded. The plant DR, throw and target come from the reset key alone, so every estimator flies
the same episodes. The baselines run through cascade_b0.Runner on the same reset keys. Each row
carries the success count with a Wilson 95% interval, mean return and length, terminated rate and
termination causes (the first in cascade_b0.CAUSES order). final_eval.json is written when
train.py did not write one and is otherwise checked against the re-evaluation (rows observed
through their training estimator only).

Per config: mean and sample std over seeds, the Wilson interval of the pooled episodes and a t
interval over seeds for the paired diff against the reference. --contrast A:B pairs config A
against config B seed by seed on the same episodes.

Writes to OUT (default RUNS): summary.json (baselines, per-run rows with health and wall times,
per-config statistics, contrasts), summary.md (the tables), heldout_episodes.npz (per-episode
outcomes), curves.json and curves.png (in-training eval success and return vs iteration, with the
held-out finals and baselines).
"""

import argparse
import json
import math
import os
import re

import numpy as np

Z95 = 1.959964
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
ESTIMATORS = ("mekf18", "mekf6", "truth")
T3_PROTOCOL = {
    "pairs": [(1.0, 1.0), (0.5, 1.0), (0.0, 1.0)],
    "episodes": 4096,
    "seed": 20011,
    "baselines": "b0-nominal,b0-dr,b1-dr",
    "reference": "b0-dr",
}
PRESETS = {
    "t2": {
        "runs": "~/drone-policy-runs",
        "names": "main,A1,A2",
        "pairs": [(1.0, 1.0), (0.0, 1.0)],
        "episodes": 2048,
        "seed": 10007,
        "baselines": "",
        "reference": "",
        "run_dir": [],
        "cross": [],
        "contrast": [],
    },
    "t3": {
        "runs": "~/drone-policy-runs/t3",
        "names": "S-main,R-main",
        **T3_PROTOCOL,
        "run_dir": [],
        "cross": [],
        "contrast": [],
    },
    "t4": {
        "runs": "~/drone-policy-runs/t4",
        "names": "S-main8k,S-main8k-intol1,S-A1,S-A2,S-main",
        **T3_PROTOCOL,
        "run_dir": ["S-main=~/drone-policy-runs/t3"],
        "cross": ["S-A2=mekf6", "S-main=truth", "S-main8k=truth"],
        "contrast": [
            "S-main8k:S-main",
            "S-main8k-intol1:S-main8k",
            "S-A1:S-main",
            "S-A2:S-main",
            "S-A2@mekf6:S-main",
            "S-A2@mekf6:S-A2",
            "S-main@truth:S-main",
            "S-main8k@truth:S-main8k",
        ],
    },
}
BASELINES = {  # id -> (cascade_b0 controller, gain set, label)
    "b0-nominal": ("b0", "nominal", "B0 nominal"),
    "b0-dr": ("b0", "dr", "B0 DR"),
    "b1-nominal": ("b1", "nominal", "B1 nominal"),
    "b1-dr": ("b1", "dr", "B1 DR"),
}
METRICS = ("mean_return", "success_rate", "terminated_rate", "mean_length")


def _pair(text: str) -> tuple[float, float]:
    dr, throw = (float(x) for x in text.split(","))
    return dr, throw


def _kv(text: str, sep: str = "=") -> tuple[str, str]:
    k, v = text.split(sep, 1)
    return k, v


def _6g(value) -> float:
    return float(f"{float(value):.6g}")


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _time_v(stderr_path: str, pattern: str) -> str | None:
    try:
        with open(stderr_path) as f:
            m = re.search(pattern, f.read())
    except FileNotFoundError:
        return None
    return m.group(1) if m else None


def elapsed_s(stderr_path: str) -> float | None:
    """Wall clock of the whole process from /usr/bin/time -v."""
    text = _time_v(stderr_path, r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")
    if text is None:
        return None
    parts = [float(x) for x in text.split(":")]
    return round(sum(p * 60**i for i, p in enumerate(reversed(parts))), 2)


def peak_rss_gb(stderr_path: str) -> float | None:
    text = _time_v(stderr_path, r"Maximum resident set size \(kbytes\): (\d+)")
    return None if text is None else round(int(text) / 2**20, 2)


def train_wall_s(stdout_path: str) -> float | None:
    """Training-loop wall time printed by train.py (excludes the final eval)."""
    try:
        with open(stdout_path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if line.startswith('{"done"'):
            return json.loads(line)["wall_s"]
    return None


def load_policy(path: str):
    """nn.load_policy (kept here for the fork scripts that import it from eval_runs)."""
    import nn

    return nn.load_policy(path)


def health(log: list[dict], epochs: int | None = None) -> dict:
    """Update health over a run's log; a KL stop is an iteration that ran fewer than `epochs`."""
    keys = ("approx_kl", "value_loss", "policy_loss", "entropy", "grad_norm")
    kl_stops = [r["iter"] for r in log if epochs is not None and r.get("epochs", epochs) < epochs]
    return {
        "iters": len(log),
        "actor_frozen_iters": sum(r.get("actor_lr") == 0 for r in log),
        "kl_stop_iters": len(kl_stops),
        "kl_stop_at": kl_stops[:20],
        "nonfinite_steps": sum(r["nonfinite_steps"] for r in log),
        "skipped_updates": sum(r["skipped_updates"] for r in log),
        "replay_envs": sum(r["replay_envs"] for r in log),
        "null_metrics": sum(r.get(k) is None for r in log for k in keys),
        "max_approx_kl": max(
            (r["approx_kl"] for r in log if r["approx_kl"] is not None), default=None
        ),
    }


def mean_std(values: list[float]) -> list[float | None]:
    """[mean, sample std (ddof 1)]; the std is None for a single value."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return [round(mean, 6), None]
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return [round(mean, 6), round(std, 6)]


def seed_ci(values: list[float]) -> list[float] | None:
    """95% t interval of the mean over seeds; None below 2 seeds."""
    n = len(values)
    if n < 2:
        return None
    mean, std = mean_std(values)
    half = T95.get(n - 1, Z95) * std / math.sqrt(n)
    return [round(mean - half, 6), round(mean + half, 6)]


def wilson(k: int, n: int, z: float = Z95) -> list[float]:
    p, d = k / n, 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 6), round(centre + half, 6)]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Success of a minus b on the same episodes, with the standard error of the mean diff."""
    a, b = a.astype(bool), b.astype(bool)
    d = a.astype(np.float64) - b.astype(np.float64)
    se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else 0.0
    return {
        "diff": round(float(d.mean()), 6),
        "se": round(float(se), 6),
        "only": int(np.sum(a & ~b)),
        "ref_only": int(np.sum(b & ~a)),
    }


def make_rollout(env, num_envs: int, obs_clip: float, devices: int):
    """-> rollout(actor, obs_rms, key, dr, throw) -> per-episode numpy arrays (ret, length,
    success, terminated, cause: index into cascade_b0.CAUSES, -1 if not terminated), on the
    episodes and with the actions of ppo.make_evaluate(env, num_envs, deterministic=True)."""
    import jax
    import jax.numpy as jnp
    import nn
    import ppo
    import residual
    from env import TERM_OMEGA, TERM_POS, TERM_TILT, TERM_Z
    from jax.sharding import PartitionSpec as P

    assert num_envs % devices == 0, "num_envs must divide over devices"
    f32 = jnp.float32
    plant = env.env if isinstance(env, residual.ResidualEnv) else env
    v_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    v_step = jax.vmap(env.step)

    def flags(sim, target):  # cascade_b0.CAUSES order
        pos, vel = sim[plant.i_pos], sim[plant.i_vel]
        q, p = pos[:4], pos[4:]
        cos_tilt = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
        return jnp.stack(
            [
                ~plant.finite(sim),
                cos_tilt < np.cos(TERM_TILT),
                jnp.linalg.norm(vel[:3]) > TERM_OMEGA,
                p[2] < TERM_Z,
                jnp.linalg.norm(p - target) > TERM_POS,
            ]
        ).astype(jnp.int32)

    v_flags = jax.vmap(flags)

    def episodes(actor, obs_rms, keys, dr_scale, throw_scale):
        state, obs, _ = v_reset(keys, dr_scale, throw_scale)
        false = jnp.zeros(keys.shape[0], bool)

        def step(carry, _):
            state, obs, alive, ret, length, success, terminated, cause = carry
            mean, _ = ppo.actor_dist(actor, nn.normalize(obs_rms, obs, obs_clip))
            state, obs, _, reward, done, info = v_step(state, jnp.clip(mean, -1.0, 1.0))
            ended = alive & done.astype(bool)
            ret = ret + jnp.where(alive, ppo._finite(reward), 0.0)
            length = length + alive.astype(jnp.int32)
            success = success | (ended & info["success"].astype(bool))
            term = ended & info["terminated"].astype(bool)
            es = state.env if isinstance(state, residual.ResidualState) else state
            first = jnp.argmax(v_flags(es.sim, es.target), axis=-1).astype(jnp.int32)
            cause = jnp.where(term, first, cause)
            carry = (state, obs.astype(f32), alive & ~ended, ret, length, success)
            return (*carry, terminated | term, cause), None

        zeros = jnp.zeros(keys.shape[0], f32)
        carry = (state, obs.astype(f32), ~false, zeros, zeros.astype(jnp.int32), false, false)
        carry, _ = jax.lax.scan(
            step, (*carry, jnp.full(keys.shape[0], -1, jnp.int32)), None, env.episode_steps
        )
        return carry[3:]

    episodes = ppo._sharded(episodes, devices, (P(), P(), P("b"), P(), P()), P("b"))

    @jax.jit
    def run(actor, obs_rms, key, dr_scale, throw_scale):
        k_reset, _ = jax.random.split(key)  # make_evaluate's reset keys
        return episodes(actor, obs_rms, jax.random.split(k_reset, num_envs), dr_scale, throw_scale)

    to_mesh, to_first = ppo._movers(devices)

    def rollout(actor, obs_rms, key, dr, throw):
        out = run(*to_mesh((actor, obs_rms, key)), f32(dr), f32(throw))
        out = jax.device_get(to_first(out))
        return dict(zip(("ret", "length", "success", "terminated", "cause"), out))

    return rollout


def run_baselines(env, ids: list[str], raw_keys, pairs) -> dict:
    """{id: [per-episode arrays per pair]} from cascade_b0.Runner on the given reset keys."""
    import cascade_b0
    import jax

    keys = jax.random.wrap_key_data(raw_keys)
    runners, out = {}, {}
    for b in ids:
        controller, gain_set, _ = BASELINES[b]
        if controller not in runners:
            runners[controller] = cascade_b0.Runner(env, controller)
        gains = cascade_b0.as_batch(cascade_b0.GAIN_SETS[gain_set], keys.shape[0])
        out[b] = []
        for dr, throw in pairs:
            o = runners[controller](keys, gains, dr, throw)
            first = np.where(o["cause"].any(-1), np.argmax(o["cause"], axis=-1), -1)
            ep = {k: np.asarray(o[k]) for k in ("ret", "length", "success", "terminated")}
            ep["cause"] = np.where(ep["terminated"], first, -1).astype(np.int32)
            out[b].append(ep)
    return out


def episode_stats(ep: dict, summarize, causes: tuple[str, ...]) -> dict:
    """Summary of per-episode arrays; the means are make_evaluate's (float32 on one device)."""
    import jax

    s = jax.device_get(summarize(*(ep[k] for k in ("ret", "length", "success", "terminated"))))
    n, k = int(ep["success"].size), int(np.sum(ep["success"]))
    term = ep["terminated"].astype(bool)
    counts = {c: int(np.sum(term & (ep["cause"] == i))) for i, c in enumerate(causes)}
    assert sum(counts.values()) == int(term.sum()), "every termination needs a cause"
    return {
        "episodes": n,
        "successes": k,
        **{m: _6g(s[m]) for m in METRICS},
        "wilson95": wilson(k, n),
        "causes": counts,
    }


def contrast(rows: list[dict], per_episode: dict, a: str, b: str, tags: list[str]) -> dict:
    """Config a minus config b, seed by seed on the same episodes: per-seed paired success diffs,
    their mean +- std and t interval over seeds, and the terminated-rate diff."""
    ra = {r["seed"]: r for r in rows if r["name"] == a}
    rb = {r["seed"]: r for r in rows if r["name"] == b}
    seeds = sorted(set(ra) & set(rb))
    out = {"a": a, "b": b, "seeds": seeds}
    if not seeds:
        return out
    for i, t in enumerate(tags):
        per_seed = [
            paired(per_episode[ra[s]["run"]][i]["success"], per_episode[rb[s]["run"]][i]["success"])
            for s in seeds
        ]
        diffs = [p["diff"] for p in per_seed]
        out[f"per_seed@{t}"] = per_seed
        out[f"diff@{t}"] = mean_std(diffs)
        out[f"diff_ci95@{t}"] = seed_ci(diffs)
        term = [
            ra[s]["final_eval"][i]["terminated_rate"] - rb[s]["final_eval"][i]["terminated_rate"]
            for s in seeds
        ]
        out[f"terminated_diff@{t}"] = mean_std(term)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grid", choices=sorted(PRESETS), default="t2", help="sets the defaults")
    parser.add_argument("--runs")
    parser.add_argument("--names")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--pairs", nargs="+", type=_pair)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--baselines", help="comma list of " + ", ".join(BASELINES))
    parser.add_argument("--baseline-estimator", default="mekf6")
    parser.add_argument("--reference", help="baseline every row is paired against")
    parser.add_argument("--note", action="append", default=[], metavar="NAME=TEXT")
    parser.add_argument("--no-reeval", action="store_true", help="use final_eval.json as written")
    parser.add_argument(
        "--run-dir", action="append", metavar="NAME=DIR", help="read NAME's runs from DIR"
    )
    parser.add_argument(
        "--estimator", choices=ESTIMATORS, help="observe every run through this estimator"
    )
    parser.add_argument(
        "--cross", action="append", metavar="NAME=EST", help="add rows NAME@EST observed via EST"
    )
    parser.add_argument(
        "--contrast", action="append", metavar="A:B", help="config A minus config B, seed-paired"
    )
    parser.add_argument("--out", help="output directory (default RUNS)")
    args = parser.parse_args(argv)
    for k, v in PRESETS[args.grid].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.runs = os.path.expanduser(args.runs)
    args.out = os.path.expanduser(args.out) if args.out else args.runs
    args.baselines = [b for b in args.baselines.split(",") if b]
    for b in args.baselines:
        if b not in BASELINES:
            parser.error(f"unknown baseline {b}")
    if args.reference and args.reference not in args.baselines:
        parser.error("--reference must be one of --baselines")
    if args.no_reeval and args.reference:
        parser.error("--no-reeval has no per-episode outcomes to pair; pass --reference ''")
    args.run_dir = {k: os.path.expanduser(v) for k, v in map(_kv, args.run_dir)}
    args.cross = [_kv(c) for c in args.cross]
    for _, est in args.cross:
        if est not in ESTIMATORS:
            parser.error(f"unknown estimator {est}")
    args.contrast = [_kv(c, ":") for c in args.contrast]
    if args.no_reeval and (args.cross or args.contrast or args.estimator):
        parser.error("--no-reeval cannot change the estimator or pair configs")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.devices > 1:
        flag = f"--xla_force_host_platform_device_count={args.devices}"
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flag}".strip()
    import jax
    import jax.numpy as jnp
    from cascade_b0 import CAUSES
    from env import DroneEnv
    from residual import ResidualEnv

    f32 = jnp.float32
    names = args.names.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    pairs = list(args.pairs)
    tags = [f"{dr:g},{throw:g}" for dr, throw in pairs]
    key = jax.random.PRNGKey(args.seed)
    raw_keys = jax.random.split(jax.random.split(key)[0], args.episodes)
    os.makedirs(args.out, exist_ok=True)

    @jax.jit
    def summarize(ret, length, success, terminated):  # as ppo.make_evaluate
        return {
            "mean_return": ret.mean(),
            "success_rate": success.astype(f32).mean(),
            "terminated_rate": terminated.astype(f32).mean(),
            "mean_length": length.astype(f32).mean(),
        }

    plants: dict[str, DroneEnv] = {}

    def plant(estimator: str) -> DroneEnv:
        if estimator not in plants:
            plants[estimator] = DroneEnv(estimator=estimator)
        return plants[estimator]

    per_episode: dict[str, list[dict]] = {}
    vs = f"vs_{args.reference}" if args.reference else None
    baselines = []
    if args.baselines:
        outs = run_baselines(plant(args.baseline_estimator), args.baselines, raw_keys, pairs)
        per_episode.update(outs)
        for b in args.baselines:
            controller, gain_set, label = BASELINES[b]
            evals = []
            for (dr, throw), ep in zip(pairs, outs[b]):
                ev = {"dr": dr, "throw": throw, **episode_stats(ep, summarize, CAUSES)}
                if vs and b != args.reference:
                    i = len(evals)
                    ev[vs] = paired(ep["success"], outs[args.reference][i]["success"])
                evals.append(ev)
            row = {"id": b, "label": label, "controller": controller, "gain_set": gain_set}
            row.update(estimator=args.baseline_estimator, final_eval=evals)
            baselines.append(row)
            print(json.dumps(row), flush=True)

    # (checkpoint name, config name, estimator override)
    groups = [(n, n, args.estimator) for n in names]
    groups += [(n, f"{n}@{est}", est) for n, est in args.cross]
    rollouts: dict[tuple, tuple] = {}
    rows, curves, missing = [], {}, []
    for name, group, override in groups:
        for seed in seeds:
            ckpt = f"{name}_s{seed}"
            run = ckpt if group == name else f"{ckpt}@{override}"
            d = os.path.join(args.run_dir.get(name, args.runs), ckpt)
            try:
                with open(os.path.join(d, "done")) as f:
                    status = int(f.read().strip())
            except FileNotFoundError:
                missing.append(run)
                print(json.dumps({"run": run, "missing": True}))
                continue
            with open(os.path.join(d, "config.json")) as f:
                config = json.load(f)
            log = read_jsonl(os.path.join(d, "log.jsonl"))
            res = config.get("residual")
            eval_est = override or config["estimator"]
            row = {"run": run, "name": group, "checkpoint": ckpt, "dir": d, "seed": seed}
            row.update(exit_status=status, estimator=config["estimator"], eval_estimator=eval_est)
            row.update(dr=config["dr"], throw=config["throw"])
            row.update(residual=res, iters=config["ppo"]["total_iters"])
            row["in_tol_weight"] = config.get("in_tol_weight")
            row["health"] = health(log, config["ppo"]["epochs"])
            wall = train_wall_s(os.path.join(d, "stdout.log"))
            row["process_wall_s"] = elapsed_s(os.path.join(d, "stderr.log"))
            row["train_wall_s"] = wall
            row["steps_per_s"] = round(log[-1]["env_steps"] / wall) if wall else None
            sps = sorted(r["steps_per_s"] for r in log[1:])
            row["median_iter_steps_per_s"] = sps[len(sps) // 2] if sps else None
            row["peak_rss_gb"] = peak_rss_gb(os.path.join(d, "stderr.log"))

            path = os.path.join(d, "final_eval.json")
            written = None
            if os.path.exists(path):
                with open(path) as f:
                    written = json.load(f)
            if args.no_reeval:
                evals = written["evals"]
                assert [(e["dr"], e["throw"]) for e in evals] == pairs, run
                for e in evals:
                    k = round(e["success_rate"] * written["episodes"])
                    e.update(episodes=written["episodes"], successes=k)
                    e["wilson95"] = wilson(k, written["episodes"])
            else:
                obs_clip = config["ppo"]["obs_clip"]
                ek = (eval_est, res and res["base"], res and res["scale"], obs_clip)
                if ek not in rollouts:
                    env = plant(eval_est)
                    if res:
                        env = ResidualEnv(env, res["base"], res["scale"])
                    rollouts[ek] = (env, make_rollout(env, args.episodes, obs_clip, args.devices))
                env, rollout = rollouts[ek]
                if res:
                    row["residual_gains_match"] = list(env.gains) == list(res["gains"])
                actor, stat, clip, exact = load_policy(os.path.join(d, "policy.npz"))
                assert clip == obs_clip
                row["obs_std_exact"] = exact
                evals, per_episode[run] = [], []
                for i, (dr, throw) in enumerate(pairs):
                    ep = rollout(actor, stat, key, dr, throw)
                    per_episode[run].append(ep)
                    ev = {"dr": dr, "throw": throw, **episode_stats(ep, summarize, CAUSES)}
                    if vs:
                        ev[vs] = paired(ep["success"], per_episode[args.reference][i]["success"])
                    evals.append(ev)
                if eval_est != config["estimator"]:
                    row["final_eval_source"] = f"eval_runs ({eval_est} observations)"
                elif written is None:
                    result = {
                        "episodes": args.episodes,
                        "seed": args.seed,
                        "deterministic": True,
                        "evals": [{k: e[k] for k in ("dr", "throw", *METRICS)} for e in evals],
                    }
                    with open(path, "w") as f:
                        json.dump(result, f, indent=2)
                    row["final_eval_source"] = "eval_runs"
                else:
                    same = written["episodes"] == args.episodes and written["seed"] == args.seed
                    same &= [(e["dr"], e["throw"]) for e in written["evals"]] == pairs
                    row["final_eval_source"] = "train.py"
                    row["reeval_max_abs_diff"] = (
                        max(
                            abs(a[k] - b[k])
                            for a, b in zip(written["evals"], evals)
                            for k in METRICS
                        )
                        if same
                        else None
                    )
            row["final_eval"] = evals
            rows.append(row)
            if group == name:
                fields = ("iter", "dr_scale", "throw_scale", "eval_success_rate")
                fields += ("eval_mean_return", "eval_terminated_rate", "eval_mean_length")
                evals_log = [r for r in log if "eval_success_rate" in r]
                curves[run] = [{k: r.get(k) for k in fields} for r in evals_log]
            print(json.dumps(row), flush=True)

    configs = []
    for group_name in dict.fromkeys(g for _, g, _ in groups):
        group = [r for r in rows if r["name"] == group_name]
        if not group:
            continue
        entry = {"name": group_name, "seeds": [r["seed"] for r in group]}
        entry.update(estimator=group[0]["estimator"], eval_estimator=group[0]["eval_estimator"])
        entry["iters"] = group[0]["iters"]
        for i, t in enumerate(tags):
            for k in METRICS:
                entry[f"{k}@{t}"] = mean_std([r["final_eval"][i][k] for r in group])
            k_sum = sum(r["final_eval"][i]["successes"] for r in group)
            n_sum = sum(r["final_eval"][i]["episodes"] for r in group)
            entry[f"pooled_wilson95@{t}"] = wilson(k_sum, n_sum)
            if vs:
                diffs = [r["final_eval"][i][vs]["diff"] for r in group]
                entry[f"{vs}_diff@{t}"] = mean_std(diffs)
                entry[f"{vs}_diff_ci95@{t}"] = seed_ci(diffs)
        for k in ("train_wall_s", "process_wall_s", "steps_per_s"):
            vals = [r[k] for r in group if r[k] is not None and r["run"] == r["checkpoint"]]
            if vals:
                entry[k] = [None if x is None else round(x, 1) for x in mean_std(vals)]
        configs.append(entry)
    contrasts = [contrast(rows, per_episode, a, b, tags) for a, b in args.contrast]
    notes = dict(n.split("=", 1) for n in args.note)
    summary = {
        "pairs": pairs,
        "episodes": args.episodes,
        "seed": args.seed,
        "deterministic": True,
        "causes": list(CAUSES),
        "reference": args.reference or None,
        "baselines": baselines,
        "runs": rows,
        "configs": configs,
        "contrasts": contrasts,
        "missing": missing,
        "notes": notes,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "curves.json"), "w") as f:
        json.dump(curves, f, indent=2)
    if per_episode and not args.no_reeval:
        arrays = {
            f"{label}|{t}|{k}": v
            for label, eps in per_episode.items()
            for t, ep in zip(tags, eps)
            for k, v in ep.items()
        }
        np.savez_compressed(os.path.join(args.out, "heldout_episodes.npz"), **arrays)
    text = markdown(summary)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    plot(curves, rows, baselines, names, seeds, pairs, os.path.join(args.out, "curves.png"))


def _md_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _causes(counts: dict | None) -> str:
    if counts is None:
        return "n/a"
    return ", ".join(f"{c} {v}" for c, v in counts.items() if v) or "-"


def _pm(ms: list, digits: int, signed: bool = False) -> str:
    fmt = f"+.{digits}f" if signed else f".{digits}f"
    if ms[1] is None:
        return f"{ms[0]:{fmt}} (1 seed)"
    return f"{ms[0]:{fmt}} +- {ms[1]:.{digits}f}"


def _t95(ms: list, n: int) -> list | None:
    """t 95% interval of a mean over n seeds from [mean, SD]; None below 2 seeds."""
    if n < 2 or ms[1] is None:
        return None
    half = T95.get(n - 1, Z95) * ms[1] / math.sqrt(n)
    return [ms[0] - half, ms[0] + half]


def _ci(ci: list | None, digits: int = 4) -> str:
    return "-" if ci is None else f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _est(row: dict) -> str:
    if "controller" in row:  # baseline
        return "truth-fed" if row["controller"] == "b1" else row["estimator"]
    train, obs = row["estimator"], row.get("eval_estimator", row["estimator"])
    return train if obs == train else f"{train} -> {obs} obs"


def markdown(s: dict) -> str:
    vs = f"vs_{s['reference']}" if s["reference"] else None
    ref_label = BASELINES[s["reference"]][2] if vs else None
    title = f"# Held-out evaluation: {s['episodes']} episodes per scale (seed {s['seed']}), "
    title += "deterministic, the same episodes for every row"
    legend = "Success is count/N with a Wilson 95% interval (episode noise only). Causes count "
    legend += "terminated episodes by their first cause. Estimator is the training estimator, "
    legend += "then the observing one when they differ (cross-estimator rows, NAME@EST)."
    if vs:
        legend += f" Paired is the success difference against {ref_label} on the same episodes, "
        legend += "+- its standard error, with (row-only / reference-only) successes."
    lines = [title, "", legend]
    tags = [f"{dr:g},{throw:g}" for dr, throw in s["pairs"]]
    for i, t in enumerate(tags):
        head = ["controller", "estimator", "success", "Wilson 95%", "mean return", "terminated"]
        head += ["causes"] + ([f"paired vs {ref_label}"] if vs else [])
        lines += ["", f"## (dr, throw) = ({t})", "", _md_row(head), _md_row(["---"] * len(head))]
        entries = [(b["label"], _est(b), b["final_eval"][i]) for b in s["baselines"]]
        entries += [(r["run"], _est(r), r["final_eval"][i]) for r in s["runs"]]
        for label, est, ev in entries:
            lo, hi = ev["wilson95"]
            count = f"{ev['success_rate']:.4f} ({ev['successes']}/{ev.get('episodes')})"
            cells = [label, est, count, f"[{lo:.4f}, {hi:.4f}]", f"{ev['mean_return']:.1f}"]
            cells += [f"{ev['terminated_rate']:.4f}", _causes(ev.get("causes"))]
            if vs:
                p = ev.get(vs)
                cells.append(
                    f"{p['diff']:+.4f} +- {p['se']:.4f} ({p['only']}/{p['ref_only']})"
                    if p
                    else "reference"
                )
            lines.append(_md_row(cells))

    lines += [
        "",
        "## Per config: mean +- SD (sample std, ddof 1) over training seeds",
        "",
        (
            "+- is the SD over seeds, not an interval. The t 95% intervals are over seeds: the "
            "uncertainty of the config mean. Pooled Wilson counts all seeds' episodes together, "
            "so it reflects episode noise only and understates the uncertainty of the mean."
        ),
        "",
    ]
    head = ["config", "estimator", "iters", "seeds", "(dr, throw)", "success mean +- SD"]
    head += ["success t 95% over seeds", "pooled Wilson 95% (episode noise only)"]
    head += ["mean return", "terminated"]
    head += [f"paired diff vs {ref_label}", "t 95% over seeds"] if vs else []
    lines += [_md_row(head), _md_row(["---"] * len(head))]
    for b in s["baselines"]:
        for i, t in enumerate(tags):
            ev = b["final_eval"][i]
            lo, hi = ev["wilson95"]
            cells = [b["label"], _est(b), "-", "-", f"({t})", f"{ev['success_rate']:.4f}", "-"]
            cells += [f"[{lo:.4f}, {hi:.4f}]", f"{ev['mean_return']:.1f}"]
            cells += [f"{ev['terminated_rate']:.4f}"]
            if vs:
                p = ev.get(vs)
                cells += [f"{p['diff']:+.4f}" if p else "reference", "-"]
            lines.append(_md_row(cells))
    for e in s["configs"]:
        est = e.get("estimator", "")
        if e.get("eval_estimator", est) != est:
            est = f"{est} -> {e['eval_estimator']} obs"
        for t in tags:
            lo, hi = e.get(f"pooled_wilson95@{t}", [math.nan, math.nan])
            cells = [e["name"], est, e.get("iters", ""), ",".join(map(str, e["seeds"])), f"({t})"]
            sr = e[f"success_rate@{t}"]
            ci = _t95(sr, len(e["seeds"]))
            ci_cell = "-" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            cells += [_pm(sr, 4), ci_cell, f"[{lo:.4f}, {hi:.4f}]"]
            cells += [_pm(e[f"mean_return@{t}"], 1), _pm(e[f"terminated_rate@{t}"], 4)]
            if vs:
                cells += [_pm(e[f"{vs}_diff@{t}"], 4, True), _ci(e.get(f"{vs}_diff_ci95@{t}"))]
            lines.append(_md_row(cells))
    if s["notes"]:
        lines.append("")
    lines += [f"- {name}: {note}" for name, note in s["notes"].items()]

    contrasts = [c for c in s.get("contrasts", []) if c["seeds"]]
    if contrasts:
        lines += ["", "## Seed-paired contrasts: config A minus config B on the same episodes", ""]
        lines.append(
            "Per seed: paired success diff +- its standard error. Mean +- std and the t 95% "
            "interval are over seeds; terminated is the terminated-rate diff."
        )
        lines.append("")
        head = ["A - B", "seeds", "(dr, throw)", "success diff", "t 95% over seeds"]
        head += ["per seed", "terminated diff"]
        lines += [_md_row(head), _md_row(["---"] * len(head))]
        for c in contrasts:
            for t in tags:
                per = ", ".join(
                    f"s{sd} {p['diff']:+.4f} +- {p['se']:.4f}"
                    for sd, p in zip(c["seeds"], c[f"per_seed@{t}"])
                )
                cells = [f"{c['a']} - {c['b']}", ",".join(map(str, c["seeds"])), f"({t})"]
                cells += [_pm(c[f"diff@{t}"], 4, True), _ci(c[f"diff_ci95@{t}"]), per]
                cells.append(_pm(c[f"terminated_diff@{t}"], 4, True))
                lines.append(_md_row(cells))
    if s.get("missing"):
        lines += ["", "Not found (no done marker): " + ", ".join(s["missing"])]

    lines += ["", "## Training health and cost", ""]
    head = ["run", "iters", "actor frozen", "KL stops", "skipped updates", "nonfinite steps"]
    head += ["replay envs", "max KL", "train wall s", "process wall s", "steps/s", "peak RSS GB"]
    lines += [_md_row(head), _md_row(["---"] * len(head))]
    for r in s["runs"]:
        if r["run"] != r.get("checkpoint", r["run"]):
            continue
        h = r["health"]
        at = f" (iters {', '.join(map(str, h['kl_stop_at']))})" if h["kl_stop_iters"] else ""
        cells = [r["run"], h["iters"], h["actor_frozen_iters"], f"{h['kl_stop_iters']}{at}"]
        cells += [h["skipped_updates"], h["nonfinite_steps"], h["replay_envs"]]
        cells += [h["max_approx_kl"], r["train_wall_s"], r["process_wall_s"], r["steps_per_s"]]
        cells += [r["peak_rss_gb"]]
        lines.append(_md_row(cells))
    return "\n".join(lines) + "\n"


def plot(curves, rows, baselines, names, seeds, pairs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = [n for n in names if any(f"{n}_s{s}" in curves for s in seeds)] or names
    width = 5.5 if len(names) <= 3 else 4.6
    fig, axes = plt.subplots(
        2, len(names), figsize=(width * len(names), 8), squeeze=False, sharex="col"
    )
    final = {r["run"]: r["final_eval"] for r in rows}
    markers = ("*", "D", "s", "^", "v")
    styles = ("--", ":", "-.", (0, (6, 2)))
    for j, name in enumerate(names):
        top, bottom = axes[0, j], axes[1, j]
        for seed in seeds:
            run = f"{name}_s{seed}"
            if run not in curves:
                continue
            c = curves[run]
            it = [p["iter"] + 1 for p in c]
            (line,) = top.plot(
                it,
                [p["eval_success_rate"] for p in c],
                marker="o",
                ms=2.5,
                lw=1,
                label=f"seed {seed}",
            )
            colour = line.get_color()
            bottom.plot(
                it, [p["eval_mean_return"] for p in c], marker="o", ms=2.5, lw=1, color=colour
            )
            x = it[-1] * 1.03
            for ev, mk in zip(final[run], markers):
                top.plot(x, ev["success_rate"], mk, color=colour, ms=7, alpha=0.8)
                bottom.plot(x, ev["mean_return"], mk, color=colour, ms=7, alpha=0.8)
        dr, throw = pairs[0]
        for b, ls in zip(baselines, styles):
            ev = b["final_eval"][0]
            label = f"{b['label']} held-out ({dr:g},{throw:g})"
            top.axhline(ev["success_rate"], color="0.3", ls=ls, lw=1.2, label=label)
            bottom.axhline(ev["mean_return"], color="0.3", ls=ls, lw=1.2)
        top.set_title(name)
        top.set_ylabel("eval success rate")
        bottom.set_ylabel("eval mean return")
        bottom.set_xlabel("iteration (in-training eval: 1024 episodes at the current scales)")
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for (dr, throw), mk in zip(pairs, markers):
        handles.append(Line2D([], [], marker=mk, color="k", ls="none", ms=6))
        labels.append(f"final held-out ({dr:g},{throw:g})")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
