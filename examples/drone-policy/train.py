"""Train the drone-policy PPO agent on DroneEnv.

The shipped policy (policy/; 40 min of training with two such runs sharing 32 cores):

    python examples/drone-policy/train.py --estimator mekf6 --dr curriculum --throw curriculum \
        --curriculum-iters 300 --iters 8000 --num-envs 1024 --steps 32 --devices 8 \
        --eval-every 100 --eval-envs 1024 --final-eval 1,1 0.5,1 0,1 --final-eval-envs 4096 \
        --final-eval-seed 20011 --lr 1.5e-4 --lr-end 3e-5 --ent-coef 1e-3 \
        --init-std 0.1,0.1,0.1,0.2 --in-tol-weight 0.25 --critic-head-gain 0.01 \
        --actor-warmup 20 --seed 2 --out runs/policy_s2

Other forms:

    python examples/drone-policy/train.py --estimator mekf6 --seed 0 --devices 8 --out runs/main_s0
    python examples/drone-policy/train.py --nominal --seed 0 --out runs/a1_s0
    python examples/drone-policy/train.py --residual dr --residual-scale 0.3 --out runs/res_s0
    python examples/drone-policy/train.py --quick --out runs/quick

--dr schedules the plant, sensor and latency randomization and --throw the start throw; each is
"curriculum" (0 -> 1 over --curriculum-iters) or "0". --nominal (the no-DR ablation) is --dr 0
with the throw curriculum. --devices N splits the env batch over N XLA host devices (set before
jax is imported).
--residual BASE trains a residual over the B0 cascade with that gain set (residual.ResidualEnv);
policy.npz then also carries the base controller, and residual.residual_action runs B0 + NN.
Prints one JSON line per iteration (also appended to OUT/log.jsonl) and writes OUT/policy.npz and
OUT/config.json every 100 iterations and at the end. The final policy is then evaluated
deterministically at each --final-eval DR,THROW pair on fresh held-out episodes
(OUT/final_eval.json).
"""

import argparse
import json
import math
import os
import time

CHECKPOINT_EVERY = 100
SCHEDULES = ("curriculum", "0")
FINAL_EVAL_SEED = 10007


class InTolWeight:
    """DroneEnv whose reward pays `weight` instead of env.IN_TOL_BONUS while the drone is inside
    the success tolerances; physics, termination and info are unchanged."""

    def __init__(self, env, weight: float):
        self.env, self.weight = env, weight

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, key, dr_scale, throw_scale=None):
        return self.env.reset(key, dr_scale, throw_scale)

    def step(self, state, action):
        import env as drone_env
        import jax.numpy as jnp

        state, obs, critic, reward, done, info = self.env.step(state, action)
        sim, zero = state.sim, jnp.zeros(self.env.act_dim)
        # finite=True: info["terminated"] already covers a diverged step
        m = drone_env.evaluate(
            sim[self.env.i_pos], sim[self.env.i_vel], state.target, zero, zero, jnp.bool_(True)
        )
        extra = self.weight - drone_env.IN_TOL_BONUS
        bonus = extra * (m.in_window_ok & ~info["terminated"])
        return state, obs, critic, reward + bonus.astype(reward.dtype), done, info


def schedule(kind: str, curriculum_iters: int):
    if kind == "0":
        return lambda it: 0.0
    if curriculum_iters <= 0:
        return lambda it: 1.0
    return lambda it: min(1.0, it / curriculum_iters)


def _clean(value):
    value = float(value)
    return float(f"{value:.6g}") if math.isfinite(value) else None


def train(
    env,
    cfg,
    seed=0,
    dr_scale_fn=None,
    out=None,
    meta=None,
    log=print,
    throw_scale_fn=None,
    eval_every=0,
    eval_envs=1024,
    eval_env=None,
):
    """Run cfg.total_iters PPO iterations on env. throw_scale_fn defaults to dr_scale_fn. With
    eval_every > 0, every eval_every iterations also runs eval_envs deterministic episodes at the
    current scales on eval_env (default env; eval_* fields). -> (TrainState, list of
    per-iteration records)."""
    import jax
    import jax.numpy as jnp
    import ppo

    dr_scale_fn = dr_scale_fn or (lambda it: 1.0)
    throw_scale_fn = throw_scale_fn or dr_scale_fn
    meta = {
        "seed": seed,
        "obs_dim": env.obs_dim,
        "critic_dim": env.critic_dim,
        "act_dim": env.act_dim,
        "episode_steps": env.episode_steps,
        **(meta or {}),
    }
    key = jax.random.PRNGKey(seed)
    ts = ppo.init(env, cfg, key, jnp.float32(dr_scale_fn(0)), jnp.float32(throw_scale_fn(0)))
    train_iter = ppo.make_train_iter(env, cfg)
    evaluate = None
    if eval_every > 0:
        evaluate = ppo.make_evaluate(
            eval_env or env, eval_envs, obs_clip=cfg.obs_clip, devices=cfg.devices
        )
    extra_arrays = env.policy_arrays() if hasattr(env, "policy_arrays") else None
    if out is not None:
        os.makedirs(out, exist_ok=True)
    steps_per_iter = cfg.num_envs * cfg.num_steps
    records = []
    for it in range(cfg.total_iters):
        dr_scale, throw_scale = float(dr_scale_fn(it)), float(throw_scale_fn(it))
        start = time.perf_counter()
        ts, metrics = train_iter(ts, jnp.float32(dr_scale), jnp.float32(throw_scale))
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        record = {
            "iter": it,
            "env_steps": (it + 1) * steps_per_iter,
            "mean_return": _clean(metrics["mean_return"]),
            "mean_length": _clean(metrics["mean_length"]),
            "finished_episodes": int(metrics["finished_episodes"]),
            "success_rate": _clean(metrics["success_rate"]),
            "terminated_rate": _clean(metrics["terminated_rate"]),
            "approx_kl": _clean(metrics["approx_kl"]),
            "entropy": _clean(metrics["entropy"]),
            "value_loss": _clean(metrics["value_loss"]),
            "lr": _clean(metrics["lr"]),
            "actor_lr": _clean(metrics["actor_lr"]),
            "steps_per_s": round(steps_per_iter / elapsed, 1),
            "dr_scale": dr_scale,
            "throw_scale": throw_scale,
            "policy_loss": _clean(metrics["policy_loss"]),
            "clip_frac": _clean(metrics["clip_frac"]),
            "grad_norm": _clean(metrics["grad_norm"]),
            "skipped_updates": int(metrics["skipped_updates"]),
            "epochs": int(metrics["epochs"]),
            "explained_var": _clean(metrics["explained_var"]),
            "ret_std": _clean(metrics["ret_std"]),
            "nonfinite_steps": int(metrics["nonfinite_steps"]),
            "replay_envs": int(metrics["replay_envs"]),
        }
        if evaluate is not None and (it + 1) % eval_every == 0:
            k_eval = jax.random.fold_in(jax.random.PRNGKey(seed + 1), it)
            scales = (jnp.float32(dr_scale), jnp.float32(throw_scale))
            ev = jax.device_get(evaluate(ts.params["actor"], ts.obs_rms, k_eval, *scales))
            record.update({f"eval_{k}": _clean(v) for k, v in ev.items()})
        records.append(record)
        line = json.dumps(record)
        if log is not None:
            log(line)
        if out is None:
            continue
        with open(os.path.join(out, "log.jsonl"), "a" if it else "w") as f:
            f.write(line + "\n")
        if (it + 1) % CHECKPOINT_EVERY == 0 or it + 1 == cfg.total_iters:
            info = {"iter": it + 1, "env_steps": record["env_steps"]}
            info.update(dr_scale=dr_scale, throw_scale=throw_scale)
            ppo.save_policy(out, ts, cfg, {**meta, **info}, extra_arrays)
    return ts, records


def final_eval(env, ts, cfg, pairs, num_envs=2048, seed=FINAL_EVAL_SEED):
    """Deterministic eval of the actor on num_envs fresh episodes per (dr, throw) pair; every pair
    uses the same reset keys. -> list of records."""
    import jax
    import jax.numpy as jnp
    import ppo

    evaluate = ppo.make_evaluate(env, num_envs, obs_clip=cfg.obs_clip, devices=cfg.devices)
    key = jax.random.PRNGKey(seed)
    rows = []
    for dr, throw in pairs:
        scales = (jnp.float32(dr), jnp.float32(throw))
        ev = jax.device_get(evaluate(ts.params["actor"], ts.obs_rms, key, *scales))
        rows.append({"dr": dr, "throw": throw, **{k: _clean(v) for k, v in ev.items()}})
    return rows


def _pair(text: str) -> tuple[float, float]:
    dr, throw = (float(x) for x in text.split(","))
    return dr, throw


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--estimator", choices=("mekf18", "mekf6", "truth"), default="mekf6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, help="PPO iterations (default 1500, --quick 500)")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32, help="policy steps per env per iteration")
    parser.add_argument("--dr", choices=SCHEDULES, default="curriculum", help="plant/sensor DR")
    parser.add_argument("--throw", choices=SCHEDULES, default="curriculum", help="start throw")
    parser.add_argument(
        "--curriculum-iters", type=int, help="scales ramp 0 -> 1 over these (default 300)"
    )
    parser.add_argument(
        "--nominal", action="store_true", help="no-DR ablation: --dr 0, throw curriculum"
    )
    parser.add_argument("--quick", action="store_true", help="500 iterations, 100-iter curriculum")
    parser.add_argument("--devices", type=int, default=1, help="XLA host devices for the rollout")
    parser.add_argument("--eval-every", type=int, default=0, help="deterministic eval period")
    parser.add_argument("--eval-envs", type=int, default=1024)
    parser.add_argument(
        "--final-eval", nargs="*", type=_pair, default=[(1.0, 1.0), (0.0, 1.0)], metavar="DR,THROW"
    )
    parser.add_argument("--final-eval-envs", type=int, default=2048)
    parser.add_argument("--final-eval-seed", type=int, default=FINAL_EVAL_SEED)
    parser.add_argument("--lr", type=float, help="initial learning rate (default 3e-4)")
    parser.add_argument("--lr-end", type=float, help="final learning rate (default 3e-5)")
    parser.add_argument("--ent-coef", type=float, help="entropy bonus (default 1e-3)")
    parser.add_argument(
        "--init-std", help="initial action std: one value or p,q,r,c (default 0.1,0.1,0.1,0.2)"
    )
    parser.add_argument(
        "--in-tol-weight",
        type=float,
        help="training reward bonus inside the success tolerances (default env.IN_TOL_BONUS; "
        "evals keep the env reward)",
    )
    parser.add_argument("--actor-head-gain", type=float, help="actor output init gain (0.01)")
    parser.add_argument("--critic-head-gain", type=float, help="critic output init gain (1.0)")
    parser.add_argument(
        "--actor-warmup", type=int, help="iterations with the actor frozen, critic training (0)"
    )
    parser.add_argument(
        "--residual", choices=("dr", "nominal"), help="train a residual over B0 with this gain set"
    )
    parser.add_argument(
        "--residual-scale", type=float, default=0.3, help="residual action scale (default 0.3)"
    )
    parser.add_argument(
        "--out", help="output directory (default runs/<estimator>_<schedules>_s<seed>)"
    )
    args = parser.parse_args(argv)
    if args.nominal:
        args.dr, args.throw = "0", "curriculum"
    return args


def setup(args: argparse.Namespace, env):
    """-> (PPOConfig, meta, train_env, eval_env) for parsed args on a DroneEnv."""
    import env as drone_env
    import ppo

    iters = args.iters or (500 if args.quick else 1500)
    curriculum = args.curriculum_iters
    if curriculum is None:
        curriculum = 100 if args.quick else 300
    knobs = (
        ("lr", args.lr),
        ("lr_end", args.lr_end),
        ("ent_coef", args.ent_coef),
        ("actor_head_gain", args.actor_head_gain),
        ("critic_head_gain", args.critic_head_gain),
        ("actor_warmup", args.actor_warmup),
    )
    overrides = {k: v for k, v in knobs if v is not None}
    if args.init_std:
        log_std = tuple(math.log(float(s)) for s in args.init_std.split(","))
        overrides["init_log_std"] = log_std[0] if len(log_std) == 1 else log_std
    cfg = ppo.PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.steps,
        total_iters=iters,
        devices=args.devices,
        **overrides,
    )
    in_tol = drone_env.IN_TOL_BONUS if args.in_tol_weight is None else args.in_tol_weight
    meta = {
        "estimator": args.estimator,
        "nominal": args.nominal,
        "dr": args.dr,
        "throw": args.throw,
        "curriculum_iters": curriculum,
        "quick": args.quick,
        "in_tol_weight": in_tol,
        "residual": None,
    }
    train_env, eval_env = env, env
    if in_tol != drone_env.IN_TOL_BONUS:
        train_env = InTolWeight(env, in_tol)
    if args.residual:
        from residual import ResidualEnv

        train_env = ResidualEnv(train_env, args.residual, args.residual_scale)
        eval_env = ResidualEnv(env, args.residual, args.residual_scale)
        meta["residual"] = eval_env.config()
    return cfg, meta, train_env, eval_env


def main(argv=None):
    args = parse_args(argv)
    if args.devices > 1:
        flag = f"--xla_force_host_platform_device_count={args.devices}"
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flag}".strip()
    from env import DroneEnv  # imported after XLA_FLAGS: env.py builds the elodin world

    tag = "nominal" if args.nominal else f"dr{args.dr}_throw{args.throw}".replace("curriculum", "c")
    if args.residual:
        tag += f"_res{args.residual}"
    out = args.out or os.path.join("runs", f"{args.estimator}_{tag}_s{args.seed}")

    env = DroneEnv(estimator=args.estimator)
    cfg, meta, train_env, eval_env = setup(args, env)
    curriculum = meta["curriculum_iters"]
    start = time.perf_counter()
    ts, _ = train(
        train_env,
        cfg,
        args.seed,
        schedule(args.dr, curriculum),
        out,
        meta,
        throw_scale_fn=schedule(args.throw, curriculum),
        eval_every=args.eval_every,
        eval_envs=args.eval_envs,
        eval_env=eval_env,
    )
    wall = round(time.perf_counter() - start, 1)
    if args.final_eval:
        rows = final_eval(
            eval_env, ts, cfg, args.final_eval, args.final_eval_envs, args.final_eval_seed
        )
        result = {
            "episodes": args.final_eval_envs,
            "seed": args.final_eval_seed,
            "deterministic": True,
            "evals": rows,
        }
        with open(os.path.join(out, "final_eval.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({"final_eval": rows}))
    print(json.dumps({"done": out, "wall_s": wall}))


if __name__ == "__main__":
    main()
