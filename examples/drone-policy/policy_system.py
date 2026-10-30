"""A drone-policy controller as an in-graph el.system, and worlds of thrown drones flying it.

The controller is a policy directory (policy.npz plus its config.json, such as the shipped
policy/ or a training run), a policy.npz path, or a baseline id (b0-nominal, b0-dr, b1-nominal,
b1-dr). system(controller) goes first in the 300 Hz chain (policy_world.system's `policy`) and
acts on the first tick of every TICKS_PER_STEP, counted by a float phase: it reads the
policy_obs the previous tick ended with and holds policy_action in between, which is the env's
timing (env.step writes the action before the first of its ticks). The forward is the float64
deploy path (nn.policy_action, residual.residual_action or the cascade) with jnp constants.

Cranelift: the hold selects with a predicate broadcast to the action shape. A rank-0 predicate
(jnp.where(scalar, vec, vec)) is miscompiled when it lands in a pointer-ABI function, which
happened in every truth-estimator world: lanes 1..3 of the action read stack padding.
"""

import json
import os
import typing as ty
from dataclasses import dataclass, field

import cascade_b0
import elodin as el
import jax
import jax.numpy as jnp
import nn
import numpy as np
import policy_io
import policy_world
from dr import TARGET, dr_sample, throw_sample
from env import TICKS_PER_STEP

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED = os.path.join(HERE, "policy")  # policy.npz + config.json of the shipped policy
CONFIG_NAMES = ("config.json", "policy_config.json")  # train config next to policy.npz

BASELINES = {  # id -> (cascade_b0 controller, gain set)
    "b0-nominal": ("b0", "nominal"),
    "b0-dr": ("b0", "dr"),
    "b1-nominal": ("b1", "nominal"),
    "b1-dr": ("b1", "dr"),
}

PolicyPhase = ty.Annotated[
    jax.Array, el.Component("policy_phase", el.ComponentType.F64, metadata={"priority": 300})
]


@dataclass
class PolicyState(el.Archetype):
    policy_phase: PolicyPhase = field(default_factory=lambda: jnp.float64(0.0))


def load(controller: str) -> dict:
    """Baseline id -> {"baseline": id}; policy.npz path (or its directory) -> its arrays, plus
    the training estimator from the config.json (or policy_config.json) next to it."""
    if controller in BASELINES:
        return {"baseline": controller}
    path = os.path.join(controller, "policy.npz") if os.path.isdir(controller) else controller
    policy = dict(np.load(path))
    for name in CONFIG_NAMES:
        config = os.path.join(os.path.dirname(path), name)
        if os.path.exists(config):
            with open(config) as f:
                policy["estimator"] = json.load(f)["estimator"]
            break
    return policy


def forward(policy: ty.Mapping) -> tuple[ty.Callable, bool]:
    """(fn(obs, critic) -> action in [-1, 1], needs_critic) in the obs dtype (float64 in-graph)."""
    if "baseline" in policy:
        kind, gain_set = BASELINES[policy["baseline"]]
        gains = cascade_b0.GAIN_SETS[gain_set]
        if kind == "b1":
            return (lambda obs, critic: cascade_b0.b1_action(obs, critic, gains)), True
        return (lambda obs, critic: cascade_b0.b0_action(obs, gains)), False
    arrays = {k: v for k, v in policy.items() if k != "estimator"}
    if "residual_base" in arrays:
        import residual

        return (lambda obs, critic: residual.residual_action(arrays, obs, jnp)), False
    return (lambda obs, critic: nn.policy_action(arrays, obs, jnp)), False


def _hold(phase, a, new):
    act = jnp.broadcast_to(phase < 0.5, a.shape)  # see the module docstring (Cranelift)
    phase = jnp.where(phase > TICKS_PER_STEP - 1.5, 0.0, phase + 1.0)
    return jnp.where(act, new, a), phase


def system(controller: str | ty.Mapping) -> el.System:
    policy = load(controller) if isinstance(controller, str) else controller
    fn, needs_critic = forward(policy)
    if needs_critic:

        @el.map
        def policy_b1(
            obs: policy_io.PolicyObs,
            critic: policy_io.CriticObs,
            a: policy_io.PolicyAction,
            phase: PolicyPhase,
        ) -> tuple[policy_io.PolicyAction, PolicyPhase]:
            return _hold(phase, a, fn(obs, critic))

        return policy_b1

    @el.map
    def policy_step(
        obs: policy_io.PolicyObs, a: policy_io.PolicyAction, phase: PolicyPhase
    ) -> tuple[policy_io.PolicyAction, PolicyPhase]:
        return _hold(phase, a, fn(obs, None))

    return policy_step


def episode_init(key, dr_scale, throw_scale, estimator: str = "mekf6") -> dict[str, np.ndarray]:
    """drone_init values of the episode env.reset(key, dr_scale, throw_scale) starts."""
    k_dr, k_throw, k_noise = jax.random.split(key, 3)
    dr = dr_sample(k_dr, dr_scale)
    throw = throw_sample(k_throw, throw_scale)
    init = policy_world.drone_init(dr, throw, k_noise, estimator, jnp.array(TARGET))
    return {k: np.asarray(v) for k, v in init.items()}


def entity_names(n: int) -> list[str]:
    return ["drone"] if n == 1 else [f"drone_{i}" for i in range(n)]


def build_world(
    controller: str | ty.Mapping,
    n: int = 1,
    estimator: str | None = None,
    keys=None,
    seed: int = 0,
    dr_scale: float = 1.0,
    throw_scale: float = 1.0,
    schematic: bool = False,
) -> tuple[el.World, el.System]:
    """World of n drones thrown from the episodes of `keys` (default split(PRNGKey(seed), n))
    with the controller in graph; estimator defaults to the policy's training estimator."""
    policy = load(controller) if isinstance(controller, str) else controller
    estimator = estimator or policy.get("estimator", "mekf6")
    if keys is None:
        keys = jax.random.split(jax.random.PRNGKey(seed), n)
    inits = [episode_init(k, dr_scale, throw_scale, estimator) for k in keys]
    w, _ = policy_world.world(n, estimator, inits, extra=[PolicyState()])
    if schematic:
        w.schematic(build_schematic(n), "drone-policy.kdl")
    return w, policy_world.system(estimator, policy=system(policy))


def build_schematic(n: int):
    from config import Config
    from elodin import ui

    names = entity_names(n)
    target = ",".join(f"{v:g}" for v in TARGET)
    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Viewport",
                    pos=f"(0,0,0,1, {TARGET[0] + 6:g},{TARGET[1] - 6:g},{TARGET[2] + 4:g})",
                    look_at=f"(0,0,0,1, {target})",
                    show_grid=True,
                    active=True,
                ),
                ui.vsplit(
                    ui.graph(", ".join(f"{e}.world_pos" for e in names), name="world_pos"),
                    ui.graph(", ".join(f"{e}.policy_action" for e in names), name="action"),
                    share=0.4,
                ),
                name="Viewport",
            ),
        ),
        *[ui.object_3d(f"{e}.world_pos", mesh=ui.glb(Config.GLOBAL.drone_glb)) for e in names],
        theme=ui.theme(mode="dark", scheme="default"),
    )
