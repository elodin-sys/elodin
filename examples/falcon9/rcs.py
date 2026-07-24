"""Cold-gas RCS geometry and allocation (WHITEPAPER 10.3).

Eight nitrogen thrusters in two interstage pods (EST layout — unpublished,
a calibration surface). Body frame: +X nose; pods sit near the interstage
at station RCS_STATION_M, on the +/-Y sides of the hull.

Each pod carries four nozzles: +Z, -Z (pitch pair when fired on opposite
pods, roll pairs when fired same-side) and +X, -X (yaw-ish via lever...).
Layout chosen so pure pitch, yaw, and roll torques are all achievable:

  thruster  pod (+Y/-Y)  fires toward   torque content
  0         +Y           +Z             pitch- / roll-
  1         +Y           -Z             pitch+ / roll+
  2         -Y           +Z             pitch- / roll+
  3         -Y           -Z             pitch+ / roll-
  4         +Y           +Y             yaw+   (lever about Z)
  5         +Y           -Y (inboard)   yaw-
  6         -Y           -Y             yaw-
  7         -Y           +Y (inboard)   yaw+
"""

import jax
import jax.numpy as jnp
from constants import RCS_STATION_M, RCS_THRUST_PER_THRUSTER_N, STAGE1_DIAMETER_M

jax.config.update("jax_enable_x64", True)

N_RCS = 8
_R = STAGE1_DIAMETER_M / 2.0

# Positions (body frame, m) and exhaust directions; thrust force = -exhaust dir.
RCS_POS = jnp.array(
    [
        [RCS_STATION_M, +_R, 0.0],
        [RCS_STATION_M, +_R, 0.0],
        [RCS_STATION_M, -_R, 0.0],
        [RCS_STATION_M, -_R, 0.0],
        [RCS_STATION_M, +_R, 0.0],
        [RCS_STATION_M, +_R, 0.0],
        [RCS_STATION_M, -_R, 0.0],
        [RCS_STATION_M, -_R, 0.0],
    ]
)
RCS_FORCE_DIR = jnp.array(
    [
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
        [0.0, +1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, +1.0, 0.0],
    ]
)


def rcs_wrench(levels, cg_station_m, thrust_n=RCS_THRUST_PER_THRUSTER_N):
    """Total body-frame (force, torque) for thruster levels in [0, 1]."""
    forces = levels[:, None] * thrust_n * RCS_FORCE_DIR
    cg = jnp.array([cg_station_m, 0.0, 0.0])
    torques = jnp.cross(RCS_POS - cg, forces)
    return jnp.sum(forces, axis=0), jnp.sum(torques, axis=0)


def effectiveness_matrix(cg_station_m, thrust_n=RCS_THRUST_PER_THRUSTER_N) -> jnp.ndarray:
    """B (6 x N): stacked [force; torque] per unit thruster level."""
    cols = []
    for i in range(N_RCS):
        levels = jnp.zeros(N_RCS).at[i].set(1.0)
        f, t = rcs_wrench(levels, cg_station_m, thrust_n)
        cols.append(jnp.concatenate([f, t]))
    return jnp.stack(cols, axis=1)


# Thruster groups per torque axis: pairs whose off-axis torques cancel.
# (axis index, group A, group B) — signs are derived from B at build time.
_AXIS_GROUPS = ((0, (0, 3), (1, 2)), (1, (1, 3), (0, 2)), (2, (4, 7), (5, 6)))


def allocate_torque(torque_cmd_body, cg_station_m, thrust_n=RCS_THRUST_PER_THRUSTER_N):
    """Map a desired body torque to thruster levels in [0, 1].

    Deterministic axis-decomposed allocation: each torque axis has two
    dedicated thruster pairs (one per sign) whose off-axis torques cancel.
    Group torque signs are read from the effectiveness matrix, so a layout
    change cannot silently flip an axis. Minimum-impulse behavior comes from
    the valve dynamics downstream.
    """
    b = effectiveness_matrix(cg_station_m, thrust_n)[3:6, :]  # torque rows
    levels = jnp.zeros(N_RCS)
    for axis, group_a, group_b in _AXIS_GROUPS:
        cmd = torque_cmd_body[axis]
        auth_a = b[axis, group_a[0]] + b[axis, group_a[1]]
        auth_b = b[axis, group_b[0]] + b[axis, group_b[1]]
        # Pick the group whose torque sign matches the command's sign.
        use_a = jnp.sign(cmd) == jnp.sign(auth_a)
        auth = jnp.where(use_a, jnp.abs(auth_a), jnp.abs(auth_b))
        lvl = jnp.clip(jnp.abs(cmd) / jnp.maximum(auth, 1e-9), 0.0, 1.0)
        # Minimum-impulse floor: a demand below 2% of authority is not worth
        # opening valves for (cold-gas budget protection).
        active = jnp.abs(cmd) > 0.02 * auth
        for i in group_a:
            levels = levels.at[i].add(jnp.where(active & use_a, lvl, 0.0))
        for i in group_b:
            levels = levels.at[i].add(jnp.where(active & ~use_a, lvl, 0.0))
    return jnp.clip(levels, 0.0, 1.0)
