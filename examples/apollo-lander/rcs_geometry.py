"""Apollo LM RCS visualization geometry.

The KDL thruster ``direction`` vectors are exhaust directions. The reaction
force on the vehicle is therefore the opposite vector, and ``position x force``
defines the body-axis torque sign each visible jet contributes.

Offsets are schematic Z-up (ENU), matching ``apollo-lander.kdl`` after the
Y-up GLB is lifted with ``y_up_to_schematic``: mesh (x, y, z) → schematic
(x, -z, y).
"""

from __future__ import annotations

# (position, exhaust direction) — same order and values as apollo-lander.kdl.
RCS_THRUSTERS = (
    ((1.089, 1.506, 0.870), (0.0, 1.0, 0.0)),
    ((1.388, 1.361, 0.874), (1.0, 0.0, 0.0)),
    ((-1.180, 1.369, 0.527), (0.0, 0.0, -1.0)),
    ((-1.360, 1.361, 0.855), (-1.0, 0.0, 0.0)),
    ((-1.062, 1.507, 0.858), (0.0, 1.0, 0.0)),
    ((-1.249, -1.230, 1.127), (0.0, 0.0, 1.0)),
    ((-1.429, -1.280, 0.916), (-1.0, 0.0, 0.0)),
    ((-1.232, -1.428, 0.912), (0.0, -1.0, 0.0)),
    ((1.297, -1.243, 1.127), (0.0, 0.0, 1.0)),
    ((1.296, -1.243, 0.694), (0.0, 0.0, -1.0)),
    ((1.484, -1.260, 0.905), (1.0, 0.0, 0.0)),
    ((1.314, -1.442, 0.908), (0.0, -1.0, 0.0)),
    ((1.207, 1.369, 1.201), (0.0, 0.0, 1.0)),
    ((1.207, 1.369, 0.527), (0.0, 0.0, -1.0)),
    ((-1.249, -1.230, 0.694), (0.0, 0.0, -1.0)),
    ((-1.180, 1.369, 1.200), (0.0, 0.0, 1.0)),
)


def _torque_axis_sign(
    position: tuple[float, float, float], direction: tuple[float, float, float]
) -> tuple[int, float]:
    px, py, pz = position
    dx, dy, dz = direction
    torque = (
        py * (-dz) - pz * (-dy),
        pz * (-dx) - px * (-dz),
        px * (-dy) - py * (-dx),
    )
    axis = max(range(3), key=lambda i: abs(torque[i]))
    return axis, 1.0 if torque[axis] >= 0.0 else -1.0


RCS_THRUSTER_AXIS = tuple(_torque_axis_sign(p, d)[0] for p, d in RCS_THRUSTERS)
RCS_THRUSTER_SIGN = tuple(_torque_axis_sign(p, d)[1] for p, d in RCS_THRUSTERS)
RCS_THRUSTER_VIZ_MIN_RAW_LEVEL = 0.001

if len(RCS_THRUSTER_AXIS) != len(RCS_THRUSTER_SIGN):
    raise RuntimeError("RCS thruster axis/sign tables must have the same length")
if len(RCS_THRUSTERS) != 16:
    raise RuntimeError("RCS thruster table must match the 16 KDL nozzles")


def rcs_thruster_levels(torque_norm: tuple[float, float, float]) -> tuple[float, ...]:
    """Return per-nozzle visualization levels from normalized body torque."""

    levels = []
    for axis, sign in zip(RCS_THRUSTER_AXIS, RCS_THRUSTER_SIGN):
        raw_level = max(0.0, torque_norm[axis] * sign)
        if raw_level <= RCS_THRUSTER_VIZ_MIN_RAW_LEVEL:
            levels.append(0.0)
        else:
            levels.append(raw_level**0.5)
    return tuple(levels)
