"""Apollo LM RCS visualization geometry.

The KDL thruster ``direction`` vectors are exhaust directions. The reaction
force on the vehicle is therefore the opposite vector, and ``position x force``
defines the body-axis torque sign each visible jet contributes.
"""

from __future__ import annotations

RCS_THRUSTER_AXIS = (1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0)
RCS_THRUSTER_SIGN = (
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
)

if len(RCS_THRUSTER_AXIS) != len(RCS_THRUSTER_SIGN):
    raise RuntimeError("RCS thruster axis/sign tables must have the same length")


def rcs_thruster_levels(torque_norm: tuple[float, float, float]) -> tuple[float, ...]:
    """Return per-nozzle visualization levels from normalized body torque."""

    return tuple(
        max(0.0, torque_norm[axis] * sign)
        for axis, sign in zip(RCS_THRUSTER_AXIS, RCS_THRUSTER_SIGN)
    )
