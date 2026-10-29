"""Dzhanibekov-effect example: 

The Dzhanibekov effect (also known as the tennis-racket effect or the
intermediate axis theorem) states that rotation about an object's
intermediate axis is unstable. That is, if we have principal axes
x, y, and z, with moments Ix, Iy, and Iz such that Ix<Iy<Iz (which
axes we call "x", "y", or "z" doesn't matter), rotations about body
y will be unstable. This manifests in the body periodically 
rotating about the ZX plane by 180 degrees. The effect was famously
"discovered" by Soviet cosmonaut Vladimir Dzhanibekov when he
noticed the effect on a wing nut spun about its intermediate
axis on the Salyut 7 spacecraft in 1985.

The instability is a direct consequence of Euler's equations
of rigid body motion. Demonstrating this effect in Elodin
shows that these equations are implemented properly in
calc_accel.

This specific example tries to recreate a famous video in physics
classrooms around the globe, immortalized on the wikipedia page
for the tennis-racket theorem. You can find the video at
https://en.wikipedia.org/wiki/Tennis_racket_theorem.

In the video, we're looking at some type of hand wrench. In our sim,
the rigid body is a large cylinder with a smaller cylinder attached at a
right angle at the large cylinder's midpoint, forming a "T": the smaller
cylinder (the stem) protrudes from the midpoint of the large cylinder (the
bar). The two cylinders have the same density.

Body axes:
- Y: along the smaller cylinder (the spin axis, intermediate moment)
- Z: along the large cylinder (lowest moment)
- X: completes the right-handed frame (highest moment)

The body spins about body Y at one revolution per 0.75 seconds, with a small
perturbation on the other axes to induce the instability.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp

SIM_TIME_STEP = 1.0 / 240.0
MAX_TICKS = 2880  # 12.0 seconds

# Geometry and material (meters, kg/m^3).
LARGE_LENGTH_M = 0.80  # large cylinder (the bar), along body Z
LARGE_RADIUS_M = 0.065
SMALL_LENGTH_M = 0.40  # smaller cylinder (the stem), along body Y, measured
# from the large cylinder's center to the stem tip (part of it is inside
# the bar). Deliberately shorter than the bar.
SMALL_RADIUS_M = 0.05
DENSITY_KG_M3 = 1000.0

# Spin about body Y: one revolution per 0.75 seconds, plus a small
# perturbation on X and Z to seed the instability.
SPIN_PERIOD_S = 0.75
INITIAL_WORLD_OMEGA = jnp.array(
    [0.08, 2.0 * jnp.pi / SPIN_PERIOD_S, 0.08], dtype=jnp.float64
)


def _solid_cylinder_mass(radius_m: float, length_m: float) -> float:
    return DENSITY_KG_M3 * jnp.pi * radius_m**2 * length_m


def _axial_inertia(mass: float, radius_m: float) -> float:
    """Moment about the cylinder symmetry axis: (1/2) M r^2."""
    return 0.5 * mass * radius_m**2


def _transverse_inertia(mass: float, radius_m: float, length_m: float) -> float:
    """Moment about a transverse axis through its center: M (3 r^2 + L^2) / 12."""
    return mass * (3.0 * radius_m**2 + length_m**2) / 12.0


_large_mass = _solid_cylinder_mass(LARGE_RADIUS_M, LARGE_LENGTH_M)
_small_mass = _solid_cylinder_mass(SMALL_RADIUS_M, SMALL_LENGTH_M)
TOTAL_MASS_KG = _large_mass + _small_mass

# Center of mass on body Y, measured from the large cylinder's center. The
# large cylinder's center sits at _large_center_y (< 0) relative to the COM,
# and the smaller cylinder's center sits at _small_center_y (> 0).
_com_offset_y = _small_mass * (SMALL_LENGTH_M / 2.0) / TOTAL_MASS_KG
_large_center_y = -_com_offset_y
_small_center_y = SMALL_LENGTH_M / 2.0 - _com_offset_y

# Principal moments about the COM. Displacements are along Y only, so the
# parallel-axis term M d^2 adds to Ix and Iz but not to Iy.
INERTIA_DIAG_KG_M2 = jnp.array(
    [
        # Ix (highest): transverse moments of both cylinders + parallel axis
        _transverse_inertia(_large_mass, LARGE_RADIUS_M, LARGE_LENGTH_M)
        + _transverse_inertia(_small_mass, SMALL_RADIUS_M, SMALL_LENGTH_M)
        + _large_mass * _large_center_y**2
        + _small_mass * _small_center_y**2,
        # Iy (intermediate, spin axis): transverse bar + axial stem
        _transverse_inertia(_large_mass, LARGE_RADIUS_M, LARGE_LENGTH_M)
        + _axial_inertia(_small_mass, SMALL_RADIUS_M),
        # Iz (lowest): axial bar + transverse stem + parallel axis
        _axial_inertia(_large_mass, LARGE_RADIUS_M)
        + _transverse_inertia(_small_mass, SMALL_RADIUS_M, SMALL_LENGTH_M)
        + _large_mass * _large_center_y**2
        + _small_mass * _small_center_y**2,
    ],
    dtype=jnp.float64,
)

BodyOmega = ty.Annotated[
    jax.Array,
    el.Component(
        "body_omega",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "wx,wy,wz"},
    ),
]

# World transforms of the two cylinder mesh frames. The schematic cylinder
# primitive is a Bevy cylinder, which runs along Bevy Y. The editor maps
# Bevy Y onto sim Z (ENU up), so an unrotated cylinder renders along body Z:
# - the large cylinder lies along body Z (no extra rotation),
# - the smaller cylinder lies along body Y, so its mesh frame is the body
#   frame rotated by -90 degrees about X (the cylinder's sim-Z axis maps
#   onto body Y).
# Each mesh center is offset from the body origin (the COM) along body Y.
LargePos = ty.Annotated[
    el.SpatialTransform,
    el.Component(
        "large_pos",
        el.ComponentType.SpatialPosF64,
        metadata={"element_names": "qx,qy,qz,qw,x,y,z"},
    ),
]

SmallPos = ty.Annotated[
    el.SpatialTransform,
    el.Component(
        "small_pos",
        el.ComponentType.SpatialPosF64,
        metadata={"element_names": "qx,qy,qz,qw,x,y,z"},
    ),
]

_SMALL_ROT = el.Quaternion.from_axis_angle(
    jnp.array([1.0, 0.0, 0.0]), -jnp.pi / 2.0
)
_LARGE_CENTER_BODY = jnp.array([0.0, _large_center_y, 0.0])
_SMALL_CENTER_BODY = jnp.array([0.0, _small_center_y, 0.0])


@el.dataclass
class HandleDiagnostics(el.Archetype):
    body_omega: BodyOmega = field(
        default_factory=lambda: jnp.zeros(3, dtype=jnp.float64)
    )
    large_pos: LargePos = field(
        default_factory=lambda: el.SpatialTransform(
            angular=el.Quaternion.identity(), linear=_LARGE_CENTER_BODY
        )
    )
    small_pos: SmallPos = field(
        default_factory=lambda: el.SpatialTransform(
            angular=_SMALL_ROT, linear=_SMALL_CENTER_BODY
        )
    )


@el.map
def update_body_omega(pos: el.WorldPos, vel: el.WorldVel) -> BodyOmega:
    return pos.angular().inverse() @ vel.angular()


@el.map
def update_large_pos(pos: el.WorldPos) -> LargePos:
    return el.SpatialTransform(
        angular=pos.angular(),
        linear=pos.linear() + pos.angular() @ _LARGE_CENTER_BODY,
    )


@el.map
def update_small_pos(pos: el.WorldPos) -> SmallPos:
    return el.SpatialTransform(
        angular=pos.angular() * _SMALL_ROT,
        linear=pos.linear() + pos.angular() @ _SMALL_CENTER_BODY,
    )


def build_world() -> tuple[el.World, el.System]:
    w = el.World()
    w.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=el.Quaternion.identity(),
                    linear=jnp.zeros(3),
                ),
                world_vel=el.SpatialMotion(angular=INITIAL_WORLD_OMEGA),
                inertia=el.SpatialInertia(
                    TOTAL_MASS_KG, inertia=INERTIA_DIAG_KG_M2
                ),
            ),
            HandleDiagnostics(body_omega=INITIAL_WORLD_OMEGA),
        ],
        name="T-Handle",
        id="handle",
    )

    w.schematic(
        """
        hsplit {
            viewport name=Viewport pos="handle.world_pos + (0,0,0,0, 2.5, 0.0, 1.2)" look_at="handle.world_pos" hdr=#true show_grid=#true share=0.55
            vsplit share=0.45 {
                graph "handle.body_omega[0],handle.body_omega[1],handle.body_omega[2]" name="Body Omega (rad/s)"
                graph "handle.world_vel[0],handle.world_vel[1],handle.world_vel[2]" name="World Omega (rad/s)"
            }
        }
        object_3d handle.large_pos {
            cylinder radius=0.065 height=0.8 {
                color 33 150 243
            }
        }
        object_3d handle.small_pos {
            cylinder radius=0.05 height=0.4 {
                color 255 152 0
            }
        }
        """,
        "dzhanibekov-effect.kdl",
    )

    sys = (
        el.six_dof(SIM_TIME_STEP, integrator=el.Integrator.Rk4)
        | update_body_omega
        | update_large_pos
        | update_small_pos
    )
    return w, sys


w, sys = build_world()

if __name__ == "__main__":
    w.run(
        sys,
        simulation_rate=1.0 / SIM_TIME_STEP,
        default_playback_speed=1.0,
        max_ticks=MAX_TICKS,
    )
