"""Rebuild examples/db-client/schematic.kdl via elodin.ui (Phases 1–2).

Watch live::

    elodin ui watch examples/db-client/schematic.py --db 127.0.0.1:2240
"""

from __future__ import annotations

import jax.numpy as jnp

import elodin.ui as ui
from elodin.ui import Expr, Schema


def packed_covariance_cholesky(cov):
    """3×3 Cholesky of the editor packed SPD 6-vector ``[p00,p10,p20,p11,p21,p22]``.

    ``jnp.linalg.cholesky`` lowers to LAPACK FFI, which Cranelift rejects.
    """
    p00, p10, p20 = cov[0], cov[1], cov[2]
    p11, p21, p22 = cov[3], cov[4], cov[5]
    l00 = jnp.sqrt(p00)
    l10 = p10 / l00
    l20 = p20 / l00
    l11 = jnp.sqrt(p11 - l10 * l10)
    l21 = (p21 - l20 * l10) / l11
    l22 = jnp.sqrt(p22 - l20 * l20 - l21 * l21)
    z = jnp.float64(0.0)
    return jnp.array([[l00, z, z], [l10, l11, z], [l20, l21, l22]])


@ui.kernel
def covariance_cholesky(cov):
    return packed_covariance_cholesky(cov)


@ui.kernel
def apply_transform(pos, transform):
    """Apply a row-major 4×4 homogeneous transform to a 3D position."""
    T = jnp.array(
        [
            [transform[0], transform[1], transform[2], transform[3]],
            [transform[4], transform[5], transform[6], transform[7]],
            [transform[8], transform[9], transform[10], transform[11]],
            [transform[12], transform[13], transform[14], transform[15]],
        ]
    )
    p = jnp.array([pos[0], pos[1], pos[2], jnp.float64(1.0)])
    return (T @ p)[:3]


def build() -> ui.Schematic:
    # Typed expressions (Phase 2): still emit EQL strings into KDL.
    world_pos = Expr("drone.world_pos")
    chase_pos = world_pos + Expr("(0,0,0,0, 0.4, 0.4, 0.25)")
    schema = Schema.from_json(
        {
            "components": {
                "drone.nav.covariance": {
                    "shape": [6],
                    "prim_type": "f64",
                },
                "drone.nav.position": {
                    "shape": [3],
                    "prim_type": "f64",
                },
                "drone.nav.transform": {
                    "shape": [16],
                    "prim_type": "f64",
                },
            }
        }
    )
    chol = covariance_cholesky(schema["drone.nav.covariance"])
    rotated = apply_transform(schema["drone.nav.position"], schema["drone.nav.transform"])

    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Chase",
                    pos=chase_pos,
                    look_at=world_pos,
                    show_grid=True,
                    active=True,
                    share=0.55,
                ),
                ui.vsplit(
                    ui.graph("drone.imu.accel", name="Accelerometer (m/s^2)"),
                    ui.graph("drone.imu.gyro", name="Gyroscope (rad/s)"),
                    ui.graph(
                        "drone.nav.speed",
                        name="Ground speed, stream-derived (m/s)",
                    ),
                    share=0.45,
                ),
                name="Flight",
            ),
            ui.vsplit(
                ui.graph("drone.battery.voltage", name="Battery (V)"),
                ui.graph("drone.motor.rpm", name="Motor RPM"),
                ui.graph("drone.status.armed", name="Armed"),
                ui.graph(
                    "drone.status.mode",
                    name="Flight mode (1=hover, 2=cruise)",
                ),
                name="Status",
            ),
            ui.vsplit(
                ui.graph(world_pos, name="World pos (quaternion + xyz)"),
                ui.graph(rotated, name="Transformed position (T @ xyz)"),
                ui.graph(chol, name="Nav covariance Cholesky"),
                name="Pose",
            ),
        ),
        ui.object_3d(
            world_pos,
            mesh=ui.glb("crazyflie.glb", scale=0.7),
            animate=[
                ui.joint(
                    f"Root.Propeller_{i}",
                    rotation_vector=Expr(f"(0, drone.propeller_angle[{i}], 0)"),
                )
                for i in range(4)
            ],
        ),
        ui.object_3d(
            world_pos,
            mesh=ui.ellipsoid(
                error_covariance_cholesky=chol,
                color=ui.color(64, 180, 255, 40),
                error_confidence_interval=70.0,
                show_grid=True,
                grid_color=ui.color(180, 230, 255, 160),
            ),
        ),
        ui.object_3d(
            rotated,
            mesh=ui.sphere(radius=0.06, color=ui.color(255, 180, 64)),
        ),
        ui.line_3d(world_pos, line_width=2.0, color="yalk"),
        coordinate=ui.coordinate(frame="ENU"),
        theme=ui.theme(mode="dark", scheme="default"),
        timeline=ui.timeline(),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
