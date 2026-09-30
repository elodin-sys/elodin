"""Python schematic for the display-kernels example.

Watch live while the sim is running::

    elodin ui watch examples/display-kernels/schematic.py --db 127.0.0.1:2240
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
def covariance_stats(cov):
    """``[det(P), trace(P)]`` from the packed SPD 6-vector."""
    p00, p10, p20 = cov[0], cov[1], cov[2]
    p11, p21, p22 = cov[3], cov[4], cov[5]
    det = (
        p00 * (p11 * p22 - p21 * p21)
        - p10 * (p10 * p22 - p20 * p21)
        + p20 * (p10 * p21 - p11 * p20)
    )
    return jnp.array([det, p00 + p11 + p22])


def build() -> ui.Schematic:
    world_pos = Expr("craft.world_pos")
    chase_pos = world_pos + Expr("(0,0,0,0, 6,-6,4)")
    schema = Schema.from_json(
        {
            "components": {
                "craft.error_covariance": {
                    "shape": [6],
                    "prim_type": "f64",
                },
            }
        }
    )
    cov = schema["craft.error_covariance"]
    chol = covariance_cholesky(cov)
    stats = covariance_stats(cov)

    return ui.schematic(
        ui.hsplit(
            ui.tabs(
                ui.viewport(
                    name="Viewport",
                    pos=chase_pos,
                    look_at=world_pos,
                    show_grid=True,
                    active=True,
                ),
                ui.inspector(),
            ),
            ui.vsplit(
                ui.graph(stats, name="Uncertainty volume (det, trace)"),
                ui.graph("craft.error_covariance", name="Packed covariance (EQL)"),
            ),
        ),
        ui.object_3d(
            world_pos,
            mesh=ui.sphere(radius=0.12, color=ui.color(255, 180, 64)),
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
        ui.line_3d(world_pos, line_width=2.0, color="yalk"),
        coordinate=ui.coordinate(frame="ENU"),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
