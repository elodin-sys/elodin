"""Typed schematic for the ellipsoid frustum-intersection example.

Layout matches `ellipsoid.kdl`. Object meshes match the `__OBJECT_MESH__`
substitution in `sim.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import elodin.ui as ui

ELLIPSOID_SCALE = (0.9, 0.9, 0.38)
DRONE_GLB = "talon-quad-v2.glb"
DRONE_SCALE = 0.65


def build(
    *,
    ellipsoid_scale: Sequence[float] = ELLIPSOID_SCALE,
    drone_glb: str = DRONE_GLB,
    drone_scale: float = DRONE_SCALE,
) -> ui.Schematic:
    sx, sy, sz = ellipsoid_scale
    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Viewport Source",
                    near=0.05,
                    far=6.0,
                    active=True,
                    show_grid=True,
                    create_frustum=True,
                    frustums_color="yalk",
                    projection_color="mint",
                    frustums_thickness=0.006,
                    frustums_up_marker="highlight",
                    pos="(0,0,0,1, -3,-0.5,2)",
                    look_at="(0,0,0,0, 0,0,0)",
                ),
                ui.viewport(
                    name="Target View",
                    active=True,
                    show_grid=True,
                    show_frustums=True,
                    pos="(0,0,0,1, 2,2,1.5)",
                    look_at="(0,0,0,0, 0,0,0)",
                ),
                ui.sensor_view(
                    "drone.scene_cam",
                    name="Sensor Camera",
                ),
                name="Frustums",
            ),
        ),
        ui.object_3d(
            "ellipsoid.world_pos",
            mesh=ui.ellipsoid(
                scale=f"({sx}, {sy}, {sz})",
                color=ui.color(0, 188, 212, 28),
                show_grid=True,
                grid_color=ui.color(255, 255, 255, 120),
            ),
        ),
        ui.object_3d(
            "drone.world_pos",
            mesh=ui.glb(
                drone_glb,
                scale=drone_scale,
            ),
        ),
        theme=ui.theme(
            mode="dark",
            scheme="default",
        ),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
