"""Typed schematic for the sensor-camera bouncing-balls example.

Layout and object meshes match `sensor-camera.kdl`.
"""

from __future__ import annotations

import elodin.ui as ui

BALL_RADIUS = 0.3
BALLS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("cam_ball_a", (0, 220, 220)),
    ("cam_ball_b", (220, 0, 220)),
    ("ball_1", (255, 140, 0)),
    ("ball_2", (255, 255, 100)),
    ("ball_3", (100, 255, 100)),
)


def build(*, ball_radius: float = BALL_RADIUS) -> ui.Schematic:
    return ui.schematic(
        ui.hsplit(
            ui.viewport(
                name="Main",
                pos="(0,0,0,0, 14,14,10)",
                look_at="(0,0,0,0, 0,0,1)",
                show_grid=True,
                show_frustums=True,
            ),
            ui.vsplit(
                ui.sensor_view("cam_ball_a.scene_cam", name="RGB Camera (Cyan Ball)"),
                ui.sensor_view("cam_ball_b.thermal_cam", name="Thermal (Magenta Ball)"),
            ),
        ),
        *(
            ui.object_3d(
                f"{name}.world_pos",
                mesh=ui.sphere(radius=ball_radius, color=ui.color(*rgb)),
            )
            for name, rgb in BALLS
        ),
        ui.object_3d(
            "(0,0,0,1, 0,0,0)",
            mesh=ui.plane(width=12, depth=12, color=ui.color(60, 120, 60)),
        ),
        timeline=ui.timeline(follow_latest=True),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
