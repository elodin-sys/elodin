"""What the editor and the overhead sensor camera show: every agent and object as a sphere."""

from __future__ import annotations

import elodin.ui as ui

CAMERA = "camera.overhead"
AGENT_COLOR = ui.color(0, 200, 255)
OBJECT_COLOR = ui.color(255, 140, 0)


def at(position: str) -> str:
    """A pose at the 3-vector component `position`, unrotated."""
    return f"(0,0,0,1, {position}[0], {position}[1], {position}[2])"


def build(agent_count: int, object_count: int, camera: bool) -> ui.Schematic:
    scene = ui.viewport(
        name="Scene",
        pos="(0,0,0,0, 0,-130,95)",
        look_at="(0,0,0,0, 0,0,25)",
        show_grid=True,
        show_frustums=True,
    )
    return ui.schematic(
        ui.hsplit(scene, ui.sensor_view(CAMERA, name="Overhead camera")) if camera else scene,
        *(
            ui.object_3d(
                at(f"agent_{index:03d}.position"),
                mesh=ui.sphere(radius=1.5, color=AGENT_COLOR),
            )
            for index in range(agent_count)
        ),
        *(
            ui.object_3d(
                at(f"object_{index:03d}.object_position"),
                mesh=ui.sphere(radius=1.5, color=OBJECT_COLOR),
            )
            for index in range(object_count)
        ),
        ui.object_3d(
            "(0,0,0,1, 0,0,-2)",
            mesh=ui.plane(width=100, depth=100, color=ui.color(40, 70, 40)),
        ),
        timeline=ui.timeline(follow_latest=True),
    )
