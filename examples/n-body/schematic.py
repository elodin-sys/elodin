"""Python-authored solar-system UI, independent of simulation initialization.

Watch live::

    elodin ui watch examples/n-body/schematic.py --db 127.0.0.1:2240
"""

from collections.abc import Sequence

import elodin.ui as ui
from body_metadata import (
    AU_IN_KM,
    SUN_COLOR,
    SUN_RADIUS_KM,
    TRUTH_COLOR,
    Body,
    load_bodies,
)


def _body_scene(
    name: str, radius_km: float, color_rgb: tuple[int, int, int], icon: str
) -> list[ui.Object3D | ui.Line3d | ui.VectorArrow]:
    """A simulated body, its orbital trail, and its short label arrow."""
    position = f"{name}.world_pos"
    color = ui.color(*color_rgb)
    # Preserve the eight-decimal AU radii from the original KDL generator.
    radius = round(radius_km / AU_IN_KM, 8)
    return [
        ui.object_3d(
            position,
            mesh=ui.sphere(radius=radius, color=color),
            icon=ui.icon(
                builtin=icon,
                visibility=ui.visibility_range(min=1.0, fade_distance=5.0),
                color=color,
            ),
        ),
        ui.line_3d(position, line_width=1.5, perspective=False, color=color),
        ui.vector_arrow(
            "(0,0,0.03)",
            origin=position,
            scale=1.0,
            name=name.replace("_", " "),
            show_name=True,
            thickness=0.02,
            label_position="1.0",
            color=color,
        ),
    ]


def _truth_scene(body: Body) -> list[ui.Object3D | ui.Line3d]:
    position = f"truth_{body.name}.truth_world_pos"
    radius = round(body.meta.radius_km / AU_IN_KM * 0.6, 8)
    return [
        ui.object_3d(
            position,
            mesh=ui.sphere(radius=radius, color=ui.color(*TRUTH_COLOR, 120)),
        ),
        ui.line_3d(position, line_width=1.0, perspective=False, color=ui.color(*TRUTH_COLOR, 80)),
    ]


def build(bodies: Sequence[Body] | None = None) -> ui.Schematic:
    """Build for the loaded bodies, or discover them from the configured CSVs.

    The simulation passes its actual body list. Standalone CLI/watch builds read
    only CSV identities; they do not initialize a world or mutate sim globals.
    """
    if bodies is None:
        bodies = load_bodies()

    scene = _body_scene("sun", SUN_RADIUS_KM, SUN_COLOR, "wb_sunny")
    for body in bodies:
        scene.extend(
            _body_scene(body.name, body.meta.radius_km, body.meta.color_rgb, body.meta.icon)
        )
        scene.extend(_truth_scene(body))

    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.tabs(
                    ui.viewport(
                        name="SolarSystem",
                        pos="(0,0,0,1, -6,6,6)",
                        look_at="(0,0,0,1, 0,0,0)",
                        hdr=True,
                        show_grid=True,
                        active=True,
                    ),
                    ui.viewport(
                        name="TopDown",
                        pos="(0,0,0,1, 0,0,25)",
                        look_at="(0,0,0,1, 0,0,0)",
                        hdr=True,
                        show_grid=True,
                    ),
                    share=0.75,
                ),
                ui.tabs(
                    ui.graph(
                        "earth.world_pos[4],earth.world_pos[5],earth.world_pos[6],"
                        "truth_earth.truth_world_pos[4],truth_earth.truth_world_pos[5],"
                        "truth_earth.truth_world_pos[6]",
                        name="Earth vs Truth (AU)",
                    ),
                    ui.hierarchy(),
                    ui.inspector(),
                    share=0.25,
                ),
            ),
        ),
        *scene,
        coordinate=ui.coordinate(frame="ENU"),
        timeline=ui.timeline(follow_latest=True),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
