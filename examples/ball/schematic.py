"""Python schematic matching the ball example's KDL.

Watch live::

    elodin ui watch examples/ball/schematic.py --db 127.0.0.1:2240
"""

from __future__ import annotations

import elodin.ui as ui

# Named ``yalk`` at alpha 100. ``ui.color`` takes 0–255 components.
_YALK_100 = ui.color(255, 230, 51, 100)


def build(*, frame: str = "ENU") -> ui.Schematic:
    if frame == "ENU":
        cam_pos = "(0,0,0,0, 8,2,4)"
        look_at = "(0,0,0,0, 0,0,3)"
        scene_frame = "ENU"
        extra_frame = None
    elif frame == "NED":
        cam_pos = "(0,0,0,0, 8,2,-4)"
        look_at = "(0,0,0,0, 0,0,-3)"
        scene_frame = "NED"
        extra_frame = "NED"
    else:
        raise ValueError(f"unsupported schematic frame {frame!r}")

    return ui.schematic(
        ui.hsplit(
            ui.tabs(
                ui.viewport(
                    name="Viewport",
                    pos=cam_pos,
                    look_at=look_at,
                    hdr=True,
                    show_grid=True,
                    active=True,
                    frame=extra_frame,
                ),
                ui.inspector(),
            ),
        ),
        ui.object_3d(
            "ball.world_pos",
            mesh=ui.sphere(radius=0.2, color="orange"),
            frame=scene_frame,
        ),
        ui.line_3d(
            "ball.world_pos",
            line_width=2.0,
            color="white",
            frame=extra_frame,
        ),
        ui.vector_arrow(
            "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]",
            origin="ball.world_pos",
            scale=1.0,
            name="Ball Velocity",
            show_name=True,
            label_position="0.3m",
            color=_YALK_100,
            frame=extra_frame,
        ),
        ui.object_3d(
            "(0,0,0,1, 0,0,0)",
            mesh=ui.plane(width=2000, depth=2000, color=ui.color(32, 128, 32, 125)),
            frame=extra_frame,
        ),
        coordinate=ui.coordinate(frame=scene_frame),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
