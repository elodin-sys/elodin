"""Typed schematic for the n-body solar-system example.

Layout matches `solar-system-template.kdl`. Body and sun objects match
`body.template.kdl` and `sun.template.kdl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import elodin.ui as ui

TRUTH_RGB = (180, 180, 180)
SUN_COLOR = (255, 220, 120)
AU_IN_KM = 149_597_870.7
SUN_RADIUS_AU = 696_340.0 / AU_IN_KM


class BodyVisual(NamedTuple):
    name: str
    radius: float
    color: tuple[int, int, int]
    icon: str
    label: str
    truth_radius: float


def _rgb(color: tuple[int, int, int], alpha: int | None = None) -> ui.Color:
    if alpha is None:
        return ui.color(*color)
    return ui.color(*color, alpha)


def _labeled_body(
    eql: str,
    *,
    radius: float,
    color: tuple[int, int, int],
    icon: str,
    label: str,
    line_width: float,
) -> list[object]:
    painted = _rgb(color)
    return [
        ui.object_3d(
            eql,
            mesh=ui.sphere(radius=radius, color=painted),
            icon=ui.icon(
                builtin=icon,
                color=painted,
                visibility=ui.visibility_range(min=1.0, fade_distance=5.0),
            ),
        ),
        ui.line_3d(eql, line_width=line_width, perspective=False, color=painted),
        ui.vector_arrow(
            "(0,0,0.03)",
            origin=eql,
            name=label,
            color=painted,
            show_name=True,
            thickness=0.02,
            label_position="1.0",
        ),
    ]


def _truth_body(name: str, radius: float) -> list[object]:
    eql = f"truth_{name}.truth_world_pos"
    return [
        ui.object_3d(
            eql,
            mesh=ui.sphere(radius=radius, color=_rgb(TRUTH_RGB, 120)),
        ),
        ui.line_3d(
            eql,
            line_width=1.0,
            perspective=False,
            color=_rgb(TRUTH_RGB, 80),
        ),
    ]


def _layout() -> ui.Panel:
    return ui.tabs(
        ui.hsplit(
            ui.tabs(
                ui.viewport(
                    name="SolarSystem",
                    fov=45.0,
                    active=True,
                    show_grid=True,
                    show_arrows=True,
                    hdr=True,
                    pos="(0,0,0,1, -6,6,6)",
                    look_at="(0,0,0,1, 0,0,0)",
                ),
                ui.viewport(
                    name="TopDown",
                    fov=45.0,
                    show_grid=True,
                    show_arrows=True,
                    hdr=True,
                    pos="(0,0,0,1, 0,0,25)",
                    look_at="(0,0,0,1, 0,0,0)",
                ),
                share=0.75,
            ),
            ui.tabs(
                ui.graph(
                    "earth.world_pos[4],earth.world_pos[5],earth.world_pos[6],truth_earth.truth_world_pos[4],truth_earth.truth_world_pos[5],truth_earth.truth_world_pos[6]",
                    name="Earth vs Truth (AU)",
                ),
                ui.hierarchy(),
                ui.inspector(),
                share=0.25,
            ),
        ),
    )


def build(
    bodies: Sequence[BodyVisual] = (),
    *,
    sun_radius: float = SUN_RADIUS_AU,
    sun_color: tuple[int, int, int] = SUN_COLOR,
) -> ui.Schematic:
    elems: list[object] = [
        _layout(),
        *_labeled_body(
            "sun.world_pos",
            radius=sun_radius,
            color=sun_color,
            icon="wb_sunny",
            label="sun",
            line_width=1.5,
        ),
    ]
    for body in bodies:
        elems.extend(
            _labeled_body(
                f"{body.name}.world_pos",
                radius=body.radius,
                color=body.color,
                icon=body.icon,
                label=body.label,
                line_width=1.5,
            )
        )
        elems.extend(_truth_body(body.name, body.truth_radius))
    return ui.schematic(
        *elems,
        coordinate=ui.coordinate(frame="ENU"),
        timeline=ui.timeline(follow_latest=True),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
