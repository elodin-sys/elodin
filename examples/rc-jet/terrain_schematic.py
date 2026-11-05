"""Mojave terrain selection shared by the RC-jet layouts."""

import os

import elodin.ui as ui


TERRAIN_MODE_ENV = "ELODIN_RC_JET_TERRAIN"
STATIC_TERRAIN_POSE = (
    "(0,0,0,1,-2429898.6572,-4606953.4052,3670007.6523)"
    ".translate_world((0,0,-2).enu_to_ecef_vector())"
)


def terrain_mode() -> str:
    mode = os.environ.get(TERRAIN_MODE_ENV, "static").strip().lower()
    if mode not in {"static", "atlas"}:
        raise ValueError(f"{TERRAIN_MODE_ENV} must be 'static' or 'atlas', got {mode!r}")
    return mode


def terrain():
    mode = terrain_mode()
    if mode == "static":
        return ui.object_3d(
            STATIC_TERRAIN_POSE,
            mesh=ui.glb("mojave_rc_field.glb", rotate=(90.0, 0.0, 0.0)),
            frame="ECEF",
            frame_orientation="ECEF",
            orientation="absolute",
        )
    return ui.world_mesh(
        "mojave_rc_field",
        translate=(0.0, 0.0, 528.0),
        frame="ENU",
        visible=True,
    )
