"""Static Mojave terrain shared by the RC-jet layouts."""

from elodin import ui

STATIC_TERRAIN_POSE = (
    "(0,0,0,1,-2429898.6572,-4606953.4052,3670007.6523)"
    ".translate_world((0,0,-2).enu_to_ecef_vector())"
)


def terrain():
    return ui.object_3d(
        STATIC_TERRAIN_POSE,
        mesh=ui.glb("mojave_rc_field.glb", rotate=(90.0, 0.0, 0.0)),
        frame="ECEF",
        frame_orientation="ECEF",
        orientation="absolute",
    )
