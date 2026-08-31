"""Rebuild examples/db-client/schematic.kdl via elodin.ui (Phases 1–2).

Watch live::

    elodin ui watch examples/db-client/schematic.py --db 127.0.0.1:2240
"""

from __future__ import annotations

import elodin.ui as ui
from elodin.ui import Expr


def build() -> ui.Schematic:
    # Typed expressions (Phase 2): still emit EQL strings into KDL.
    world_pos = Expr("drone.world_pos")
    chase_pos = world_pos + Expr("(0,0,0,0, 0.4, 0.4, 0.25)")

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
                ui.graph("drone.battery.voltage", name="Battery BBB(V)"),
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
                ui.graph("drone.nav.speed", name="Ground speed (m/s)"),
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
        ui.line_3d(world_pos, line_width=2.0, color="yalk"),
        coordinate=ui.coordinate(frame="ENU"),
        theme=ui.theme(mode="dark", scheme="default"),
        timeline=ui.timeline(),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
