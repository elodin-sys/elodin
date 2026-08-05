"""Rebuild examples/db-client/schematic.kdl via elodin.ui (Phase 1 G1)."""

from __future__ import annotations

import elodin.ui as ui


def build() -> ui.Schematic:
    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Chase",
                    pos="drone.world_pos + (0,0,0,0, 0.4, 0.4, 0.25)",
                    look_at="drone.world_pos",
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
                ui.graph("drone.battery.voltage", name="Battery (V)"),
                ui.graph("drone.motor.rpm", name="Motor RPM"),
                ui.graph("drone.status.armed", name="Armed"),
                ui.graph(
                    "drone.status.mode",
                    name="Flight mode (1=hover, 2=cruise)",
                ),
                name="Status",
            ),
            ui.vsplit(
                ui.graph("drone.world_pos", name="World pos (quaternion + xyz)"),
                ui.graph("drone.nav.speed", name="Ground speed (m/s)"),
                name="Pose",
            ),
        ),
        ui.object_3d(
            "drone.world_pos",
            mesh=ui.glb("crazyflie.glb", scale=0.7),
            animate=[
                ui.joint(
                    "Root.Propeller_0",
                    rotation_vector="(0, drone.propeller_angle[0], 0)",
                ),
                ui.joint(
                    "Root.Propeller_1",
                    rotation_vector="(0, drone.propeller_angle[1], 0)",
                ),
                ui.joint(
                    "Root.Propeller_2",
                    rotation_vector="(0, drone.propeller_angle[2], 0)",
                ),
                ui.joint(
                    "Root.Propeller_3",
                    rotation_vector="(0, drone.propeller_angle[3], 0)",
                ),
            ],
        ),
        ui.line_3d("drone.world_pos", line_width=2.0, color="yalk"),
        coordinate=ui.coordinate(frame="ENU"),
        theme=ui.theme(mode="dark", scheme="default"),
        timeline=ui.timeline(),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
