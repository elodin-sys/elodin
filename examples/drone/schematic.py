"""Typed schematic for the drone example.

The drone GLB follows `Config.GLOBAL.drone_glb`.
"""

from __future__ import annotations

import elodin.ui as ui

DEFAULT_DRONE_GLB = "talon-quad-v2.glb"


def _body_axis(vector: str, name: str):
    return ui.vector_arrow(
        vector,
        origin="drone.world_pos",
        scale=1.0,
        name=name,
        body_frame=True,
    )


def build(*, drone_glb: str = DEFAULT_DRONE_GLB) -> ui.Schematic:
    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Viewport",
                    pos="drone.world_pos + (0,0,0,0, 2,2,2)",
                    look_at="drone.world_pos",
                    show_grid=True,
                    active=True,
                ),
                ui.vsplit(
                    ui.graph("drone.angle_desired", name="angle_desired"),
                    ui.graph(
                        "drone.world_pos.q0, drone.world_pos.q1, "
                        "drone.world_pos.q2, drone.world_pos.q3, "
                        "drone.attitude_target",
                        name="World Pos",
                    ),
                    ui.graph("drone.ang_vel_setpoint"),
                    share=0.4,
                ),
                name="Viewport",
            ),
            ui.vsplit(
                ui.graph("drone.gyro"),
                ui.graph("drone.accel"),
                ui.graph("drone.magnetometer"),
                name="Sensor Panel",
            ),
        ),
        ui.window(
            ui.tabs(
                ui.hsplit(
                    name="Motor Panel",
                    ui.vsplit(
                        ui.graph("drone.motor_input"),
                        ui.graph("drone.motor_pwm"),
                        ui.graph("drone.motor_rpm"),
                        share=0.4,
                    ),
                    ui.graph("drone.thrust"),
                ),
            ),
            title="Motor Panel",
        ),
        ui.window(
            ui.hsplit(
                name="Rate Control Panel",
                ui.vsplit(
                    ui.graph("drone.rate_pid_state"),
                    ui.component_monitor(component_name="drone.rate_pid_state"),
                ),
                ui.vsplit(
                    ui.graph(
                        "drone.gyro, drone.ang_vel_setpoint",
                        name="Drone: rate_control",
                    ),
                ),
            ),
            title="Rate Control Panel",
        ),
        _body_axis("(1, 0, 0)", "Drone X"),
        _body_axis("(0, 1, 0)", "Drone Y"),
        _body_axis("(0, 0, 1)", "Drone Z"),
        ui.object_3d(
            "drone.world_pos",
            mesh=ui.glb(drone_glb),
            icon=ui.icon(
                builtin="flight",
                color=ui.color(0, 188, 212),
                visibility=ui.visibility_range(min=500.0),
            ),
        ),
        theme=ui.theme(mode="dark", scheme="default"),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
