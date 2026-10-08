"""Python schematic for the Betaflight SITL editor."""

from __future__ import annotations

import elodin.ui as ui

from course import Course, gate_objects
from fpv_camera import MSG


def build_schematic(
    course: Course, *, audit_enabled: bool, camera_enabled: bool = False
) -> ui.Schematic:
    """Build the default, course, camera, or referee-audit layout."""

    if course.gates and audit_enabled:
        # The oblique view shows the full opening and world-X departure.
        pos = "(0,0,0,1, 3,-4,3.5)"
        look_at = "(9,0,2)"
    elif course.gates:
        # Frame the opt-in course; the default keeps its original chase camera.
        pos = "(0,0,0,1, 4,-2,3)"
        look_at = "(10,0,1.8)"
    else:
        pos = "drone.world_pos + (0,0,0,0, 10,10,5)"
        look_at = "drone.world_pos"

    if audit_enabled:
        accel_graph = ui.graph("drone.last_gate_passed", name="Referee: Last Gate Passed")
        gyro_graph = ui.graph("drone.gate_pass_times", name="Referee: Gate Pass Times")
    else:
        accel_graph = ui.graph("drone.accel", name="Accelerometer")
        gyro_graph = ui.graph("drone.gyro", name="Gyroscope")

    left_panels: list[ui.Panel] = []
    if camera_enabled:
        left_panels.append(ui.sensor_view(MSG, name="FPV Camera"))
    left_panels.extend(
        [
            ui.graph("drone.motor_command", name="Motor Commands (from Betaflight)"),
            ui.graph("drone.motor_thrust", name="Motor Thrust"),
            accel_graph,
        ]
    )
    right_panels = [
        ui.graph("drone.world_pos.linear()", name="Position (ENU)"),
        ui.graph("drone.world_vel.linear()", name="Velocity"),
        gyro_graph,
    ]

    elements: list[ui.Panel | ui.Object3D] = [
        ui.tabs(
            ui.hsplit(
                ui.viewport(
                    name="Viewport",
                    pos=pos,
                    look_at=look_at,
                    show_grid=True,
                    active=True,
                ),
                ui.vsplit(*left_panels, share=0.3),
                ui.vsplit(*right_panels, share=0.3),
                name="Viewport",
            )
        ),
        ui.object_3d(
            "drone.world_pos",
            mesh=ui.glb("edu-450-v2-drone.glb", scale=10.0),
            sensor_visible=not camera_enabled,
        ),
    ]
    if camera_enabled:
        elements.append(
            ui.object_3d(
                "(0,0,0,1, 0,0,0)",
                mesh=ui.plane(width=40, depth=40, color=ui.color(70, 90, 70)),
            )
        )
    elements.extend(gate_objects(course))
    return ui.schematic(*elements)


if __name__ == "__main__":
    from course import course_from_name

    print(build_schematic(course_from_name("none"), audit_enabled=False).emit_kdl())
