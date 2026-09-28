"""Pure KDL builder for the Betaflight SITL editor schematic."""

from course import Course, gate_schematic
from fpv_camera import MSG


def build_schematic(course: Course, *, audit_enabled: bool, camera_enabled: bool = False) -> str:
    """Build the default, course, camera, or referee-audit layout without text replacement."""

    camera = 'pos="drone.world_pos + (0,0,0,0, 10,10,5)" look_at="drone.world_pos"'
    if course.gates:
        # Frame the opt-in course; the default keeps its original chase camera.
        camera = 'pos="(0,0,0,1, 4,-2,3)" look_at="(10,0,1.8)"'
        if audit_enabled:
            # The oblique view shows the full opening and world-X departure.
            camera = 'pos="(0,0,0,1, 3,-4,3.5)" look_at="(9,0,2)"'

    frustum = " show_frustums=#true" if camera_enabled else ""
    fpv_pane = f'\n                sensor_view "{MSG}" name="FPV Camera"' if camera_enabled else ""
    accel_graph = 'graph "drone.accel" name="Accelerometer"'
    gyro_graph = 'graph "drone.gyro" name="Gyroscope"'
    if audit_enabled:
        accel_graph = 'graph "drone.last_gate_passed" name="Referee: Last Gate Passed"'
        gyro_graph = 'graph "drone.gate_pass_times" name="Referee: Gate Pass Times"'

    schematic = f"""
    tabs {{
        hsplit name = "Viewport" {{
            viewport name=Viewport {camera} show_grid=#true{frustum} active=#true
            vsplit share=0.3 {{{fpv_pane}
                graph "drone.motor_command" name="Motor Commands (from Betaflight)"
                graph "drone.motor_thrust" name="Motor Thrust"
                {accel_graph}
            }}
            vsplit share=0.3 {{
                graph "drone.world_pos.linear()" name="Position (ENU)"
                graph "drone.world_vel.linear()" name="Velocity"
                {gyro_graph}
            }}
        }}
    }}
    object_3d drone.world_pos {{
        glb path="edu-450-v2-drone.glb" scale=10.0
    }}
    """
    if camera_enabled:
        schematic += """
    object_3d "(0,0,0,1, 0,0,0)" {
        plane width=40 depth=40 { color 70 90 70 }
    }
    """
    if course.gates:
        schematic += "\n" + gate_schematic(course) + "\n"
    return schematic
