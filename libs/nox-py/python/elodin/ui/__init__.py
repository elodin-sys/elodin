"""Typed schematic builders that emit canonical KDL (Phase 1).

Author schematics in Python; the editor continues to consume KDL artifacts.
Expressions are still EQL strings — typed expressions arrive in Phase 2.
"""

from __future__ import annotations

from elodin.elodin import ui as _native

Schematic = _native.Schematic
Panel = _native.Panel
Object3D = _native.Object3D
Line3d = _native.Line3d
VectorArrow = _native.VectorArrow
WorldMesh = _native.WorldMesh
Window = _native.Window
Mesh = _native.Mesh
Joint = _native.Joint
Coordinate = _native.Coordinate
Theme = _native.Theme
Timeline = _native.Timeline

schematic = _native.schematic
from_kdl = _native.from_kdl
write = _native.write
push = _native.push

coordinate = _native.coordinate
theme = _native.theme
timeline = _native.timeline
tabs = _native.tabs
hsplit = _native.hsplit
vsplit = _native.vsplit
graph = _native.graph
viewport = _native.viewport
component_monitor = _native.component_monitor
inspector = _native.inspector
hierarchy = _native.hierarchy
schematic_tree = _native.schematic_tree
data_overview = _native.data_overview
video_stream = _native.video_stream
sensor_view = _native.sensor_view
log_stream = _native.log_stream
action_pane = _native.action_pane
query_table = _native.query_table
query_plot = _native.query_plot
glb = _native.glb
joint = _native.joint
object_3d = _native.object_3d
line_3d = _native.line_3d
vector_arrow = _native.vector_arrow
world_mesh = _native.world_mesh
window = _native.window

__all__ = [
    "Schematic",
    "Panel",
    "Object3D",
    "Line3d",
    "VectorArrow",
    "WorldMesh",
    "Window",
    "Mesh",
    "Joint",
    "Coordinate",
    "Theme",
    "Timeline",
    "schematic",
    "from_kdl",
    "write",
    "push",
    "coordinate",
    "theme",
    "timeline",
    "tabs",
    "hsplit",
    "vsplit",
    "graph",
    "viewport",
    "component_monitor",
    "inspector",
    "hierarchy",
    "schematic_tree",
    "data_overview",
    "video_stream",
    "sensor_view",
    "log_stream",
    "action_pane",
    "query_table",
    "query_plot",
    "glb",
    "joint",
    "object_3d",
    "line_3d",
    "vector_arrow",
    "world_mesh",
    "window",
]
