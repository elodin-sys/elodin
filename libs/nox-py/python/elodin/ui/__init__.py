"""Typed schematic builders that emit canonical KDL.

Author schematics in Python; the editor continues to consume KDL artifacts.
"""

from __future__ import annotations

from elodin.elodin import ui as _native

from .expr import (
    ComponentHandle,
    Expr,
    ExprError,
    as_eql_strings,
    pose,
    sym_mat3,
    tuple_expr,
)
from .schema import Schema

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
to_python = _native.to_python
write = _native.write
push = _native.push
set_build_error = _native.set_build_error
overlay_key = _native.overlay_key
apply_overlay = _native.apply_overlay
extract_overlay = _native.extract_overlay

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
    "Expr",
    "ExprError",
    "ComponentHandle",
    "Schema",
    "pose",
    "sym_mat3",
    "tuple_expr",
    "as_eql_strings",
    "schematic",
    "from_kdl",
    "to_python",
    "write",
    "push",
    "set_build_error",
    "overlay_key",
    "apply_overlay",
    "extract_overlay",
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
