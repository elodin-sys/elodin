use crate::Color;
use impeller2::component::Asset;
use impeller2::types::{ComponentId, EntityId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Panel {
    Viewport(Viewport),
    VSplit(Split),
    HSplit(Split),
    Graph(Graph),
    ComponentMonitor(ComponentMonitor),
    ActionPane(ActionPane),
    SQLTable(SQLTable),
    SQLPlot(SQLPlot),
    Tabs(Vec<Panel>),
    Inspector,
    Hierarchy,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Split {
    pub panels: Vec<Panel>,
    pub shares: HashMap<usize, f32>,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Viewport {
    pub fov: f32,
    pub active: bool,
    pub show_grid: bool,
    pub hdr: bool,
    pub name: Option<String>,
    pub pos: Option<String>,
    pub look_at: Option<String>,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            fov: 45.0,
            active: false,
            show_grid: false,
            hdr: false,
            name: None,
            pos: None,
            look_at: None,
        }
    }
}

impl Asset for Panel {
    const NAME: &'static str = "panel";
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Graph {
    pub eql: String,
    pub name: Option<String>,
    #[serde(default)]
    pub graph_type: GraphType,
    pub auto_y_range: bool,
    pub y_range: Range<f64>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Hash, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum GraphType {
    #[default]
    Line,
    Point,
    Bar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Line3d {
    pub eql: String,
    pub line_width: f32,
    pub color: Color,
    pub perspective: bool,
}

impl Asset for Line3d {
    const NAME: &'static str = "line_3d";
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Camera;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct VectorArrow {
    pub id: ComponentId,
    pub range: Range<usize>,
    pub color: Color,
    pub attached: bool,
    pub body_frame: bool,
    pub scale: f32,
}

impl Asset for VectorArrow {
    const NAME: &'static str = "arrow";
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct BodyAxes {
    pub entity_id: EntityId,
    pub scale: f32,
}

impl Asset for BodyAxes {
    const NAME: &'static str = "body_axes";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Mesh {
    Sphere { radius: f32 },
    Box { x: f32, y: f32, z: f32 },
    Cylinder { radius: f32, height: f32 },
}

impl Mesh {
    pub fn cuboid(x: f32, y: f32, z: f32) -> Self {
        Self::Box { x, y, z }
    }

    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Glb(pub String);

impl Asset for Mesh {
    const NAME: &'static str = "mesh";
}

impl Asset for Glb {
    const NAME: &'static str = "glb";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Material {
    pub base_color: Color,
}

impl Material {
    pub fn color(r: f32, g: f32, b: f32) -> Self {
        Material {
            base_color: Color { r, g, b },
        }
    }
}

impl Asset for Material {
    const NAME: &'static str = "material";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Object3D {
    Glb(String),
    Mesh { mesh: Mesh, material: Material },
}

impl Asset for Object3D {
    const NAME: &'static str = "mesh_source";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ComponentMonitor {
    pub component_id: ComponentId,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct SQLTable {
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ActionPane {
    pub label: String,
    pub lua: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct SQLPlot {
    pub query: String,
    pub refresh_interval: Duration,
    pub auto_refresh: bool,
}
