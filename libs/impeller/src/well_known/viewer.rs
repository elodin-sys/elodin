use nox::{ArrayRepr, Quaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::{Asset, ComponentId, EntityId};

use super::Color;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Panel {
    Viewport(Viewport),
    VSplit(Split),
    HSplit(Split),
    Graph(Graph),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Split {
    pub panels: Vec<Panel>,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Viewport {
    pub track_entity: Option<EntityId>,
    pub track_rotation: bool,
    pub fov: f32,
    pub active: bool,
    pub pos: Vector3<f32, ArrayRepr>,
    pub rotation: Quaternion<f32, ArrayRepr>,
    pub show_grid: bool,
    pub hdr: bool,
    pub name: Option<String>,
}

impl Viewport {
    pub fn looking_at(mut self, pos: Vector3<f32, ArrayRepr>) -> Self {
        let dir = pos - self.pos;
        let dir = Vector3::new(dir.x(), dir.z(), -dir.y());
        self.rotation = Quaternion::look_at_rh(dir, Vector3::y_axis()).inverse();
        self
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            track_entity: None,
            fov: 45.0,
            active: false,
            pos: Vector3::new(5.0, 5.0, 10.0),
            rotation: Quaternion::identity(),
            track_rotation: true,
            show_grid: false,
            hdr: false,
            name: None,
        }
    }
}

impl Asset for Panel {
    const ASSET_NAME: &'static str = "panel";
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Graph {
    pub entities: Vec<GraphEntity>,
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphEntity {
    pub entity_id: EntityId,
    pub components: Vec<GraphComponent>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphComponent {
    pub component_id: ComponentId,
    pub indexes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Line3d {
    pub entity: EntityId,
    pub component_id: ComponentId,
    pub index: [usize; 3],
    pub line_width: f32,
    pub color: Color,
    pub perspective: bool,
}

impl Asset for Line3d {
    const ASSET_NAME: &'static str = "line_3d";
}
