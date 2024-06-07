use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::{Asset, ComponentId, EntityId};

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
    pub pos: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub show_grid: bool,
    pub hdr: bool,
}

impl Viewport {
    pub fn looking_at(mut self, pos: Vector3<f32>) -> Self {
        let dir = pos - self.pos;
        self.rotation = UnitQuaternion::look_at_rh(&dir.xzy(), &Vector3::y()).inverse();
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
            rotation: UnitQuaternion::identity(),
            track_rotation: true,
            show_grid: false,
            hdr: false,
        }
    }
}

impl Asset for Panel {
    const ASSET_NAME: &'static str = "panel";
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Graph {
    pub entities: Vec<GraphEntity>,
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
