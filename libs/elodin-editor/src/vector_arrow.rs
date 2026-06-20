use bevy::{math::DVec3, prelude::*};
use impeller2_wkt::ComponentValue;
use nox::ArrayBuf;
use std::collections::HashMap;

use crate::object_3d::CompiledExpr;

#[derive(Component)]
pub struct ViewportArrow {
    pub camera: Entity,
}

/// The two pose entities carrying an arrow's start and end as `WorldPos`,
/// placed by the canonical position pipeline (`sync_pos` -> Geo* -> Transform).
#[derive(Component)]
pub struct ArrowEndpoints {
    pub start: Entity,
    pub end: Entity,
}

/// Marker on endpoint entities pointing back at the owning arrow entity.
#[derive(Component)]
pub struct ArrowEndpoint {
    pub owner: Entity,
}

/// Last successfully rendered arrow pose, reused when endpoint reads fail transiently.
#[derive(Clone)]
pub struct CachedArrowPose {
    pub direction_world: Vec3,
    /// Shaft direction in the start endpoint's local space (mesh +Y).
    pub local_rotation: Quat,
}

#[derive(Component, Default)]
pub struct VectorArrowState {
    pub vector_expr: Option<CompiledExpr>,
    pub origin_expr: Option<CompiledExpr>,
    /// Whether the last expression evaluation produced a drawable arrow.
    pub valid: bool,
    pub cached_pose: Option<CachedArrowPose>,
    pub visuals: HashMap<Entity, ArrowVisual>,
    pub label: Option<Entity>,
    /// Label offset from the arrow root, cached in render_vector_arrow for UI sync.
    pub label_offset: Option<Vec3>,
    pub label_name: Option<String>,
    pub label_color: Option<Color>,
    pub label_scope: ArrowLabelScope,
}

#[derive(Clone)]
pub struct ArrowVisual {
    pub root: Entity,
    pub shaft: Entity,
    pub head: Entity,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum ArrowLabelScope {
    #[default]
    Global,
    Viewport,
}

pub fn component_value_tail_to_vec3(value: &ComponentValue) -> Option<DVec3> {
    match value {
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            tail_to_vec3(data.iter().copied().map(f64::from))
        }
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            tail_to_vec3(data.iter().copied())
        }
        _ => None,
    }
}

fn tail_to_vec3(mut iter: impl DoubleEndedIterator<Item = f64>) -> Option<DVec3> {
    let z = iter.next_back()?;
    let y = iter.next_back()?;
    let x = iter.next_back()?;
    Some(DVec3::new(x, y, z))
}
