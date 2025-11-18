use bevy::{math::DVec3, prelude::*};
use impeller2_wkt::{ComponentValue, WorldPos};
use nox::{ArrayBuf, Quaternion, Vector3};

use crate::WorldPosExt;
use crate::object_3d::CompiledExpr;

#[derive(Component, Default)]
pub struct VectorArrowState {
    pub vector_expr: Option<CompiledExpr>,
    pub origin_expr: Option<CompiledExpr>,
    pub visual: Option<ArrowVisual>,
    pub label: Option<Entity>,
}

#[derive(Clone)]
pub struct ArrowVisual {
    pub root: Entity,
    pub shaft: Entity,
    pub head: Entity,
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
    let world = WorldPos {
        att: Quaternion::identity(),
        pos: Vector3::new(x, y, z),
    };
    Some(world.bevy_pos())
}
