use bevy::prelude::*;
use impeller2_wkt::ComponentValue;
use nox::ArrayBuf;

use crate::object_3d::CompiledExpr;

#[derive(Component, Default)]
pub struct VectorArrowState {
    pub vector_expr: Option<CompiledExpr>,
    pub origin_expr: Option<CompiledExpr>,
}

pub fn component_value_tail_to_vec3(value: &ComponentValue) -> Option<Vec3> {
    match value {
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            tail_to_vec3(data.iter().copied().map(|v| v as f32))
        }
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            tail_to_vec3(data.iter().copied().map(|v| v as f32))
        }
        _ => None,
    }
}

fn tail_to_vec3(mut iter: impl DoubleEndedIterator<Item = f32>) -> Option<Vec3> {
    let z = iter.next_back()?;
    let y = iter.next_back()?;
    let x = iter.next_back()?;
    Some(Vec3::new(x, y, z))
}
