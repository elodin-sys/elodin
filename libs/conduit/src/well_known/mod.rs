use nalgebra::{Quaternion, Vector3};
use ndarray::{array, CowArray, Ix1};
use serde::{Deserialize, Serialize};
use smallvec::smallvec;

use crate::{ComponentType, ComponentValue, PrimitiveTy};

#[cfg(feature = "bevy")]
mod bevy_conv;

mod metadata;
mod pbr;
mod primitives;

pub use metadata::*;
pub use pbr::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct WorldPos {
    pub att: Quaternion<f64>,
    pub pos: Vector3<f64>,
}

impl crate::Component for WorldPos {
    fn component_id() -> crate::ComponentId {
        crate::ComponentId::new("world_pos")
    }

    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: PrimitiveTy::F64,
            shape: smallvec![7],
        }
    }

    fn component_value<'a>(&self) -> crate::ComponentValue<'a> {
        let arr = array![
            self.att.coords.x,
            self.att.coords.y,
            self.att.coords.z,
            self.att.coords.w,
            self.pos.x,
            self.pos.y,
            self.pos.z
        ]
        .into_dyn();
        ComponentValue::F64(CowArray::from(arr))
    }

    fn from_component_value(value: crate::ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let crate::ComponentValue::F64(arr) = value else {
            return None;
        };
        if arr.shape() != [7] {
            return None;
        }
        let arr = arr.into_dimensionality::<Ix1>().ok()?;
        let arr = arr.as_slice()?;
        Some(WorldPos {
            att: Quaternion::new(arr[3], arr[0], arr[1], arr[2]),
            pos: Vector3::new(arr[4], arr[5], arr[6]),
        })
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct TraceAnchor {
    pub anchor: Vector3<f64>,
}

impl crate::Component for TraceAnchor {
    fn component_id() -> crate::ComponentId {
        crate::ComponentId::new("trace_anchor")
    }

    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: PrimitiveTy::F64,
            shape: smallvec![3],
        }
    }

    fn component_value<'a>(&self) -> crate::ComponentValue<'a> {
        let arr = array![self.anchor.x, self.anchor.y, self.anchor.z].into_dyn();
        ComponentValue::F64(CowArray::from(arr))
    }

    fn from_component_value(value: crate::ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let crate::ComponentValue::F64(arr) = value else {
            return None;
        };
        if arr.shape() != [3] {
            return None;
        }
        let arr = arr.into_dimensionality::<Ix1>().ok()?;
        let arr = arr.as_slice()?;
        Some(TraceAnchor {
            anchor: Vector3::new(arr[0], arr[1], arr[2]),
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::UnitQuaternion;

    use crate::Component;

    use super::*;

    #[test]
    fn test_world_pos() {
        let world_pos = WorldPos {
            att: *UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2),
            pos: Vector3::new(1.0, 2.0, 3.0),
        };
        let val = world_pos.component_value();
        let world_pos_2 = WorldPos::from_component_value(val).unwrap();
        assert_eq!(world_pos, world_pos_2);
    }
}
