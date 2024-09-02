use ndarray::{array, CowArray, Ix1};
use nox::{ArrayRepr, Quaternion, Tensor, Vector3};
use smallvec::smallvec;

use crate::{ComponentType, ComponentValue, PrimitiveTy, ValueRepr};

#[cfg(feature = "bevy")]
mod bevy_conv;

mod camera;
mod metadata;
mod pbr;
mod viewer;

pub use camera::*;
pub use metadata::*;
pub use pbr::*;
pub use viewer::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct WorldPos {
    pub att: Quaternion<f64, ArrayRepr>,
    pub pos: Vector3<f64, ArrayRepr>,
}

impl crate::Component for WorldPos {
    const NAME: &'static str = "world_pos";
    const ASSET: bool = false;

    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: PrimitiveTy::F64,
            shape: smallvec![7],
        }
    }
}

impl ValueRepr for WorldPos {
    type ValueDim = ndarray::Ix1;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        let [qx, qy, qz, w] = self.att.parts().map(Tensor::into_buf);
        let [x, y, z] = self.pos.parts().map(Tensor::into_buf);
        let arr = array![qx, qy, qz, w, x, y, z];
        ComponentValue::F64(CowArray::from(arr))
    }

    fn from_component_value<D: ndarray::Dimension>(
        value: crate::ComponentValue<'_, D>,
    ) -> Option<Self>
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_world_pos() {
        let world_pos = WorldPos {
            att: Quaternion::from_axis_angle(Vector3::y_axis(), std::f64::consts::FRAC_PI_2),
            pos: Vector3::new(1.0, 2.0, 3.0),
        };
        let val = world_pos.component_value();
        let world_pos_2 = WorldPos::from_component_value(val).unwrap();
        assert_eq!(world_pos, world_pos_2);
    }
}
