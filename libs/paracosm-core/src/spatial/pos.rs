use bevy::prelude::{Quat, Vec3};
use elo_conduit::{cid, ComponentType, ComponentValue};
use nalgebra::{matrix, ArrayStorage, Matrix, UnitQuaternion, Vector3, U1, U7};
use serde::{Deserialize, Serialize};

use super::SpatialTransform;

#[derive(Clone, Copy, Debug, PartialEq, Default, Deserialize, Serialize)]
pub struct SpatialPos {
    pub pos: Vector3<f64>,
    pub att: UnitQuaternion<f64>,
}

impl SpatialPos {
    pub fn new(pos: Vector3<f64>, att: UnitQuaternion<f64>) -> Self {
        Self { pos, att }
    }

    pub fn linear(pos: Vector3<f64>) -> SpatialPos {
        SpatialPos {
            pos,
            att: UnitQuaternion::identity(),
        }
    }

    #[inline]
    pub fn transform(self) -> SpatialTransform {
        SpatialTransform {
            linear: self.pos,
            angular: self.att,
        }
    }

    pub fn bevy(self, scale: f32) -> bevy::prelude::Transform {
        let SpatialPos { pos, att } = self;
        bevy::prelude::Transform {
            translation: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) * scale,
            rotation: Quat::from_xyzw(att.i as f32, att.j as f32, att.k as f32, att.w as f32),
            ..Default::default()
        }
    }

    pub fn vector(&self) -> Matrix<f64, U7, U1, ArrayStorage<f64, 7, 1>> {
        matrix![
            self.att[0];
            self.att[1];
            self.att[2];
            self.att[3];
            self.pos[0];
            self.pos[1];
            self.pos[2]
        ]
    }
}

impl elo_conduit::Component for SpatialPos {
    fn component_id() -> elo_conduit::ComponentId {
        cid!(31;spatial_pos)
    }

    fn component_type() -> elo_conduit::ComponentType {
        ComponentType::SpatialPosF64
    }

    fn component_value<'a>(&self) -> elo_conduit::ComponentValue<'a> {
        elo_conduit::ComponentValue::SpatialPosF64((*self.att, self.pos))
    }

    fn from_component_value(value: elo_conduit::ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let ComponentValue::SpatialPosF64((att, pos)) = value else {
            return None;
        };
        Some(Self {
            pos,
            att: UnitQuaternion::new_normalize(att),
        })
    }
}
