use bevy::prelude::{Quat, Vec3};
use nalgebra::{UnitQuaternion, Vector3};

use super::SpatialTransform;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpatialPos {
    pub pos: Vector3<f64>,
    pub att: UnitQuaternion<f64>,
}

impl SpatialPos {
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
}
