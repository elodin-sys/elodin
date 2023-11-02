use nalgebra::{
    ArrayStorage, Dyn, Matrix, MatrixView, Quaternion, UnitQuaternion, Vector6, U1, U6, U7,
};

use crate::tree::{Joint, JointType};

use super::SpatialPos;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GeneralizedPos {
    pub(crate) dof: u8,
    pub(crate) is_quat: bool,
    pub(crate) inner: Matrix<f64, U7, U1, ArrayStorage<f64, 7, 1>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GeneralizedMotion {
    pub(crate) dof: u8,
    pub(crate) inner: Vector6<f64>,
}

impl GeneralizedMotion {
    pub fn vector(&self) -> MatrixView<f64, Dyn, Dyn, U1, U6> {
        self.inner.view((0, 0), (self.dof as usize, 1))
    }

    pub fn integrate(&mut self, accel: &GeneralizedMotion, dt: f64) {
        assert_eq!(
            self.dof, accel.dof,
            "accel and vel must have same num of dofs"
        );
        self.inner += dt * accel.inner;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GeneralizedForce {
    pub(crate) dof: u8,
    pub(crate) inner: Vector6<f64>,
}

impl GeneralizedForce {
    pub fn vector(&self) -> MatrixView<f64, Dyn, Dyn, U1, U6> {
        self.inner.view((0, 0), (self.dof as usize, 1))
    }
}

impl GeneralizedPos {
    pub fn integrate(&mut self, vel: &GeneralizedMotion, dt: f64) {
        let (mut pos, vel) = if self.is_quat {
            assert!(
                self.dof >= 4,
                "quaternions coords must have at least 4 dofs"
            );
            assert!(
                vel.dof >= 3,
                "vel must have at least 3 dofs when integrating quat coord"
            );
            let quat = self.inner.fixed_view::<4, 1>(0, 0);
            let quat = Quaternion::from_vector(quat.into_owned());
            let new_quat =
                quat + dt * 0.5 * Quaternion::new(0., vel.inner.x, vel.inner.y, vel.inner.z) * quat;
            let new_quat = new_quat.normalize();

            let mut view = self.inner.fixed_view_mut::<4, 1>(0, 0);
            view.copy_from(new_quat.as_vector());
            (
                self.inner.view_mut((4, 0), (self.dof as usize - 4, 1)),
                vel.inner.view((3, 0), (vel.dof as usize - 3, 1)),
            )
        } else {
            assert_eq!(self.dof, vel.dof, "vel and pos must have same num of dofs");
            (
                self.inner.view_mut((0, 0), (self.dof as usize, 1)),
                vel.inner.view((0, 0), (self.dof as usize, 1)),
            )
        };
        pos += dt * vel;
    }

    pub fn to_spatial(&self, joint: &Joint) -> SpatialPos {
        match joint.joint_type {
            JointType::Free => {
                assert_eq!(self.dof, 7, "free joints need 7 dofs");
                let att = self.inner.fixed_view::<4, 1>(0, 0).into_owned();
                let att = Quaternion::from_vector(att);
                let att = UnitQuaternion::new_normalize(att);
                let pos = self.inner.fixed_view::<3, 1>(4, 0).into_owned();
                SpatialPos { pos, att }
            }
            JointType::Revolute { axis } => {
                assert_eq!(self.dof, 1, "revolute joints should have a single dof");
                let angle = self.inner[0];
                let att = UnitQuaternion::from_axis_angle(&axis, angle);
                SpatialPos {
                    pos: joint.pos,
                    att,
                }
            }
            JointType::Sphere => {
                assert_eq!(self.dof, 4, "sphere joints need 4 dofs");
                let att = self.inner.fixed_view::<4, 1>(0, 0).into_owned();
                let att = Quaternion::from_vector(att);
                let att = UnitQuaternion::new_normalize(att);
                SpatialPos {
                    pos: joint.pos,
                    att,
                }
            }
            JointType::Fixed => {
                assert_eq!(self.dof, 0, "fixed joints have 0 dofs");
                SpatialPos::default()
            }
        }
    }
}
