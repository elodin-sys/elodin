use nalgebra::{ArrayStorage, Dyn, Matrix, MatrixView, Vector3, Vector6, U1, U6};
use std::ops::Mul;

use super::{GeneralizedForce, GeneralizedMotion, SpatialForce, SpatialMotion, Transpose};

#[derive(Debug)]
pub struct SpatialSubspace {
    pub(crate) cols: usize,
    pub(crate) inner: Matrix<f64, U6, U6, ArrayStorage<f64, 6, 6>>,
}

impl SpatialSubspace {
    pub fn transpose(self) -> Transpose<Self> {
        Transpose(self)
    }

    pub fn matrix(&self) -> MatrixView<'_, f64, Dyn, Dyn, U1, U6> {
        self.inner.view((0, 0), (6, self.cols as usize))
    }
}

impl Mul<SpatialForce> for Transpose<SpatialSubspace> {
    type Output = GeneralizedForce;

    fn mul(self, rhs: SpatialForce) -> Self::Output {
        let mat = self.0.matrix();
        let out = mat.transpose() * rhs.vector();
        let dof = out.shape().0 as u8;
        let mut inner = Vector6::zeros();
        for (row, o) in out.row_iter().zip(inner.iter_mut()) {
            *o = row.to_scalar();
        }
        GeneralizedForce { dof, inner }
    }
}

impl Mul<GeneralizedMotion> for SpatialSubspace {
    type Output = SpatialMotion;

    fn mul(self, rhs: GeneralizedMotion) -> Self::Output {
        self * &rhs
    }
}

impl<'a> Mul<&'a GeneralizedMotion> for SpatialSubspace {
    type Output = SpatialMotion;

    fn mul(self, rhs: &'a GeneralizedMotion) -> Self::Output {
        let out = self.matrix() * rhs.vector();
        let vel = Vector3::new(out[0], out[1], out[2]);
        let ang_vel = Vector3::new(out[3], out[4], out[5]);
        SpatialMotion { vel, ang_vel }
    }
}
