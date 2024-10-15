use core::ops::Add;

use crate::{Field, Matrix, OwnedRepr, Quaternion, RealField, Scalar, Vector};

/// Modified Rodrigues Parameters
pub struct MRP<T: Field, R: OwnedRepr>(pub Vector<T, 3, R>);

impl<T: Field, R: OwnedRepr> Default for MRP<T, R> {
    fn default() -> Self {
        MRP(Vector::zeros())
    }
}

impl<'a, T: Field, R: OwnedRepr> From<&'a Quaternion<T, R>> for MRP<T, R> {
    fn from(quat: &'a Quaternion<T, R>) -> Self {
        let w = quat.0.get(3);
        let vec: Vector<T, 3, R> = quat.0.fixed_slice(&[0]);
        let inner = vec / (w + T::one());
        MRP(inner)
    }
}

impl<T: Field, R: OwnedRepr> From<Quaternion<T, R>> for MRP<T, R> {
    fn from(quat: Quaternion<T, R>) -> Self {
        MRP::from(&quat)
    }
}

impl<T: RealField, R: OwnedRepr> MRP<T, R> {
    /// Constructs a new MRP from individual scalar components.
    pub fn new(
        x: impl Into<Scalar<T, R>>,
        y: impl Into<Scalar<T, R>>,
        z: impl Into<Scalar<T, R>>,
    ) -> Self {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let inner = Vector::from_arr([x, y, z]);
        MRP(inner)
    }

    /// Returns the four parts (components) of the quaternion as scalars.
    pub fn parts(&self) -> [Scalar<T, R>; 3] {
        self.0.parts()
    }

    pub fn from_rot_matrix(mat: Matrix<T, 3, 3, R>) -> Self {
        // source: Analytical Mechanics of Space Systems, Fourth Edition, Hanspeter Schaub, John L. Junkins
        // eq (3.154)
        let trace = mat.get([0, 0]) + mat.get([1, 1]) + mat.get([2, 2]);
        let zeta = (T::one() + trace).sqrt();
        let vec = Vector::from_scalars([
            mat.get([1, 2]) - mat.get([2, 1]),
            mat.get([2, 0]) - mat.get([0, 2]),
            mat.get([0, 1]) - mat.get([1, 0]),
        ]);
        let inner = -vec / (&zeta * (&zeta + T::two()));
        MRP(inner)
    }
}

impl<T: RealField, R: OwnedRepr> Add for MRP<T, R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        MRP(self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: OwnedRepr> Add<MRP<T, R>> for &'a MRP<T, R> {
    type Output = MRP<T, R>;

    fn add(self, rhs: MRP<T, R>) -> Self::Output {
        MRP(&self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: OwnedRepr> Add<&'a MRP<T, R>> for MRP<T, R> {
    type Output = MRP<T, R>;

    fn add(self, rhs: &'a MRP<T, R>) -> Self::Output {
        MRP(self.0 + &rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{array, tensor, ArrayRepr};

    use super::*;

    #[test]
    fn test_mrp_from_rot() {
        // value from mrp2rot.m - https://github.com/mlourakis/MRPs/blob/master/mrp2rot.m
        let rot = tensor![
            [-0.3061, 0.4898, 0.8163],
            [0.8163, -0.3061, 0.4898],
            [0.4898, 0.8163, -0.3061]
        ];
        let mrp: MRP<f64, ArrayRepr> = MRP::from_rot_matrix(rot);
        assert_relative_eq!(mrp.0.inner(), &array![0.5, 0.5, 0.5], epsilon = 1e-3);
        // value from mrp2rot.m
        let rot = tensor![
            [1.0, 0.0, 0.0],
            [0.0, -0.7041, 0.7101],
            [0.0, -0.7101, -0.7041],
        ];
        let mrp: MRP<f64, ArrayRepr> = MRP::from_rot_matrix(rot);
        assert_relative_eq!(mrp.0.inner(), &array![-0.6666, 0.0, 0.0], epsilon = 1e-4);
        // value from rot2mrp
    }
}
