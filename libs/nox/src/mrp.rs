use std::ops::Add;

use crate::{Field, Quaternion, RealField, Repr, Scalar, Vector};

/// Modified Rodrigues Parameters
pub struct MRP<T: Field, R: Repr>(pub Vector<T, 3, R>);

impl<T: Field, R: Repr> Default for MRP<T, R> {
    fn default() -> Self {
        MRP(Vector::zeros())
    }
}

impl<'a, T: Field, R: Repr> From<&'a Quaternion<T, R>> for MRP<T, R> {
    fn from(quat: &'a Quaternion<T, R>) -> Self {
        let w = quat.0.get(3);
        let vec: Vector<T, 3, R> = quat.0.fixed_slice(&[0]);
        let inner = vec / (w + T::one());
        MRP(inner)
    }
}

impl<T: Field, R: Repr> From<Quaternion<T, R>> for MRP<T, R> {
    fn from(quat: Quaternion<T, R>) -> Self {
        MRP::from(&quat)
    }
}

impl<T: Field, R: Repr> MRP<T, R> {
    /// Constructs a new MRP from individual scalar components.
    pub fn new(
        x: impl Into<Scalar<T, R>>,
        y: impl Into<Scalar<T, R>>,
        z: impl Into<Scalar<T, R>>,
    ) -> Self {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let inner = Vector::from_arr([&x, &y, &z]);
        MRP(inner)
    }

    /// Returns the four parts (components) of the quaternion as scalars.
    pub fn parts(&self) -> [Scalar<T, R>; 3] {
        self.0.parts()
    }
}

impl<T: RealField, R: Repr> Add for MRP<T, R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        MRP(self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: Repr> Add<MRP<T, R>> for &'a MRP<T, R> {
    type Output = MRP<T, R>;

    fn add(self, rhs: MRP<T, R>) -> Self::Output {
        MRP(&self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: Repr> Add<&'a MRP<T, R>> for MRP<T, R> {
    type Output = MRP<T, R>;

    fn add(self, rhs: &'a MRP<T, R>) -> Self::Output {
        MRP(self.0 + &rhs.0)
    }
}
