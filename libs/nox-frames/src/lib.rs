#[cfg(feature = "earth")]
pub mod earth;
#[cfg(feature = "earth")]
pub mod iers;

use core::ops::Mul;

use nox::{
    ArrayDim, Const, DefaultRepr, Dim, DotDim, Elem, Field, Matrix, OwnedRepr, RealField,
    ShapeConstraint, Tensor,
};

pub trait Frame {}

pub struct Transform<T: Field, A: Frame, B: Frame, R: OwnedRepr = DefaultRepr> {
    dcm: Matrix<T, 3, 3, R>,
    phantom: std::marker::PhantomData<(A, B)>,
}

impl<T: Field, A: Frame, B: Frame, R: OwnedRepr> From<Matrix<T, 3, 3, R>>
    for Transform<T, A, B, R>
{
    fn from(val: Matrix<T, 3, 3, R>) -> Self {
        Transform {
            dcm: val,
            phantom: std::marker::PhantomData,
        }
    }
}

pub type DCM<T, A, B, R = DefaultRepr> = Transform<T, A, B, R>;

impl<T: RealField, A: Frame, B: Frame, R: OwnedRepr> DCM<T, A, B, R> {
    pub fn dot<D2>(
        &self,
        right: &Tensor<T, D2, R>,
    ) -> Tensor<T, <ShapeConstraint as DotDim<(Const<3>, Const<3>), D2>>::Output, R>
    where
        T: Field + Elem,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<(Const<3>, Const<3>), D2>,
        <ShapeConstraint as DotDim<(Const<3>, Const<3>), D2>>::Output: Dim + ArrayDim,
    {
        self.dcm.dot(right)
    }
}

impl<T: RealField, A: Frame, B: Frame, C: Frame, R: OwnedRepr> Mul<DCM<T, A, B, R>>
    for DCM<T, B, C, R>
{
    type Output = DCM<T, A, C, R>;

    fn mul(self, rhs: DCM<T, A, B, R>) -> Self::Output {
        DCM {
            dcm: self.dcm.dot(&rhs.dcm),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: RealField, A: Frame, B: Frame, R: OwnedRepr> Transform<T, A, B, R> {
    fn inverse(&self) -> Transform<T, B, A, R> {
        Transform {
            dcm: self.dcm.transpose(),
            phantom: std::marker::PhantomData,
        }
    }
}
