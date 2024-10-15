//! Provides a Matrix type alias with convenience functions for converting to various representations.
use core::marker::PhantomData;

use crate::Const;
use crate::{
    DefaultRepr, Dim, Error, NonScalarDim, NonTupleDim, OwnedRepr, RealField, SquareDim, Tensor,
    Vector,
};

/// Type alias for a tensor that specifically represents a matrix.
pub type Matrix<T, const R: usize, const C: usize, P = DefaultRepr> =
    Tensor<T, (Const<R>, Const<C>), P>;

pub type Matrix3<T, R = DefaultRepr> = Matrix<T, 3, 3, R>;
pub type Matrix3x6<T, R = DefaultRepr> = Matrix<T, 3, 6, R>;
pub type Matrix4<T, R = DefaultRepr> = Matrix<T, 4, 4, R>;
pub type Matrix5<T, R = DefaultRepr> = Matrix<T, 5, 5, R>;
pub type Matrix6<T, R = DefaultRepr> = Matrix<T, 6, 6, R>;
pub type Matrix6x3<T, R = DefaultRepr> = Matrix<T, 6, 3, R>;

impl<const R: usize, const C: usize, T: RealField, Rep: OwnedRepr> Matrix<T, R, C, Rep> {
    pub fn from_rows(rows: [Vector<T, C, Rep>; R]) -> Self {
        let arr = rows.map(|x| x.inner);
        let inner = Rep::concat_many(arr, 0);
        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: RealField, D: Dim, R: OwnedRepr> Tensor<T, (D, D), R>
where
    D: NonTupleDim + NonScalarDim,
    (D, D): Dim + SquareDim<SideDim = D>,
{
    pub fn try_inverse(&self) -> Result<Self, Error> {
        let shape = R::shape(&self.inner);
        match shape.as_ref().first() {
            Some(2) => {
                let m11 = self.get([0, 0]);
                let m12 = self.get([0, 1]);
                let m21 = self.get([1, 0]);
                let m22 = self.get([1, 1]);

                let determinant = &m11 * &m22 - &m21 * &m12;

                let parts = [
                    m22 / &determinant,
                    -m12 / &determinant,
                    -m21 / &determinant,
                    m11 / &determinant,
                ];
                Ok(Tensor::from_scalars_with_shape(parts, &[2, 2]))
            }
            Some(3) => {
                let m11 = self.get([0, 0]);
                let m12 = self.get([0, 1]);
                let m13 = self.get([0, 2]);

                let m21 = self.get([1, 0]);
                let m22 = self.get([1, 1]);
                let m23 = self.get([1, 2]);

                let m31 = self.get([2, 0]);
                let m32 = self.get([2, 1]);
                let m33 = self.get([2, 2]);

                let minor_m12_m23 = &m22 * &m33 - &m32 * &m23;
                let minor_m11_m23 = &m21 * &m33 - &m31 * &m23;
                let minor_m11_m22 = &m21 * &m32 - &m31 * &m22;

                let determinant =
                    &m11 * &minor_m12_m23 - &m12 * &minor_m11_m23 + &m13 * &minor_m11_m22;

                let parts = [
                    &minor_m12_m23 / &determinant,
                    (&m13 * &m32 - &m33 * &m12) / &determinant,
                    (&m12 * &m23 - &m22 * &m13) / &determinant,
                    -&minor_m11_m23 / &determinant,
                    (&m11 * &m33 - &m31 * &m13) / &determinant,
                    (&m13 * &m21 - &m23 * &m11) / &determinant,
                    &minor_m11_m22 / &determinant,
                    (&m12 * &m31 - &m32 * &m11) / &determinant,
                    (m11 * m22 - m21 * m12) / &determinant,
                ];
                Ok(Tensor::from_scalars_with_shape(parts, &[3, 3]))
            }
            _ => R::try_lu_inverse(&self.inner).map(Tensor::from_inner),
        }
    }
}

impl<T: RealField, R: OwnedRepr> Matrix3<T, R> {
    pub fn look_at_rh(dir: impl Into<Vector<T, 3, R>>, up: impl Into<Vector<T, 3, R>>) -> Self {
        let dir = dir.into() * T::neg_one();
        let up = up.into();
        // apply gram-schmidt orthogonalization to create a rot matrix
        let z = dir.normalize();
        let x = up.cross(&z).normalize();
        let y = z.cross(&x);
        Self::from_rows([x, y, z])
    }
}

#[cfg(test)]
mod tests {

    use crate::{tensor, Client, CompFn, Vector, Vector3};

    use super::*;

    #[test]
    fn test_add() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a + b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0]], tensor![[2.0, 3.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[3.0, 5.0]]);
    }

    #[test]
    fn test_sub() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a - b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0]], tensor![[2.0, 3.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[-1.0, -1.0]]);
    }

    #[test]
    fn test_mul() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 2, 2>, b: Matrix<f32, 2, 2>| a * b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                tensor![[1.0f32, 2.0], [2.0, 3.0]],
                tensor![[2.0, 3.0], [4.0, 5.0]],
            )
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[2., 6.], [8., 15.]]);
    }

    #[test]
    fn test_fixed_slice() {
        let client = Client::cpu().unwrap();
        fn slice(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 1> {
            mat.fixed_slice::<(Const<1>, Const<1>)>(&[0, 2])
        }
        let comp = slice.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0, 3.0, 4.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[3.0]])
    }

    #[test]
    fn test_index() {
        let client = Client::cpu().unwrap();
        fn index(mat: Matrix<f32, 4, 3>, index: Vector<u32, 2>) -> Matrix<f32, 2, 3> {
            mat.index(index)
        }
        let comp = index.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let a = tensor![[0., 1., 2.], [2., 3., 4.], [4., 5., 6.], [7., 8., 9.]];
        let out = exec.run(&client, a, tensor![1, 2]).unwrap().to_host();
        assert_eq!(out, tensor![[2., 3., 4.], [4., 5., 6.]]);

        let out = exec.run(&client, a, tensor![0, 3]).unwrap().to_host();
        assert_eq!(out, tensor![[0., 1., 2.], [7., 8., 9.]]);
    }

    #[test]
    fn test_eye() {
        let client = Client::cpu().unwrap();
        let comp = (|| Matrix::<f32, 3, 3, _>::eye()).build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec.run(&client).unwrap().to_host();
        assert_eq!(
            out,
            tensor![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn test_diag() {
        let client = Client::cpu().unwrap();
        let comp = (|| Matrix::<f32, 3, 3, _>::from_diag(tensor![1.0, 4.0, 8.0].into()))
            .build()
            .unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec.run(&client).unwrap().to_host();
        assert_eq!(
            out,
            tensor![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]]
        );
    }

    #[test]
    fn test_from_rows() {
        let client = Client::cpu().unwrap();
        let comp = (|| {
            Matrix::<f32, 3, 3, _>::from_rows([
                tensor![1.0, 2.0, 3.0].into(),
                tensor![4.0, 5.0, 6.0].into(),
                tensor![7.0, 8.0, 9.0].into(),
            ])
        })
        .build()
        .unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec.run(&client).unwrap().to_host();
        assert_eq!(
            out,
            tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        );
    }

    // #[test]
    // fn test_inverse() {
    //     let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    //     let out = a.try_inverse().unwrap();
    //     assert_eq!(out, tensor![[-2.0, 1.0], [1.5, -0.5]]);
    //     let a = tensor![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    //     let out = a.try_inverse().unwrap();
    //     assert_eq!(out, a);
    // }

    #[test]
    fn test_look_at() {
        // source: nalgebra
        // `Rotation3::look_at_rh(&Vector3::new(1.0,2.0,3.0).normalize(), &Vector3::y_axis())`
        let expected = tensor![
            [
                -0.9486832980505139,
                -0.16903085094570333,
                -0.2672612419124244
            ],
            [0.0, 0.8451542547285167, -0.5345224838248488],
            [0.31622776601683794, -0.50709255283711, -0.8017837257372732]
        ]
        .transpose();
        assert_eq!(
            Matrix3::look_at_rh(Vector3::new(1.0, 2.0, 3.0).normalize(), Vector3::y_axis()),
            expected
        )
    }
}
