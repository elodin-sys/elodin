//! Provides a Matrix type alias with convenience functions for converting to various representations.
use crate::{DefaultRepr, Error, RealField, Repr, Tensor};
use nalgebra::Const;

/// Type alias for a tensor that specifically represents a matrix.
pub type Matrix<T, const R: usize, const C: usize, P = DefaultRepr> =
    Tensor<T, (Const<R>, Const<C>), P>;

impl<T: RealField, const N: usize, R: Repr> Matrix<T, N, N, R> {
    pub fn try_inverse(&self) -> Result<Self, Error> {
        match N {
            2 => {
                let m11 = self.get((0, 0));
                let m12 = self.get((0, 1));
                let m21 = self.get((1, 0));
                let m22 = self.get((1, 1));

                let determinant = &m11 * &m22 - &m21 * &m12;

                let parts = [
                    m22 / &determinant,
                    -m12 / &determinant,
                    -m21 / &determinant,
                    m11 / &determinant,
                ];
                Ok(Tensor::from_scalars(parts))
            }
            3 => {
                let m11 = self.get((0, 0));
                let m12 = self.get((0, 1));
                let m13 = self.get((0, 2));

                let m21 = self.get((1, 0));
                let m22 = self.get((1, 1));
                let m23 = self.get((1, 2));

                let m31 = self.get((2, 0));
                let m32 = self.get((2, 1));
                let m33 = self.get((2, 2));

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
                Ok(Tensor::from_scalars(parts))
            }
            _ => R::try_lu_inverse(&self.inner).map(Tensor::from_inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{matrix, vector, ArrayStorage};

    use crate::{tensor, Client, CompFn, ToHost, Vector};

    use super::*;

    #[test]
    fn test_add() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a + b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![3.0, 5.0]);
    }

    #[test]
    fn test_sub() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a - b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![-1.0, -1.0]);
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
                matrix![1.0f32, 2.0; 2.0, 3.0],
                matrix![2.0, 3.0; 4.0, 5.0],
            )
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![2., 6.; 8., 15.]);
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
            .run(&client, matrix![1.0f32, 2.0, 3.0, 4.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![3.0])
    }

    #[test]
    fn test_index() {
        let client = Client::cpu().unwrap();
        fn index(mat: Matrix<f32, 4, 3>, index: Vector<u32, 2>) -> Matrix<f32, 2, 3> {
            mat.index(index)
        }
        let comp = index.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let a = matrix![0., 1., 2.;
                        2., 3., 4.;
                        4., 5., 6.;
                        7., 8., 9.];
        let out: nalgebra::Matrix<f32, Const<2>, Const<3>, ArrayStorage<f32, 2, 3>> =
            exec.run(&client, a, vector![1, 2]).unwrap().to_host();
        assert_eq!(
            out,
            matrix![2., 3., 4.;
                    4., 5., 6.]
        );

        let out: nalgebra::Matrix<f32, Const<2>, Const<3>, ArrayStorage<f32, 2, 3>> =
            exec.run(&client, a, vector![0, 3]).unwrap().to_host();
        assert_eq!(
            out,
            matrix![0., 1., 2.;
                    7., 8., 9.]
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
}
