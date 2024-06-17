//! Provides a Matrix type alias with convenience functions for converting to various representations.
use crate::{DefaultRepr, Tensor};
use nalgebra::Const;

/// Type alias for a tensor that specifically represents a matrix.
pub type Matrix<T, const R: usize, const C: usize, P = DefaultRepr> =
    Tensor<T, (Const<R>, Const<C>), P>;

// pub trait Dot<Rhs = Self> {
//     type Output;

//     fn dot(self, rhs: Rhs) -> Self::Output;
// }

// impl<T, const R: usize, const C: usize> Dot for Matrix<T, R, C, Op>
// where
//     T: NativeType + NalgebraScalar + ArrayElement,
// {
//     type Output = Matrix<T, R, C, Op>;

//     fn dot(self, rhs: Self) -> Self::Output {
//         let inner = Noxpr::dot(self.inner, &rhs.inner);
//         Matrix {
//             inner,
//             phantom: PhantomData,
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use nalgebra::{matrix, vector, ArrayStorage};

    use crate::{Client, CompFn, ToHost, Vector};

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
}
