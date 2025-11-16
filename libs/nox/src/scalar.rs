//! Provides functionality for managing scalar tensors, including operations and transformations between host and client representations.
use crate::DefaultRepr;
use crate::{ScalarDim, Tensor};

/// Type alias for a scalar tensor with a specific type `T`, an underlying representation `P`.
pub type Scalar<T, P = DefaultRepr> = Tensor<T, ScalarDim, P>;

#[cfg(all(test, feature = "noxpr"))]
mod tests {
    use crate::{Client, CompFn};

    use super::*;

    #[test]
    fn test_sqrt_log_opt() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Scalar<f32>| a.sqrt().log()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, Scalar::from(core::f32::consts::PI))
            .unwrap()
            .to_host();
        assert_eq!(out, 0.5723649.into());
    }
}
