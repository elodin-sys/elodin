//! Provides functionality for managing scalar tensors.
use crate::DefaultRepr;
use crate::{ScalarDim, Tensor};

/// Type alias for a scalar tensor with a specific type `T`, an underlying representation `P`.
pub type Scalar<T, P = DefaultRepr> = Tensor<T, ScalarDim, P>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_log_opt() {
        let a = Scalar::from(core::f32::consts::PI);
        let result = a.sqrt().log();
        // ln(sqrt(PI)) ≈ 0.5723649
        assert!((result.into_buf() - 0.5723649f32).abs() < 0.0001f32);
    }
}
