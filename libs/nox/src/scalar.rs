//! Provides functionality for managing scalar tensors, including operations and transformations between host and client representations.
use crate::DefaultRepr;
use crate::{ScalarDim, Tensor};

/// Type alias for a scalar tensor with a specific type `T`, an underlying representation `P`.
pub type Scalar<T, P = DefaultRepr> = Tensor<T, ScalarDim, P>;
