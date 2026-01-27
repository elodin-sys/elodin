//! Provides the `Comp` struct which encapsulates a compiled computation with type information.
use core::marker::PhantomData;

use crate::NoxprFn;

/// Represents a compiled computation, parameterized over input and return types.
/// The actual execution is handled by the IREE runtime in nox-py.
pub struct Comp<T, R> {
    pub func: NoxprFn,
    pub(crate) phantom: PhantomData<(T, R)>,
}
