//! Provides the `Comp` struct which encapsulates an XLA computation with type information.
use core::marker::PhantomData;

use crate::{Client, Exec};

/// Represents an XLA computation, parameterized over input and return types.
pub struct Comp<T, R> {
    pub comp: xla::XlaComputation,
    pub(crate) phantom: PhantomData<(T, R)>,
}

impl<T, R> Comp<T, R> {
    /// Converts the computation to a human-readable HLO (High Level Optimizer) text format.
    pub fn to_hlo_text(&self) -> Result<String, xla::Error> {
        self.comp.to_hlo_text()
    }

    /// Compiles the XLA computation into a kernel.
    pub fn compile(&self, client: &Client) -> Result<Exec<T, R>, xla::Error> {
        let exec = client.compile(&self.comp)?;
        Ok(Exec {
            exec,
            phantom: PhantomData,
        })
    }
}
