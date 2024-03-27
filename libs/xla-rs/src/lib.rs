#![recursion_limit = "2048"]
//! Rust bindings for XLA (Accelerated Linear Algebra).
//!
//! [XLA](https://www.tensorflow.org/xla) is a compiler library for Machine Learning. It can be
//! used to run models efficiently on GPUs, TPUs, and on CPUs too.
//!
//! [`XlaOp`]s are used to build a computation graph. This graph can built into a
//! [`XlaComputation`]. This computation can then be compiled into a [`PjRtLoadedExecutable`] and
//! then this executable can be run on a [`PjRtClient`]. [`Literal`] values are used to represent
//! tensors in the host memory, and [`PjRtBuffer`] represent views of tensors/memory on the
//! targeted device.
//!
//! The following example illustrates how to build and run a simple computation.
//! ```ignore
//! // Create a CPU client.
//! let client = xla::PjRtClient::cpu()?;
//!
//! // A builder object is used to store the graph of XlaOp.
//! let builder = xla::XlaBuilder::new("test-builder");
//!
//! // Build a simple graph summing two constants.
//! let cst20 = xla_builder.constant_r0(20f32);
//! let cst22 = xla_builder.constant_r0(22f32);
//! let sum = (cst20 + cst22)?;
//!
//! // Create a computation from the final node.
//! let sum= sum.build()?;
//!
//! // Compile this computation for the target device and then execute it.
//! let result = client.compile(&sum)?;
//! let result = &result.execute::<xla::Literal>(&[])?;
//!
//! // Retrieve the resulting value.
//! let result = result[0][0].to_literal_sync()?.to_vec::<f32>()?;
//! ```

mod buffer;
mod builder;
mod client;
mod computation;
mod element_type;
mod error;
mod executable;
mod hlo_module;
mod literal;
mod native_type;
mod op;
mod shape;

pub use buffer::*;
pub use builder::*;
pub use client::*;
pub use computation::*;
pub use element_type::*;
pub use error::{Error, Result, Status};
pub use executable::*;
pub use hlo_module::*;
pub use literal::*;
pub use native_type::*;
pub use op::*;
pub use shape::*;

extern crate lapack_src as _;

#[derive(Debug, Copy, Clone)]
pub enum TfLogLevel {
    Info,
    Warning,
    Error,
    Fatal,
}

impl TfLogLevel {
    fn as_env_variable_str(&self) -> &'static str {
        match self {
            Self::Info => "0",
            Self::Warning => "1",
            Self::Error => "2",
            Self::Fatal => "3",
        }
    }
}

pub fn set_tf_min_log_level(log_level: TfLogLevel) {
    std::env::set_var("TF_CPP_MIN_LOG_LEVEL", log_level.as_env_variable_str())
}

#[cfg(test)]
mod tests;
