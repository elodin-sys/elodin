#![doc = include_str!("README.md")]
pub mod batch_dfs;
// We only use the batch module for testing purposes.
#[cfg(test)]
mod batch;
mod builder;
mod comp;
mod comp_fn;
mod node;
mod repr;
mod scalar;
mod tensor;
mod vector;

#[cfg(feature = "jax")]
mod py;

pub use builder::*;
pub use comp::*;
pub use comp_fn::*;
pub use node::*;
pub use repr::*;
pub use tensor::*;
