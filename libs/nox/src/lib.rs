#![allow(clippy::arc_with_non_send_sync)]

mod builder;
mod client;
mod comp;
mod comp_fn;
mod constant;
mod error;
mod exec;
mod fields;
mod matrix;
mod noxpr;
mod param;
mod quaternion;
mod scalar;
mod spatial;
mod tensor;
mod transfer;
mod vector;

#[cfg(feature = "jax")]
pub mod jax;

pub use builder::*;
pub use client::*;
pub use comp::*;
pub use comp_fn::*;
pub use constant::*;
pub use error::*;
pub use exec::*;
pub use fields::*;
pub use matrix::*;
pub use noxpr::*;
pub use param::*;
pub use quaternion::*;
pub use scalar::*;
pub use spatial::*;
pub use tensor::*;
pub use transfer::*;
pub use vector::*;

pub use nalgebra;
pub use xla;
