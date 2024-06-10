#![recursion_limit = "1024"]
#![allow(clippy::arc_with_non_send_sync)]

mod array;
mod builder;
mod client;
mod comp;
mod comp_fn;
mod error;
mod exec;
mod fields;
mod matrix;
mod mrp;
mod noxpr;
mod quaternion;
mod repr;
mod scalar;
mod spatial;
mod tensor;
mod transfer;
mod vector;

#[cfg(feature = "jax")]
pub mod jax;

pub use array::*;
pub use builder::*;
pub use client::*;
pub use comp::*;
pub use comp_fn::*;
pub use error::*;
pub use exec::*;
pub use fields::*;
pub use matrix::*;
pub use mrp::*;
pub use noxpr::*;
pub use quaternion::*;
pub use repr::*;
pub use scalar::*;
pub use spatial::*;
pub use tensor::*;
pub use transfer::*;
pub use vector::*;

pub use nalgebra;
pub use xla;
