#![recursion_limit = "1024"]
#![allow(clippy::arc_with_non_send_sync)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod array;
mod dim;
mod error;
mod fields;
mod literal;
mod matrix;
mod mrp;
mod quaternion;
mod repr;
mod scalar;
mod spatial;
mod tensor;
mod vector;

pub mod utils;

pub use array::prelude::*;
pub use dim::*;
pub use error::*;
pub use fields::*;
pub use literal::*;
pub use matrix::*;
pub use mrp::*;
pub use quaternion::*;
pub use repr::*;
pub use scalar::*;
pub use spatial::*;
pub use tensor::*;
pub use vector::*;

#[cfg(feature = "jax")]
pub mod jax;

#[cfg(feature = "noxpr")]
pub mod noxpr;
#[cfg(feature = "noxpr")]
pub use noxpr::*;

#[doc(hidden)]
pub mod doctest;

#[cfg(feature = "noxpr")]
pub use crate::noxpr::Op as DefaultRepr;

#[cfg(not(feature = "noxpr"))]
pub use crate::array::ArrayRepr as DefaultRepr;
