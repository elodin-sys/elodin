#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod buf;
pub mod com_de;
pub mod error;
pub mod table;
pub mod types;
