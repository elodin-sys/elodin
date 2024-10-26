#![allow(dead_code)]
// sourced from https://github.com/rust-lang/futures-rs/blob/master/futures-task/src/noop_waker.rs
//! Utilities for creating zero-cost wakers that don't do anything.

use core::ptr::null;
use core::task::{RawWaker, RawWakerVTable, Waker};

unsafe fn noop_clone(_data: *const ()) -> RawWaker {
    noop_raw_waker()
}

unsafe fn noop(_data: *const ()) {}

const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

const fn noop_raw_waker() -> RawWaker {
    RawWaker::new(null(), &NOOP_WAKER_VTABLE)
}

#[inline]
pub fn noop_waker() -> Waker {
    unsafe { Waker::from_raw(noop_raw_waker()) }
}
