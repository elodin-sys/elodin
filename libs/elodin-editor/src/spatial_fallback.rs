//! Dev-only fallback spatial layer for builds without `big_space`.
//!
//! With the `big_space` feature gated to opt-in, the editor mostly avoids
//! references to floating-origin types altogether. The two aliases below
//! still let a handful of query filters expand to a no-op in this mode.

pub type WithoutFloatingOrigin = ();
#[allow(dead_code)]
pub type WithFloatingOrigin = ();
