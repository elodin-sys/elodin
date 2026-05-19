//! Dev-only fallback spatial layer for builds without `big_space`.
//!
//! All grid-cell coordinates collapse to a single cell. This keeps the editor
//! type-compatible for no-default-feature builds, but it is not a product mode.

use bevy::{math::DVec3, prelude::*};

pub type WithoutFloatingOrigin = ();
pub type WithFloatingOrigin = ();

