#![doc(html_root_url = "https://docs.rs/bevy_geo_frames/0.1.0")]
// #![doc = include_str!("../README.md")]
//#![forbid(missing_docs)]
#[cfg(feature = "bevy")]
mod geo;
#[cfg(feature = "bevy")]
pub use geo::*;

/// Coordinate frames used in the sim.
///
/// Units: meters, seconds.
/// Bevy world: +X=East, +Y=Up, +Z=South (so North = -Z).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "strum",
    derive(strum_macros::IntoStaticStr, strum_macros::EnumString)
)]

#[cfg_attr(feature = "bevy", derive(bevy::prelude::Reflect))]
pub enum GeoFrame {
    /// East-North-Up: +X=East, +Y=North, +Z=Up
    ENU,
    /// North-East-Down: +X=North, +Y=East, +Z=Down
    NED,
    /// Earth-Centered Earth-Fixed
    /// +X through (lat=0, lon=0) equator
    /// +Y through (lat=0, lon=90°E) equator
    /// +Z through North Pole
    ECEF,
    // Leaving out these time-dependent coordinate frames for the moment.

    // /// Earth-Centered Inertial
    // /// +X to vernal equinox, +Y 90°E, +Z North Pole
    // ECI,
    // /// Geocentric Celestial Reference Frame (inertial, J2000)
    // /// Sometimes called the International Celestial Reference Frame (ICRF)
    // /// Approximated as ECI here.
    // GCRF,
}

#[cfg(feature = "big_space")]
pub mod big_space;

pub mod prelude;
