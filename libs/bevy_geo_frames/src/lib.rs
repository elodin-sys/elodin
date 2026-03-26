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
/// Bevy world: +X=East, +Y=Up, +Z=South
///
/// Note: there was a temptation to codify Bevy's coordinate system as East
/// (+X), Up (+Y), South (+Z) or EUS, but that's not a standard coordinate
/// system and Bevy's coordinates aren't actually related to the cardinal
/// directions. To enforce it as a special case it has `to_bevy()` and
/// `from_bevy()` but is not itself codified as a coordinate system.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Reflect))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "strum",
    derive(strum_macros::IntoStaticStr, strum_macros::EnumString)
)]
#[cfg_attr(feature = "default_enu", derive(Default))]
pub enum GeoFrame {
    #[cfg_attr(feature = "default_enu", default)]
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

    // Since this is aerospace, we may want a solar system centered coordinate
    // system like HCI or BCRS / ICRF at some point, which is fine but perhaps
    // breaks the naming for it being a "Geo" or "Earth" frame.
}

/// Provide a means of specifying a default possibly. Meant to be used with
/// `Option<GeoFrame>`.
pub trait OrDefault {
    fn or_default(self) -> Self;
}

impl OrDefault for Option<GeoFrame> {
    /// If [GeoFrame] impls `Default` and its given `None`, it will return the
    /// `Default`.
    ///
    /// If [GeoFrame] does not impl `Default`, this is an identity function.
    fn or_default(self) -> Option<GeoFrame> {
        #[cfg(feature = "default_enu")]
        { self.or(Some(GeoFrame::default())) }
        #[cfg(not(feature = "default_enu"))]
        { self }
    }
}

#[cfg(feature = "big_space")]
pub mod big_space;

pub mod prelude;
