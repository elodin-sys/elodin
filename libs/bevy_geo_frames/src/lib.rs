#![doc(html_root_url = "https://docs.rs/bevy_geo_frames/0.1.0")]
// #![doc = include_str!("../README.md")]
//#![forbid(missing_docs)]

pub use map_3d::Ellipsoid;

mod transforms;

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

/// Where the Bevy world origin lives on Earth.
///
/// Used to turn ECEF positions into local ENU, then ENU → Bevy.
#[derive(Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Reflect))]
pub struct GeoOrigin {
    /// Geodetic latitude [rad]
    pub latitude: f64,
    /// Geodetic longitude [rad]
    pub longitude: f64,
    /// Altitude above mean radius [m]
    pub altitude: f64,
    #[cfg_attr(feature = "bevy", reflect(ignore))]
    /// Planet/body shape model (currently used primarily for reference radius).
    pub ellipsoid: Ellipsoid,
}

impl GeoOrigin {
    /// Uses default Earth radius.
    pub fn new_from_degrees(latitude_deg: f64, longitude_deg: f64, altitude: f64) -> Self {
        let latitude = latitude_deg.to_radians();
        let longitude = longitude_deg.to_radians();
        Self {
            latitude,
            longitude,
            altitude,
            ..Default::default()
        }
    }

    /// Provide an ellipsoid.
    pub fn with_ellipsoid(mut self, shape: Ellipsoid) -> Self {
        self.ellipsoid = shape;
        self
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Reflect))]
pub enum RotationKind {
    #[default]
    /// Re-express the local→frame attitude as a rotation operator in Bevy
    /// (`bevy_R * att * bevy_R⁻¹`). Identity attitude therefore leaves the
    /// mesh in its authored Bevy / glTF axes (Y-up).
    Relative,
    /// Compose the local→frame attitude with the frame→Bevy basis
    /// (`bevy_R * att`). Identity attitude aligns mesh local axes with the
    /// schematic frame (ENU: +X east, +Y north, +Z up).
    Absolute,
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
        {
            self.or(Some(GeoFrame::default()))
        }
        #[cfg(not(feature = "default_enu"))]
        {
            self
        }
    }
}

#[cfg(feature = "big_space")]
pub mod big_space;

#[cfg(feature = "bevy")]
pub mod solar;

pub mod prelude;
