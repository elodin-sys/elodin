//! Plane-mode ENU / NED / ECEF transforms (no Bevy ECS).
//!
//! Uses [`bevy_math`] so crates like `eql` can depend on `bevy_geo_frames`
//! without enabling the `bevy` feature.

#![allow(non_snake_case)]

use crate::{GeoFrame, GeoOrigin};
use bevy_math::{DMat3, DMat4, DVec3};

impl GeoFrame {
    /// ECEF coordinates of the geodetic origin.
    pub fn origin_ecef(origin: &GeoOrigin) -> DVec3 {
        map_3d::enu2ecef(
            0.0,
            0.0,
            0.0,
            origin.latitude,
            origin.longitude,
            origin.altitude,
            &origin.ellipsoid,
        )
        .into()
    }

    /// Provides the matrix `${ecef}_R_{from}`.
    ///
    /// See [this reference](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates).
    pub fn ecef_R_(from: &Self, origin: &GeoOrigin) -> DMat3 {
        use std::f64::consts::FRAC_PI_2;
        if *from == GeoFrame::ECEF {
            return DMat3::IDENTITY;
        }

        let ecef_R_enu = DMat3::from_rotation_z(FRAC_PI_2 + origin.longitude)
            * DMat3::from_rotation_x(FRAC_PI_2 - origin.latitude);
        match from {
            GeoFrame::ECEF => DMat3::IDENTITY,
            GeoFrame::ENU => ecef_R_enu,
            GeoFrame::NED => ecef_R_enu * Self::enu_R_ned(),
        }
    }

    #[inline]
    pub fn enu_R_ned() -> DMat3 {
        DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Z)
    }

    #[inline]
    pub fn ned_R_enu() -> DMat3 {
        DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Z)
    }

    /// Plane-mode rotation `${self}_R_{from}`.
    pub fn plane_R_(&self, from: &GeoFrame, origin: &GeoOrigin) -> DMat3 {
        use GeoFrame::*;
        match (*from, *self) {
            (x, y) if x == y => DMat3::IDENTITY,
            (ENU, NED) => Self::ned_R_enu(),
            (NED, ENU) => Self::enu_R_ned(),
            (ECEF, x) => Self::ecef_R_(&x, origin).inverse(),
            (x, ECEF) => Self::ecef_R_(&x, origin),
            (x, y) => unreachable!("{x:?} -> {y:?}"),
        }
    }

    /// Plane-mode origin offset `${self}_O_{from}`.
    pub fn plane_O_(&self, from: &GeoFrame, origin: &GeoOrigin) -> DVec3 {
        let origin_ecef = Self::origin_ecef(origin);
        match (from, *self) {
            (GeoFrame::ECEF, GeoFrame::ENU | GeoFrame::NED) => {
                -self.plane_R_(from, origin) * origin_ecef
            }
            (GeoFrame::ENU | GeoFrame::NED, GeoFrame::ECEF) => origin_ecef,
            _ => DVec3::ZERO,
        }
    }

    /// Plane-mode affine `${self}_M_{from}` (`out = R * in + O`).
    pub fn plane_M_(&self, from: &GeoFrame, origin: &GeoOrigin) -> DMat4 {
        let R = self.plane_R_(from, origin);
        let O = self.plane_O_(from, origin);
        DMat4::from_mat3_translation(R, O)
    }
}
