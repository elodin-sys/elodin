//! Camera-driven visibility and density curves for cinematic Earth.

use bevy::math::Vec3;

/// WGS84 semi-major (equatorial) axis [m].
pub const WGS84_A_M: f64 = 6_378_137.0;
/// WGS84 semi-minor (polar) axis [m].
pub const WGS84_B_M: f64 = 6_356_752.314_245;

/// Altitude where the sky starts opening to space.
pub const SPACE_VIS_START_M: f32 = 20_000.0;
/// Altitude span of the pad-to-space transition (fully space at 80 km).
pub const SPACE_VIS_SPAN_M: f32 = 60_000.0;
/// Atmosphere density multiplier on the pad (full column).
pub const ATMO_DENSITY_PAD: f32 = 1.0;
/// Atmosphere density multiplier once the limb is a disc (LEO look).
pub const ATMO_DENSITY_LEO: f32 = 0.16;
/// Density quantum: the scattering LUT regenerates only on step changes.
pub const DENSITY_STEP: f32 = 0.01;

/// Sun elevation vs `up`: 1 = noon, −1 = midnight.
pub fn sun_elevation(to_sun: Vec3, up: Vec3) -> f32 {
    let up = up.normalize_or(Vec3::Y);
    to_sun.normalize_or(up).dot(up)
}

/// Returns sinE relative to the visible limb.
pub fn limb_relative_elevation(sin_e: f32, altitude_m: f32, surface_radius_m: f32) -> f32 {
    let radius = surface_radius_m.max(1.0);
    let height = altitude_m.max(0.0);
    let cos_dip = (radius / (radius + height)).clamp(-1.0, 1.0);
    let elevation = sin_e.clamp(-1.0, 1.0).asin();
    (elevation + cos_dip.acos()).sin()
}

/// Star visibility from limb-relative sun elevation.
pub fn star_visibility(elevation: f32) -> f32 {
    (-elevation / 0.15).clamp(0.0, 1.0)
}

/// Nightglow stays off through dusk.
pub fn nightglow_visibility(elevation: f32) -> f32 {
    ((-0.05 - elevation) / 0.3).clamp(0.0, 1.0)
}

/// Local `n·sun` where city lights start (sun ~3° below the local horizon).
pub const CITY_NIGHT_START: f32 = -0.05;
/// Local `n·sun` where city lights are fully on (sun ~17.5° below).
pub const CITY_NIGHT_FULL: f32 = -0.30;

/// Maps local `n · sun` to city-light visibility.
pub fn city_night_mask(s: f32) -> f32 {
    ((CITY_NIGHT_START - s) / (CITY_NIGHT_START - CITY_NIGHT_FULL)).clamp(0.0, 1.0)
}

/// 0 below 20 km, 1 above 80 km.
pub fn space_visibility(altitude_m: f32) -> f32 {
    ((altitude_m - SPACE_VIS_START_M) / SPACE_VIS_SPAN_M).clamp(0.0, 1.0)
}

/// Full column on the pad, 0.16 once the limb is a disc.
pub fn atmosphere_density(altitude_m: f32) -> f32 {
    ATMO_DENSITY_PAD + (ATMO_DENSITY_LEO - ATMO_DENSITY_PAD) * space_visibility(altitude_m)
}

/// [`atmosphere_density`] snapped to [`DENSITY_STEP`] quanta.
pub fn quantize_density(altitude_m: f32) -> f32 {
    let raw = atmosphere_density(altitude_m);
    (raw / DENSITY_STEP).round() * DENSITY_STEP
}

/// WGS84 surface radius for a geocentric latitude sine.
pub fn geocentric_surface_radius_m(sin_lat_gc: f64) -> f64 {
    let s2 = sin_lat_gc.clamp(-1.0, 1.0).powi(2);
    let c2 = 1.0 - s2;
    1.0 / (c2 / (WGS84_A_M * WGS84_A_M) + s2 / (WGS84_B_M * WGS84_B_M)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_curve_matches_pad_and_leo() {
        assert_eq!(atmosphere_density(0.0), ATMO_DENSITY_PAD);
        assert!((atmosphere_density(80_000.0) - ATMO_DENSITY_LEO).abs() < 1e-5);
        assert!((atmosphere_density(400_000.0) - ATMO_DENSITY_LEO).abs() < 1e-5);
        // Midpoint of the ramp.
        let mid = atmosphere_density(50_000.0);
        assert!((mid - 0.58).abs() < 1e-3, "got {mid}");
    }

    #[test]
    fn quantized_density_steps() {
        let d = quantize_density(50_000.0);
        assert!((d / DENSITY_STEP - (d / DENSITY_STEP).round()).abs() < 1e-4);
    }

    #[test]
    fn star_visibility_day_night() {
        assert_eq!(star_visibility(1.0), 0.0);
        assert_eq!(star_visibility(-1.0), 1.0);
        assert!(star_visibility(-0.075) > 0.0 && star_visibility(-0.075) < 1.0);
    }

    #[test]
    fn stars_wash_out_at_the_limb() {
        // Sun disk on (or above) the visible limb: camera stops down, sky black.
        assert_eq!(star_visibility(0.0), 0.0);
        assert_eq!(star_visibility(0.05), 0.0);
        assert_eq!(star_visibility(-0.15), 1.0);
    }

    #[test]
    fn nightglow_lags_stars() {
        // Just below the limb stars are already up but nightglow is not.
        assert!(star_visibility(-0.03) > 0.0);
        assert_eq!(nightglow_visibility(-0.03), 0.0);
        assert_eq!(nightglow_visibility(-1.0), 1.0);
    }

    #[test]
    fn space_visibility_ramp() {
        assert_eq!(space_visibility(0.0), 0.0);
        assert_eq!(space_visibility(20_000.0), 0.0);
        assert_eq!(space_visibility(80_000.0), 1.0);
        assert_eq!(space_visibility(400_000.0), 1.0);
    }

    #[test]
    fn geocentric_radius_matches_wgs84() {
        assert!((geocentric_surface_radius_m(0.0) - WGS84_A_M).abs() < 1.0);
        assert!((geocentric_surface_radius_m(1.0) - WGS84_B_M).abs() < 1.0);
        // LC-39A's geocentric radius is about 6,373.2 km.
        let lat_gc = 28.4_f64.to_radians(); // geocentric ~0.2 deg below geodetic
        let r = geocentric_surface_radius_m(lat_gc.sin());
        assert!((r - 6_373_290.0).abs() < 5_000.0, "got {r}");
    }

    #[test]
    fn sun_elevation_dot() {
        assert_eq!(sun_elevation(Vec3::Y, Vec3::Y), 1.0);
        assert_eq!(sun_elevation(Vec3::NEG_Y, Vec3::Y), -1.0);
        assert_eq!(sun_elevation(Vec3::X, Vec3::Y), 0.0);
    }

    #[test]
    fn city_night_mask_is_zero_in_day_and_at_start() {
        assert_eq!(city_night_mask(1.0), 0.0);
        assert_eq!(city_night_mask(0.0), 0.0);
        assert_eq!(city_night_mask(CITY_NIGHT_START), 0.0);
    }

    #[test]
    fn city_night_mask_is_one_at_full_night() {
        assert_eq!(city_night_mask(CITY_NIGHT_FULL), 1.0);
        assert_eq!(city_night_mask(-1.0), 1.0);
    }

    #[test]
    fn city_night_mask_is_half_mid_band() {
        let mid = 0.5 * (CITY_NIGHT_START + CITY_NIGHT_FULL);
        assert!((city_night_mask(mid) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn city_night_mask_is_monotone() {
        let mut prev = city_night_mask(1.0);
        for i in 0..=40 {
            let s = 1.0 - i as f32 * 0.05;
            let v = city_night_mask(s);
            assert!(v + 1e-6 >= prev, "s={s} v={v} prev={prev}");
            prev = v;
        }
    }

    #[test]
    fn limb_dip_is_zero_on_the_pad() {
        let radius = WGS84_A_M as f32;
        assert!((limb_relative_elevation(-0.34, 0.0, radius) + 0.34).abs() < 1e-5);
        assert!((limb_relative_elevation(0.5, 0.0, radius) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn limb_dip_at_leo_rewrites_sunrise() {
        let radius = WGS84_A_M as f32;
        let altitude = 400_000.0;
        let dip = (radius / (radius + altitude)).acos();
        assert!(
            (dip.to_degrees() - 19.8).abs() < 0.3,
            "dip {} deg",
            dip.to_degrees()
        );

        let at_limb = limb_relative_elevation(-0.34, altitude, radius);
        assert!(at_limb.abs() < 0.02, "got {at_limb}");

        let morning = limb_relative_elevation(0.05, altitude, radius);
        assert!((morning - 0.39).abs() < 0.02, "got {morning}");
        assert_eq!(star_visibility(morning), 0.0);
        assert_eq!(nightglow_visibility(morning), 0.0);

        let night = limb_relative_elevation(-0.71, altitude, radius);
        assert_eq!(star_visibility(night), 1.0);
        assert_eq!(nightglow_visibility(night), 1.0);
    }
}
