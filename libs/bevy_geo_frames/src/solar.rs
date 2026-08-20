//! Solar position accurate to about 0.05°.

use bevy::math::DVec3;

/// Days from Unix epoch 00:00 to J2000.0 (2000-01-01 12:00 TT ≈ UTC).
const UNIX_DAYS_TO_J2000: f64 = 10_957.5;
const MICROS_PER_DAY: f64 = 86_400e6;

fn wrap_deg(deg: f64) -> f64 {
    deg.rem_euclid(360.0)
}

/// Unit vector toward the sun in ECEF at `unix_micros`.
pub fn sun_direction_ecef(unix_micros: i64) -> DVec3 {
    let d = unix_micros as f64 / MICROS_PER_DAY - UNIX_DAYS_TO_J2000;
    let mean_lon = 280.460 + 0.985_647_4 * d;
    let g = wrap_deg(357.528 + 0.985_600_3 * d).to_radians();
    let lambda = (wrap_deg(mean_lon + 1.915 * g.sin() + 0.020 * (2.0 * g).sin())).to_radians();
    let eps = (23.439 - 4.0e-7 * d).to_radians();
    let (sin_l, cos_l) = lambda.sin_cos();
    let (sin_e, cos_e) = eps.sin_cos();
    let eq = DVec3::new(cos_l, cos_e * sin_l, sin_e * sin_l);
    let gmst = wrap_deg(280.460_618_37 + 360.985_647_366_29 * d).to_radians();
    let (s, c) = gmst.sin_cos();
    DVec3::new(c * eq.x + s * eq.y, -s * eq.x + c * eq.y, eq.z).normalize()
}

/// Geocentric subsolar latitude and longitude, degrees (−90..90, −180..180).
pub fn subsolar_lat_lon_deg(unix_micros: i64) -> (f64, f64) {
    let d = sun_direction_ecef(unix_micros);
    (d.z.asin().to_degrees(), d.y.atan2(d.x).to_degrees())
}

/// Solar elevation in degrees at a geodetic latitude/longitude (spherical up).
pub fn sun_elevation_deg(unix_micros: i64, lat_deg: f64, lon_deg: f64) -> f64 {
    let to_sun = sun_direction_ecef(unix_micros);
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    to_sun.dot(up).clamp(-1.0, 1.0).asin().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2000-01-01 12:00:00 UTC (J2000.0).
    const J2000_NOON_US: i64 = 946_728_000_000_000;
    /// 2026-03-20 12:00:00 UTC.
    const EQUINOX_2026_NOON_US: i64 = 1_774_008_000_000_000;
    /// 2026-03-20 09:20:00 UTC (subsolar ~40°E; morning-side fixture).
    const EQUINOX_2026_0920_US: i64 = 1_773_998_400_000_000;
    /// 2026-06-21 12:00:00 UTC.
    const SOLSTICE_2026_NOON_US: i64 = 1_782_043_200_000_000;

    fn assert_near(got: f64, expect: f64, tol: f64, label: &str) {
        assert!(
            (got - expect).abs() < tol,
            "{label}: got {got}, expected {expect} ± {tol}"
        );
    }

    #[test]
    fn direction_is_unit() {
        let d = sun_direction_ecef(J2000_NOON_US);
        assert!((d.length() - 1.0).abs() < 1e-12, "len {}", d.length());
    }

    #[test]
    fn j2000_noon_subsolar() {
        let (lat, lon) = subsolar_lat_lon_deg(J2000_NOON_US);
        assert_near(lat, -23.0, 0.5, "J2000 lat");
        assert_near(lon, 0.8, 0.5, "J2000 lon");
    }

    #[test]
    fn march_2026_equinox_subsolar_lat() {
        let (lat, _) = subsolar_lat_lon_deg(EQUINOX_2026_NOON_US);
        assert_near(lat, 0.0, 0.5, "2026-03-20 lat");
    }

    #[test]
    fn june_2026_solstice_subsolar_lat() {
        let (lat, _) = subsolar_lat_lon_deg(SOLSTICE_2026_NOON_US);
        assert_near(lat, 23.44, 0.5, "2026-06-21 lat");
    }

    #[test]
    fn cube_sat_start_subsolar_east_of_x() {
        let (lat, lon) = subsolar_lat_lon_deg(EQUINOX_2026_0920_US);
        assert_near(lat, 0.0, 0.5, "09:20 lat");
        // Equation of time is a couple of degrees; 09:20 UTC is ~40°E mean sun.
        assert_near(lon, 40.0, 3.0, "09:20 lon");
    }

    #[test]
    fn crs12_pad_elevation_is_near_noon() {
        // 2017-08-14T16:31:37Z at LC-39A — the falcon9 recording epoch.
        const CRS12_US: i64 = 1_502_728_297_000_000;
        let el = sun_elevation_deg(CRS12_US, 28.60839, -80.60433);
        assert_near(el, 71.0, 1.0, "CRS-12 LC-39A elevation");
    }

    #[test]
    fn ecef_sun_moves_west_fifteen_deg_per_hour() {
        let (_, lon0) = subsolar_lat_lon_deg(J2000_NOON_US);
        let hour = 3_600_000_000;
        let (_, lon1) = subsolar_lat_lon_deg(J2000_NOON_US + hour);
        let mut delta = lon0 - lon1;
        if delta < 0.0 {
            delta += 360.0;
        }
        assert_near(delta, 15.0, 0.5, "hourly west drift");
    }
}
