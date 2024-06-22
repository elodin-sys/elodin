use hifitime::{Duration, Epoch, Unit, MJD_OFFSET};
use nox::{Array, ArrayRepr, Matrix, Vector};

use crate::{iers::IERS, Frame, DCM};

/// Celestial Intermediate Reference System
pub struct CIRS;
impl Frame for CIRS {}

/// Terrestrial Intermediate Reference System
pub struct TIRS;
impl Frame for TIRS {}

/// International Terrestrial Reference Frame
pub struct ITRF;
impl Frame for ITRF {}

pub type ECEF = ITRF;

/// Geocentric Celestial Reference Frame
pub struct GCRF;
impl Frame for GCRF {}

/// North East Down
pub struct NED;
impl Frame for NED {}

fn epoch_to_mjd_ut1(time: Epoch, iers: &IERS) -> f64 {
    let mjd_utc = time.to_mjd_utc_days();
    let ut1_utc = iers.get_ut1_utc(mjd_utc).unwrap_or_default();
    let ut1_utc = Duration::from_f64(ut1_utc, Unit::Second);
    (time + ut1_utc).to_mjd_utc_days()
}

pub fn earth_rotation_iers(time: Epoch, iers: &IERS) -> DCM<f64, CIRS, TIRS, ArrayRepr> {
    let mjd_ut1 = epoch_to_mjd_ut1(time, iers);
    let era = unsafe { rsofa::iauEra00(MJD_OFFSET, mjd_ut1) };
    let mut rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    unsafe { rsofa::iauRz(era, rot.as_mut_ptr()) }
    DCM::from(Matrix::from_inner(Array { buf: rot }))
}

pub fn polar_motion_iers(time: Epoch, iers: &IERS) -> Option<DCM<f64, TIRS, ITRF, ArrayRepr>> {
    let mjd_utc = time.to_mjd_utc_days();
    let mjd_tt = time.to_mjd_tt_days();
    let [pm_x, pm_y] = iers.get_pm(mjd_utc)?;
    let s_prime = unsafe { rsofa::iauSp00(MJD_OFFSET, mjd_tt) };
    let mut rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    unsafe {
        rsofa::iauPom00(pm_x, pm_y, s_prime, rot.as_mut_ptr());
    };
    Some(DCM::from(Matrix::from_buf(rot)))
}

pub fn bias_precession_nutation_iers(time: Epoch, iers: &IERS) -> DCM<f64, GCRF, CIRS, ArrayRepr> {
    let [mut x, mut y, mut s] = [0.0, 0.0, 0.0];
    let mjd_utc = time.to_mjd_utc_days();
    let mjd_tt = time.to_mjd_tt_days();
    unsafe {
        rsofa::iauXys06a(MJD_OFFSET, mjd_tt, &mut x, &mut y, &mut s);
    }
    let [dx, dy] = iers.get_nutation(mjd_utc).unwrap_or_default();
    x += dx;
    y += dy;
    let mut rot = [[0.0; 3]; 3];
    unsafe {
        rsofa::iauC2ixys(x, y, s, rot.as_mut_ptr());
    };
    DCM::from(Matrix::from_buf(rot))
}

pub fn eci_to_ecef_iers(time: Epoch, iers: &IERS) -> DCM<f64, GCRF, ITRF, ArrayRepr> {
    polar_motion_iers(time, iers).unwrap_or_else(|| {
        DCM::from(Matrix::from_buf([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]))
    }) * earth_rotation_iers(time, iers)
        * bias_precession_nutation_iers(time, iers)
}

pub fn eci_to_ecef(time: Epoch) -> DCM<f64, GCRF, ITRF, ArrayRepr> {
    let iers = IERS::global();
    eci_to_ecef_iers(time, iers)
}

pub fn ecef_to_eci(time: Epoch) -> DCM<f64, ITRF, GCRF, ArrayRepr> {
    eci_to_ecef(time).inverse() // NOTE: replace /frm this with transpose
}

pub fn ecef_to_ned(lat: f64, long: f64) -> DCM<f64, ECEF, NED, ArrayRepr> {
    // source: https://github.com/zbai/MATLAB-Groves/blob/master/MATLAB/NED_to_ECEF.m
    // const R_0: f64 = 6378137.0;
    // const E: f64 = 0.0818191908425;
    // let r_e = R_0 / f64::sqrt(1.0 - (E * f64::sin(lat)).powi(2));
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_long, cos_long) = long.sin_cos();

    let c_e_n = [
        [-sin_lat * cos_long, -sin_lat * sin_long, cos_lat],
        [-sin_long, cos_long, 0.0],
        [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat],
    ];

    DCM::from(Matrix::from_buf(c_e_n))
}

pub fn ned_to_ecef(lat: f64, long: f64) -> DCM<f64, NED, ECEF, ArrayRepr> {
    ecef_to_ned(lat, long).inverse()
}

/// Calculate the unit vector pointing at the sun in ECI
/// NOTE: eventually this should be moved to something like nox-astro
pub fn sun_vec(epoch: Epoch) -> Vector<f64, 3, ArrayRepr> {
    // source: vallado + https://astronomy.stackexchange.com/a/37199
    let centuries = epoch.to_tdb_centuries_since_j2000();
    let mean_long = 280.4606184 + 36000.77005361 * centuries;
    let mean_anomoly = 357.5277233 + 35999.05034 * centuries;
    let eclipitic_long = mean_long
        + 1.914666471 * f64::sin(mean_anomoly.to_radians())
        + 0.918994643 * f64::sin(2.0 * mean_anomoly.to_radians());
    let obliquity = 23.43929 - (46.8093 / 3600.0) * centuries;
    let (sin_eclipitic_long, cos_eclipitic_long) = eclipitic_long.to_radians().sin_cos();
    let (sin_obliquity, cos_obliquity) = obliquity.to_radians().sin_cos();
    let x = cos_eclipitic_long;
    let y = cos_obliquity * sin_eclipitic_long;
    let z = sin_obliquity * sin_eclipitic_long;
    Vector::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nox::{array, tensor};

    use super::*;

    #[test]
    fn test_eci_to_ecef() {
        let epoch = Epoch::from_gregorian_utc(2019, 1, 4, 12, 0, 0, 0);
        let x_eci = tensor![-2981784.0, 5207055.0, 3161595.0];
        let eci_to_ecef = eci_to_ecef(epoch);
        println!("{:?}", eci_to_ecef.dcm);
        let x_ecef = eci_to_ecef.dot(&x_eci);
        let expected = tensor![-5762648.74320628, -1682708.43849581, 3156027.93288401];

        assert_relative_eq!(x_ecef.inner(), expected.inner(), epsilon = 1e-2);
    }

    #[test]
    fn test_earth_rotation() {
        let epoch = Epoch::from_gregorian_utc(2019, 1, 4, 12, 0, 0, 0);
        let earth_rot = earth_rotation_iers(epoch, IERS::global());
        let expected_rot = tensor![
            [0.23457505, -0.97209801, 0.],
            [0.97209801, 0.23457505, 0.],
            [0., 0., 1.]
        ];
        assert_relative_eq!(earth_rot.dcm.inner(), expected_rot.inner(), epsilon = 1e-5);
    }

    #[test]
    fn test_ned_to_ecef() {
        let [lat, long] = [40.29959f64.to_radians(), -111.72822f64.to_radians()];
        let ecef = ned_to_ecef(lat, long).dot(&tensor![4.0, 5.0, 6.0]);
        assert_relative_eq!(
            ecef.inner(),
            &array![7.2966, 4.8032, -0.8300],
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_sun_pos() {
        let epoch = Epoch::from_gregorian_utc(2019, 1, 4, 12, 0, 0, 0);
        let sun_vec = sun_vec(epoch);
        let expected = tensor![
            0.23061245658276283,
            -0.8770198065823603,
            -0.3801863810692247
        ]
        .normalize(); // source astropy

        assert_relative_eq!(sun_vec.inner(), expected.inner(), epsilon = 1e-2)
    }
}
