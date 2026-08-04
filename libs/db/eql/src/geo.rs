//! Plane-mode ENU / NED / ECEF transforms for EQL SQL emission.
//!
//! Matches [`bevy_geo_frames`] Plane-mode `GeoFrame::_M_` / `_R_` (WGS84).

use map_3d::Ellipsoid;

/// Geodetic origin used for ECEF ↔ local conversions (degrees / metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoOrigin {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
}

impl GeoOrigin {
    pub fn new(latitude_deg: f64, longitude_deg: f64, altitude_m: f64) -> Self {
        Self {
            latitude_deg,
            longitude_deg,
            altitude_m,
        }
    }

    fn lat_rad(self) -> f64 {
        self.latitude_deg.to_radians()
    }

    fn lon_rad(self) -> f64 {
        self.longitude_deg.to_radians()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameId {
    Enu,
    Ned,
    Ecef,
}

impl FrameId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Enu => "ENU",
            Self::Ned => "NED",
            Self::Ecef => "ECEF",
        }
    }
}

/// 3×3 row-major rotation plus translation: `out = R * in + t`.
#[derive(Debug, Clone, Copy)]
pub struct Affine3 {
    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
}

impl Affine3 {
    pub fn identity() -> Self {
        Self {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0, 0.0, 0.0],
        }
    }

    pub fn transform_point(self, p: [f64; 3]) -> [f64; 3] {
        let r = self.r;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + self.t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + self.t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + self.t[2],
        ]
    }

    pub fn transform_vector(self, v: [f64; 3]) -> [f64; 3] {
        let r = self.r;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }

    fn from_rotation_translation(r: [[f64; 3]; 3], t: [f64; 3]) -> Self {
        Self { r, t }
    }

    fn from_rotation(r: [[f64; 3]; 3]) -> Self {
        Self {
            r,
            t: [0.0, 0.0, 0.0],
        }
    }
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat_vec(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

fn mat_transpose(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

/// ENU ↔ NED: same as `bevy_geo_frames` (`from_cols(Y, X, -Z)` → row-major).
fn enu_ned_r() -> [[f64; 3]; 3] {
    // Columns [Y, X, -Z] as rows: out = [[0,1,0],[1,0,0],[0,0,-1]] * in
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
}

fn rotation_x(theta: f64) -> [[f64; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rotation_z(theta: f64) -> [[f64; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// `${ecef}_R_{from}` — same construction as `GeoFrame::ecef_R_`.
fn ecef_r(from: FrameId, origin: GeoOrigin) -> [[f64; 3]; 3] {
    use std::f64::consts::FRAC_PI_2;
    if from == FrameId::Ecef {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let ecef_r_enu = mat_mul(
        rotation_z(FRAC_PI_2 + origin.lon_rad()),
        rotation_x(FRAC_PI_2 - origin.lat_rad()),
    );
    match from {
        FrameId::Ecef => unreachable!(),
        FrameId::Enu => ecef_r_enu,
        FrameId::Ned => mat_mul(ecef_r_enu, enu_ned_r()),
    }
}

fn origin_ecef(origin: GeoOrigin) -> [f64; 3] {
    map_3d::enu2ecef(
        0.0,
        0.0,
        0.0,
        origin.lat_rad(),
        origin.lon_rad(),
        origin.altitude_m,
        &Ellipsoid::WGS84,
    )
    .into()
}

/// `${to}_R_{from}`
fn frame_r(from: FrameId, to: FrameId, origin: Option<GeoOrigin>) -> Result<[[f64; 3]; 3], String> {
    if from == to {
        return Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
    match (from, to) {
        (FrameId::Enu, FrameId::Ned) | (FrameId::Ned, FrameId::Enu) => Ok(enu_ned_r()),
        (FrameId::Ecef, local @ (FrameId::Enu | FrameId::Ned)) => {
            let origin = origin.ok_or_else(|| missing_origin(from, to))?;
            Ok(mat_transpose(ecef_r(local, origin))) // inverse of orthogonal R
        }
        (local @ (FrameId::Enu | FrameId::Ned), FrameId::Ecef) => {
            let origin = origin.ok_or_else(|| missing_origin(from, to))?;
            Ok(ecef_r(local, origin))
        }
        _ => unreachable!(),
    }
}

fn missing_origin(from: FrameId, to: FrameId) -> String {
    format!(
        "{}→{} conversion requires a geo origin (schematic `coordinate` lat/lon/alt)",
        from.as_str(),
        to.as_str()
    )
}

/// Affine `${to}_M_{from}` for positions (Plane / WGS84).
pub fn point_affine(
    from: FrameId,
    to: FrameId,
    origin: Option<GeoOrigin>,
) -> Result<Affine3, String> {
    if from == to {
        return Ok(Affine3::identity());
    }
    let r = frame_r(from, to, origin)?;
    let t = match (from, to) {
        (FrameId::Enu, FrameId::Ned) | (FrameId::Ned, FrameId::Enu) => [0.0, 0.0, 0.0],
        (FrameId::Ecef, FrameId::Enu | FrameId::Ned) => {
            let origin = origin.ok_or_else(|| missing_origin(from, to))?;
            // Plane: O = -R * origin_ecef
            let o = origin_ecef(origin);
            let ro = mat_vec(r, o);
            [-ro[0], -ro[1], -ro[2]]
        }
        (FrameId::Enu | FrameId::Ned, FrameId::Ecef) => {
            let origin = origin.ok_or_else(|| missing_origin(from, to))?;
            origin_ecef(origin)
        }
        _ => unreachable!(),
    };
    Ok(Affine3::from_rotation_translation(r, t))
}

/// Rotation `${to}_R_{from}` for free vectors / directions.
pub fn direction_affine(
    from: FrameId,
    to: FrameId,
    origin: Option<GeoOrigin>,
) -> Result<Affine3, String> {
    Ok(Affine3::from_rotation(frame_r(from, to, origin)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enu_ned_swap_point() {
        let a = point_affine(FrameId::Enu, FrameId::Ned, None).unwrap();
        let out = a.transform_point([1.0, 2.0, 3.0]);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn ecef_origin_is_zero_ned() {
        let origin = GeoOrigin::new(28.5, -80.6, 0.0);
        let ecef = point_affine(FrameId::Ned, FrameId::Ecef, Some(origin))
            .unwrap()
            .transform_point([0.0, 0.0, 0.0]);
        let a = point_affine(FrameId::Ecef, FrameId::Ned, Some(origin)).unwrap();
        let ned = a.transform_point(ecef);
        assert!(
            (ned[0] * ned[0] + ned[1] * ned[1] + ned[2] * ned[2]).sqrt() < 1e-6,
            "got {ned:?}"
        );
    }

    #[test]
    fn direction_ignores_origin_translation() {
        let origin = GeoOrigin::new(28.5, -80.6, 0.0);
        let point = point_affine(FrameId::Ecef, FrameId::Ned, Some(origin)).unwrap();
        let dir = direction_affine(FrameId::Ecef, FrameId::Ned, Some(origin)).unwrap();
        let v = [1000.0, 0.0, 0.0];
        let p = point.transform_point(v);
        let d = dir.transform_vector(v);
        assert!(
            (p[0] - d[0]).abs() > 1.0 || (p[1] - d[1]).abs() > 1.0 || (p[2] - d[2]).abs() > 1.0
        );
    }

    #[test]
    fn matches_map3d_ned_offset() {
        let origin = GeoOrigin::new(28.5, -80.6, 0.0);
        let ned = [10.0, -3.0, 2.0];
        let ecef = point_affine(FrameId::Ned, FrameId::Ecef, Some(origin))
            .unwrap()
            .transform_point(ned);
        let via_map: [f64; 3] = map_3d::ned2ecef(
            ned[0],
            ned[1],
            ned[2],
            origin.lat_rad(),
            origin.lon_rad(),
            origin.altitude_m,
            &Ellipsoid::WGS84,
        )
        .into();
        let err = ((ecef[0] - via_map[0]).powi(2)
            + (ecef[1] - via_map[1]).powi(2)
            + (ecef[2] - via_map[2]).powi(2))
        .sqrt();
        assert!(
            err < 1e-6,
            "affine vs map_3d err={err}: {ecef:?} vs {via_map:?}"
        );
    }
}
