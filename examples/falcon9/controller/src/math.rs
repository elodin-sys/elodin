//! Minimal f64 vector/quaternion algebra and WGS84 geodesy for the FSW.
//! Conventions match the plant: Hamilton quaternions stored [x, y, z, w],
//! q maps body-frame vectors into ECEF.

pub const WGS84_A: f64 = 6_378_137.0;
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
pub const MU_EARTH: f64 = 3.986_004_418e14;
pub const OMEGA_EARTH: f64 = 7.292_115e-5;

pub type V3 = [f64; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}

/// Quaternion [x, y, z, w].
pub type Quat = [f64; 4];

pub const QUAT_IDENT: Quat = [0.0, 0.0, 0.0, 1.0];

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

pub fn quat_conj(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn quat_normalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        QUAT_IDENT
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

pub fn quat_rotate(q: Quat, v: V3) -> V3 {
    // v' = q v q*
    let qv = [q[0], q[1], q[2]];
    let t = scale(cross(qv, v), 2.0);
    add(add(v, scale(t, q[3])), cross(qv, t))
}

pub fn quat_rotate_inv(q: Quat, v: V3) -> V3 {
    quat_rotate(quat_conj(q), v)
}

pub fn quat_from_axis_angle(axis: V3, angle: f64) -> Quat {
    let a = normalize(axis);
    let (s, c) = (angle * 0.5).sin_cos();
    [a[0] * s, a[1] * s, a[2] * s, c]
}

/// Integrate body-frame angular rate over dt: q <- q * exp(omega dt / 2).
pub fn quat_integrate(q: Quat, omega_body: V3, dt: f64) -> Quat {
    let angle = norm(omega_body) * dt;
    if angle < 1e-12 {
        return q;
    }
    let dq = quat_from_axis_angle(omega_body, angle);
    quat_normalize(quat_mul(q, dq))
}

/// Shortest-arc quaternion rotating unit vector `from` onto unit vector `to`.
pub fn quat_between(from: V3, to: V3) -> Quat {
    let c = dot(from, to).clamp(-1.0, 1.0);
    if c > 1.0 - 1e-12 {
        return QUAT_IDENT;
    }
    if c < -1.0 + 1e-12 {
        // Antipodal: rotate 180 deg about any axis normal to `from`.
        let axis = if from[0].abs() < 0.9 {
            normalize(cross(from, [1.0, 0.0, 0.0]))
        } else {
            normalize(cross(from, [0.0, 1.0, 0.0]))
        };
        return quat_from_axis_angle(axis, std::f64::consts::PI);
    }
    let axis = normalize(cross(from, to));
    quat_from_axis_angle(axis, c.acos())
}

/// Geodetic latitude, longitude, ellipsoid height from ECEF (Bowring).
pub fn ecef_to_geodetic(r: V3) -> (f64, f64, f64) {
    let (x, y, z) = (r[0], r[1], r[2]);
    let lon = y.atan2(x);
    let p = x.hypot(y);
    let b = WGS84_A * (1.0 - WGS84_F);
    let ep2 = WGS84_E2 / (1.0 - WGS84_E2);
    let mut beta = z.atan2((1.0 - WGS84_F) * p);
    let mut lat = beta;
    for _ in 0..4 {
        lat = (z + ep2 * b * beta.sin().powi(3)).atan2(p - WGS84_E2 * WGS84_A * beta.cos().powi(3));
        beta = ((1.0 - WGS84_F) * lat.tan()).atan();
    }
    let sin_lat = lat.sin();
    let w = (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let alt = p * lat.cos() + z * sin_lat - WGS84_A * w;
    (lat, lon, alt)
}

pub fn geodetic_to_ecef(lat: f64, lon: f64, alt: f64) -> V3 {
    let n = WGS84_A / (1.0 - WGS84_E2 * lat.sin() * lat.sin()).sqrt();
    [
        (n + alt) * lat.cos() * lon.cos(),
        (n + alt) * lat.cos() * lon.sin(),
        (n * (1.0 - WGS84_E2) + alt) * lat.sin(),
    ]
}

/// Local (north, east, down) unit vectors at a geodetic point, in ECEF.
pub fn ned_basis(lat: f64, lon: f64) -> [V3; 3] {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    [
        [-sl * co, -sl * so, cl],
        [-so, co, 0.0],
        [-cl * co, -cl * so, -sl],
    ]
}

/// Point-mass gravitation.
pub fn gravity(r: V3) -> V3 {
    let n = norm(r);
    scale(r, -MU_EARTH / (n * n * n))
}

/// Rotating-frame fictitious acceleration: -2 w x v - w x (w x r).
pub fn frame_accel(r: V3, v: V3) -> V3 {
    let w = [0.0, 0.0, OMEGA_EARTH];
    sub(scale(cross(w, v), -2.0), cross(w, cross(w, r)))
}

/// US76-lite density for q-bar gating (two-piece exponential; EST).
pub fn density(alt_m: f64) -> f64 {
    let h = alt_m.max(0.0);
    if h < 25_000.0 {
        1.225 * (-h / 8_440.0).exp()
    } else {
        0.0642 * (-(h - 25_000.0) / 6_580.0).exp()
    }
}
