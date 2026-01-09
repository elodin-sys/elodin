use bevy::math::{DMat3, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use map_3d::{ecef2enu, enu2ecef, geodetic2ecef, Ellipsoid};

/// Default Earth radius in meters (approximate mean radius).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Planet/body shape model used for things like gravity scaling and debug rendering.
///
/// NOTE: Coordinate conversion in this crate currently always uses `Ellipsoid::WGS84`
/// from `map_3d` regardless of this setting. `Shape` is primarily used as a
/// configurable "reference radius" until we fully plumb shape into the conversion
/// routines.
#[derive(Debug, Clone, Copy)]
pub enum Shape {
    /// A spherical body with a single reference radius.
    Sphere { radius: f64 },
    /// An ellipsoidal body (from `map_3d`).
    Ellipsoid(Ellipsoid),
}

impl Shape {

    /// Ellipsoid to use for lat/lon <-> ECEF conversions.
    pub fn ellipsoid(self) -> Ellipsoid {
        match self {
            Shape::Ellipsoid(e) => e,
            // Until `map_3d` exposes a spherical ellipsoid, use WGS84 for conversions.
            // Shape::Sphere { .. } => Ellipsoid::WGS84,
-            Shape::Sphere { radius } => Ellipsoid::UnitSphere,
        }
    }

    /// Reference radius used for simple gravity scaling and visualization.
    fn approx_radius(&self) -> f64 {
        match self {
            Shape::Ellipsoid(_) => EARTH_RADIUS_M,
            Shape::Sphere { radius } => radius
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape::Ellipsoid(Ellipsoid::WGS84)
    }
}


/// Coordinate frames used in the sim.
///
/// Units: meters, seconds.
/// Bevy world is treated as EUS: +X=East, +Y=Up, +Z=South (so North = -Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoFrame {
    /// Bevy world: +X=East, +Y=Up, +Z=South
    EUS,

    /// East-North-Up: +X=East, +Y=North, +Z=Up
    ENU,
    /// North-East-Down: +X=North, +Y=East, +Z=Down
    NED,

    /// Earth-Centered Earth-Fixed
    /// +X through (lat=0, lon=0) equator
    /// +Y through (lat=0, lon=90°E) equator
    /// +Z through North Pole
    ECEF,
    /// Earth-Centered Inertial
    /// +X to vernal equinox, +Y 90°E, +Z North Pole
    ECI,
    /// Geocentric Celestial Reference Frame (inertial, J2000)
    /// Sometimes called the International Celestial Reference Frame (ICRF)
    /// Approximated as ECI here.
    GCRF,
}

/// Where the Bevy world origin lives on Earth.
///
/// Used to turn ECEF positions into local ENU, then ENU → Bevy.
#[derive(Debug, Clone, Copy)]
pub struct GeoOrigin {
    /// Geodetic latitude [rad]
    pub lat: f64,
    /// Geodetic longitude [rad]
    pub lon: f64,
    /// Altitude above mean radius [m]
    pub alt_m: f64,
    /// Planet/body shape model (currently used primarily for reference radius).
    pub shape: Shape,
    /// Origin in ECEF coordinates [m]
    pub ecef_origin: DVec3,
}

impl GeoOrigin {
    /// Simple spherical Earth model (good enough for games).
    /// Uses default Earth radius.
    pub fn new_from_degrees(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self::new_from_degrees_with_radius(lat_deg, lon_deg, alt_m, EARTH_RADIUS_M)
    }

    /// Simple spherical Earth model with custom radius.
    /// Useful for demonstrations with smaller values.
    pub fn new_from_degrees_with_radius(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        radius: f64,
    ) -> Self {
        Self::new_from_degrees_with_shape(
            lat_deg,
            lon_deg,
            alt_m,
            Shape::Sphere { radius },
        )
    }

    /// Construct an origin with an explicit shape model.
    pub fn new_from_degrees_with_shape(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        shape: Shape,
    ) -> Self {
        let lat = lat_deg.to_radians();
        let lon = lon_deg.to_radians();

        // Use map_3d to compute ECEF position of the origin
        // Note: until we fully plumb Shape into conversions, this will use WGS84 unless
        // the caller passes an explicit Ellipsoid shape.
        let (x, y, z) = geodetic2ecef(lat, lon, alt_m, shape.ellipsoid());
        let ecef_origin = DVec3::new(x, y, z);

        Self {
            lat,
            lon,
            alt_m,
            shape,
            ecef_origin,
        }
    }
}

/// Global geospatial context:
/// * origin (for ENU/ECEF)
/// * Earth rotation model (for ECI/GCRF <-> ECEF).
#[derive(Resource, Debug, Clone, Copy)]
pub struct GeoContext {
    pub origin: GeoOrigin,
    /// Earth rotation angle at t=0 [rad]; 0 means ECI/ECEF axes aligned at startup.
    pub theta0_rad: f64,
    /// Earth rotation rate [rad/s]; sidereal ≈ 7.2921150e-5 rad/s.
    pub earth_rot_rate_rad_per_s: f64,
}

impl Default for GeoContext {
    fn default() -> Self {
        Self::new_from_degrees(0.0, 0.0, 0.0, 0.0)
    }
}

impl GeoContext {
    /// Construct with a given origin and an initial Earth angle.
    /// For more realism you can plug in GMST at startup as theta0_rad.
    /// Uses default Earth radius.
    pub fn new_from_degrees(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        theta0_rad: f64,
    ) -> Self {
        Self::new_from_degrees_with_radius(lat_deg, lon_deg, alt_m, theta0_rad, EARTH_RADIUS_M)
    }

    /// Construct with a given origin, initial Earth angle, and custom radius.
    /// Useful for demonstrations with smaller radius values.
    pub fn new_from_degrees_with_radius(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        theta0_rad: f64,
        radius: f64,
    ) -> Self {
        let origin = GeoOrigin::new_from_degrees_with_radius(lat_deg, lon_deg, alt_m, radius);
        Self::new_from_origin(origin, theta0_rad)
    }

    /// Construct with a given origin, initial Earth angle, and explicit shape.
    pub fn new_from_degrees_with_shape(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        theta0_rad: f64,
        shape: Shape,
    ) -> Self {
        let origin = GeoOrigin::new_from_degrees_with_shape(lat_deg, lon_deg, alt_m, shape);
        Self::new_from_origin(origin, theta0_rad)
    }

    fn new_from_origin(origin: GeoOrigin, theta0_rad: f64) -> Self {
        Self {
            origin,
            theta0_rad,
            earth_rot_rate_rad_per_s: 7.292_115_0e-5, // Earth sidereal spin
        }
    }

    /// Rotation matrix Rz(angle) about +Z axis:
    /// x' = cosθ x - sinθ y
    /// y' = sinθ x + cosθ y
    /// z' = z
    pub(crate) fn rot_z(angle: f64) -> DMat3 {
        let (s, c) = angle.sin_cos();
        DMat3::from_cols(
            DVec3::new(c, s, 0.0),   // image of x-axis
            DVec3::new(-s, c, 0.0),  // image of y-axis
            DVec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Convert ECI -> ECEF at simulation time t [s].
    ///
    /// ECEF = Rz(theta) * ECI, with theta = theta0 + ω⊕ t.
    pub fn eci_to_ecef(&self, r_eci: DVec3, t_seconds: f64) -> DVec3 {
        let theta = self.theta0_rad + self.earth_rot_rate_rad_per_s * t_seconds;
        Self::rot_z(theta) * r_eci
    }

    /// Convert GCRF -> ECI.
    ///
    /// For now we approximate GCRF == ECI (J2000), so this is identity.
    pub fn gcrf_to_eci(&self, r_gcrf: DVec3, _t_seconds: f64) -> DVec3 {
        r_gcrf
    }

    /// ECEF -> local ENU at the origin.
    pub fn ecef_to_enu(&self, r_ecef: DVec3) -> DVec3 {
        let (e, n, u) = ecef2enu(
            r_ecef.x,
            r_ecef.y,
            r_ecef.z,
            self.origin.lat,
            self.origin.lon,
            self.origin.alt_m,
            Ellipsoid::WGS84,
        );
        DVec3::new(e, n, u)
    }
}

impl GeoFrame {
    /// ENU: (E, N, U) -> Bevy(EUS): (E, U, -N)
    #[inline]
    fn enu_vec_to_bevy(v: Vec3) -> Vec3 {
        Vec3::new(v.x, v.z, -v.y)
    }

    /// NED: (N, E, D) -> Bevy(EUS): (E, -D, -N)
    #[inline]
    fn ned_vec_to_bevy(v: Vec3) -> Vec3 {
        Vec3::new(v.y, -v.z, -v.x)
    }

    /// Bevy(EUS) -> ENU: (E, U, -N) -> (E, N, U)
    #[inline]
    fn bevy_to_enu_vec(v: Vec3) -> Vec3 {
        Vec3::new(v.x, -v.z, v.y)
    }

    /// Bevy(EUS) -> NED: (E, -D, -N) -> (N, E, D)
    #[inline]
    fn bevy_to_ned_vec(v: Vec3) -> Vec3 {
        Vec3::new(-v.z, v.x, -v.y)
    }

    pub fn convert_to(&self, v: Vec3, from_frame: &GeoFrame, ctx: &GeoContext, t_seconds: f64) -> Vec3 {
        if from_frame == self {
            v
        } else {
            let w = from_frame.to_bevy_vec(v, ctx, t_seconds);
            self.from_bevy_vec(w, ctx, t_seconds)
        }
    }

    /// Convert a vector (typically a position) from this frame into Bevy (EUS).
    ///
    /// `ctx` provides origin + Earth rotation.
    /// `t_seconds` is simulation time, e.g. `time.elapsed_secs_f64()`.
    pub fn to_bevy_vec(self, v: Vec3, ctx: &GeoContext, t_seconds: f64) -> Vec3 {
        match self {
            GeoFrame::EUS => v,
            GeoFrame::ENU => Self::enu_vec_to_bevy(v),
            GeoFrame::NED => Self::ned_vec_to_bevy(v),
            GeoFrame::ECEF => {
                // ECEF -> ENU (local origin) -> Bevy
                let ecef = DVec3::new(v.x as f64, v.y as f64, v.z as f64);
                let enu_d = ctx.ecef_to_enu(ecef);
                let enu = Vec3::new(enu_d.x as f32, enu_d.y as f32, enu_d.z as f32);
                Self::enu_vec_to_bevy(enu)
            }
            GeoFrame::ECI => {
                // ECI -> ECEF (time-dependent) -> ENU -> Bevy
                let r_eci = DVec3::new(v.x as f64, v.y as f64, v.z as f64);
                let r_ecef = ctx.eci_to_ecef(r_eci, t_seconds);
                let enu_d = ctx.ecef_to_enu(r_ecef);
                let enu = Vec3::new(enu_d.x as f32, enu_d.y as f32, enu_d.z as f32);
                Self::enu_vec_to_bevy(enu)
            }
            GeoFrame::GCRF => {
                // GCRF -> ECI (currently identity) -> ECEF -> ENU -> Bevy
                let r_gcrf = DVec3::new(v.x as f64, v.y as f64, v.z as f64);
                let r_eci = ctx.gcrf_to_eci(r_gcrf, t_seconds);
                let r_ecef = ctx.eci_to_ecef(r_eci, t_seconds);
                let enu_d = ctx.ecef_to_enu(r_ecef);
                let enu = Vec3::new(enu_d.x as f32, enu_d.y as f32, enu_d.z as f32);
                Self::enu_vec_to_bevy(enu)
            }
        }
    }

    /// Gravity vector in this frame, in m/s² (simple model).
    /// 
    /// `radius` is the geodetic radius of the planet in meters.
    pub fn gravity_accel(self, pos_in_frame: Vec3, radius: f64) -> Vec3 {
        const G0: f32 = 9.80665;
        let radius_m_f32 = radius as f32;

        match self {
            GeoFrame::EUS => Vec3::new(0.0, -G0, 0.0),
            GeoFrame::ENU => Vec3::new(0.0, 0.0, -G0),
            GeoFrame::NED => Vec3::new(0.0, 0.0, G0),
            GeoFrame::ECEF | GeoFrame::ECI => {
                let r2 = pos_in_frame.length_squared();
                if r2 == 0.0 {
                    return Vec3::ZERO;
                }
                let r = r2.sqrt();
                let dir_to_center = -pos_in_frame / r;
                let scale = G0 * (radius_m_f32 * radius_m_f32) / r2;
                dir_to_center * scale
            }
            GeoFrame::GCRF => Vec3::ZERO,
        }
    }

    /// Convert a vector (typically a position) from Bevy (EUS) into this frame.
    ///
    /// `ctx` provides origin + Earth rotation.
    /// `t_seconds` is simulation time, e.g. `time.elapsed_secs_f64()`.
    pub fn from_bevy_vec(self, v_bevy: Vec3, ctx: &GeoContext, t_seconds: f64) -> Vec3 {
        match self {
            GeoFrame::EUS => v_bevy,
            GeoFrame::ENU => Self::bevy_to_enu_vec(v_bevy),
            GeoFrame::NED => Self::bevy_to_ned_vec(v_bevy),
            GeoFrame::ECEF => {
                // Bevy -> ENU -> ECEF
                let enu = Self::bevy_to_enu_vec(v_bevy);
                let enu_d = DVec3::new(enu.x as f64, enu.y as f64, enu.z as f64);
                // Use map_3d to convert ENU to ECEF
                let (x, y, z) = enu2ecef(
                    enu_d.x,
                    enu_d.y,
                    enu_d.z,
                    ctx.origin.lat,
                    ctx.origin.lon,
                    ctx.origin.alt_m,
                    Ellipsoid::WGS84,
                );
                Vec3::new(x as f32, y as f32, z as f32)
            }
            GeoFrame::ECI => {
                // Bevy -> ENU -> ECEF -> ECI
                let enu = Self::bevy_to_enu_vec(v_bevy);
                let enu_d = DVec3::new(enu.x as f64, enu.y as f64, enu.z as f64);
                // Use map_3d to convert ENU to ECEF
                let (x, y, z) = enu2ecef(
                    enu_d.x,
                    enu_d.y,
                    enu_d.z,
                    ctx.origin.lat,
                    ctx.origin.lon,
                    ctx.origin.alt_m,
                    Ellipsoid::WGS84,
                );
                let ecef = DVec3::new(x, y, z);
                // Reverse ECI -> ECEF: ecef = Rz(theta) * eci, so eci = Rz(-theta) * ecef
                let theta = ctx.theta0_rad + ctx.earth_rot_rate_rad_per_s * t_seconds;
                let eci = GeoContext::rot_z(-theta) * ecef;
                Vec3::new(eci.x as f32, eci.y as f32, eci.z as f32)
            }
            GeoFrame::GCRF => {
                // Bevy -> ENU -> ECEF -> ECI -> GCRF (currently identity)
                let enu = Self::bevy_to_enu_vec(v_bevy);
                let enu_d = DVec3::new(enu.x as f64, enu.y as f64, enu.z as f64);
                // Use map_3d to convert ENU to ECEF
                let (x, y, z) = enu2ecef(
                    enu_d.x,
                    enu_d.y,
                    enu_d.z,
                    ctx.origin.lat,
                    ctx.origin.lon,
                    ctx.origin.alt_m,
                    Ellipsoid::WGS84,
                );
                let ecef = DVec3::new(x, y, z);
                let theta = ctx.theta0_rad + ctx.earth_rot_rate_rad_per_s * t_seconds;
                let eci = GeoContext::rot_z(-theta) * ecef;
                // GCRF == ECI currently
                Vec3::new(eci.x as f32, eci.y as f32, eci.z as f32)
            }
        }
    }

    /// Gravity vector expressed in Bevy's EUS coordinates.
    pub fn gravity_in_bevy(
        self,
        pos_in_frame: Vec3,
        ctx: &GeoContext,
        t_seconds: f64,
    ) -> Vec3 {
        let g_frame = self.gravity_accel(pos_in_frame, ctx.origin.shape.approx_radius());
        self.to_bevy_vec(g_frame, ctx, t_seconds)
    }

    /// Convert an angular velocity vector from this frame into Bevy’s EUS frame.
    pub fn to_bevy_ang_vel(
        self,
        w: Vec3,
        ctx: &GeoContext,
        t_seconds: f64,
    ) -> Vec3 {
        self.to_bevy_vec(w, ctx, t_seconds)
    }

    /// Returns the rotation that converts *this frame’s basis*
    /// into Bevy’s world-space EUS basis.
    pub fn basis_to_eus_quat(self, ctx: &GeoContext, t_seconds: f64) -> Quat {
        let m = self.basis_to_eus_mat3(ctx, t_seconds);
        Quat::from_mat3(&m)
    }

    /// Basis transform as Mat3. For now we implement simple cases and treat
    /// ECEF/ECI/GCRF via ENU near-origin.
    pub fn basis_to_eus_mat3(self, _ctx: &GeoContext, _t_seconds: f64) -> Mat3 {
        match self {
            GeoFrame::EUS => Mat3::IDENTITY,
            GeoFrame::ENU => {
                // Columns are frame basis vectors expressed in EUS.
                // ENU: e_hat = +X, n_hat = -Z, u_hat = +Y
                Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y)
            }
            GeoFrame::NED => {
                // NED: n_hat = -Z, e_hat = +X, d_hat = -Y
                Mat3::from_cols(Vec3::NEG_Z, Vec3::X, Vec3::NEG_Y)
            }
            // For these, a fully correct basis would require time-dependent
            // Earth orientation.
            GeoFrame::ECEF | GeoFrame::ECI | GeoFrame::GCRF => todo!(),
        }
    }
}

/// Per-entity geo position:
///   0: which frame the coords are in
///   1: position in that frame (ENU, NED, ECEF, ECI, GCRF, EUS).
#[derive(Component)]
pub struct GeoTranslation(pub GeoFrame, pub Vec3);

/// Per-entity geo velocity:
///   0: frame
///   1: velocity in that frame (m/s).
#[derive(Component)]
pub struct GeoVelocity(pub GeoFrame, pub Vec3);

/// Per-entity geo orientation:
///   0: frame the quaternion is expressed in
///   1: rotation from local -> that frame
#[derive(Component)]
pub struct GeoRotation(pub GeoFrame, pub Quat);

/// Per-entity angular velocity in some frame, in rad/s.
#[derive(Component)]
pub struct GeoAngularVelocity(pub GeoFrame, pub Vec3);

/// Plugin wiring: sets up `GeoContext` and systems that run
/// *before* transform propagation.
pub struct GeoFramePlugin {
    /// Where is world-space (0,0,0) on Earth?
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub origin_alt_m: f64,
    /// Initial Earth rotation angle at t=0 [rad].
    pub theta0_rad: f64,
    /// Planet/body shape model (currently used primarily for reference radius).
    pub shape: Shape,
}

impl Default for GeoFramePlugin {
    fn default() -> Self {
        Self {
            origin_lat_deg: 0.0,
            origin_lon_deg: 0.0,
            origin_alt_m: 0.0,
            theta0_rad: 0.0,
            shape: Shape::default(),
        }
    }
}

impl Plugin for GeoFramePlugin {
    fn build(&self, app: &mut App) {
        let ctx = GeoContext::new_from_degrees_with_shape(
            self.origin_lat_deg,
            self.origin_lon_deg,
            self.origin_alt_m,
            self.theta0_rad,
            self.shape,
        );

        app.insert_resource(ctx)
            // Integrate in frame space each Update
            .add_systems(Update, (integrate_geo_motion, integrate_geo_orientation))
            // Then convert to Bevy before transform propagation
            .add_systems(
                PostUpdate,
                (apply_geo_translation, apply_geo_rotation)
                    .chain()
                    .before(TransformSystem::TransformPropagate),
            );
    }
}

/// System: integrate motion in *frame* coordinates from GeoVelocity.
pub fn integrate_geo_motion(
    time: Res<Time>,
    ctx: Res<GeoContext>,
    mut q: Query<(&mut GeoTranslation, &GeoVelocity)>,
) {
    let dt = time.delta_secs();
    let t = time.elapsed_secs_f64();
    for (mut geo_pos, geo_vel) in &mut q {
        let v = dbg!(geo_pos.0.convert_to(geo_vel.1, &geo_vel.0, &ctx, t));
        geo_pos.1 += v * dt;
    }
}

/// Integrate `GeoRotation` in *frame space* using `GeoAngularVelocity`.
pub fn integrate_geo_orientation(
    time: Res<Time>,
    mut q: Query<(&mut GeoRotation, &GeoAngularVelocity)>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    for (mut geo_rot, ang) in &mut q {
        if geo_rot.0 != ang.0 {
            continue;
        }

        let omega = ang.1; // rad/s in that frame
        let speed = omega.length();
        if speed == 0.0 {
            continue;
        }

        let axis = omega / speed;
        let angle = speed * dt; // radians this frame

        let delta = Quat::from_axis_angle(axis, angle);
        geo_rot.1 = delta * geo_rot.1;
    }
}

/// System: convert `GeoTranslation` into `Transform.translation`
/// right before Bevy propagates transforms through the hierarchy.
pub fn apply_geo_translation(
    ctx: Res<GeoContext>,
    time: Res<Time>,
    mut q: Query<(&GeoTranslation, &mut Transform)>,
) {
    let t = time.elapsed_secs_f64();
    for (geo, mut transform) in &mut q {
        transform.translation = geo.0.to_bevy_vec(geo.1, &ctx, t);
    }
}

/// System: convert `GeoRotation` into `Transform.rotation`.
pub fn apply_geo_rotation(
    ctx: Res<GeoContext>,
    time: Res<Time>,
    mut q: Query<(&GeoRotation, &mut Transform)>,
) {
    let t = time.elapsed_secs_f64();

    for (geo_rot, mut transform) in &mut q {
        let frame = geo_rot.0;
        let local_rot = geo_rot.1;

        let frame_to_eus = frame.basis_to_eus_quat(&ctx, t);
        transform.rotation = frame_to_eus * local_rot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> GeoContext {
        GeoContext::new_from_degrees(0.0, 0.0, 0.0, 0.0)
    }

    #[test]
    fn eus_identity_mapping() {
        let ctx = dummy_ctx();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let mapped = GeoFrame::EUS.to_bevy_vec(v, &ctx, 0.0);
        assert_eq!(mapped, v);
    }

    #[test]
    fn enu_to_eus_axes() {
        let ctx = dummy_ctx();
        // east, north, up in ENU
        let east = Vec3::new(1.0, 0.0, 0.0);
        let north = Vec3::new(0.0, 1.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);

        let east_eus = GeoFrame::ENU.to_bevy_vec(east, &ctx, 0.0);
        let north_eus = GeoFrame::ENU.to_bevy_vec(north, &ctx, 0.0);
        let up_eus = GeoFrame::ENU.to_bevy_vec(up, &ctx, 0.0);

        assert_eq!(east_eus, Vec3::new(1.0, 0.0, 0.0));   // +X
        assert_eq!(north_eus, Vec3::new(0.0, 0.0, -1.0)); // -Z
        assert_eq!(up_eus, Vec3::new(0.0, 1.0, 0.0));     // +Y
    }

    #[test]
    fn ned_to_eus_axes() {
        let ctx = dummy_ctx();
        // north, east, down in NED
        let north = Vec3::new(1.0, 0.0, 0.0);
        let east = Vec3::new(0.0, 1.0, 0.0);
        let down = Vec3::new(0.0, 0.0, 1.0);

        let north_eus = GeoFrame::NED.to_bevy_vec(north, &ctx, 0.0);
        let east_eus = GeoFrame::NED.to_bevy_vec(east, &ctx, 0.0);
        let down_eus = GeoFrame::NED.to_bevy_vec(down, &ctx, 0.0);

        assert_eq!(north_eus, Vec3::new(0.0, 0.0, -1.0)); // -Z
        assert_eq!(east_eus, Vec3::new(1.0, 0.0, 0.0));   // +X
        assert_eq!(down_eus, Vec3::new(0.0, -1.0, 0.0));  // -Y
    }

    #[test]
    fn gravity_in_enu_and_eus() {
        let ctx = dummy_ctx();
        let pos = Vec3::ZERO;

        let g_enu = GeoFrame::ENU.gravity_accel(pos, ctx.origin.shape.approx_radius());
        let g_eus = GeoFrame::EUS.gravity_accel(pos, ctx.origin.shape.approx_radius());

        assert!((g_enu - Vec3::new(0.0, 0.0, -9.80665)).length() < 1e-5);
        assert!((g_eus - Vec3::new(0.0, -9.80665, 0.0)).length() < 1e-5);

        let g_enu_world = GeoFrame::ENU.gravity_in_bevy(pos, &ctx, 0.0);
        let g_eus_world = GeoFrame::EUS.gravity_in_bevy(pos, &ctx, 0.0);

        // ENU gravity mapped into EUS should match direct EUS gravity
        assert!((g_enu_world - g_eus_world).length() < 1e-4);
    }

    #[test]
    fn integrate_geo_motion_basic() {
        let mut world = World::new();
        // Initialize Time resource - in Bevy 0.16, Res<Time> resolves to Time<Real>
        world.init_resource::<Time>();
        world.init_resource::<GeoContext>();
        // Advance time to have a delta
        let mut time = world.resource_mut::<Time>();
        time.advance_by(std::time::Duration::from_secs(1));

        world.spawn((
            GeoTranslation(GeoFrame::ENU, Vec3::ZERO),
            GeoVelocity(GeoFrame::ENU, Vec3::new(1.0, 0.0, 0.0)),
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(integrate_geo_motion);

        schedule.run(&mut world);

        let mut q = world.query::<&GeoTranslation>();
        let geo = q.single(&world).unwrap();
        assert_eq!(geo.1, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn round_trip_eus() {
        let ctx = dummy_ctx();
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::EUS.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::EUS.from_bevy_vec(bevy, &ctx, 0.0);
        assert!((round_trip - original).length() < 1e-5);
    }

    #[test]
    fn round_trip_enu() {
        let ctx = dummy_ctx();
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ENU.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::ENU.from_bevy_vec(bevy, &ctx, 0.0);
        assert!((round_trip - original).length() < 1e-5);
    }

    #[test]
    fn round_trip_ned() {
        let ctx = dummy_ctx();
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::NED.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::NED.from_bevy_vec(bevy, &ctx, 0.0);
        assert!((round_trip - original).length() < 1e-5);
    }

    #[test]
    fn round_trip_ecef() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ECEF.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::ECEF.from_bevy_vec(bevy, &ctx, 0.0);
        // ECEF involves coordinate transformations, so use a slightly larger tolerance
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn round_trip_eci() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ECI.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::ECI.from_bevy_vec(bevy, &ctx, 0.0);
        // ECI involves time-dependent transformations, so use a slightly larger tolerance
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn round_trip_gcrf() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = Vec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::GCRF.to_bevy_vec(original, &ctx, 0.0);
        let round_trip = GeoFrame::GCRF.from_bevy_vec(bevy, &ctx, 0.0);
        // GCRF involves time-dependent transformations, so use a slightly larger tolerance
        assert!((round_trip - original).length() < 1e-3);
    }
}
