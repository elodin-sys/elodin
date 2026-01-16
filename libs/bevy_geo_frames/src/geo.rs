use bevy::math::{DMat3, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use map_3d::{ecef2enu, enu2ecef, Ellipsoid};

/// Default Earth radius in meters (approximate mean radius).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Earth sidereal spin
pub const EARTH_SIDEREAL_SPIN: f64 = 7.292_115_0e-5;

pub const DEFAULT_RENDER: GeoFrame = GeoFrame::ECEF;

/// Planet/body shape model used for things like gravity scaling and debug rendering.
///
/// NOTE: Any time we call into `map_3d` we must provide an `Ellipsoid`. We always
/// source that from `Shape::ellipsoid()` so `Shape::Sphere { .. }` can use
/// `Ellipsoid::UnitSphere` (and you can scale the resulting ECEF/ENU numbers however
/// you like in your game).
#[derive(Debug, Clone, Copy)]
pub enum Shape {
    /// A spherical body with a single reference radius.
    Sphere { radius: f64 },
    /// An ellipsoidal body (from `map_3d`).
    Ellipsoid(Ellipsoid),
}

impl Shape {

    /// Ellipsoid to use for latitude/longitude <-> ECEF conversions.
    pub fn ellipsoid(self) -> Ellipsoid {
        match self {
            Shape::Ellipsoid(e) => e,
            Shape::Sphere { .. } => Ellipsoid::UnitSphere,
        }
    }

    /// Reference radius used for simple gravity scaling and visualization.
    pub fn approx_radius(&self) -> f64 {
        match self {
            Shape::Ellipsoid(_) => EARTH_RADIUS_M,
            Shape::Sphere { radius } => *radius
        }
    }

    /// A radius scale factor
    fn scale_factor(&self) -> f64 {
        match self {
            Shape::Ellipsoid(_) => 1.0,
            Shape::Sphere { radius } => *radius
        }
    }


}

impl Default for Shape {
    fn default() -> Self {
        Shape::Ellipsoid(Ellipsoid::WGS84)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
/// How should these coordinates be presented.
pub enum Present {
    #[default]
    /// Present them from the perspective of a Plane on the ground.
    ///
    /// In some sense the latitude and longitude can just be ignored.
    ///
    /// e.g., ENU (x,y,z) -> Bevy (x, z, -y)
    Plane,
    /// Present them from the perspective of a Sphere.
    ///
    /// Here the latitude and longitude place us on the sphere, and then we use
    /// the coordinates from there.
    Sphere,
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
#[derive(Default, Debug, Clone, Copy)]
pub struct GeoOrigin {
    /// Geodetic latitude [rad]
    pub latitude: f64,
    /// Geodetic longitude [rad]
    pub longitude: f64,
    /// Altitude above mean radius [m]
    pub altitude: f64,
    /// Planet/body shape model (currently used primarily for reference radius).
    pub shape: Shape,
}

impl GeoOrigin {
    /// Simple spherical Earth model (good enough for games).
    /// Uses default Earth radius.
    pub fn new_from_degrees(
        latitude_deg: f64,
        longitude_deg: f64,
        altitude: f64,
    ) -> Self {
        let latitude = latitude_deg.to_radians();
        let longitude = longitude_deg.to_radians();
        Self {
            latitude,
            longitude,
            altitude,
            shape: Shape::default(),
        }
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
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
    /// Time in seconds
    pub time: f64,
    /// Presentation mode: plane or sphere
    pub present: Present,
}

impl Default for GeoContext {
    fn default() -> Self {
        let origin = GeoOrigin::default();
        Self {
            origin,
            theta0_rad: 0.0,
            earth_rot_rate_rad_per_s: EARTH_SIDEREAL_SPIN,
            time: 0.0,
            present: Present::default()
        }
    }
}

impl From<GeoOrigin> for GeoContext {
    fn from(origin: GeoOrigin) -> Self {
        let mut ctx = GeoContext::default();
        ctx.origin = origin;
        ctx
    }
}

impl GeoContext {
    pub fn with_rotation_angle(mut self, theta0_rad: f64) -> Self {
        self.theta0_rad = theta0_rad;
        self
    }

    fn stable_lat(&self) -> f64 {
        self.origin.latitude
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
    pub fn eci_to_ecef(&self, r_eci: DVec3) -> DVec3 {
        let theta = self.theta0_rad + self.earth_rot_rate_rad_per_s * self.time;
        Self::rot_z(theta) * r_eci
    }

    /// Convert ECEF -> ECI at simulation time `self.time`.
    pub fn ecef_to_eci(&self, r_ecef: DVec3) -> DVec3 {
        let theta = self.theta0_rad + self.earth_rot_rate_rad_per_s * self.time;
        Self::rot_z(-theta) * r_ecef
    }

    /// Convert GCRF -> ECI.
    ///
    /// For now we approximate GCRF == ECI (J2000), so this is identity.
    pub fn gcrf_to_eci(&self, r_gcrf: DVec3, _t_seconds: f64) -> DVec3 {
        r_gcrf
    }

    /// Returns a longitude used for local ENU bases that is stable at the poles.
    ///
    /// At |latitude| ≈ 90°, longitude becomes ill-defined and can cause the ENU basis
    /// to flip as `longitude` varies. This function smoothly suppresses the effect of
    /// longitude as `cos(latitude)` goes to zero, yielding consistent results at the poles
    /// while matching the true longitude away from them.
    fn stable_lon(&self) -> f64 {
        self.origin.longitude
        // let (sin_lat, cos_lat) = self.origin.latitude.sin_cos();
        // let (sin_lon, cos_lon) = self.origin.longitude.sin_cos();
        // let denom = cos_lon * cos_lat + sin_lat.abs();
        // (sin_lon * cos_lat).atan2(denom)
    }

    /// ECEF -> local ENU at the origin.
    pub fn ecef_to_enu(&self, r_ecef: DVec3) -> DVec3 {
        let (e, n, u) = ecef2enu(
            r_ecef.x,
            r_ecef.y,
            r_ecef.z,
            self.stable_lat(),
            self.stable_lon(),
            self.origin.altitude,
            self.origin.shape.ellipsoid(),
        );
        DVec3::new(e, n, u)
    }
}

impl GeoFrame {
    /// ENU: (E, N, U) -> local EUS: (E, U, -N)
    #[inline]
    fn enu_to_eus(v_enu: DVec3) -> DVec3 {
        DVec3::new(v_enu.x, v_enu.z, -v_enu.y)
    }

    /// NED: (N, E, D) -> local EUS: (E, -D, -N)
    #[inline]
    fn ned_to_eus(v_ned: DVec3) -> DVec3 {
        DVec3::new(v_ned.y, -v_ned.z, -v_ned.x)
    }

    /// local EUS: (E, U, S) -> ENU: (E, N, U) with N = -S
    #[inline]
    fn eus_to_enu(v_eus: DVec3) -> DVec3 {
        DVec3::new(v_eus.x, -v_eus.z, v_eus.y)
    }

    /// local EUS: (E, U, S) -> NED: (N, E, D) with N = -S, D = -U
    #[inline]
    fn eus_to_ned(v_eus: DVec3) -> DVec3 {
        DVec3::new(-v_eus.z, v_eus.x, -v_eus.y)
    }

    #[inline]
    fn eus_to_bevy(v_eus: DVec3) -> Vec3 {
        Vec3::new(v_eus.x as f32, v_eus.y as f32, v_eus.z as f32)
    }

    #[inline]
    fn bevy_to_eus(v_bevy: Vec3) -> DVec3 {
        DVec3::new(v_bevy.x as f64, v_bevy.y as f64, v_bevy.z as f64)
    }

    /// Converts ENU (East-North-Up) coordinates to a 3D position on a sphere in Bevy coordinates
    ///
    /// This function takes ENU coordinates relative to a reference point and converts them
    /// to a position on a sphere suitable for rendering in Bevy.
    ///
    /// ## Inputs:
    /// - e = east coordinate [m] from reference point
    /// - n = north coordinate [m] from reference point
    /// - u = up coordinate [m] from reference point
    /// - lat0 = reference latitude [rad] of the reference point
    /// - lon0 = reference longitude [rad] of the reference point
    /// - alt0 = reference altitude [m] of the reference point
    /// - r_ellips = reference ellipsoid (e.g., Ellipsoid::WGS84)
    /// - sphere_scale = scale factor to convert meters to Bevy units (e.g., 1.0 for 1:1, 0.001 for mm to m)
    ///
    /// ## Outputs:
    /// - (x, y, z) tuple suitable for Bevy's Vec3 representing position on the sphere
    ///   - Coordinates are in Bevy units (scaled by sphere_scale)
    ///   - X points right (East)
    ///   - Y points up
    ///   - Z points forward (negative North in typical Bevy convention)
    ///
    /// ## Example:
    /// ```rust
    /// use bevy::prelude::*;
    /// use bevy::math::f64::DVec3;
    /// use map_3d::{Ellipsoid};
    /// use bevy_geo_frames::{GeoPosition, GeoOrigin, GeoContext, GeoFrame};
    /// use std::f64::consts::PI;
    ///
    /// // Reference point: New York (40.7°N, -73.9°W, 0m altitude)
    /// let lat0 = 40.7_f64.to_radians();
    /// let lon0 = -73.9_f64.to_radians();
    /// let alt0 = 0.0;
    /// let context = GeoOrigin::new_from_degrees(lat0, lon0, alt0).into();
    ///
    /// // ENU coordinates: 100m east, 50m north, 10m up from reference
    /// let v = GeoPosition(GeoFrame::ENU, DVec3::new(100.0, 50.0, 10.0)).to_bevy_sphere(
    ///     &context,
    /// );
    /// // In Bevy: Vec3::new(x as f32, y as f32, z as f32)
    /// ```
    // pub fn to_bevy_sphere(
    //     &self,
    //     v: DVec3,
    //     ctx: &GeoContext,
    // ) -> DVec3 {
    //     todo!()
        // let scale = ctx.origin.scale_factor();
        // let w = GeoFrame::ECEF.convert_to(v, ctx);
        // w.yzx()

        // // Convert into the local EUS offset first.
        // let local_eus = self.to_eus(v, ctx);
        // let local = DVec3::new(local_eus.x, local_eus.y, local_eus.z);

        // // Rotate local offsets into sphere presentation: yaw by longitude, then pitch by (latitude - 90°).
        // let pitch = ctx.origin.latitude - std::f64::consts::FRAC_PI_2;
        // let (sin_pitch, cos_pitch) = pitch.sin_cos();
        // let (sin_lon, cos_lon) = ctx.origin.longitude.sin_cos();

        // let local_pitched = DVec3::new(
        //     local.x,
        //     cos_pitch * local.y - sin_pitch * local.z,
        //     sin_pitch * local.y + cos_pitch * local.z,
        // );
        // let local_rotated = DVec3::new(
        //     cos_lon * local_pitched.x + sin_lon * local_pitched.z,
        //     local_pitched.y,
        //     -sin_lon * local_pitched.x + cos_lon * local_pitched.z,
        // );

        // let origin_local = DVec3::new(0.0, scale, 0.0);
        // let origin_pitched = DVec3::new(
        //     origin_local.x,
        //     cos_pitch * origin_local.y - sin_pitch * origin_local.z,
        //     sin_pitch * origin_local.y + cos_pitch * origin_local.z,
        // );
        // let origin_rotated = DVec3::new(
        //     cos_lon * origin_pitched.x + sin_lon * origin_pitched.z,
        //     origin_pitched.y,
        //     -sin_lon * origin_pitched.x + cos_lon * origin_pitched.z,
        // );

        // origin_rotated + local_rotated
    // }
    fn to_eus(self, v: DVec3, ctx: &GeoContext) -> DVec3 {
        match self {
            GeoFrame::EUS => v,
            GeoFrame::ENU => Self::enu_to_eus(v),
            GeoFrame::NED => Self::ned_to_eus(v),
            GeoFrame::ECEF => Self::enu_to_eus(ctx.ecef_to_enu(v)),
            GeoFrame::ECI => {
                let ecef = ctx.eci_to_ecef(v);
                Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            }
            GeoFrame::GCRF => {
                let eci = ctx.gcrf_to_eci(v, ctx.time);
                let ecef = ctx.eci_to_ecef(eci);
                Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            }
        }
    }

    fn from_eus(self, v_eus: DVec3, ctx: &GeoContext) -> DVec3 {
        match self {
            GeoFrame::EUS => v_eus,
            GeoFrame::ENU => Self::eus_to_enu(v_eus),
            GeoFrame::NED => Self::eus_to_ned(v_eus),
            GeoFrame::ECEF => {
                let enu = Self::eus_to_enu(v_eus);
                let (x, y, z) = enu2ecef(
                    enu.x,
                    enu.y,
                    enu.z,
                    ctx.stable_lat(),
                    ctx.stable_lon(),
                    ctx.origin.altitude,
                    ctx.origin.shape.ellipsoid(),
                );
                DVec3::new(x, y, z)
            }
            GeoFrame::ECI => {
                let ecef = GeoFrame::ECEF.from_eus(v_eus, ctx);
                ctx.ecef_to_eci(ecef)
            }
            GeoFrame::GCRF => {
                // GCRF == ECI for now
                let ecef = GeoFrame::ECEF.from_eus(v_eus, ctx);
                ctx.ecef_to_eci(ecef)
            }
        }
    }

    /// Convert a DVec3 (position/velocity) from `from_frame` into `self`.
    pub fn convert_to(&self, v: DVec3, from_frame: GeoFrame, ctx: &GeoContext) -> DVec3 {
        if from_frame == *self {
            return v;
        }
        use GeoFrame::*;
        match (from_frame, *self) {
            (x, y) if x == y => v,
            (ECEF, ECI) => ctx.eci_to_ecef(v),
            (ECEF, GCRF) => ctx.eci_to_ecef(ctx.gcrf_to_eci(v, ctx.time)),
            (ENU, ECEF) => map_3d::enu2ecef(
                    v.x,
                    v.y,
                    v.z,
                    ctx.stable_lat(),
                    ctx.stable_lon(),
                    ctx.origin.altitude,
                    ctx.origin.shape.ellipsoid(),
                ).into(),
            (NED, ECEF) => map_3d::ned2ecef(
                    v.x,
                    v.y,
                    v.z,
                    ctx.stable_lat(),
                    ctx.stable_lon(),
                    ctx.origin.altitude,
                    ctx.origin.shape.ellipsoid(),
                ).into(),
            (ECEF, NED) => {
                let eus = Self::ned_to_eus(v);
                let enu = Self::eus_to_enu(eus);
                map_3d::enu2ecef(
                    enu.x,
                    enu.y,
                    enu.z,
                    ctx.stable_lat(),
                    ctx.stable_lon(),
                    ctx.origin.altitude,
                    ctx.origin.shape.ellipsoid(),
                )
                .into()
            }
            (ECEF, ENU) => map_3d::ecef2enu(
                    v.x,
                    v.y,
                    v.z,
                    ctx.stable_lat(),
                    ctx.stable_lon(),
                    ctx.origin.altitude,
                    ctx.origin.shape.ellipsoid(),
                ).into(),
            (ECEF, EUS) => v.yzx(),
            // (ECI, EUS) =>
            (EUS, ECEF) => v.zxy(),
            (EUS, NED) => Self::eus_to_ned(v),
            (EUS, ENU) => Self::eus_to_enu(v),
            (NED, EUS) => Self::ned_to_eus(v),
            (ENU, EUS) => Self::enu_to_eus(v),
            (EUS, ECI) => {
                let ecef = ctx.eci_to_ecef(v);
                Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            }
            (EUS, GCRF) => {
                let eci = ctx.gcrf_to_eci(v, ctx.time);
                let ecef = ctx.eci_to_ecef(eci);
                Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            }
            (x, y) => todo!("{x:?} -> {y:?}"),
            // GeoFrame::EUS => v,
            // GeoFrame::ENU => Self::enu_to_eus(v),
            // GeoFrame::NED => Self::ned_to_eus(v),
            // GeoFrame::ECEF => Self::enu_to_eus(ctx.ecef_to_enu(v)),
            // GeoFrame::ECI => {
            //     let ecef = ctx.eci_to_ecef(v);
            //     Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            // }
            // GeoFrame::GCRF => {
            //     let eci = ctx.gcrf_to_eci(v, ctx.time);
            //     let ecef = ctx.eci_to_ecef(eci);
            //     Self::enu_to_eus(ctx.ecef_to_enu(ecef))
            // }
        }
        // let eus = from_frame.to_eus(v, ctx);
        // self.from_eus(eus, ctx)
    }

    /// Convert a position in this frame into Bevy world-space translation.
    /// Internally we keep everything in `DVec3` and only cast at the boundary.
    pub fn to_bevy_pos(self, v: DVec3, ctx: &GeoContext) -> Vec3 {
        Self::eus_to_bevy(self.to_eus(v, ctx))
    }

    /// Convert a Bevy world-space translation into a position in this frame.
    pub fn from_bevy_pos(self, v_bevy: Vec3, ctx: &GeoContext) -> DVec3 {
        self.from_eus(Self::bevy_to_eus(v_bevy), ctx)
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
    /// `time` is simulation time, e.g. `time.elapsed_secs_f64()`.
    // NOTE: legacy `from_bevy_vec` removed in favor of `from_bevy_pos`.

    /// Gravity vector expressed in Bevy's EUS coordinates.
    pub fn gravity_in_bevy(
        self,
        pos_in_frame: Vec3,
        ctx: &GeoContext,
    ) -> Vec3 {
        let g_frame = self.gravity_accel(pos_in_frame, ctx.origin.shape.approx_radius());
        match self {
            GeoFrame::EUS => g_frame,
            GeoFrame::ENU => Vec3::new(g_frame.x, g_frame.z, -g_frame.y),
            GeoFrame::NED => Vec3::new(g_frame.y, -g_frame.z, -g_frame.x),
            // For these frames, a correct mapping would need a proper local basis at the origin.
            GeoFrame::ECEF | GeoFrame::ECI | GeoFrame::GCRF => Vec3::ZERO,
        }
    }

    /// Convert an angular velocity vector from this frame into Bevy’s EUS frame.
    pub fn to_bevy_ang_vel(
        self,
        w: Vec3,
        _ctx: &GeoContext,
    ) -> Vec3 {
        match self {
            GeoFrame::EUS => w,
            GeoFrame::ENU => Vec3::new(w.x, w.z, -w.y),
            GeoFrame::NED => Vec3::new(w.y, -w.z, -w.x),
            GeoFrame::ECEF | GeoFrame::ECI | GeoFrame::GCRF => Vec3::ZERO,
        }
    }

    /// Returns the rotation that converts *this frame’s basis*
    /// into Bevy’s world-space EUS basis.
    pub fn basis_to_eus_quat(self, ctx: &GeoContext) -> Quat {
        let m = self.basis_to_eus_mat3(ctx);
        Quat::from_mat3(&m)
    }

    /// Basis transform as Mat3. For now we implement simple cases and treat
    /// ECEF/ECI/GCRF via ENU near-origin.
    pub fn basis_to_eus_mat3(self, _ctx: &GeoContext) -> Mat3 {
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
pub struct GeoPosition(pub GeoFrame, pub DVec3);

impl GeoPosition {
    pub fn to_bevy_sphere(&self, context: &GeoContext) -> DVec3 {
        // let scale = 1.0; //* context.origin.shape.scale_factor();
        let w = GeoFrame::ECEF.convert_to(self.1, self.0, context);
        w.yzx()
    }

    pub fn to_bevy_plane(&self, context: &GeoContext) -> DVec3 {
        let scale = context.origin.shape.scale_factor();
        // GeoFrame::EUS.convert_to(self.1/scale, self.0, context)
        GeoFrame::EUS.convert_to(self.1, self.0, context)
    }
}

/// Per-entity geo velocity:
///   0: frame
///   1: velocity in that frame (m/s).
#[derive(Component)]
pub struct GeoVelocity(pub GeoFrame, pub DVec3);

/// Per-entity geo orientation:
///   0: frame the quaternion is expressed in
///   1: rotation from local -> that frame
#[derive(Component)]
pub struct GeoRotation(pub GeoFrame, pub Quat);

/// Per-entity angular velocity in some frame, in rad/s.
#[derive(Component)]
pub struct GeoAngularVelocity(pub GeoFrame, pub Vec3);

#[derive(Default)]
/// Plugin wiring: sets up `GeoContext` and systems that run
/// *before* transform propagation.
pub struct GeoFramePlugin {
    pub context: Option<GeoContext>,
    pub origin: Option<GeoOrigin>,
}

impl Plugin for GeoFramePlugin {
    fn build(&self, app: &mut App) {
        let mut ctx = self.context.unwrap_or_default();
        if let Some(origin) = self.origin {
            ctx.origin = origin;
        }
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
    mut ctx: ResMut<GeoContext>,
    mut q: Query<(&mut GeoPosition, &GeoVelocity)>,
) {
    ctx.time = time.elapsed_secs_f64();
    let dt = time.delta_secs_f64();
    for (mut geo_pos, geo_vel) in &mut q {
        let v = geo_pos.0.convert_to(geo_vel.1, geo_vel.0, &ctx);
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

/// System: convert `GeoPosition` into `Transform.translation`
/// right before Bevy propagates transforms through the hierarchy.
pub fn apply_geo_translation(
    mut ctx: ResMut<GeoContext>,
    time: Res<Time>,
    mut q: Query<(&GeoPosition, &mut Transform)>,
) {
    ctx.time = time.elapsed_secs_f64();
    let ctx_ref: &GeoContext = &*ctx;
    for (geo, mut transform) in &mut q {
        // let pos_in_render = render.convert_to(geo.1, geo.0, ctx_ref);
        transform.translation = match ctx_ref.present {
            Present::Plane => geo.to_bevy_plane(ctx_ref).as_vec3(),
            Present::Sphere => geo.to_bevy_sphere(ctx_ref).as_vec3()
        };
    }
}

/// System: convert `GeoRotation` into `Transform.rotation`.
pub fn apply_geo_rotation(
    ctx: Res<GeoContext>,
    mut q: Query<(&GeoRotation, &mut Transform)>,
) {
    for (geo_rot, mut transform) in &mut q {
        let frame = geo_rot.0;
        let local_rot = geo_rot.1;

        let frame_to_eus = frame.basis_to_eus_quat(&ctx);
        transform.rotation = frame_to_eus * local_rot;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> GeoContext {
        GeoContext::default()
    }
    macro_rules! assert_approx_eq {
        ($x: expr, $y: expr) => {
            assert_approx_eq!($x, $y, 1e-5);
        };
        ($x: expr, $y: expr, $e: expr) => {
            let a = $x;
            let b = $y;
            let eps = $e;
            assert!((a - b).length() <= eps,
                    "got {:?} expected {:?}", a, b);
        };
        ($x: expr, $y: expr, $e: expr, $l: expr) => {
            let a = $x;
            let b = $y;
            let eps = $e;
            assert!((a - b).length() <= eps,
                    "{}: got {:?} expected {:?}", $l, a, b);
        };
    }

    fn assert_vec3_close(label: &str, a: Vec3, b: Vec3, eps: f32) {
        assert!(
            (a - b).length() <= eps,
            "{label}: got {a:?}, expected {b:?}"
        );
    }

    #[test]
    fn eus_identity_mapping() {
        let ctx = dummy_ctx();
        let v = DVec3::new(1.0, 2.0, 3.0);
        let mapped = GeoFrame::EUS.to_bevy_pos(v, &ctx);
        assert_eq!(mapped, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn enu_to_eus_axes() {
        let ctx = dummy_ctx();
        // east, north, up in ENU
        let east = DVec3::new(1.0, 0.0, 0.0);
        let north = DVec3::new(0.0, 1.0, 0.0);
        let up = DVec3::new(0.0, 0.0, 1.0);

        let east_eus = GeoFrame::ENU.to_bevy_pos(east, &ctx);
        let north_eus = GeoFrame::ENU.to_bevy_pos(north, &ctx);
        let up_eus = GeoFrame::ENU.to_bevy_pos(up, &ctx);

        assert_eq!(east_eus, Vec3::new(1.0, 0.0, 0.0));   // +X
        assert_eq!(north_eus, Vec3::new(0.0, 0.0, -1.0)); // -Z
        assert_eq!(up_eus, Vec3::new(0.0, 1.0, 0.0));     // +Y
    }

    #[test]
    fn ned_to_eus_axes() {
        let ctx = dummy_ctx();
        // north, east, down in NED
        let north = DVec3::new(1.0, 0.0, 0.0);
        let east = DVec3::new(0.0, 1.0, 0.0);
        let down = DVec3::new(0.0, 0.0, 1.0);

        let north_eus = GeoFrame::NED.to_bevy_pos(north, &ctx);
        let east_eus = GeoFrame::NED.to_bevy_pos(east, &ctx);
        let down_eus = GeoFrame::NED.to_bevy_pos(down, &ctx);

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

        let g_enu_world = GeoFrame::ENU.gravity_in_bevy(pos, &ctx);
        let g_eus_world = GeoFrame::EUS.gravity_in_bevy(pos, &ctx);

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
            GeoPosition(GeoFrame::ENU, DVec3::ZERO),
            GeoVelocity(GeoFrame::ENU, DVec3::new(1.0, 0.0, 0.0)),
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(integrate_geo_motion);

        schedule.run(&mut world);

        let mut q = world.query::<&GeoPosition>();
        let geo = q.single(&world).unwrap();
        assert_eq!(geo.1, DVec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn round_trip_eus() {
        let ctx = dummy_ctx();
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::EUS.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::EUS.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-9);
    }

    #[test]
    fn round_trip_enu() {
        let ctx = dummy_ctx();
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ENU.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::ENU.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-9);
    }

    #[test]
    fn round_trip_ned() {
        let ctx = dummy_ctx();
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::NED.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::NED.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-9);
    }

    #[test]
    fn round_trip_ecef() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ECEF.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::ECEF.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn round_trip_eci() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::ECI.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::ECI.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn round_trip_gcrf() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoFrame::GCRF.to_bevy_pos(original, &ctx);
        let round_trip = GeoFrame::GCRF.from_bevy_pos(bevy, &ctx);
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn zero_conversions_plane_and_sphere() {
        let origins = [
            (
                "equator",
                GeoOrigin::new_from_degrees(0.0, 0.0, 0.0),
                [
                    (GeoFrame::EUS, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ECI, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                    (GeoFrame::GCRF, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                ],
            ),
            (
                "north_pole",
                GeoOrigin::new_from_degrees(90.0, 0.0, 0.0),
                [
                    (GeoFrame::EUS, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
                    // (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ECI, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                    (GeoFrame::GCRF, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                ],
            ),
            (
                "south_pole",
                GeoOrigin::new_from_degrees(-90.0, 0.0, 0.0),
                [
                    (GeoFrame::EUS, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                    (GeoFrame::ECI, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                    (GeoFrame::GCRF, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO),
                ],
            ),
        ];
        let eps = 1e-5;

        for (label, origin, expectations) in origins {
            let ctx: GeoContext = origin.with_shape(Shape::Sphere { radius: 1.0 }).into();
            for (frame, expected_plane, expected_sphere) in expectations {
                // let zero = GeoPosition(frame, DVec3::ZERO);
                let zero = GeoPosition(frame, -DVec3::ZERO);
                let plane = zero.to_bevy_plane(&ctx).as_vec3();
                // assert_approx_eq(
                //     &format!("{frame:?} plane zero ({label})"),
                //     plane,
                //     expected_plane,
                //     eps,
                // );
                assert_approx_eq!(
                    plane,
                    expected_plane,
                    eps,
                    format!("{frame:?} plane zero ({label})")
                );

                let sphere = zero.to_bevy_sphere(&ctx).as_vec3();
                assert_approx_eq!(
                    sphere,
                    expected_sphere,
                    eps,
                    format!("{frame:?} sphere zero ({label})")
                );
            }
        }
    }

    #[ignore]
    #[test]
    fn present_plane_and_sphere_at_equator_origin() {
        let radius = 1.0;
        let ctx: GeoContext = GeoOrigin::new_from_degrees(0.0, 0.0, 0.0)
            .with_shape(Shape::Sphere { radius })
            .into();
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (
                GeoFrame::EUS,
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(3.0, -3.0, -1.0),
            ),
            (
                GeoFrame::ENU,
                Vec3::new(1.0, 3.0, -2.0),
                Vec3::new(4.0, 2.0, -1.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
                Vec3::new(-2.0, 1.0, -2.0),
            ),
            (
                GeoFrame::ECEF,
                Vec3::new(2.0, 0.0, -3.0),
                Vec3::new(1.0, 3.0, -2.0),
            ),
            (
                GeoFrame::ECI,
                Vec3::new(2.0, 0.0, -3.0),
                Vec3::new(1.0, 3.0, -2.0),
            ),
            (
                GeoFrame::GCRF,
                Vec3::new(2.0, 0.0, -3.0),
                Vec3::new(1.0, 3.0, -2.0),
            ),
        ];

        for (frame, expected_plane, expected_sphere) in cases {
            let plane = frame.to_bevy_pos(v, &ctx);
            assert_vec3_close(
                &format!("{frame:?} plane (equator)"),
                plane,
                expected_plane,
                eps,
            );

            let sphere_d = GeoPosition(frame, v).to_bevy_sphere(&ctx);
            let sphere = Vec3::new(sphere_d.x as f32, sphere_d.y as f32, sphere_d.z as f32);
            assert_vec3_close(
                &format!("{frame:?} sphere (equator)"),
                sphere,
                expected_sphere,
                eps,
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_north_pole() {
        let radius = 1.0;
        let ctx: GeoContext = GeoOrigin::new_from_degrees(90.0, 0.0, 0.0)
            .with_shape(Shape::Sphere { radius })
            .into();
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (
                GeoFrame::EUS,
                Vec3::new(1.0, 2.0, 3.0),
            ),
            (
                GeoFrame::ENU,
                Vec3::new(1.0, 3.0, -2.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
            ),
            (
                GeoFrame::ECEF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::ECI,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::GCRF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
        ];

        for (frame, expected_plane) in cases {
            let plane = frame.to_bevy_pos(v, &ctx);
            assert_vec3_close(
                &format!("{frame:?} plane (north pole)"),
                plane,
                expected_plane,
                eps,
            );

            let sphere_d = GeoPosition(frame, v).to_bevy_sphere(&ctx);
            let sphere = Vec3::new(sphere_d.x as f32, sphere_d.y as f32, sphere_d.z as f32);
            assert_vec3_close(
                &format!("{frame:?} sphere (north pole)"),
                sphere,
                expected_plane + Vec3::Y,
                eps,
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_north_pole_180() {
        let radius = 1.0;
        let ctx: GeoContext = GeoOrigin::new_from_degrees(90.0, 180.0, 0.0)
            .with_shape(Shape::Sphere { radius })
            .into();
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            // (
            //     GeoFrame::EUS,
            //     Vec3::new(1.0, 2.0, 3.0),
            // ),
            (
                GeoFrame::ENU,
                Vec3::new(1.0, 3.0, -2.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
            ),
            (
                GeoFrame::ECEF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::ECI,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::GCRF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
        ];

        for (frame, expected_plane) in cases {
            let plane = frame.to_bevy_pos(v, &ctx);
            assert_vec3_close(
                &format!("{frame:?} plane (north pole)"),
                plane,
                expected_plane,
                eps,
            );

            let sphere = GeoPosition(frame, v).to_bevy_sphere(&ctx).as_vec3();
            assert_vec3_close(
                &format!("{frame:?} sphere (north pole)"),
                sphere,
                Vec3::new(-expected_plane.x, expected_plane.y + radius as f32, -expected_plane.z),
                eps,
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_south_pole() {
        let radius = 1.0;
        let ctx: GeoContext = GeoOrigin::new_from_degrees(-90.0, 0.0, 0.0)
            .with_shape(Shape::Sphere { radius })
            .into();
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (
                GeoFrame::EUS,
                Vec3::new(1.0, 2.0, 3.0),
            ),
            (
                GeoFrame::ENU,
                Vec3::new(1.0, 3.0, -2.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
            ),
            (
                GeoFrame::ECEF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::ECI,
                Vec3::new(2.0, 2.0, 1.0),
            ),
            (
                GeoFrame::GCRF,
                Vec3::new(2.0, 2.0, 1.0),
            ),
        ];

        for (frame, expected_plane) in cases {
            let plane = frame.to_bevy_pos(v, &ctx);
            assert_vec3_close(
                &format!("{frame:?} plane (south pole)"),
                plane,
                expected_plane,
                eps,
            );

            // let sphere_d = frame.to_bevy_sphere(v, &ctx);
            // let sphere = Vec3::new(sphere_d.x as f32, sphere_d.y as f32, sphere_d.z as f32);
            // assert_vec3_close(
            //     &format!("{frame:?} sphere (south pole)"),
            //     sphere,
            //     Vec3::new(expected_plane.x, -expected_plane.y - radius as f32, -expected_plane.z),
            //     eps,
            // );
        }
    }
}
