use bevy::math::{DMat3, DMat4, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use map_3d::{ecef2enu, Ellipsoid};
/// Earth sidereal spin
pub const EARTH_SIDEREAL_SPIN: f64 = 7.292_115_0e-5;

pub const DEFAULT_RENDER: GeoFrame = GeoFrame::ECEF;

/// Return the approximate radius of the ellipsoid.
pub fn approx_radius(ellipsoid: &Ellipsoid) -> f64 {
    ellipsoid.parameters().0
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
/// Bevy world: +X=East, +Y=Up, +Z=South (so North = -Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeoFrame {
    /// East-North-Up: +X=East, +Y=North, +Z=Up
    ENU,
    /// North-East-Down: +X=North, +Y=East, +Z=Down
    NED,
    /// Earth-Centered Earth-Fixed
    /// +X through (lat=0, lon=0) equator
    /// +Y through (lat=0, lon=90°E) equator
    /// +Z through North Pole
    ECEF,
    // Leaving out these time-dependent
    // /// Earth-Centered Inertial
    // /// +X to vernal equinox, +Y 90°E, +Z North Pole
    // ECI,
    // /// Geocentric Celestial Reference Frame (inertial, J2000)
    // /// Sometimes called the International Celestial Reference Frame (ICRF)
    // /// Approximated as ECI here.
    // GCRF,
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
    pub ellipsoid: Ellipsoid,
}

impl GeoOrigin {
    /// Simple spherical Earth model (good enough for games).
    /// Uses default Earth radius.
    pub fn new_from_degrees(latitude_deg: f64, longitude_deg: f64, altitude: f64) -> Self {
        let latitude = latitude_deg.to_radians();
        let longitude = longitude_deg.to_radians();
        Self {
            latitude,
            longitude,
            altitude,
            ..default()
        }
    }

    pub fn with_ellipsoid(mut self, shape: Ellipsoid) -> Self {
        self.ellipsoid = shape;
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
            present: Present::default(),
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

    pub fn with_present(mut self, present: Present) -> Self {
        self.present = present;
        self
    }

    /// Rotation matrix Rz(angle) about +Z axis:
    /// x' = cosθ x - sinθ y
    /// y' = sinθ x + cosθ y
    /// z' = z
    pub(crate) fn rot_z(angle: f64) -> DMat3 {
        let (s, c) = angle.sin_cos();
        DMat3::from_cols(
            DVec3::new(c, s, 0.0),  // image of x-axis
            DVec3::new(-s, c, 0.0), // image of y-axis
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
    /// ECEF -> local ENU at the origin.
    pub fn ecef_to_enu(&self, r_ecef: DVec3) -> DVec3 {
        let (e, n, u) = ecef2enu(
            r_ecef.x,
            r_ecef.y,
            r_ecef.z,
            self.origin.latitude,
            self.origin.longitude,
            self.origin.altitude,
            &self.origin.ellipsoid,
        );
        DVec3::new(e, n, u)
    }
}

impl GeoFrame {
    /// Provides the transformation matrix ${bevy}_M_{from}$ of the coordinate frame.
    pub fn bevy_M_(from: &Self, context: &GeoContext) -> DMat4 {
        let R = Self::bevy_R_(from, context);
        let O = Self::bevy_O_(from, context);
        // DMat4::from_mat3_translation(R, O);
        DMat4::from_cols(
            R.x_axis.extend(0.0),
            R.y_axis.extend(0.0),
            R.z_axis.extend(0.0),
            O.extend(1.0),
        )
    }

    /// Provides the origin vector ${bevy}_O_{from}$ of the coordinate frame.
    pub fn bevy_O_(from: &Self, context: &GeoContext) -> DVec3 {
        match context.present {
            Present::Plane => DVec3::ZERO,
            Present::Sphere => {
                match from {
                    GeoFrame::ENU | GeoFrame::NED => {
                        let bevy_R_enu = Self::bevy_R_(&GeoFrame::ENU, context);
                        approx_radius(&context.origin.ellipsoid) * bevy_R_enu.z_axis
                    }
                    // For these, a fully correct basis would require time-dependent
                    // Earth orientation.
                    GeoFrame::ECEF => DVec3::ZERO,
                }
            }
        }
    }

    /// Provides the rotation matrix ${bevy}_R_{from}$.
    pub fn bevy_R_(from: &Self, context: &GeoContext) -> DMat3 {
        let bevy_R_ecef = DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y);
        match context.present {
            Present::Plane => match from {
                GeoFrame::ENU => {
                    // Columns are frame basis vectors expressed in Bevy world space.
                    // ENU: e_hat = +X, n_hat = -Z, u_hat = +Y
                    DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y)
                }
                GeoFrame::NED => {
                    // NED: n_hat = -Z, e_hat = +X, d_hat = -Y
                    DMat3::from_cols(DVec3::NEG_Z, DVec3::X, DVec3::NEG_Y)
                }
                // For these, a fully correct basis would require time-dependent
                // Earth orientation.
                GeoFrame::ECEF => bevy_R_ecef,
            },
            Present::Sphere => bevy_R_ecef * Self::ecef_R_(from, &context.origin),
        }
    }

    /// Provides the matrix ${ecef}_R_{self}$.
    pub fn ecef_R_(from: &Self, origin: &GeoOrigin) -> DMat3 {
        use std::f64::consts::FRAC_PI_2;
        if *from == GeoFrame::ECEF {
            return DMat3::IDENTITY;
        }
        let ecef_R_enu = DMat3::from_rotation_z(-(FRAC_PI_2 + origin.longitude))
            * DMat3::from_rotation_x(-(FRAC_PI_2 - origin.latitude));
        match from {
            GeoFrame::ECEF => DMat3::IDENTITY,
            GeoFrame::ENU => ecef_R_enu,
            GeoFrame::NED => ecef_R_enu * Self::enu_R_ned(),
        }
    }

    #[inline]
    fn enu_R_ned() -> DMat3 {
        DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Z)
    }

    #[inline]
    fn enu_to_ned(v_enu: DVec3) -> DVec3 {
        DVec3::new(v_enu.y, v_enu.x, -v_enu.z)
    }

    #[inline]
    fn ned_to_enu(v_ned: DVec3) -> DVec3 {
        DVec3::new(v_ned.y, v_ned.x, -v_ned.z)
    }

    /// Convert a DVec3 (position/velocity) from `from_frame` into `self`.
    pub fn convert_to(&self, v: DVec3, from_frame: GeoFrame, ctx: &GeoContext) -> DVec3 {
        use GeoFrame::*;
        match (from_frame, *self) {
            (x, y) if x == y => v,
            (ENU, ECEF) => map_3d::enu2ecef(
                v.x,
                v.y,
                v.z,
                ctx.origin.latitude,
                ctx.origin.longitude,
                ctx.origin.altitude,
                &ctx.origin.ellipsoid,
            )
            .into(),
            (NED, ECEF) => map_3d::ned2ecef(
                v.x,
                v.y,
                v.z,
                ctx.origin.latitude,
                ctx.origin.longitude,
                ctx.origin.altitude,
                &ctx.origin.ellipsoid,
            )
            .into(),
            (ECEF, NED) => {
                let enu = map_3d::ecef2enu(
                    v.x,
                    v.y,
                    v.z,
                    ctx.origin.latitude,
                    ctx.origin.longitude,
                    ctx.origin.altitude,
                    &ctx.origin.ellipsoid,
                )
                .into();
                Self::enu_to_ned(enu)
            }
            (ECEF, ENU) => map_3d::ecef2enu(
                v.x,
                v.y,
                v.z,
                ctx.origin.latitude,
                ctx.origin.longitude,
                ctx.origin.altitude,
                &ctx.origin.ellipsoid,
            )
            .into(),
            (ENU, NED) => Self::enu_to_ned(v),
            (NED, ENU) => Self::ned_to_enu(v),
            (x, y) => unreachable!("{x:?} -> {y:?}"),
        }
    }
}

/// Per-entity geo position:
///   0: which frame the coords are in
///   1: position in that frame (ENU, NED, ECEF).
#[derive(Component)]
pub struct GeoPosition(pub GeoFrame, pub DVec3);

impl GeoPosition {
    pub fn to_bevy(&self, context: &GeoContext) -> DVec3 {
        GeoFrame::bevy_M_(&self.0, context).transform_point3(self.1)
    }

    pub fn from_bevy(frame: GeoFrame, v_bevy: impl Into<DVec3>, context: &GeoContext) -> Self {
        let v = v_bevy.into();
        GeoPosition(
            frame,
            GeoFrame::bevy_M_(&frame, context)
                .inverse()
                .transform_point3(v),
        )
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
                (
                    apply_geo_translation, //apply_geo_rotation
                )
                    .chain()
                    .before(TransformSystem::TransformPropagate),
            );
    }
}

/// System: integrate motion in *frame* coordinates from GeoVelocity.
pub fn integrate_geo_motion(
    time: Res<Time>,
    ctx: ResMut<GeoContext>,
    mut q: Query<(&mut GeoPosition, &GeoVelocity)>,
) {
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
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform)>,
) {
    let ctx_ref: &GeoContext = &*ctx;
    for (geo, mut transform) in &mut q {
        // let pos_in_render = render.convert_to(geo.1, geo.0, ctx_ref);
        transform.translation = geo.to_bevy(ctx_ref).as_vec3();
    }
}

/// System: convert `GeoRotation` into `Transform.rotation`.
pub fn apply_geo_rotation(ctx: Res<GeoContext>, mut q: Query<(&GeoRotation, &mut Transform)>) {
    for (geo_rot, mut transform) in &mut q {
        let frame = geo_rot.0;
        let local_rot = geo_rot.1;

        let frame_to_eus = todo!(); // frame.basis_to_eus_quat(&ctx);
                                    // transform.rotation = frame_to_eus * local_rot;
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
            assert!((a - b).length() <= eps, "got {:?} expected {:?}", a, b);
        };
        ($x: expr, $y: expr, $e: expr, $l: expr) => {
            let a = $x;
            let b = $y;
            let eps = $e;
            assert!(
                (a - b).length() <= eps,
                "{}: got {:?} expected {:?}",
                $l,
                a,
                b
            );
        };
    }

    #[test]
    fn enu_to_eus_axes() {
        let ctx = dummy_ctx();
        // east, north, up in ENU
        let east = DVec3::new(1.0, 0.0, 0.0);
        let north = DVec3::new(0.0, 1.0, 0.0);
        let up = DVec3::new(0.0, 0.0, 1.0);

        let east_eus = GeoPosition(GeoFrame::ENU, east).to_bevy(&ctx).as_vec3();
        let north_eus = GeoPosition(GeoFrame::ENU, north).to_bevy(&ctx).as_vec3();
        let up_eus = GeoPosition(GeoFrame::ENU, up).to_bevy(&ctx).as_vec3();

        assert_eq!(east_eus, Vec3::new(1.0, 0.0, 0.0)); // +X
        assert_eq!(north_eus, Vec3::new(0.0, 0.0, -1.0)); // -Z
        assert_eq!(up_eus, Vec3::new(0.0, 1.0, 0.0)); // +Y
    }

    #[test]
    fn ned_to_eus_axes() {
        let ctx = dummy_ctx();
        // north, east, down in NED
        let north = DVec3::new(1.0, 0.0, 0.0);
        let east = DVec3::new(0.0, 1.0, 0.0);
        let down = DVec3::new(0.0, 0.0, 1.0);

        let north_eus = GeoPosition(GeoFrame::NED, north).to_bevy(&ctx).as_vec3();
        let east_eus = GeoPosition(GeoFrame::NED, east).to_bevy(&ctx).as_vec3();
        let down_eus = GeoPosition(GeoFrame::NED, down).to_bevy(&ctx).as_vec3();

        assert_eq!(north_eus, Vec3::new(0.0, 0.0, -1.0)); // -Z
        assert_eq!(east_eus, Vec3::new(1.0, 0.0, 0.0)); // +X
        assert_eq!(down_eus, Vec3::new(0.0, -1.0, 0.0)); // -Y
    }

    #[test]
    fn enu_r_ned_mul_vector_123() {
        let v_ned = DVec3::new(1.0, 2.0, 3.0);
        let v_enu = GeoFrame::enu_R_ned() * v_ned;
        assert_approx_eq!(v_enu, DVec3::new(2.0, 1.0, -3.0));
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
    fn round_trip_enu() {
        let ctx = dummy_ctx();
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoPosition(GeoFrame::ENU, original).to_bevy(&ctx).as_vec3();
        let round_trip = GeoPosition::from_bevy(GeoFrame::ENU, bevy, &ctx).1;
        assert!((round_trip - original).length() < 1e-9);
    }

    #[test]
    fn round_trip_ned() {
        let ctx = dummy_ctx();
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoPosition(GeoFrame::NED, original).to_bevy(&ctx).as_vec3();
        let round_trip = GeoPosition::from_bevy(GeoFrame::NED, bevy, &ctx).1;
        assert!((round_trip - original).length() < 1e-9);
    }

    #[test]
    fn round_trip_ecef() {
        let ctx = dummy_ctx();
        // Use unique coordinates to verify the conversion works correctly
        let original = DVec3::new(1.0, 2.0, 3.0);
        let bevy = GeoPosition(GeoFrame::ECEF, original)
            .to_bevy(&ctx)
            .as_vec3();
        let round_trip = GeoPosition::from_bevy(GeoFrame::ECEF, bevy, &ctx).1;
        assert!((round_trip - original).length() < 1e-3);
    }

    #[test]
    fn zero_conversions_plane_and_sphere() {
        let origins = [
            (
                "equator",
                GeoOrigin::new_from_degrees(0.0, 0.0, 0.0),
                [
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                ],
            ),
            (
                "north_pole",
                GeoOrigin::new_from_degrees(90.0, 0.0, 0.0),
                [
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
                    // (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                ],
            ),
            (
                "south_pole",
                GeoOrigin::new_from_degrees(-90.0, 0.0, 0.0),
                [
                    (GeoFrame::ENU, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::NED, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)),
                    (GeoFrame::ECEF, Vec3::ZERO, Vec3::ZERO),
                ],
            ),
        ];
        let eps = 1e-5;

        for (label, origin, expectations) in origins {
            let ctx_plane: GeoContext = origin
                .with_ellipsoid(Ellipsoid::Sphere { radius: 1.0 })
                .into();
            let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);
            for (frame, expected_plane, expected_sphere) in expectations {
                let zero = GeoPosition(frame, DVec3::ZERO);
                let plane = zero.to_bevy(&ctx_plane).as_vec3();
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

                let sphere = zero.to_bevy(&ctx_sphere).as_vec3();
                assert_approx_eq!(
                    sphere,
                    expected_sphere,
                    eps,
                    format!("{frame:?} sphere zero ({label})")
                );
            }
        }
    }

    #[test]
    fn present_plane_and_sphere_at_equator_origin() {
        let radius = 1.0;
        let ctx_plane: GeoContext = GeoOrigin::new_from_degrees(0.0, 0.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius })
            .into();
        let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (
                GeoFrame::ENU,
                Vec3::new(1.0, 3.0, -2.0),
                Vec3::new(4.0, -2.0, 1.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
                Vec3::new(-2.0, -1.0, 2.0),
            ),
            (
                GeoFrame::ECEF,
                Vec3::new(1.0, 3.0, -2.0),
                Vec3::new(1.0, 3.0, -2.0),
            ),
        ];

        for (frame, expected_plane, expected_sphere) in cases {
            let plane = GeoPosition(frame, v).to_bevy(&ctx_plane).as_vec3();
            assert_approx_eq!(
                plane,
                expected_plane,
                eps,
                format!("{frame:?} plane (equator)")
            );

            let sphere = GeoPosition(frame, v).to_bevy(&ctx_sphere).as_vec3();
            assert_approx_eq!(
                sphere,
                expected_sphere,
                eps,
                format!("{frame:?} sphere (equator)")
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_north_pole() {
        let radius = 1.0;
        let ctx_plane: GeoContext = GeoOrigin::new_from_degrees(90.0, 270.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius })
            .into();
        let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (GeoFrame::ENU, Vec3::new(1.0, 3.0, -2.0)),
            (GeoFrame::NED, Vec3::new(2.0, -3.0, -1.0)),
            (GeoFrame::ECEF, Vec3::new(1.0, 3.0, -2.0)),
        ];

        for (frame, expected_plane) in cases {
            let plane = GeoPosition(frame, v).to_bevy(&ctx_plane).as_vec3();
            assert_approx_eq!(
                plane,
                expected_plane,
                eps,
                format!("{frame:?} plane (north pole)")
            );

            let sphere = GeoPosition(frame, v).to_bevy(&ctx_sphere).as_vec3();
            assert_approx_eq!(
                sphere,
                if frame == GeoFrame::ECEF {
                    expected_plane
                } else {
                    expected_plane + Vec3::Y
                },
                eps,
                format!("{frame:?} sphere (north pole)")
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_north_pole_180() {
        let radius = 1.0;
        let ctx_plane: GeoContext = GeoOrigin::new_from_degrees(90.0, 180.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius })
            .into();
        let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (GeoFrame::ENU, Vec3::new(1.0, 3.0, -2.0)),
            (GeoFrame::NED, Vec3::new(2.0, -3.0, -1.0)),
            (GeoFrame::ECEF, Vec3::new(1.0, 3.0, -2.0)),
        ];

        for (frame, expected_plane) in cases {
            let plane = GeoPosition(frame, v).to_bevy(&ctx_plane).as_vec3();
            assert_approx_eq!(
                plane,
                expected_plane,
                eps,
                format!("{frame:?} plane (north pole)")
            );

            let sphere = GeoPosition(frame, v).to_bevy(&ctx_sphere).as_vec3();
            assert_approx_eq!(
                sphere,
                if frame == GeoFrame::ECEF {
                    expected_plane
                } else {
                    Vec3::new(
                        expected_plane.z,
                        expected_plane.y + radius as f32,
                        -expected_plane.x,
                    )
                },
                eps,
                format!("{frame:?} sphere (north pole)")
            );
        }
    }

    #[test]
    fn present_plane_and_sphere_at_south_pole() {
        let radius = 1.0;
        let ctx_plane: GeoContext = GeoOrigin::new_from_degrees(-90.0, 0.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius })
            .into();
        let ctx_plane = ctx_plane.with_present(Present::Plane);
        let ctx_sphere: GeoContext = GeoOrigin::new_from_degrees(-90.0, 0.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius })
            .into();
        let ctx_sphere = ctx_sphere.with_present(Present::Sphere);
        let v = DVec3::new(1.0, 2.0, 3.0);
        let eps = 1e-4;

        let cases = [
            (GeoFrame::ENU, Vec3::new(1.0, 3.0, -2.0)),
            (GeoFrame::NED, Vec3::new(2.0, -3.0, -1.0)),
            (GeoFrame::ECEF, Vec3::new(1.0, 3.0, -2.0)),
        ];

        for (frame, expected_plane) in cases {
            let plane = GeoPosition(frame, v).to_bevy(&ctx_plane).as_vec3();
            assert_approx_eq!(
                plane,
                expected_plane,
                eps,
                format!("{frame:?} plane (south pole)")
            );

            let sphere = GeoPosition(frame, v).to_bevy(&ctx_sphere).as_vec3();
            assert_approx_eq!(
                sphere,
                if frame == GeoFrame::ECEF {
                    expected_plane
                } else {
                    Vec3::new(
                        expected_plane.z,
                        -expected_plane.y - radius as f32,
                        expected_plane.x,
                    )
                },
                eps,
                format!("{frame:?} sphere (south pole)")
            );
        }
    }
}
