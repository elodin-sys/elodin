#![allow(non_snake_case)]
use bevy::math::{DMat3, DMat4, DQuat, DVec3};
use bevy::prelude::*;
use map_3d::Ellipsoid;

/// Earth sidereal spin
pub const EARTH_SIDEREAL_SPIN: f64 = 7.292_115_0e-5;

/// Return the approximate radius of the ellipsoid.
pub fn approx_radius(ellipsoid: &Ellipsoid) -> f64 {
    ellipsoid.parameters().0
}

/// Return the approximate radius of the ellipsoid.
pub fn radius(ellipsoid: &Ellipsoid, latitude: f64) -> f64 {
    map_3d::get_radius_normal(latitude, ellipsoid)
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
#[cfg_attr(feature = "strum", derive(strum_macros::IntoStaticStr, strum_macros::EnumString))]
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
    // Leaving out these time-dependent coordinate frames for the moment.

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
#[derive(Default, Debug, Clone, Copy, Reflect)]
pub struct GeoOrigin {
    /// Geodetic latitude [rad]
    pub latitude: f64,
    /// Geodetic longitude [rad]
    pub longitude: f64,
    /// Altitude above mean radius [m]
    pub altitude: f64,
    #[reflect(ignore)]
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
            ..default()
        }
    }

    /// Provide an ellipsoid.
    pub fn with_ellipsoid(mut self, shape: Ellipsoid) -> Self {
        self.ellipsoid = shape;
        self
    }
}

/// Global geospatial context:
/// * origin (for ENU/ECEF)
/// * Earth rotation model (for ECI/GCRF <-> ECEF).
#[derive(Resource, Debug, Clone, Reflect)]
pub struct GeoContext {
    /// The geographic origin of the coordinate system on Earth.
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
        GeoContext {
            origin,
            ..default()
        }
    }
}

impl GeoContext {
    /// Provide an initial rotation angle.
    pub fn with_rotation_angle(mut self, theta0_rad: f64) -> Self {
        self.theta0_rad = theta0_rad;
        self
    }

    /// Provide a presentation mode.
    pub fn with_present(mut self, present: Present) -> Self {
        self.present = present;
        self
    }

    /// Convert ECI -> ECEF at simulation time t [s].
    pub fn eci_R_ecef(&self) -> DMat3 {
        let theta = self.theta0_rad + self.earth_rot_rate_rad_per_s * self.time;
        DMat3::from_rotation_z(theta)
    }
}

impl GeoFrame {
    /// Provides the transformation matrix ${bevy}_M_{from}$ from a coordinate
    /// frame.
    pub fn bevy_M_(from: &Self, context: &GeoContext) -> DMat4 {
        let R = Self::bevy_R_(from, context);
        let O = Self::bevy_O_(from, context);
        DMat4::from_mat3_translation(R, O)
    }

    /// The general tranformation matrix for ${self}_M_{from}$ of the two
    /// coordinate frames.
    pub fn _M_(&self, from: &GeoFrame, context: &GeoContext) -> DMat4 {
        let R = self._R_(from, context);
        let O = self._O_(from, context);
        DMat4::from_mat3_translation(R, O)
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

    /// Provides the origin vector ${self}_O_{from}$ of the coordinate frame.
    pub fn _O_(&self, from: &Self, context: &GeoContext) -> DVec3 {
        match context.present {
            Present::Plane => DVec3::ZERO,
            Present::Sphere => {
                match (from, *self) {
                    (GeoFrame::ECEF, GeoFrame::ENU) => {
                        -approx_radius(&context.origin.ellipsoid) * DVec3::Z
                    }
                    (GeoFrame::ECEF, GeoFrame::NED) => {
                        approx_radius(&context.origin.ellipsoid) * DVec3::Z
                    }
                    (GeoFrame::ENU | GeoFrame::NED, GeoFrame::ECEF) => {
                        let ecef_R_enu = Self::ecef_R_(&GeoFrame::ENU, &context.origin);
                        approx_radius(&context.origin.ellipsoid) * ecef_R_enu.z_axis
                    }
                    // For these, a fully correct basis would require time-dependent
                    // Earth orientation.
                    _ => DVec3::ZERO,
                }
            }
        }
    }

    /// Provides the origin vector ${bevy}_O_{from}$ of the coordinate frame.
    pub fn ecef_O_(from: &Self, context: &GeoContext) -> DVec3 {
        match context.present {
            Present::Plane => DVec3::ZERO,
            Present::Sphere => {
                match from {
                    GeoFrame::ENU | GeoFrame::NED => {
                        let ecef_R_enu = Self::ecef_R_(&GeoFrame::ENU, &context.origin);
                        approx_radius(&context.origin.ellipsoid) * ecef_R_enu.z_axis
                    }
                    // For these, a fully correct basis would require time-dependent
                    // Earth orientation.
                    GeoFrame::ECEF => DVec3::ZERO,
                }
            }
        }
    }

    /// The general rotation matrix for ${self}_R_{from}$ of the two
    /// coordinate frames.
    pub fn _R_(&self, from: &GeoFrame, context: &GeoContext) -> DMat3 {
        use GeoFrame::*;
        match (*from, *self) {
            (x, y) if x == y => DMat3::IDENTITY,
            (ENU, NED) => DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Y),
            (NED, ENU) => DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Y),
            (ECEF, x) => Self::ecef_R_(&x, &context.origin),
            (x, ECEF) => Self::ecef_R_(&x, &context.origin).inverse(),
            (x, y) => unreachable!("{x:?} -> {y:?}"),
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

    /// The tranformation matrix for $ecef_M_{from}$ of the two
    /// coordinate frames.
    pub fn ecef_M_(from: &Self, context: &GeoContext) -> DMat4 {
        let R = Self::ecef_R_(from, &context.origin);
        let O = Self::ecef_O_(from, context);
        DMat4::from_mat3_translation(R, O)
    }

    /// Provides the matrix ${ecef}_R_{self}$.
    ///
    /// Given from this [reference.](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    pub fn ecef_R_(from: &Self, origin: &GeoOrigin) -> DMat3 {
        use std::f64::consts::FRAC_PI_2;
        if *from == GeoFrame::ECEF {
            return DMat3::IDENTITY;
        }

        // The reference uses this formula.
        //
        // $ \begin{bmatrix} x \\ y \\ z \end{bmatrix} = R_3[-(\pi/2 + \lambda)]~R_1[-(\pi/2 - \varphi)]\begin{bmatrix} E \\ N \\ U \end{bmatrix}
        //
        // Implementing on inspection results in this code:
        //
        // let ecef_R_enu = DMat3::from_rotation_z(-(FRAC_PI_2 + origin.longitude))
        //      * DMat3::from_rotation_x(-(FRAC_PI_2 - origin.latitude));
        //
        // However, the matrix implementations differ. Essentially the signs are
        // flipped in the rotation matrices.
        //
        // `DMat3::from_rotation_x(-\theta) = R_1[\theta]`
        let ecef_R_enu = DMat3::from_rotation_z(FRAC_PI_2 + origin.longitude)
            * DMat3::from_rotation_x(FRAC_PI_2 - origin.latitude);
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

    #[allow(dead_code)]
    #[inline]
    fn ned_R_enu() -> DMat3 {
        DMat3::from_cols(DVec3::Y, DVec3::X, DVec3::NEG_Z)
    }
}

/// Per-entity geo position:
///   0: which frame the coords are in
///   1: position in that frame (ENU, NED, ECEF).
#[derive(Debug, Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GeoPosition(pub GeoFrame, pub DVec3);

impl GeoPosition {
    /// Convert position to Bevy.
    pub fn to_bevy(&self, context: &GeoContext) -> DVec3 {
        GeoFrame::bevy_M_(&self.0, context).transform_point3(self.1)
    }

    /// Convert position from Bevy.
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
#[derive(Debug, Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GeoVelocity(pub GeoFrame, pub DVec3);

impl GeoVelocity {

    /// Convert vector to Bevy.
    pub fn to_bevy(&self, context: &GeoContext) -> DVec3 {
        GeoFrame::bevy_R_(&self.0, context) * self.1
    }

    /// Convert vector from Bevy.
    pub fn from_bevy(frame: &GeoFrame, v_bevy: impl Into<DVec3>, context: &GeoContext) -> Self {
        let v = v_bevy.into();
        let w = GeoFrame::bevy_R_(frame, context).transpose() * v;
        GeoVelocity(*frame, GeoFrame::bevy_R_(frame, context).transpose() * w)
    }
}

/// Per-entity geo orientation:
///   0: frame the quaternion is expressed in
///   1: rotation from local -> that frame
#[derive(Debug, Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GeoRotation(pub GeoFrame, pub DQuat);

/// Per-entity angular velocity in some frame, in rad/s.
#[derive(Debug, Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GeoAngularVelocity(pub GeoFrame, pub DVec3);

/// Plugin wiring: sets up `GeoContext` and systems that run
/// *before* transform propagation.
pub struct GeoFramePlugin {
    /// An initial [GeoContext]
    pub context: Option<GeoContext>,
    /// An initial [GeoOrigin]
    pub origin: Option<GeoOrigin>,
    /// Add systems that apply the transforms.
    ///
    /// In some cases these will need to be setup manually.
    pub apply_transforms: bool,
}

impl Default for GeoFramePlugin {
    fn default() -> Self {
        GeoFramePlugin {
            context: None,
            origin: None,
            apply_transforms: true,
        }
    }
}

impl Plugin for GeoFramePlugin {
    fn build(&self, app: &mut App) {
        let mut ctx = self.context.clone().unwrap_or_default();
        if let Some(origin) = self.origin {
            ctx.origin = origin;
        }
        #[cfg(feature = "inspector")]
        app
            .register_type_data::<f64, bevy_inspector_egui::inspector_egui_impls::InspectorEguiImpl>();
        app
            .register_type::<GeoPosition>()
            .register_type::<GeoRotation>()
            .register_type::<GeoVelocity>()
            .register_type::<GeoAngularVelocity>()
            .insert_resource(ctx)
            // Integrate in frame space each Update
            .add_systems(Update, (integrate_geo_motion, integrate_geo_orientation));
        if self.apply_transforms {
            // Then convert to Bevy before transform propagation
            #[cfg(not(feature = "big_space"))]
            app.add_systems(
                PostUpdate,
                (
                    touch_geo_on_context_change,
                    apply_transforms,
                    apply_geo_rotation,
                )
                    .chain()
                    .before(TransformSystems::Propagate),
            );

            // We ought not to do this here because it relies on the generic
            // parameter.

            // #[cfg(feature = "big_space")]
            // app.add_plugins(crate::big_space::plugin::<i128>);
        }
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
        let R = geo_pos.0._R_(&geo_vel.0, &ctx);
        let v = R * geo_vel.1;
        geo_pos.1 += v * dt;
    }
}

/// Integrate `GeoRotation` in *frame space* using `GeoAngularVelocity`.
pub fn integrate_geo_orientation(
    time: Res<Time>,
    ctx: ResMut<GeoContext>,
    mut q: Query<(&mut GeoRotation, &GeoAngularVelocity)>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }

    for (mut geo_rot, ang) in &mut q {
        let rot0_R_ang0 = geo_rot.0._R_(&ang.0, &ctx);
        if geo_rot.0 != ang.0 {
            // We're punting.
            continue;
        }
        let omega = rot0_R_ang0 * ang.1;
        let delta = DQuat::from_scaled_axis(omega * dt);
        geo_rot.1 = delta * geo_rot.1;
    }
}

/// When the [GeoContext] changes, touch all the [GeoPosition] and [GeoRotation]
/// objects so they will be updated.
pub fn touch_geo_on_context_change(
    ctx: Res<GeoContext>,
    mut pos_query: Query<&mut GeoPosition>,
    mut rot_query: Query<&mut GeoRotation>,
) {
    if !ctx.is_changed() {
        return;
    }

    for mut pos in &mut pos_query {
        pos.set_changed();
    }
    for mut rot in &mut rot_query {
        rot.set_changed();
    }
}

/// System: convert `GeoPosition` into `Transform.translation` right before Bevy
/// propagates transforms through the hierarchy.
pub fn apply_transforms(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform), Changed<GeoPosition>>,
) {
    for (geo, mut transform) in &mut q {
        transform.translation = geo.to_bevy(&ctx).as_vec3();
    }
}

/// System: convert `GeoRotation` into `Transform.rotation`.
pub fn apply_geo_rotation(
    ctx: Res<GeoContext>,
    mut q: Query<(&GeoRotation, &mut Transform), Changed<GeoRotation>>,
) {
    for (geo_rot, mut transform) in &mut q {
        let frame = geo_rot.0;
        let local_rot = geo_rot.1;
        let bevy_R_geo0 = DQuat::from_mat3(&GeoFrame::bevy_R_(&frame, &ctx));
        transform.rotation = (bevy_R_geo0 * local_rot).as_quat();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> GeoContext {
        GeoContext::default()
    }

    #[inline]
    fn enu_to_ned(v_enu: DVec3) -> DVec3 {
        DVec3::new(v_enu.y, v_enu.x, -v_enu.z)
    }

    #[inline]
    fn ned_to_enu(v_ned: DVec3) -> DVec3 {
        DVec3::new(v_ned.y, v_ned.x, -v_ned.z)
    }

    fn convert_pos(to: GeoFrame, from: GeoFrame, v: DVec3, ctx: &GeoContext) -> DVec3 {
        to._M_(&from, ctx).transform_point3(v)
    }

    /// Convert a DVec3 (position/velocity) from `from_frame` into `self`.
    fn convert_pos_map_3d(to: GeoFrame, from_frame: GeoFrame, v: DVec3, ctx: &GeoContext) -> DVec3 {
        use GeoFrame::*;
        match (from_frame, to) {
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
                enu_to_ned(enu)
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
            (ENU, NED) => enu_to_ned(v),
            (NED, ENU) => ned_to_enu(v),
            (x, y) => unreachable!("{x:?} -> {y:?}"),
        }
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
    fn bevy_r_ecef_plane_and_sphere_match() {
        let origin = GeoOrigin::new_from_degrees(0.0, 0.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius: 1.0 });
        let ctx_plane: GeoContext = origin.into();
        let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);

        let bevy_R_ecef_s = GeoFrame::bevy_R_(&GeoFrame::ECEF, &ctx_sphere);

        let ecef_R_enu_s = GeoFrame::ecef_R_(&GeoFrame::ENU, &ctx_sphere.origin);
        let bevy_R_enu_s = GeoFrame::bevy_R_(&GeoFrame::ENU, &ctx_sphere);

        assert_approx_eq!(
            bevy_R_ecef_s * DVec3::X,
            DVec3::X,
            1e-9,
            "bevy_R_ecef x-axis"
        );
        assert_approx_eq!(
            bevy_R_ecef_s * DVec3::Y,
            DVec3::NEG_Z,
            1e-9,
            "bevy_R_ecef y-axis"
        );
        assert_approx_eq!(
            bevy_R_ecef_s * DVec3::Z,
            DVec3::Y,
            1e-9,
            "bevy_R_ecef z-axis"
        );

        assert_approx_eq!(ecef_R_enu_s * DVec3::X, DVec3::Y, 1e-9, "ecef_R_enu x-axis");

        assert_approx_eq!(ecef_R_enu_s * DVec3::Y, DVec3::Z, 1e-9, "ecef_R_enu y-axis");

        // The next two assertions show that we do not entirely comport with
        // what map_3d does.
        assert_approx_eq!(
            convert_pos(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Y, &ctx_sphere),
            2.0 * DVec3::X,
            1e-9,
            "convert_pos ecef _M_ y-axis"
        );

        assert_approx_eq!(
            convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Z, &ctx_sphere),
            DVec3::new(2.0, 0.0, 0.0),
            1e-9,
            "convert_pos ecef _M_ z-axis"
        );
        assert_approx_eq!(
            convert_pos(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Z, &ctx_sphere),
            DVec3::new(1.0, 1.0, 0.0),
            1e-9,
            "convert_pos ecef _M_ z-axis"
        );
        assert_approx_eq!(ecef_R_enu_s * DVec3::Y, DVec3::Z, 1e-9, "ecef_R_enu y-axis");
        assert_approx_eq!(ecef_R_enu_s * DVec3::Z, DVec3::X, 1e-9, "ecef_R_enu z-axis");

        assert_approx_eq!(
            bevy_R_enu_s * DVec3::X,
            DVec3::NEG_Z,
            1e-9,
            "bevy_R_enu x-axis"
        );
        assert_approx_eq!(bevy_R_enu_s * DVec3::Y, DVec3::Y, 1e-9, "bevy_R_enu y-axis");
        assert_approx_eq!(bevy_R_enu_s * DVec3::Z, DVec3::X, 1e-9, "bevy_R_enu z-axis");
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
                Vec3::new(4.0, 2.0, -1.0),
            ),
            (
                GeoFrame::NED,
                Vec3::new(2.0, -3.0, -1.0),
                Vec3::new(-2.0, 1.0, -2.0),
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
                        -expected_plane.z,
                        expected_plane.y + radius as f32,
                        expected_plane.x,
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
                        -expected_plane.z,
                        -expected_plane.y - radius as f32,
                        -expected_plane.x,
                    )
                },
                eps,
                format!("{frame:?} sphere (south pole)")
            );
        }
    }
}
