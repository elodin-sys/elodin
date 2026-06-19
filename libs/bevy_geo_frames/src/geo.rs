//! The mathematical convention used in this code is this:
//!
//! A rotation matrix R from frame ENU to the frame Bevy is written as
//! ${bevy}_R_{enu}. So given a vector v in ENU, we'd produce the v in Bevy with
//! a right-multiplication as $v_{bevy} = {bevy}_R_{enu} * v_{enu}$. This
//! convention was chosen so that frames can be easily checked by adjacency.
#![allow(non_snake_case)]
use crate::GeoFrame;
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
#[reflect(Resource)]
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
    fn origin_ecef(origin: &GeoOrigin) -> DVec3 {
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

    fn bevy_R_enu_plane() -> DMat3 {
        // Columns are frame basis vectors expressed in Bevy world space.
        // ENU: e_hat = +X, n_hat = -Z, u_hat = +Y
        DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y)
    }

    /// Provides the transformation matrix ${bevy}_M_{from}$ from a coordinate
    /// frame.
    pub fn bevy_M_(from: &Self, context: &GeoContext) -> DMat4 {
        let R = Self::bevy_R_(from, context);
        let O = Self::bevy_O_(from, context);
        DMat4::from_mat3_translation(R, O)
    }

    /// The general transformation matrix for ${self}_M_{from}$ of the two
    /// coordinate frames.
    pub fn _M_(&self, from: &GeoFrame, context: &GeoContext) -> DMat4 {
        let R = self._R_(from, context);
        let O = self._O_(from, context);
        DMat4::from_mat3_translation(R, O)
    }

    /// Provides the origin vector ${bevy}_O_{from}$ of the coordinate frame.
    pub fn bevy_O_(from: &Self, context: &GeoContext) -> DVec3 {
        match context.present {
            Present::Plane => match from {
                GeoFrame::ENU | GeoFrame::NED => DVec3::ZERO,
                GeoFrame::ECEF => {
                    let bevy_R_ecef = Self::bevy_R_(&GeoFrame::ECEF, context);
                    -bevy_R_ecef * Self::origin_ecef(&context.origin)
                }
            },
            Present::Sphere => {
                match from {
                    GeoFrame::ENU | GeoFrame::NED => {
                        let bevy_R_ecef = DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y);
                        bevy_R_ecef * Self::origin_ecef(&context.origin)
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
            Present::Plane => {
                let origin_ecef = Self::origin_ecef(&context.origin);
                match (from, *self) {
                    (GeoFrame::ECEF, GeoFrame::ENU | GeoFrame::NED) => {
                        -self._R_(from, context) * origin_ecef
                    }
                    (GeoFrame::ENU | GeoFrame::NED, GeoFrame::ECEF) => origin_ecef,
                    _ => DVec3::ZERO,
                }
            }
            Present::Sphere => {
                match (from, *self) {
                    (GeoFrame::ECEF, GeoFrame::ENU | GeoFrame::NED) => {
                        -self._R_(from, context) * Self::origin_ecef(&context.origin)
                    }
                    (GeoFrame::ENU | GeoFrame::NED, GeoFrame::ECEF) => {
                        Self::origin_ecef(&context.origin)
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
            Present::Plane => match from {
                GeoFrame::ENU | GeoFrame::NED => Self::origin_ecef(&context.origin),
                GeoFrame::ECEF => DVec3::ZERO,
            },
            Present::Sphere => {
                match from {
                    GeoFrame::ENU | GeoFrame::NED => Self::origin_ecef(&context.origin),
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
        use crate::GeoFrame::*;
        match (*from, *self) {
            (x, y) if x == y => DMat3::IDENTITY,
            (ENU, NED) => Self::ned_R_enu(),
            (NED, ENU) => Self::enu_R_ned(),
            // ecef_R_(x) maps x -> ECEF, so self_R_ecef is its inverse.
            (ECEF, x) => Self::ecef_R_(&x, &context.origin).inverse(),
            (x, ECEF) => Self::ecef_R_(&x, &context.origin),
            (x, y) => unreachable!("{x:?} -> {y:?}"),
        }
    }

    /// Provides the rotation matrix ${bevy}_R_{from}$.
    pub fn bevy_R_(from: &Self, context: &GeoContext) -> DMat3 {
        let bevy_R_ecef = DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y);
        match context.present {
            Present::Plane => match from {
                GeoFrame::ENU => Self::bevy_R_enu_plane(),
                GeoFrame::NED => {
                    // NED: n_hat = -Z, e_hat = +X, d_hat = -Y
                    DMat3::from_cols(DVec3::NEG_Z, DVec3::X, DVec3::NEG_Y)
                }
                GeoFrame::ECEF => {
                    Self::bevy_R_enu_plane()
                        * Self::ecef_R_(&GeoFrame::ENU, &context.origin).inverse()
                }
            },
            Present::Sphere => bevy_R_ecef * Self::ecef_R_(from, &context.origin),
        }
    }

    /// The transformation matrix for $ecef_M_{from}$ of the two
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

    /// Create from a Bevy Transform's translation.
    pub fn from_transform(frame: GeoFrame, transform: &Transform, context: &GeoContext) -> Self {
        Self::from_bevy(frame, transform.translation, context)
    }
}

impl GeoRotation {
    /// A [RotationKind::Relative] rotation expressed in `frame`.
    pub fn new(frame: GeoFrame, q: impl Into<DQuat>) -> Self {
        GeoRotation(frame, q.into(), RotationKind::default())
    }

    /// A [RotationKind::Absolute] rotation expressed in `frame`.
    pub fn absolute(frame: GeoFrame, q: impl Into<DQuat>) -> Self {
        GeoRotation(frame, q.into(), RotationKind::Absolute)
    }

    /// A [RotationKind::Relative] rotation expressed in `frame` that orients a
    /// camera (Bevy convention: forward `-Z`, up `+Y`) to look along `dir`,
    /// with the camera's up toward `up`. Both vectors are expressed in `frame`
    /// coordinates, so the computation is frame agnostic.
    ///
    /// If `up` is `None` (or collinear with `dir`), the frame's natural camera
    /// up — the frame direction that maps to Bevy `+Y` — is used.
    pub fn look_at(
        frame: GeoFrame,
        dir: impl Into<DVec3>,
        up: Option<DVec3>,
        context: &GeoContext,
    ) -> Self {
        let frame_R_bevy = GeoFrame::bevy_R_(&frame, context).transpose();
        // The camera's identity axes expressed in `frame` coordinates.
        let f0 = frame_R_bevy * DVec3::NEG_Z;
        let u0 = frame_R_bevy * DVec3::Y;

        let f = dir.into().normalize();
        let mut up = up.unwrap_or(u0);
        if f.cross(up).length_squared() < 1e-12 {
            // `up` collinear with `dir`: prefer the frame's camera up, then
            // its camera forward (orthogonal to `u0`, so always valid).
            up = if f.cross(u0).length_squared() > 1e-12 {
                u0
            } else {
                f0
            };
        }
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        // Rotation taking the identity camera axes to the desired ones.
        let m0 = DMat3::from_cols(f0.cross(u0), f0, u0);
        let m = DMat3::from_cols(s, f, u);
        GeoRotation(
            frame,
            DQuat::from_mat3(&(m * m0.transpose())),
            RotationKind::Relative,
        )
    }

    /// Convert orientation to Bevy.
    pub fn to_bevy(&self, context: &GeoContext) -> DQuat {
        let local_rot = self.1;
        let q = DQuat::from_mat3(&GeoFrame::bevy_R_(&self.0, context));
        match self.2 {
            // Re-express the rotation operator in Bevy coordinates.
            RotationKind::Relative => q * local_rot * q.conjugate(),
            // Compose with the frame's basis change into Bevy.
            RotationKind::Absolute => q * local_rot,
        }
    }

    /// Convert a [RotationKind::Relative] orientation from Bevy.
    pub fn from_bevy(frame: GeoFrame, v_bevy: impl Into<DQuat>, context: &GeoContext) -> Self {
        Self::from_bevy_kind(frame, v_bevy, context, RotationKind::default())
    }

    /// Convert orientation from Bevy, inverting [Self::to_bevy] for `kind`.
    pub fn from_bevy_kind(
        frame: GeoFrame,
        v_bevy: impl Into<DQuat>,
        context: &GeoContext,
        kind: RotationKind,
    ) -> Self {
        let v = v_bevy.into();
        let q = DQuat::from_mat3(&GeoFrame::bevy_R_(&frame, context));
        let local_rot = match kind {
            RotationKind::Relative => q.conjugate() * v * q,
            RotationKind::Absolute => q.conjugate() * v,
        };
        GeoRotation(frame, local_rot, kind)
    }

    /// Re-express the rotation in another frame, preserving the rotation it
    /// produces in Bevy and its [RotationKind].
    pub fn as_frame(&self, to_frame: GeoFrame, context: &GeoContext) -> GeoRotation {
        let R = DQuat::from_mat3(&to_frame._R_(&self.0, context));
        let local_rot = match self.2 {
            RotationKind::Relative => R * self.1 * R.conjugate(),
            RotationKind::Absolute => R * self.1,
        };
        GeoRotation(to_frame, local_rot, self.2)
    }

    /// Create from a Bevy Transform's rotation.
    pub fn from_transform(frame: GeoFrame, transform: &Transform, context: &GeoContext) -> Self {
        Self::from_bevy(frame, transform.rotation.as_dquat(), context)
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
        GeoVelocity(*frame, w)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum RotationKind {
    #[default]
    /// The rotation is relative. An identity rotation in any frame is an
    /// identity rotation in Bevy's frame.
    Relative,
    /// The rotation is absolute. An identity rotation in ENU will produce a
    /// rotation that rotates [x,y,z] to [x,z,-y] for instance.
    Absolute,
}

/// Per-entity geo orientation:
///   0: frame the quaternion is expressed in
///   1: rotation from local -> that frame
#[derive(Debug, Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GeoRotation(pub GeoFrame, pub DQuat, pub RotationKind);

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
            .register_type_data::<f32, bevy_inspector_egui::inspector_egui_impls::InspectorEguiImpl>()
            .register_type_data::<f64, bevy_inspector_egui::inspector_egui_impls::InspectorEguiImpl>();
        app.register_type::<GeoPosition>()
            .register_type::<GeoRotation>()
            .register_type::<GeoVelocity>()
            .register_type::<GeoAngularVelocity>()
            .insert_resource(ctx)
            // Integrate in frame space each Update
            .add_systems(
                Update,
                (
                    integrate_geo_motion,
                    integrate_geo_orientation,
                    touch_geo_on_context_change,
                ),
            );
        if self.apply_transforms {
            // Then convert to Bevy before transform propagation
            #[cfg(not(feature = "big_space"))]
            app.add_systems(
                PostUpdate,
                (apply_transforms, apply_geo_rotation)
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
    // PERFORMANCE NOTE: We should be able to only apply this when changed, but
    // there was an issue with the lines_3d when I did this. Removing the
    // qualifier should be possible to re-enable if we track down who else is
    // writing to the transform in the lines_3d case.
    //
    // mut q: Query<(&GeoRotation, &mut Transform), Changed<GeoRotation>>,
    mut q: Query<(&GeoRotation, &mut Transform)>,
) {
    for (geo_rot, mut transform) in &mut q {
        transform.rotation = geo_rot.to_bevy(&ctx).as_quat();
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

    fn ecef_plane_to_bevy_expected(v: DVec3, ctx: &GeoContext) -> Vec3 {
        (GeoFrame::bevy_R_(&GeoFrame::ENU, ctx)
            * convert_pos_map_3d(GeoFrame::ENU, GeoFrame::ECEF, v, ctx))
        .as_vec3()
    }

    fn geo_frames_example_ecef_from_enu(east: f64, north: f64, up: f64) -> DVec3 {
        const LAT_DEG: f64 = 34.72;
        const LON_DEG: f64 = -86.64;
        const ALT_M: f64 = 180.5;
        const WGS84_A_M: f64 = 6_378_137.0;
        const WGS84_E2: f64 = 6.6943799901413165e-3;

        let lat = LAT_DEG.to_radians();
        let lon = LON_DEG.to_radians();
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let sin_lon = lon.sin();
        let cos_lon = lon.cos();

        let n = WGS84_A_M / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        let origin = DVec3::new(
            (n + ALT_M) * cos_lat * cos_lon,
            (n + ALT_M) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + ALT_M) * sin_lat,
        );
        let delta = DVec3::new(
            -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up,
            cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up,
            cos_lat * north + sin_lat * up,
        );
        origin + delta
    }

    fn assert_ned_to_ecef_matches_map_3d(ctx: &GeoContext, ned: DVec3, label: &str) {
        let expected = convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::NED, ned, ctx);
        let actual = convert_pos(GeoFrame::ECEF, GeoFrame::NED, ned, ctx);
        assert!(
            actual.distance(expected) < 1e-6,
            "{label}: got {actual:?}, expected {expected:?}, error {}",
            actual.distance(expected)
        );
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
    fn test_as_frame() {
        let ctx = dummy_ctx();
        let geo_rotation = GeoRotation::from_bevy(GeoFrame::ENU, DQuat::IDENTITY, &ctx);
        assert_eq!(geo_rotation.1.as_quat(), Quat::IDENTITY);
        assert_eq!(geo_rotation.to_bevy(&ctx).as_quat(), Quat::IDENTITY);
    }

    /// Quaternion equality up to sign (double cover), robust to fp noise.
    macro_rules! assert_quat_eq {
        ($a:expr, $b:expr) => {
            assert_quat_eq!($a, $b, "quat mismatch")
        };
        ($a:expr, $b:expr, $($l:tt)+) => {
            let a: DQuat = $a;
            let b: DQuat = $b;
            assert!(
                a.dot(b).abs() > 1.0 - 1e-9,
                "{}: got {:?} expected {:?}",
                format!($($l)+),
                a,
                b
            );
        };
    }

    #[test]
    fn relative_identity_is_identity_in_bevy() {
        let ctx = dummy_ctx();
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            let r = GeoRotation::new(frame, DQuat::IDENTITY);
            assert_eq!(r.2, RotationKind::Relative);
            assert_quat_eq!(r.to_bevy(&ctx), DQuat::IDENTITY, "{frame:?}");
        }
    }

    #[test]
    fn absolute_identity_is_basis_change() {
        let ctx = dummy_ctx();
        // Per the RotationKind doc: identity in ENU rotates [x,y,z] to [x,z,-y].
        let r = GeoRotation::absolute(GeoFrame::ENU, DQuat::IDENTITY);
        let q = r.to_bevy(&ctx);
        assert_approx_eq!(
            q * DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(1.0, 3.0, -2.0),
            1e-9
        );
        // It must match the frame's basis-change rotation exactly.
        let basis = DQuat::from_mat3(&GeoFrame::bevy_R_(&GeoFrame::ENU, &ctx));
        assert_quat_eq!(q, basis);
    }

    /// `look_at` is computed entirely in frame coordinates; converting to Bevy
    /// must point the camera (forward `-Z`, up `+Y`) along the frame-space
    /// direction in every frame and presentation mode.
    #[test]
    fn look_at_orients_camera_in_any_frame() {
        let contexts = [
            dummy_ctx(),
            dummy_ctx().with_present(Present::Sphere),
            GeoContext::from(GeoOrigin::new_from_degrees(35.0, -110.0, 0.0))
                .with_present(Present::Sphere),
        ];
        let dir = DVec3::new(1.0, -2.0, 0.5);
        let up = DVec3::new(-0.2, 0.3, 1.0);

        for (c, ctx) in contexts.iter().enumerate() {
            for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
                let bevy_R = GeoFrame::bevy_R_(&frame, ctx);

                let r = GeoRotation::look_at(frame, dir, Some(up), ctx);
                assert_eq!(r.0, frame);
                assert_eq!(r.2, RotationKind::Relative);
                let q = r.to_bevy(ctx);

                let fwd = q * DVec3::NEG_Z;
                let expected = (bevy_R * dir).normalize();
                assert_approx_eq!(fwd, expected, 1e-9, format!("ctx {c} {frame:?} forward"));

                // Camera up must tilt toward the requested up.
                let cam_up = q * DVec3::Y;
                assert!(
                    cam_up.dot((bevy_R * up).normalize()) > 0.0,
                    "ctx {c} {frame:?}: camera up {cam_up:?} points away from requested up"
                );

                // Default up: same forward, and up matches the frame's
                // natural camera up projected off the view direction.
                let q_default = GeoRotation::look_at(frame, dir, None, ctx).to_bevy(ctx);
                let fwd_default = q_default * DVec3::NEG_Z;
                assert_approx_eq!(
                    fwd_default,
                    expected,
                    1e-9,
                    format!("ctx {c} {frame:?} default-up forward")
                );
                // Default up maps to Bevy +Y, so the camera stays right side
                // up for any mostly-horizontal view direction.
                assert!(
                    (q_default * DVec3::Y).dot(DVec3::Y) > 0.0,
                    "ctx {c} {frame:?}: default camera up flipped"
                );
            }
        }
    }

    /// Looking "north" with frame-up must be the identity attitude in both
    /// ENU and NED (their camera identity axes differ, the result must not).
    #[test]
    fn look_at_identity_per_frame() {
        let ctx = dummy_ctx();
        let cases = [
            (GeoFrame::ENU, DVec3::Y, DVec3::Z),
            (GeoFrame::NED, DVec3::X, DVec3::NEG_Z),
        ];
        for (frame, north, up) in cases {
            let r = GeoRotation::look_at(frame, north, Some(up), &ctx);
            assert_quat_eq!(r.1, DQuat::IDENTITY, "{frame:?} local");
            assert_quat_eq!(r.to_bevy(&ctx), DQuat::IDENTITY, "{frame:?} bevy");
        }
    }

    #[test]
    fn from_bevy_kind_round_trips() {
        let ctx = dummy_ctx();
        let v = DQuat::from_euler(bevy::math::EulerRot::XYZ, 0.4, -0.9, 1.3);
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            for kind in [RotationKind::Relative, RotationKind::Absolute] {
                let r = GeoRotation::from_bevy_kind(frame, v, &ctx, kind);
                assert_eq!(r.2, kind);
                assert_quat_eq!(r.to_bevy(&ctx), v, "{frame:?} {kind:?}");
            }
        }
    }

    #[test]
    fn as_frame_preserves_bevy_rotation_and_kind() {
        // Sphere mode: all frames map to Bevy consistently. (In Plane mode the
        // ECEF -> Bevy mapping is a fixed swizzle that intentionally disagrees
        // with the origin-dependent ENU mapping.)
        let ctx = dummy_ctx().with_present(Present::Sphere);
        let q = DQuat::from_euler(bevy::math::EulerRot::XYZ, 0.4, -0.9, 1.3);
        for kind in [RotationKind::Relative, RotationKind::Absolute] {
            for to_frame in [GeoFrame::NED, GeoFrame::ECEF] {
                let r = GeoRotation(GeoFrame::ENU, q, kind);
                let converted = r.as_frame(to_frame, &ctx);
                assert_eq!(converted.0, to_frame);
                assert_eq!(converted.2, kind);
                assert_quat_eq!(
                    converted.to_bevy(&ctx),
                    r.to_bevy(&ctx),
                    "{to_frame:?} {kind:?}"
                );
            }
        }
    }

    #[test]
    fn enu_r_ned_mul_vector_123() {
        let v_ned = DVec3::new(1.0, 2.0, 3.0);
        let v_enu = GeoFrame::enu_R_ned() * v_ned;
        assert_approx_eq!(v_enu, DVec3::new(2.0, 1.0, -3.0));
    }

    #[test]
    fn bevy_r_ecef_plane_converts_through_enu() {
        let origin = GeoOrigin::new_from_degrees(0.0, 0.0, 0.0)
            .with_ellipsoid(Ellipsoid::Sphere { radius: 1.0 });
        let ctx_plane: GeoContext = origin.into();
        let ctx_sphere = ctx_plane.clone().with_present(Present::Sphere);

        let bevy_R_ecef_p = GeoFrame::bevy_R_(&GeoFrame::ECEF, &ctx_plane);
        let bevy_R_ecef_s = GeoFrame::bevy_R_(&GeoFrame::ECEF, &ctx_sphere);
        let ecef_R_enu_p = GeoFrame::ecef_R_(&GeoFrame::ENU, &ctx_plane.origin);
        let ecef_R_enu_s = GeoFrame::ecef_R_(&GeoFrame::ENU, &ctx_sphere.origin);
        let bevy_R_enu_p = GeoFrame::bevy_R_(&GeoFrame::ENU, &ctx_plane);
        let bevy_R_enu_s = GeoFrame::bevy_R_(&GeoFrame::ENU, &ctx_sphere);

        let expected_bevy_R_ecef_p = bevy_R_enu_p * ecef_R_enu_p.inverse();
        assert_approx_eq!(
            bevy_R_ecef_p * DVec3::X,
            expected_bevy_R_ecef_p * DVec3::X,
            1e-9,
            "plane ECEF should go through ENU before Bevy"
        );
        assert_approx_eq!(
            bevy_R_ecef_p * DVec3::X,
            DVec3::Y,
            1e-9,
            "plane bevy_R_ecef x-axis"
        );
        assert_approx_eq!(
            bevy_R_ecef_p * DVec3::Y,
            DVec3::X,
            1e-9,
            "plane bevy_R_ecef y-axis"
        );
        assert_approx_eq!(
            bevy_R_ecef_p * DVec3::Z,
            DVec3::NEG_Z,
            1e-9,
            "plane bevy_R_ecef z-axis"
        );

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

        // 1 m north of the origin (1,0,0): north at lat/lon 0 is +Z_ecef.
        assert_approx_eq!(
            convert_pos(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Y, &ctx_sphere),
            DVec3::new(1.0, 0.0, 1.0),
            1e-9,
            "convert_pos ecef _M_ y-axis"
        );

        // 1 m above the origin (1,0,0): matches map_3d's enu2ecef.
        assert_approx_eq!(
            convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Z, &ctx_sphere),
            DVec3::new(2.0, 0.0, 0.0),
            1e-9,
            "convert_pos ecef _M_ z-axis"
        );
        assert_approx_eq!(
            convert_pos(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Z, &ctx_sphere),
            DVec3::new(2.0, 0.0, 0.0),
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
    fn ecef_plane_positions_are_local_enu_in_bevy() {
        let ctx: GeoContext = GeoOrigin::new_from_degrees(34.72, -86.64, 180.5).into();
        let cases = [
            (
                convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::ZERO, &ctx),
                Vec3::ZERO,
                "origin",
            ),
            (
                convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::X, &ctx),
                Vec3::X,
                "1 m east",
            ),
            (
                convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Y, &ctx),
                Vec3::NEG_Z,
                "1 m north",
            ),
            (
                convert_pos_map_3d(GeoFrame::ECEF, GeoFrame::ENU, DVec3::Z, &ctx),
                Vec3::Y,
                "1 m up",
            ),
        ];

        for (ecef, expected, label) in cases {
            assert_approx_eq!(
                GeoPosition(GeoFrame::ECEF, ecef).to_bevy(&ctx).as_vec3(),
                expected,
                1e-4,
                label
            );
        }
    }

    #[test]
    fn ned_to_ecef_matches_map_3d_for_nontrivial_origins() {
        let cases = [
            (
                GeoOrigin::new_from_degrees(34.72, -86.64, 180.5),
                DVec3::new(125.0, -42.0, -17.5),
                "huntsville-ish",
            ),
            (
                GeoOrigin::new_from_degrees(-23.55, 133.88, 700.0),
                DVec3::new(-310.0, 80.0, 12.0),
                "southern hemisphere",
            ),
            (
                GeoOrigin::new_from_degrees(67.9, 21.1, 420.0),
                DVec3::new(4_000.0, -1_250.0, -250.0),
                "high latitude",
            ),
        ];

        for (origin, ned, label) in cases {
            let ctx_plane: GeoContext = origin.into();
            assert_ned_to_ecef_matches_map_3d(&ctx_plane, ned, &format!("{label} plane"));

            let ctx_sphere = ctx_plane.with_present(Present::Sphere);
            assert_ned_to_ecef_matches_map_3d(&ctx_sphere, ned, &format!("{label} sphere"));
        }
    }

    #[test]
    fn geo_frames_example_distances_match_in_plane_and_sphere() {
        fn assert_example_distances(ctx: &GeoContext, label: &str) {
            let ned_origin = GeoPosition(GeoFrame::NED, DVec3::ZERO).to_bevy(ctx);
            let enu_one_meter_east = GeoPosition(GeoFrame::ENU, DVec3::X).to_bevy(ctx);
            let ecef_one_meter_up = GeoPosition(
                GeoFrame::ECEF,
                geo_frames_example_ecef_from_enu(0.0, 0.0, 1.0),
            )
            .to_bevy(ctx);

            let east_delta = enu_one_meter_east - ned_origin;
            let up_delta = ecef_one_meter_up - ned_origin;
            let bevy_R_enu = GeoFrame::bevy_R_(&GeoFrame::ENU, ctx);
            let expected_east = bevy_R_enu * DVec3::X;
            let expected_up = bevy_R_enu * DVec3::Z;

            assert_approx_eq!(
                east_delta,
                expected_east,
                1e-4,
                format!("{label}: ENU cube should be 1 m east")
            );
            assert_approx_eq!(
                up_delta,
                expected_up,
                1e-4,
                format!("{label}: ECEF cube should be 1 m up")
            );
            assert!(
                (ned_origin.distance(enu_one_meter_east) - 1.0).abs() < 1e-4,
                "{label}: NED origin to ENU east distance was {}",
                ned_origin.distance(enu_one_meter_east)
            );
            assert!(
                (ned_origin.distance(ecef_one_meter_up) - 1.0).abs() < 1e-4,
                "{label}: NED origin to ECEF up distance was {}",
                ned_origin.distance(ecef_one_meter_up)
            );
            assert!(
                (enu_one_meter_east.distance(ecef_one_meter_up) - 2.0_f64.sqrt()).abs() < 1e-4,
                "{label}: ENU east to ECEF up distance was {}",
                enu_one_meter_east.distance(ecef_one_meter_up)
            );
        }

        fn example_camera_forward_up(ctx: &GeoContext) -> (DVec3, DVec3) {
            let camera_pos = DVec3::new(4.0, 4.0, -3.0);
            let camera_look_at = DVec3::ZERO;
            let camera_dir = camera_look_at - camera_pos;
            let camera_rot =
                GeoRotation::look_at(GeoFrame::NED, camera_dir, None, ctx).to_bevy(ctx);
            (camera_rot * DVec3::NEG_Z, camera_rot * DVec3::Y)
        }

        fn projected_ecef_up(ctx: &GeoContext, camera_forward: DVec3) -> DVec3 {
            let ecef_origin = geo_frames_example_ecef_from_enu(0.0, 0.0, 0.0);
            let ecef_one_meter_up = geo_frames_example_ecef_from_enu(0.0, 0.0, 1.0);
            let ecef_down = ecef_origin - ecef_one_meter_up;
            let bevy_up = GeoFrame::bevy_R_(&GeoFrame::ECEF, ctx) * (-ecef_down.normalize());
            (bevy_up - camera_forward * bevy_up.dot(camera_forward)).normalize()
        }

        let ctx_plane: GeoContext = GeoOrigin::new_from_degrees(34.72, -86.64, 180.5).into();
        assert_example_distances(&ctx_plane, "plane");
        let (plane_camera_forward, plane_camera_up) = example_camera_forward_up(&ctx_plane);
        let plane_projected_ecef_up = projected_ecef_up(&ctx_plane, plane_camera_forward);
        let plane_up_alignment = plane_camera_up.dot(plane_projected_ecef_up);
        assert!(
            plane_up_alignment > 1.0 - 1e-9,
            "plane camera up should match projected ECEF up; camera={plane_camera_up:?}, ecef_up={plane_projected_ecef_up:?}, dot={plane_up_alignment}"
        );

        let ctx_sphere = ctx_plane.with_present(Present::Sphere);
        assert_example_distances(&ctx_sphere, "sphere");
        let (sphere_camera_forward, sphere_camera_up) = example_camera_forward_up(&ctx_sphere);
        let sphere_projected_ecef_up = projected_ecef_up(&ctx_sphere, sphere_camera_forward);
        let up_alignment = sphere_camera_up.dot(sphere_projected_ecef_up);
        assert!(
            up_alignment < 0.99,
            "sphere camera up unexpectedly matched projected ECEF up; camera={sphere_camera_up:?}, ecef_up={sphere_projected_ecef_up:?}, dot={up_alignment}"
        );
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
                let expected_plane = if frame == GeoFrame::ECEF {
                    ecef_plane_to_bevy_expected(DVec3::ZERO, &ctx_plane)
                } else {
                    expected_plane
                };
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
                ecef_plane_to_bevy_expected(v, &ctx_plane),
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
            (GeoFrame::ECEF, ecef_plane_to_bevy_expected(v, &ctx_plane)),
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
                    Vec3::new(1.0, 3.0, -2.0)
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
            (GeoFrame::ECEF, ecef_plane_to_bevy_expected(v, &ctx_plane)),
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
                    Vec3::new(1.0, 3.0, -2.0)
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
            (GeoFrame::ECEF, ecef_plane_to_bevy_expected(v, &ctx_plane)),
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
                    Vec3::new(1.0, 3.0, -2.0)
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
