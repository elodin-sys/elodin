use crate::terrain::{
    math::{coordinate::Coordinate, ellipsoid::project_point_ellipsoid, TileCoordinate, C_SQR},
    terrain_data::tile_atlas::TileAtlas,
    terrain_data::tile_tree::TileTree,
    terrain_view::TerrainViewComponents,
};
use bevy::{
    math::{DMat3, DMat4, DQuat, DVec2, DVec3, IVec2},
    prelude::*,
    render::render_resource::ShaderType,
};

/// One matrix per side, which shuffles the a, b, and c component to their corresponding position.
const SIDE_MATRICES: [DMat3; 6] = [
    DMat3::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]),
    DMat3::from_cols_array(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
    DMat3::from_cols_array(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]),
    DMat3::from_cols_array(&[0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]),
    DMat3::from_cols_array(&[0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
];

#[derive(Clone)]
pub enum TerrainKind {
    Planar {
        side_length: f64,
    },
    Spherical {
        radius: f64,
    },
    Ellipsoidal {
        ellipsoid_from_world: DMat4,
        major_axis: f64,
        minor_axis: f64,
    },
}

// Todo: keep in sync with terrain transform, make this authoritative?

#[derive(Clone)]
pub struct TerrainModel {
    pub(crate) kind: TerrainKind,
    pub(crate) min_height: f32,
    pub(crate) max_height: f32,
    translation: DVec3,
    scale: DVec3,
    rotation: DQuat,
    world_from_local: DMat4,
    local_from_world: DMat4,
}

impl TerrainModel {
    pub(crate) fn is_spherical(&self) -> bool {
        match self.kind {
            TerrainKind::Planar { .. } => false,
            TerrainKind::Spherical { .. } => true,
            TerrainKind::Ellipsoidal { .. } => true,
        }
    }

    fn from_scale_rotation_translation(
        scale: DVec3,
        rotation: DQuat,
        translation: DVec3,
        min_height: f32,
        max_height: f32,
        kind: TerrainKind,
    ) -> Self {
        let world_from_local = DMat4::from_scale_rotation_translation(scale, rotation, translation);
        let local_from_world = world_from_local.inverse();

        Self {
            kind,
            min_height,
            max_height,
            translation,
            scale,
            rotation,
            world_from_local,
            local_from_world,
        }
    }

    pub fn planar(position: DVec3, side_length: f64, min_height: f32, max_height: f32) -> Self {
        Self::from_scale_rotation_translation(
            DVec3::splat(side_length),
            DQuat::IDENTITY,
            position,
            min_height,
            max_height,
            TerrainKind::Planar { side_length },
        )
    }

    pub fn sphere(position: DVec3, radius: f64, min_height: f32, max_height: f32) -> Self {
        Self::from_scale_rotation_translation(
            DVec3::splat(radius),
            DQuat::IDENTITY,
            position,
            min_height,
            max_height,
            TerrainKind::Spherical { radius },
        )
    }

    pub fn ellipsoid(
        position: DVec3,
        major_axis: f64,
        minor_axis: f64,
        min_height: f32,
        max_height: f32,
    ) -> Self {
        let rotation = DQuat::IDENTITY; // ::from_rotation_x(45.0_f64.to_radians());
        let ellipsoid_from_world = DMat4::from_rotation_translation(rotation, position).inverse();

        Self::from_scale_rotation_translation(
            DVec3::new(major_axis, minor_axis, major_axis),
            rotation,
            position,
            min_height,
            max_height,
            TerrainKind::Ellipsoidal {
                ellipsoid_from_world,
                major_axis,
                minor_axis,
            },
        )
    }

    pub(crate) fn position_local_to_world(&self, local_position: DVec3, height: f64) -> DVec3 {
        let world_position = self.world_from_local.transform_point3(local_position);
        let world_normal = self
            .world_from_local
            .transform_vector3(if self.is_spherical() {
                local_position
            } else {
                DVec3::Y
            })
            .normalize();

        world_position + height * world_normal
    }

    pub(crate) fn position_world_to_local(&self, world_position: DVec3) -> DVec3 {
        match self.kind {
            TerrainKind::Planar { .. } => {
                DVec3::new(1.0, 0.0, 1.0) * self.local_from_world.transform_point3(world_position)
            }

            TerrainKind::Spherical { .. } => self
                .local_from_world
                .transform_point3(world_position)
                .normalize(),
            TerrainKind::Ellipsoidal {
                ellipsoid_from_world,
                major_axis,
                minor_axis,
            } => {
                let ellipsoid_position = ellipsoid_from_world.transform_point3(world_position);
                let surface_position = project_point_ellipsoid(
                    DVec3::new(major_axis, major_axis, minor_axis),
                    ellipsoid_position,
                );
                self.local_from_world
                    .transform_point3(surface_position)
                    .normalize()
            }
        }
    }

    pub(crate) fn surface_position(&self, world_position: DVec3, height: f64) -> DVec3 {
        self.position_local_to_world(self.position_world_to_local(world_position), height)
    }

    pub(crate) fn side_count(&self) -> u32 {
        if self.is_spherical() {
            6
        } else {
            1
        }
    }

    pub(crate) fn scale(&self) -> f64 {
        match self.kind {
            TerrainKind::Planar { side_length } => side_length / 2.0,
            TerrainKind::Spherical { radius } => radius,
            TerrainKind::Ellipsoidal {
                major_axis,
                minor_axis,
                ..
            } => (major_axis + minor_axis) / 2.0,
        }
    }

    #[cfg(not(feature = "high_precision"))]
    pub(crate) fn transform(&self) -> Transform {
        Transform {
            translation: self.translation.as_vec3(),
            scale: self.scale.as_vec3(),
            rotation: self.rotation.as_quat(),
        }
    }

    #[cfg(feature = "high_precision")]
    pub(crate) fn grid_transform(
        &self,
        frame: &big_space::prelude::Grid,
    ) -> big_space::prelude::CellTransformOwned {
        let (cell, translation) = frame.translation_to_grid(self.translation);

        big_space::prelude::CellTransformOwned {
            transform: Transform {
                translation,
                scale: self.scale.as_vec3(),
                rotation: self.rotation.as_quat(),
            },
            cell,
        }
    }
}

/// Parameters of the view used to compute the position of a location on the sphere's surface relative to the view.
/// This can be calculated directly using f64 operations, or approximated using a Taylor series and f32 operations.
///
/// The idea behind the approximation, is to map from st coordinates relative to the view, to world positions relative to the view.
/// Therefore, we identify a origin tile with sufficiently high lod (origin LOD), that serves as a reference, to which we can compute our relative coordinate using partly integer math.
#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub(crate) struct SideParameter {
    /// The tile index of the origin tile projected to this side.
    pub(crate) origin_xy: IVec2,
    /// The offset between the view st coordinate and the origin st coordinate.
    /// This can be used to translate from st coordinates relative to the origin tile to st coordinates relative to the view coordinate in the shader.
    pub(crate) origin_uv: Vec2,
    /// The constant coefficient of the series.
    /// Describes the offset between the location vertically under view and the view position.
    pub(crate) c: Vec3,
    /// The linear coefficient of the series with respect to s.
    pub(crate) c_s: Vec3,
    /// The linear coefficient of the series with respect to t.
    pub(crate) c_t: Vec3,
    /// The quadratic coefficient of the series with respect to s and s.
    /// This value is pre-multiplied with 0.5.
    pub(crate) c_ss: Vec3,
    /// The quadratic coefficient of the series with respect to s and t.
    pub(crate) c_st: Vec3,
    /// The quadratic coefficient of the series with respect to t and t.
    /// This value is pre-multiplied with 0.5.
    pub(crate) c_tt: Vec3,
}

#[derive(Clone, Debug, Default, ShaderType)]
pub struct TerrainModelApproximation {
    /// The reference tile, which is used to accurately determine the relative st coordinate in the shader.
    /// The tile under the view (with the origin lod) is the origin for the Taylor series.
    pub(crate) origin_lod: u32,
    pub(crate) approximate_height: f32,
    /// The parameters of the six cube sphere faces.
    pub(crate) sides: [SideParameter; 6],
}

impl TerrainModelApproximation {
    /// Computes the view parameters based on the it's world position.
    pub(crate) fn compute(
        tile_tree: &TileTree,
        tile_atlas: &TileAtlas,
    ) -> TerrainModelApproximation {
        let origin_count = TileCoordinate::count(tile_tree.origin_lod) as f64;

        // Coordinate of the location vertically below the view.
        let view_coordinate =
            Coordinate::from_world_position(tile_tree.view_world_position, &tile_atlas.model);

        // We want to approximate the position relative to the view using a second order Taylor series.
        // For that, we have to calculate the Taylor coefficients for each cube side separately.
        // As the basis, we use the view coordinate projected to the specific side.
        // Then we calculate the relative position vector and derivatives at the view coordinate.

        // u(s)=(2s-1)/sqrt(1-4cs(s-1))
        // v(t)=(2t-1)/sqrt(1-4ct(t-1))
        // l(s,t)=sqrt(1+u(s)^2+v(t)^2)
        // a(s,t)=1/l(s,t)
        // b(s,t)=u(s)/l(s,t)
        // c(s,t)=v(t)/l(s,t)

        let mut sides = [SideParameter::default(); 6];

        for (side, &sm) in SIDE_MATRICES.iter().enumerate() {
            let view_coordinate = view_coordinate.project_to_side(side as u32, &tile_atlas.model);
            let view_xy = (view_coordinate.uv * origin_count).as_ivec2();
            let view_uv = (view_coordinate.uv * origin_count).fract().as_vec2();

            let DVec2 { x: s, y: t } = view_coordinate.uv;

            let u_denom = (1.0 - 4.0 * C_SQR * s * (s - 1.0)).sqrt();
            let u = (2.0 * s - 1.0) / u_denom;
            let u_ds = 2.0 * (C_SQR + 1.0) / u_denom.powi(3);
            let u_dss = 12.0 * C_SQR * (C_SQR + 1.0) * (2.0 * s - 1.0) / u_denom.powi(5);

            let v_denom = (1.0 - 4.0 * C_SQR * t * (t - 1.0)).sqrt();
            let v = (2.0 * t - 1.0) / v_denom;
            let v_dt = 2.0 * (C_SQR + 1.0) / v_denom.powi(3);
            let v_dtt = 12.0 * C_SQR * (C_SQR + 1.0) * (2.0 * t - 1.0) / v_denom.powi(5);

            let l = (1.0 + u * u + v * v).sqrt();
            let l_ds = u * u_ds / l;
            let l_dt = v * v_dt / l;
            let l_dss = (u * u_dss * l * l + (v * v + 1.0) * u_ds * u_ds) / l.powi(3);
            let l_dst = -(u * v * u_ds * v_dt) / l.powi(3);
            let l_dtt = (v * v_dtt * l * l + (u * u + 1.0) * v_dt * v_dt) / l.powi(3);

            let a = 1.0;
            let a_ds = -l_ds;
            let a_dt = -l_dt;
            let a_dss = 2.0 * l_ds * l_ds - l * l_dss;
            let a_dst = 2.0 * l_ds * l_dt - l * l_dst;
            let a_dtt = 2.0 * l_dt * l_dt - l * l_dtt;

            let b = u;
            let b_ds = -u * l_ds + l * u_ds;
            let b_dt = -u * l_dt;
            let b_dss = 2.0 * u * l_ds * l_ds - l * (2.0 * u_ds * l_ds + u * l_dss) + u_dss * l * l;
            let b_dst = 2.0 * u * l_ds * l_dt - l * (u_ds * l_dt + u * l_dst);
            let b_dtt = 2.0 * u * l_dt * l_dt - l * u * l_dtt;

            let c = v;
            let c_ds = -v * l_ds;
            let c_dt = -v * l_dt + l * v_dt;
            let c_dss = 2.0 * v * l_ds * l_ds - l * v * l_dss;
            let c_dst = 2.0 * v * l_ds * l_dt - l * (v_dt * l_ds + v * l_dst);
            let c_dtt = 2.0 * v * l_dt * l_dt - l * (2.0 * v_dt * l_dt + v * l_dtt) + v_dtt * l * l;

            // The model matrix is used to transform the local position and directions into the corresponding world position and directions.
            // p is transformed as a point, takes the model position into account
            // the other coefficients are transformed as vectors, discards the translation
            let m = tile_atlas.model.world_from_local;
            let p = m.transform_point3(sm * DVec3::new(a, b, c) / l);
            let p_ds = m.transform_vector3(sm * DVec3::new(a_ds, b_ds, c_ds) / l.powi(2));
            let p_dt = m.transform_vector3(sm * DVec3::new(a_dt, b_dt, c_dt) / l.powi(2));
            let p_dss = m.transform_vector3(sm * DVec3::new(a_dss, b_dss, c_dss) / l.powi(3));
            let p_dst = m.transform_vector3(sm * DVec3::new(a_dst, b_dst, c_dst) / l.powi(3));
            let p_dtt = m.transform_vector3(sm * DVec3::new(a_dtt, b_dtt, c_dtt) / l.powi(3));

            sides[side] = SideParameter {
                origin_xy: view_xy,
                origin_uv: view_uv,
                c: (p - tile_tree.view_world_position).as_vec3(),
                c_s: p_ds.as_vec3(),
                c_t: p_dt.as_vec3(),
                c_ss: (p_dss / 2.0).as_vec3(),
                c_st: p_dst.as_vec3(),
                c_tt: (p_dtt / 2.0).as_vec3(),
            };
        }

        TerrainModelApproximation {
            origin_lod: tile_tree.origin_lod,
            approximate_height: tile_tree.approximate_height,
            sides,
        }
    }
}

pub fn generate_terrain_model_approximation(
    tile_trees: Res<TerrainViewComponents<TileTree>>,
    tile_atlases: Query<&TileAtlas>,
    mut terrain_model_approximations: ResMut<TerrainViewComponents<TerrainModelApproximation>>,
) {
    for (&(terrain, view), tile_tree) in tile_trees.iter() {
        let tile_atlas = tile_atlases.get(terrain).unwrap();

        terrain_model_approximations.insert(
            (terrain, view),
            TerrainModelApproximation::compute(tile_tree, tile_atlas),
        );
    }
}
