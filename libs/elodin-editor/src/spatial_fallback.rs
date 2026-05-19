//! Dev-only fallback spatial layer for builds without `big_space`.
//!
//! All grid-cell coordinates collapse to a single cell. This keeps the editor
//! type-compatible for no-default-feature builds, but it is not a product mode.

use bevy::{math::DVec3, prelude::*};

#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOrigin;

#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCell {
    pub x: i128,
    pub y: i128,
    pub z: i128,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct FloatingOriginSettings {
    grid_edge_length: f32,
}

impl FloatingOriginSettings {
    pub fn translation_to_grid(&self, translation: DVec3) -> (GridCell, Vec3) {
        (GridCell::default(), translation.as_vec3())
    }

    pub fn grid_edge_length(&self) -> f32 {
        self.grid_edge_length
    }

    pub fn grid_position_double(&self, _grid_cell: &GridCell, transform: &Transform) -> DVec3 {
        transform.translation.as_dvec3()
    }
}

pub struct FloatingOriginPlugin {
    grid_edge_length: f32,
}

impl FloatingOriginPlugin {
    pub fn new(grid_edge_length: f64, _switch_distance: f64) -> Self {
        Self {
            grid_edge_length: grid_edge_length as f32,
        }
    }
}

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FloatingOriginSettings {
            grid_edge_length: self.grid_edge_length,
        })
        .add_systems(Startup, setup_floating_origin);
    }
}

pub mod debug {
    use bevy::prelude::*;

    #[derive(Default)]
    #[allow(dead_code)]
    pub struct FloatingOriginDebugPlugin;

    impl Plugin for FloatingOriginDebugPlugin {
        fn build(&self, _app: &mut App) {}
    }
}

#[derive(Component, Default, Clone, Copy, Debug)]
pub struct LowPrecisionRoot;

pub fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::default(),
        Transform::IDENTITY,
        Name::new("floating origin"),
    ));
}

pub fn apply_big_translation(
    ctx: ResMut<bevy_geo_frames::GeoContext>,
    mut q: Query<
        (&bevy_geo_frames::GeoPosition, &mut Transform, &mut GridCell),
        Changed<bevy_geo_frames::GeoPosition>,
    >,
    floating_origin: Res<FloatingOriginSettings>,
) {
    for (geo, mut transform, mut grid_cell) in &mut q {
        let pos = geo.to_bevy(&ctx);
        let (new_grid_cell, translation) = floating_origin.translation_to_grid(pos);
        *grid_cell = new_grid_cell;
        transform.translation = translation;
    }
}
