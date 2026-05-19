//! Dev-only fallback spatial layer for builds without `big_space`.
//!
//! All grid-cell coordinates collapse to a single cell. This keeps the editor
//! type-compatible for no-default-feature builds, but it is not a product mode.

use bevy::{math::DVec3, prelude::*};

// pub type FloatingOrigin = ();
// #[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
// pub struct FloatingOrigin;
pub type WithoutFloatingOrigin = ();
pub type WithFloatingOrigin = ();

// #[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
// pub struct GridCell {
//     pub x: i128,
//     pub y: i128,
//     pub z: i128,
// }

#[derive(Resource, Clone, Copy, Debug)]
pub struct FloatingOriginSettings {
    grid_edge_length: f32,
}

impl FloatingOriginSettings {
    pub fn translation_to_grid(&self, translation: DVec3) -> ((), Vec3) {
        ((), translation.as_vec3())
    }

    pub fn grid_edge_length(&self) -> f32 {
        self.grid_edge_length
    }

    pub fn grid_position_double(&self, _grid_cell: &(), transform: &Transform) -> DVec3 {
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
        Transform::IDENTITY,
        Name::new("floating origin"),
    ));
}

