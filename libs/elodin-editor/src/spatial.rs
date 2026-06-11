//! Floating-origin spatial layer for the editor.
//!
//! This module is a thin compatibility shim that the rest of the editor
//! depends on through a single, stable surface (`GridCell`,
//! `FloatingOrigin`, `FloatingOriginSettings`, `FloatingOriginPlugin`,
//! `LowPrecisionRoot`).
//!
//! Two implementations live behind the `big_space` cargo feature:
//!
//! * `feature = "big_space"` (default, **the only supported product mode**):
//!   the surface re-exports / wraps types from `big_space` 0.12. The
//!   wrapper is needed because that release reorganised the API:
//!   `FloatingOriginSettings` is gone, replaced by a `Grid` component
//!   stored on the `BigSpace` root entity.
//! * `not(feature = "big_space")`: dummy implementations that compile but
//!   collapse the grid to a single cell. This path exists purely for
//!   development ergonomics during the Bevy migration; it is **not
//!   intended to be shipped**.
//!
//! Note: spawning the editor's `FloatingOriginPlugin` registers a
//! [`attach_parentless_grid_cells`] system. Any entity that holds a
//! [`GridCell`] and no [`ChildOf`] will be reparented to the
//! [`BigSpaceRoot`] automatically. Call sites can therefore spawn
//! grid-aware entities without explicitly setting their parent.

use bevy::{math::DVec3, prelude::*};

pub use big_space::prelude::{BigSpace, CellCoord as GridCell, FloatingOrigin, Grid};

pub use bevy_geo_frames::big_space::apply_big_translation;

pub use big_space::grid::propagation::LowPrecisionRoot;

pub mod debug {
    use bevy::prelude::*;

    #[derive(Default)]
    #[allow(dead_code)]
    pub struct FloatingOriginDebugPlugin;

    impl Plugin for FloatingOriginDebugPlugin {
        fn build(&self, _app: &mut App) {}
    }
}

pub type WithoutFloatingOrigin = Without<FloatingOrigin>;
pub type WithFloatingOrigin = With<FloatingOrigin>;

#[derive(Resource, Clone, Debug)]
pub struct FloatingOriginSettings {
    grid: Grid,
}

impl FloatingOriginSettings {
    pub fn new(grid_edge_length: f32, switching_threshold: f32) -> Self {
        Self {
            grid: Grid::new(grid_edge_length, switching_threshold),
        }
    }

    pub fn grid(&self) -> Grid {
        self.grid.clone()
    }

    pub fn translation_to_grid(&self, translation: impl Into<DVec3>) -> (GridCell, Vec3) {
        self.grid.translation_to_grid(translation)
    }

    pub fn grid_edge_length(&self) -> f32 {
        self.grid.cell_edge_length()
    }

    pub fn grid_position_double(&self, grid_cell: &GridCell, transform: &Transform) -> DVec3 {
        self.grid.grid_position_double(grid_cell, transform)
    }
}

/// Marker for the unique `BigSpace` root entity spawned by
/// [`FloatingOriginPlugin`]. Lets [`attach_parentless_grid_cells`] adopt
/// any new grid-bearing entity without having to look up the root by name.
#[derive(Component)]
pub struct BigSpaceRoot;

type ParentlessGridCellFilter = (With<GridCell>, Without<ChildOf>, Without<BigSpaceRoot>);

pub struct FloatingOriginPlugin {
    settings: FloatingOriginSettings,
}

impl FloatingOriginPlugin {
    pub fn new(grid_edge_length: f64, switching_threshold: f64) -> Self {
        Self {
            settings: FloatingOriginSettings::new(
                grid_edge_length as f32,
                switching_threshold as f32,
            ),
        }
    }
}

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.settings.clone())
            .add_plugins(big_space::prelude::BigSpaceDefaultPlugins)
            .add_systems(Startup, setup_floating_origin)
            .add_systems(PreUpdate, attach_parentless_grid_cells)
            // Also adopt right before transform propagation (and big_space's
            // PostUpdate hierarchy validation) so entities that gained a
            // `GridCell` during this frame never cross a validation pass
            // unparented.
            .add_systems(
                PostUpdate,
                attach_parentless_grid_cells.before(bevy::transform::TransformSystems::Propagate),
            );
    }
}

pub fn setup_floating_origin(mut commands: Commands, settings: Res<FloatingOriginSettings>) {
    let root = commands
        .spawn((
            BigSpace::default(),
            settings.grid(),
            BigSpaceRoot,
            Name::new("big space root"),
        ))
        .id();

    commands.spawn((
        FloatingOrigin,
        GridCell::default(),
        Transform::IDENTITY,
        ChildOf(root),
        Name::new("floating origin"),
    ));
}

/// PreUpdate system that adopts every entity carrying a [`GridCell`] but
/// no [`ChildOf`] under the unique [`BigSpaceRoot`]. This keeps callers
/// free from having to know about the root entity when they spawn
/// grid-aware entities. Runs every frame so newly spawned entities are
/// re-parented as soon as they are visible to the query.
fn attach_parentless_grid_cells(
    mut commands: Commands,
    roots: Query<Entity, With<BigSpaceRoot>>,
    entities: Query<Entity, ParentlessGridCellFilter>,
) {
    let Some(root) = roots.iter().next() else {
        return;
    };

    for entity in &entities {
        commands.entity(entity).insert(ChildOf(root));
    }
}
