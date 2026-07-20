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
//!   the surface re-exports / wraps types from `big_space` 0.13. The
//!   wrapper is needed because the 0.12 release reorganised the API:
//!   `FloatingOriginSettings` is gone, replaced by a `Grid` component
//!   stored on the `BigSpace` root entity.
//! * `not(feature = "big_space")`: dummy implementations that compile but
//!   collapse the grid to a single cell. This path exists purely for
//!   development ergonomics during the Bevy migration; it is **not
//!   intended to be shipped**.
//!
//! Prefer parenting `GridCell` entities under [`BigSpaceRoot`] at spawn
//! (via [`BigSpaceRootEntity`]). Insert-time observers cover impeller path
//! leaves that become spatial after hierarchy wiring, so we do not need
//! per-frame janitor systems.

use bevy::{
    ecs::lifecycle::{Add, Insert},
    math::DVec3,
    prelude::*,
};

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

    pub fn grid_position_double(&self, grid_cell: &GridCell, transform: &Transform) -> DVec3 {
        self.grid.grid_position_double(grid_cell, transform)
    }
}

/// Marker for the unique `BigSpace` root entity spawned by
/// [`FloatingOriginPlugin`].
#[derive(Component)]
pub struct BigSpaceRoot;

/// Handle to the unique [`BigSpaceRoot`] entity. Inserted at startup so
/// spawn sites can parent grid-bearing entities without a query.
#[derive(Resource, Clone, Copy, Debug)]
pub struct BigSpaceRootEntity(pub Entity);

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
            .add_observer(on_grid_cell_added)
            .add_observer(on_child_of_inserted_for_grid_cell);
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

    commands.insert_resource(BigSpaceRootEntity(root));

    commands.spawn((
        FloatingOrigin,
        GridCell::default(),
        Transform::IDENTITY,
        ChildOf(root),
        Name::new("floating origin"),
    ));
}

/// Parent a grid-bearing entity under the BigSpace root when the root is known.
pub fn parent_under_big_space(entity: &mut EntityCommands, root: Option<&BigSpaceRootEntity>) {
    if let Some(root) = root {
        entity.insert(ChildOf(root.0));
    }
}

type SpatialParentFilter = Or<(With<Grid>, With<GridCell>, With<BigSpace>)>;

fn parent_is_spatial(
    parent: Entity,
    root: Entity,
    spatial_parents: &Query<Entity, SpatialParentFilter>,
) -> bool {
    parent == root || spatial_parents.contains(parent)
}

/// When a [`GridCell`] is added, ensure the entity is under a valid big_space
/// parent (root if parentless or under a non-spatial impeller path segment).
fn on_grid_cell_added(
    add: On<Add, GridCell>,
    mut commands: Commands,
    root: Option<Res<BigSpaceRootEntity>>,
    child_of: Query<&ChildOf>,
    spatial_parents: Query<Entity, SpatialParentFilter>,
) {
    let Some(root) = root else {
        return;
    };
    let entity = add.entity;
    if entity == root.0 {
        return;
    }
    match child_of.get(entity) {
        Ok(c) if parent_is_spatial(c.parent(), root.0, &spatial_parents) => {}
        _ => {
            commands.entity(entity).insert(ChildOf(root.0));
        }
    }
}

/// If a [`GridCell`] entity is (re)parented under a non-spatial entity, move it
/// to the BigSpace root. Safe against recursion: the second trigger sees
/// `parent == root` and returns.
fn on_child_of_inserted_for_grid_cell(
    insert: On<Insert, ChildOf>,
    mut commands: Commands,
    root: Option<Res<BigSpaceRootEntity>>,
    cells: Query<&ChildOf, With<GridCell>>,
    spatial_parents: Query<Entity, SpatialParentFilter>,
) {
    let Some(root) = root else {
        return;
    };
    let entity = insert.entity;
    let Ok(child_of) = cells.get(entity) else {
        return;
    };
    let parent = child_of.parent();
    if parent_is_spatial(parent, root.0, &spatial_parents) {
        return;
    }
    commands.entity(entity).insert(ChildOf(root.0));
}
