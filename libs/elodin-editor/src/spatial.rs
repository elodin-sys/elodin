use bevy::{math::DVec3, prelude::*};

#[cfg(feature = "big_space")]
pub use big_space::prelude::{BigSpace, CellCoord as GridCell, FloatingOrigin, Grid};

#[cfg(feature = "big_space")]
pub use bevy_geo_frames::big_space::apply_big_translation;

#[cfg(feature = "big_space")]
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct NoPropagateRot;

#[cfg(feature = "big_space")]
pub mod propagation {
    pub use super::NoPropagateRot;
}

#[cfg(feature = "big_space")]
pub mod debug {
    use bevy::prelude::*;

    #[derive(Default)]
    #[allow(dead_code)]
    pub struct FloatingOriginDebugPlugin;

    impl Plugin for FloatingOriginDebugPlugin {
        fn build(&self, _app: &mut App) {}
    }
}

#[cfg(feature = "big_space")]
#[derive(Resource, Clone, Debug)]
pub struct FloatingOriginSettings {
    grid: Grid,
}

#[cfg(feature = "big_space")]
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

#[cfg(feature = "big_space")]
#[derive(Component)]
pub struct BigSpaceRoot;

#[cfg(feature = "big_space")]
type ParentlessGridCellFilter = (With<GridCell>, Without<ChildOf>, Without<BigSpaceRoot>);

#[cfg(feature = "big_space")]
pub struct FloatingOriginPlugin {
    settings: FloatingOriginSettings,
}

#[cfg(feature = "big_space")]
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

#[cfg(feature = "big_space")]
impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.settings.clone())
            .add_plugins(big_space::prelude::BigSpaceDefaultPlugins)
            .add_systems(Startup, setup_floating_origin)
            .add_systems(PreUpdate, attach_parentless_grid_cells);
    }
}

#[cfg(feature = "big_space")]
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

#[cfg(feature = "big_space")]
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

#[cfg(not(feature = "big_space"))]
#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOrigin;

#[cfg(not(feature = "big_space"))]
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCell {
    pub x: i128,
    pub y: i128,
    pub z: i128,
}

#[cfg(not(feature = "big_space"))]
impl Default for GridCell {
    fn default() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }
}

#[cfg(not(feature = "big_space"))]
#[derive(Resource, Clone, Copy, Debug)]
pub struct FloatingOriginSettings {
    grid_edge_length: f32,
}

#[cfg(not(feature = "big_space"))]
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

#[cfg(not(feature = "big_space"))]
pub struct FloatingOriginPlugin {
    grid_edge_length: f32,
}

#[cfg(not(feature = "big_space"))]
impl FloatingOriginPlugin {
    pub fn new(grid_edge_length: f64, _switch_distance: f64) -> Self {
        Self {
            grid_edge_length: grid_edge_length as f32,
        }
    }
}

#[cfg(not(feature = "big_space"))]
impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FloatingOriginSettings {
            grid_edge_length: self.grid_edge_length,
        })
        .add_systems(Startup, setup_floating_origin);
    }
}

#[cfg(not(feature = "big_space"))]
pub mod debug {
    use bevy::prelude::*;

    #[derive(Default)]
    #[allow(dead_code)]
    pub struct FloatingOriginDebugPlugin;

    impl Plugin for FloatingOriginDebugPlugin {
        fn build(&self, _app: &mut App) {}
    }
}

#[cfg(not(feature = "big_space"))]
pub mod propagation {
    use bevy::prelude::*;

    #[derive(Component, Default, Clone, Copy, Debug)]
    pub struct NoPropagateRot;
}

#[cfg(not(feature = "big_space"))]
pub fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::default(),
        Transform::IDENTITY,
        Name::new("floating origin"),
    ));
}

#[cfg(not(feature = "big_space"))]
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
