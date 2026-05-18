#[cfg(not(feature = "big_space"))]
use bevy::{math::DVec3, prelude::*};

#[cfg(feature = "big_space")]
pub use big_space::{FloatingOrigin, FloatingOriginPlugin, FloatingOriginSettings, GridCell};

#[cfg(feature = "big_space")]
pub use bevy_geo_frames::big_space::apply_big_translation;

#[cfg(all(feature = "big_space", feature = "debug"))]
pub mod debug {
    pub use big_space::debug::FloatingOriginDebugPlugin;
}

#[cfg(feature = "big_space")]
pub mod propagation {
    pub use big_space::propagation::NoPropagateRot;
}

#[cfg(not(feature = "big_space"))]
#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingOrigin;

#[cfg(not(feature = "big_space"))]
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCell<P = i128> {
    pub x: P,
    pub y: P,
    pub z: P,
}

#[cfg(not(feature = "big_space"))]
impl<P: Default> Default for GridCell<P> {
    fn default() -> Self {
        Self {
            x: P::default(),
            y: P::default(),
            z: P::default(),
        }
    }
}

#[cfg(not(feature = "big_space"))]
#[derive(Resource, Clone, Copy, Debug)]
pub struct FloatingOriginSettings {
    grid_edge_length: f32,
}

#[cfg(not(feature = "big_space"))]
impl FloatingOriginSettings {
    pub fn translation_to_grid<P: Default>(&self, translation: DVec3) -> (GridCell<P>, Vec3) {
        (GridCell::default(), translation.as_vec3())
    }

    pub fn grid_edge_length(&self) -> f32 {
        self.grid_edge_length
    }

    pub fn grid_position_double<P>(
        &self,
        _grid_cell: &GridCell<P>,
        transform: &Transform,
    ) -> DVec3 {
        transform.translation.as_dvec3()
    }
}

#[cfg(not(feature = "big_space"))]
pub struct FloatingOriginPlugin<P = i128> {
    grid_edge_length: f32,
    _marker: std::marker::PhantomData<P>,
}

#[cfg(not(feature = "big_space"))]
impl<P> FloatingOriginPlugin<P> {
    pub fn new(grid_edge_length: f64, _switch_distance: f64) -> Self {
        Self {
            grid_edge_length: grid_edge_length as f32,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(not(feature = "big_space"))]
impl<P: Send + Sync + 'static> Plugin for FloatingOriginPlugin<P> {
    fn build(&self, app: &mut App) {
        app.insert_resource(FloatingOriginSettings {
            grid_edge_length: self.grid_edge_length,
        });
    }
}

#[cfg(not(feature = "big_space"))]
pub mod debug {
    use bevy::prelude::*;

    #[derive(Default)]
    #[allow(dead_code)]
    pub struct FloatingOriginDebugPlugin<P = i128>(std::marker::PhantomData<P>);

    impl<P: Send + Sync + 'static> Plugin for FloatingOriginDebugPlugin<P> {
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
pub fn apply_big_translation<P: Default + Send + Sync + 'static>(
    ctx: ResMut<bevy_geo_frames::GeoContext>,
    mut q: Query<
        (
            &bevy_geo_frames::GeoPosition,
            &mut Transform,
            &mut GridCell<P>,
        ),
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
