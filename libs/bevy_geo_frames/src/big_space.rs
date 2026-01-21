#![allow(non_snake_case)]
use crate::*;
use bevy::math::{DMat3, DMat4, DQuat, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use map_3d::Ellipsoid;
// use ::big_space::{FloatingOrigin, FloatingOriginSettings, grid::cell::GridCell};
use ::big_space::{precision::GridPrecision, FloatingOrigin, FloatingOriginSettings, GridCell};

pub fn plugin<P: GridPrecision>(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (apply_little_transforms::<P>, crate::apply_geo_rotation)
            .chain()
            .before(TransformSystem::TransformPropagate),
    );

    // Note: There is not a public `SystemSet` to anchor our
    // apply_big_translation to--at least not in our homebrew branch. There is a
    // `SystemSet` in the current big_space release. Till then we'll ask the
    // user to attach their own translation.

    app.add_systems(
        PostUpdate,
        apply_big_translation::<P>, //.before(RootGlobalTransformUpdates)
    );
}

pub fn apply_little_transforms<P: GridPrecision>(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform), (Changed<GeoPosition>, Without<GridCell<P>>)>,
) {
    for (geo, mut transform) in &mut q {
        transform.translation = geo.to_bevy(&ctx).as_vec3();
    }
}

/// System: convert `GeoPosition` into `Transform.translation` right before Bevy
/// propagates transforms through the hierarchy.
pub fn apply_big_translation<P: GridPrecision>(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform, &mut GridCell<P>), Changed<GeoPosition>>,
    floating_origin: Res<FloatingOriginSettings>,
) {
    for (geo, mut transform, mut grid_cell) in &mut q {
        let pos = geo.to_bevy(&ctx);
        let (new_grid_cell, translation) = floating_origin.translation_to_grid(pos);
        *grid_cell = new_grid_cell;
        transform.translation = translation;
    }
}
