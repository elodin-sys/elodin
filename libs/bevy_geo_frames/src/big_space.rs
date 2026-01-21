#![allow(non_snake_case)]
use bevy::math::{DMat3, DMat4, DQuat, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use map_3d::Ellipsoid;
use crate::*;
// use ::big_space::{FloatingOrigin, FloatingOriginSettings, grid::cell::GridCell};
use ::big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};

pub fn plugin(app: &mut App) {
    app.add_systems(
        PostUpdate,
        crate::apply_geo_rotation
            .before(TransformSystem::TransformPropagate),
    );

    // Note: There is not a public `SystemSet` to anchor our
    // apply_geo_translation to--at least not in our homebrew branch. There is a
    // `SystemSet` in the current big_space release. Till then we'll ask the
    // user to attach their own translation.

    // app.add_systems(
    //     PostUpdate,
    //     apply_geo_translation
    //         .before(RootGlobalTransformUpdates)
    // );
}

/// System: convert `GeoPosition` into `Transform.translation` right before Bevy
/// propagates transforms through the hierarchy.
pub fn apply_geo_translation(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform, &mut GridCell<i128>), Changed<GeoPosition>>,
    floating_origin: Res<FloatingOriginSettings>,
) {
    for (geo, mut transform, mut grid_cell) in &mut q {
        let pos = geo.to_bevy(&ctx);
        let (new_grid_cell, translation) = floating_origin.translation_to_grid(pos);
        *grid_cell = new_grid_cell;
        transform.translation = translation;
    }
}
