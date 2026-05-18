//! Support for the big_space crate.
#![allow(non_snake_case)]
use crate::*;
use ::big_space::prelude::{BigSpace, CellCoord as GridCell, Grid};
use bevy::prelude::*;

/// Add the big_space systems.
pub fn plugin(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (apply_little_transforms, crate::apply_geo_rotation)
            .chain()
            .before(TransformSystems::Propagate),
    );

    app.add_systems(
        PostUpdate,
        apply_big_translation.before(TransformSystems::Propagate),
    );
}

/// Applies the same transforms as [crate::apply_transforms] but excludes
/// components with a [GridCell].
#[allow(clippy::type_complexity)]
pub fn apply_little_transforms(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform), (Changed<GeoPosition>, Without<GridCell>)>,
) {
    for (geo, mut transform) in &mut q {
        transform.translation = geo.to_bevy(&ctx).as_vec3();
    }
}

/// System: convert `GeoPosition` into `Transform.translation` right before Bevy
/// propagates transforms through the hierarchy.
pub fn apply_big_translation(
    ctx: ResMut<GeoContext>,
    mut q: Query<(&GeoPosition, &mut Transform, &mut GridCell), Changed<GeoPosition>>,
    grids: Query<&Grid, With<BigSpace>>,
) {
    let Some(grid) = grids.iter().next() else {
        return;
    };

    for (geo, mut transform, mut grid_cell) in &mut q {
        let pos = geo.to_bevy(&ctx);
        let (new_grid_cell, translation) = grid.translation_to_grid(pos);
        *grid_cell = new_grid_cell;
        transform.translation = translation;
    }
}
