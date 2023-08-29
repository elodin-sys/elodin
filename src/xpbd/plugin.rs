use bevy::prelude::*;
use bevy_ecs::schedule::SystemSet;

use crate::{Att, Pos};

use super::{
    constraints::clear_distance_lagrange,
    systems::{self, SubstepSchedule},
};

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TickSet {
    ClearConstraintLagrange,
    TickPhysics,
    SyncPos,
}

pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (clear_distance_lagrange).in_set(TickSet::ClearConstraintLagrange),
        )
        .add_systems(Update, (tick).in_set(TickSet::TickPhysics))
        .add_schedule(SubstepSchedule, systems::substep_schedule())
        .add_systems(Update, (sync_pos).in_set(TickSet::SyncPos))
        .configure_sets(
            Update,
            (
                TickSet::ClearConstraintLagrange,
                TickSet::TickPhysics,
                TickSet::SyncPos,
            )
                .chain(),
        );
    }
}

pub fn tick(world: &mut World) {
    for _ in 0..16 {
        world.run_schedule(SubstepSchedule)
    }
}

pub fn sync_pos(mut query: Query<(&mut Transform, &Pos, &Att)>) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut transform, Pos(pos), Att(att))| {
            transform.translation = Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32);
            transform.rotation =
                Quat::from_xyzw(att.i as f32, att.j as f32, att.k as f32, att.w as f32);
            // TODO: Is `Quat` a JPL quat who knows?!
        });
}
