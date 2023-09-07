use bevy::prelude::*;
use bevy_ecs::schedule::{ScheduleLabel, SystemSet};

use crate::{Att, Pos};

use super::{
    constraints::{
        clear_distance_lagrange, clear_revolute_lagrange, distance_system, gravity_system,
        revolute_damping, revolute_system,
    },
    systems::*,
    SUBSTEPS,
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
            (clear_distance_lagrange, clear_revolute_lagrange)
                .in_set(TickSet::ClearConstraintLagrange),
        )
        .add_systems(Update, (tick).in_set(TickSet::TickPhysics))
        .add_systems(Update, (sync_pos).in_set(TickSet::SyncPos))
        .configure_sets(
            Update,
            (
                TickSet::ClearConstraintLagrange,
                TickSet::TickPhysics,
                TickSet::SyncPos,
            )
                .chain(),
        )
        //.insert_resource(FixedTime::new_from_secs(1.0 / 60.0))
        .add_schedule(SubstepSchedule, substep_schedule());
    }
}

pub fn tick(world: &mut World) {
    for _ in 0..SUBSTEPS {
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

#[derive(ScheduleLabel, Debug, PartialEq, Eq, Hash, Clone)]
pub struct SubstepSchedule;

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum SubstepSet {
    CalcEffects,
    Integrate,
    SolveConstraints,
    UpdateVel,
    DampJoints,
    ClearEffects,
    UpdateTime,
}

pub fn substep_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.configure_sets(
        (
            SubstepSet::CalcEffects,
            SubstepSet::Integrate,
            SubstepSet::SolveConstraints,
            SubstepSet::UpdateVel,
            SubstepSet::DampJoints,
            SubstepSet::ClearEffects,
            SubstepSet::UpdateTime,
        )
            .chain(),
    );
    schedule.add_systems(
        (calculate_effects, calculate_sensors, gravity_system).in_set(SubstepSet::CalcEffects),
    );
    schedule.add_systems((integrate_att, integrate_pos).in_set(SubstepSet::Integrate));
    schedule.add_systems((distance_system, revolute_system).in_set(SubstepSet::SolveConstraints));
    schedule.add_systems((update_vel, update_ang_vel).in_set(SubstepSet::UpdateVel));
    schedule.add_systems((revolute_damping).in_set(SubstepSet::DampJoints));
    schedule.add_systems((clear_effects).in_set(SubstepSet::ClearEffects));
    schedule.add_systems((update_time).in_set(SubstepSet::UpdateTime));
    schedule
}
