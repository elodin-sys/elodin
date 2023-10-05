use bevy::prelude::*;
use bevy_ecs::schedule::{ScheduleLabel, SystemSet};

use crate::{history::HistoryPlugin, Att, Pos};

use super::{
    components::{Config, Paused, PhysicsFixedTime, TickMode},
    constraints::{
        clear_distance_lagrange, clear_revolute_lagrange, distance_system, gravity_system,
        revolute_damping, revolute_system,
    },
    systems::*,
};

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TickSet {
    ClearConstraintLagrange,
    TickPhysics,
    SyncPos,
}

#[derive(ScheduleLabel, Debug, PartialEq, Eq, Hash, Clone)]
pub struct PhysicsSchedule;

fn run_physics_system(world: &mut World) {
    if world.resource::<Paused>().0 {
        return;
    }
    let delta_time = world.resource::<Time>().delta();
    let mut tick_mode = world.resource_mut::<TickMode>();
    match tick_mode.as_mut() {
        TickMode::FreeRun => {
            world.run_schedule(PhysicsSchedule);
        }
        TickMode::Lockstep(l) => {
            if l.can_continue() {
                world.run_schedule(PhysicsSchedule);
            }
        }
        TickMode::Fixed => {
            let mut fixed_time = world.resource_mut::<PhysicsFixedTime>();
            fixed_time.0.tick(delta_time);
            let _ = world.try_schedule_scope(PhysicsSchedule, |world, schedule| {
                while world.resource_mut::<PhysicsFixedTime>().0.expend().is_ok() {
                    schedule.run(world);
                }
            });
        }
    }
}

#[derive(Default)]
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Paused(false));
        app.add_plugins(HistoryPlugin);
        app.add_systems(Update, run_physics_system);
        app.add_systems(
            PhysicsSchedule,
            (clear_distance_lagrange, clear_revolute_lagrange)
                .in_set(TickSet::ClearConstraintLagrange),
        );
        app.add_systems(PhysicsSchedule, (tick).in_set(TickSet::TickPhysics));
        app.add_systems(PhysicsSchedule, sync_pos.in_set(TickSet::SyncPos))
            .configure_sets(
                Update,
                (
                    TickSet::ClearConstraintLagrange,
                    TickSet::TickPhysics,
                    TickSet::SyncPos,
                )
                    .chain(),
            )
            .add_schedule(SubstepSchedule, substep_schedule());
    }
}

pub fn tick(world: &mut World) {
    let config = world.get_resource::<Config>().expect("missing config");
    for _ in 0..config.substep_count {
        world.run_schedule(SubstepSchedule)
    }
}

pub fn sync_pos(mut query: Query<(&mut Transform, &Pos, &Att)>, config: Res<Config>) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut transform, Pos(pos), Att(att))| {
            transform.translation =
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) * config.scale;
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
