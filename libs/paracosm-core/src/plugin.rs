use bevy::prelude::*;
use bevy_ecs::schedule::{ScheduleLabel, SystemSet};
use nalgebra::DMatrix;

use crate::{
    hierarchy::TopologicalSortPlugin,
    history::HistoryPlugin,
    tree::{com_system, cri_system, forward_dynamics, rne_system},
    TreeMassMatrix, WorldPos,
};

use super::{
    constraints::gravity_system,
    systems::*,
    tree::kinematic_system,
    types::{Config, Paused, PhysicsFixedTime, TickMode},
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
        app.insert_resource(TreeMassMatrix(DMatrix::zeros(6, 6))); // FIXME
        app.add_plugins(crate::bevy_transform::TransformPlugin);
        app.add_plugins(HistoryPlugin);
        app.add_plugins(TopologicalSortPlugin);
        app.add_systems(Update, run_physics_system);
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

pub fn sync_pos(mut query: Query<(&mut Transform, &WorldPos)>, config: Res<Config>) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut transform, WorldPos(pos))| {
            *transform = pos.bevy(config.scale);
        });
}

#[derive(ScheduleLabel, Debug, PartialEq, Eq, Hash, Clone)]
pub struct SubstepSchedule;

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum SubstepSet {
    ForwardKinematics,
    CoMPos,
    RecursiveNewtonEuler,
    CompositeRigidBodyInertia,
    ForwardDynamics,
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
            SubstepSet::ForwardKinematics,
            SubstepSet::CoMPos,
            SubstepSet::CalcEffects,
            SubstepSet::RecursiveNewtonEuler,
            SubstepSet::CompositeRigidBodyInertia,
            SubstepSet::ForwardDynamics,
            SubstepSet::Integrate,
            SubstepSet::UpdateVel,
            SubstepSet::ClearEffects,
            SubstepSet::UpdateTime,
        )
            .chain(),
    );
    schedule.add_systems((kinematic_system).in_set(SubstepSet::ForwardKinematics));
    schedule.add_systems((com_system).in_set(SubstepSet::CoMPos));
    schedule.add_systems(
        (calculate_effects, calculate_sensors, gravity_system).in_set(SubstepSet::CalcEffects),
    );
    schedule.add_systems((rne_system).in_set(SubstepSet::RecursiveNewtonEuler));
    schedule.add_systems((cri_system).in_set(SubstepSet::CompositeRigidBodyInertia));
    schedule.add_systems((forward_dynamics).in_set(SubstepSet::ForwardDynamics));
    schedule.add_systems((integrate_pos).in_set(SubstepSet::Integrate));
    schedule.add_systems((clear_effects).in_set(SubstepSet::ClearEffects));
    schedule.add_systems((update_time).in_set(SubstepSet::UpdateTime));
    schedule
}
