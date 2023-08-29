use crate::Time;
use bevy_ecs::{
    query::WorldQuery,
    schedule::{IntoSystemConfigs, IntoSystemSetConfigs, Schedule, ScheduleLabel, SystemSet},
    system::{Query, Res, ResMut},
};

use super::{body, components::*, constraints::distance_system};

#[derive(ScheduleLabel, Debug, PartialEq, Eq, Hash, Clone)]
pub struct SubstepSchedule;

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
enum SubstepSet {
    CalcEffects,
    Integrate,
    SolveConstraints,
    UpdateVel,
    ClearEffects,
    UpdateTime,
}

pub fn schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.configure_sets(
        (
            SubstepSet::CalcEffects,
            SubstepSet::Integrate,
            SubstepSet::SolveConstraints,
            SubstepSet::UpdateVel,
            SubstepSet::ClearEffects,
            SubstepSet::UpdateTime,
        )
            .chain(),
    );
    schedule.add_systems((calculate_effects, calculate_sensors).in_set(SubstepSet::CalcEffects));
    schedule.add_systems((integrate_att, integrate_pos).in_set(SubstepSet::Integrate));
    schedule.add_systems((distance_system).in_set(SubstepSet::SolveConstraints));
    schedule.add_systems((update_vel, update_ang_vel).in_set(SubstepSet::UpdateVel));
    schedule.add_systems((clear_effects).in_set(SubstepSet::ClearEffects));
    schedule.add_systems((update_time).in_set(SubstepSet::UpdateTime));
    schedule
}

#[derive(WorldQuery)]
#[world_query(mutable)]
struct EffectQuery {
    effect: &'static mut Effect,
    effectors: &'static Effectors,
    entity: EntityQueryReadOnly,
}

#[derive(WorldQuery)]
#[world_query(mutable)]
struct SensorQuery {
    sensors: &'static mut Sensors,
    entity: EntityQueryReadOnly,
}

fn calculate_effects(mut query: Query<EffectQuery>, time: Res<Time>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        for effector in &q.effectors.0 {
            *q.effect += effector.effect(*time, EntityStateRef::from_query(&q.entity))
        }
    })
}

fn calculate_sensors(mut query: Query<SensorQuery>, time: Res<Time>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        for effector in &mut q.sensors.0 {
            effector.sense(*time, EntityStateRef::from_query(&q.entity));
        }
    })
}

fn clear_effects(mut query: Query<&mut Effect>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        *q = Effect::default();
    });
}

fn update_time(mut time: ResMut<Time>, config: Res<Config>) {
    time.0 += config.sub_dt;
}

fn integrate_pos(
    mut query: Query<(&mut Pos, &mut PrevPos, &mut Vel, &mut Effect, &mut Mass)>,
    config: Res<Config>,
) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut pos, mut prev_pos, mut vel, effect, mass)| {
            body::integrate_pos(
                &mut pos.0,
                &mut prev_pos.0,
                &mut vel.0,
                effect.force.0,
                mass.0,
                config.sub_dt,
            )
        })
}

fn integrate_att(
    mut query: Query<(
        &mut Att,
        &mut PrevAtt,
        &mut AngVel,
        &mut Effect,
        &mut Inertia,
        &mut InverseInertia,
    )>,
    config: Res<Config>,
) {
    query.par_iter_mut().for_each_mut(
        |(mut att, mut prev_att, mut ang_vel, effect, inertia, inverse_inertia)| {
            body::integrate_att(
                &mut att.0,
                &mut prev_att.0,
                &mut ang_vel.0,
                inertia.0,
                inverse_inertia.0,
                effect.torque.0,
                config.sub_dt,
            )
        },
    )
}

fn update_vel(mut query: Query<(&Pos, &PrevPos, &mut Vel)>, config: Res<Config>) {
    query
        .par_iter_mut()
        .for_each_mut(|(pos, prev_pos, mut vel)| {
            vel.0 = body::calc_vel(pos.0, prev_pos.0, config.sub_dt);
        })
}

fn update_ang_vel(mut query: Query<(&Att, &PrevAtt, &mut AngVel)>, config: Res<Config>) {
    query
        .par_iter_mut()
        .for_each_mut(|(att, prev_att, mut ang_vel)| {
            ang_vel.0 = body::calc_ang_vel(att.0, prev_att.0, config.sub_dt);
        })
}
