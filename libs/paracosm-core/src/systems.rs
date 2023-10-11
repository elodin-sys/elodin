use crate::Time;
use bevy_ecs::{
    query::WorldQuery,
    system::{Query, Res, ResMut},
};
use bevy_utils::tracing;

use super::{body, types::*};

#[derive(WorldQuery)]
#[world_query(mutable)]
pub(crate) struct EffectQuery {
    effect: &'static mut Effect,
    effectors: &'static Effectors,
    entity: EntityQueryReadOnly,
}

#[derive(WorldQuery)]
#[world_query(mutable)]
pub(crate) struct SensorQuery {
    sensors: &'static mut Sensors,
    entity: EntityQueryReadOnly,
}

#[tracing::instrument]
pub(crate) fn calculate_effects(mut query: Query<EffectQuery>, time: Res<Time>) {
    query.iter_mut().for_each(|mut q| {
        for effector in &q.effectors.0 {
            *q.effect += effector.effect(*time, EntityStateRef::from_query(&q.entity))
        }
    })
}

#[tracing::instrument]
pub(crate) fn calculate_sensors(mut query: Query<SensorQuery>, time: Res<Time>) {
    query.iter_mut().for_each(|mut q| {
        for effector in &mut q.sensors.0 {
            effector.sense(*time, EntityStateRef::from_query(&q.entity));
        }
    })
}

#[tracing::instrument]
pub(crate) fn clear_effects(mut query: Query<&mut Effect>) {
    query.iter_mut().for_each(|mut q| {
        *q = Effect::default();
    });
}

#[tracing::instrument]
pub(crate) fn update_time(mut time: ResMut<Time>, config: Res<Config>) {
    time.0 += config.sub_dt;
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct IntQuery {
    body_pos: &'static mut BodyPos,
    prev_pos: &'static mut PrevPos,
    prev_att: &'static mut PrevAtt,
    body_vel: &'static mut BodyVel,
    effect: &'static mut Effect,
    mass: &'static mut Mass,
    inertia: &'static mut Inertia,
    inverse_inertia: &'static mut InverseInertia,
    fixed: &'static Fixed,
}

#[tracing::instrument]
pub(crate) fn integrate_pos(mut query: Query<IntQuery>, config: Res<Config>) {
    query.iter_mut().for_each(|mut query| {
        if query.fixed.0 {
            return;
        }
        body::integrate_pos(
            &mut query.body_pos.0.pos,
            &mut query.prev_pos.0,
            &mut query.body_vel.0.vel,
            query.effect.force.0,
            query.mass.0,
            config.sub_dt,
        );

        body::integrate_att(
            &mut query.body_pos.0.att,
            &mut query.prev_att.0,
            &mut query.body_vel.0.vel,
            query.inertia.0,
            query.inverse_inertia.0,
            query.effect.torque.0,
            config.sub_dt,
        );
    })
}

#[tracing::instrument]
pub(crate) fn update_vel(
    mut query: Query<(&BodyPos, &PrevPos, &mut BodyVel, &Fixed, &PrevAtt)>,
    config: Res<Config>,
) {
    query
        .iter_mut()
        .for_each(|(pos, prev_pos, mut vel, fixed, prev_att)| {
            if fixed.0 {
                return;
            }
            vel.0.vel = body::calc_vel(pos.0.pos, prev_pos.0, config.sub_dt);
            vel.0.ang_vel = body::calc_ang_vel(pos.0.att, prev_att.0, config.sub_dt);
        })
}
