use crate::Time;
use bevy_ecs::{
    query::WorldQuery,
    system::{Query, Res, ResMut},
};
use bevy_utils::tracing;

use super::types::*;

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
    joint_pos: &'static mut JointPos,
    joint_vel: &'static mut JointVel,
    effect: &'static mut Effect,
    mass: &'static mut Mass,
    inertia: &'static mut Inertia,
    inverse_inertia: &'static mut InverseInertia,
    fixed: &'static Fixed,
    joint_accel: &'static JointAccel,
}

#[tracing::instrument]
pub(crate) fn integrate_pos(mut query: Query<IntQuery>, config: Res<Config>) {
    query.iter_mut().for_each(|mut query| {
        if query.fixed.0 {
            return;
        }
        query
            .joint_vel
            .0
            .integrate(&query.joint_accel.0, config.sub_dt);
        query
            .joint_pos
            .0
            .integrate(&query.joint_vel.0, config.sub_dt);
    })
}
