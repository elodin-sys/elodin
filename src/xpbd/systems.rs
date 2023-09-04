use crate::Time;
use bevy_ecs::{
    query::WorldQuery,
    system::{Query, Res, ResMut},
};

use super::{body, components::*};

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

pub(crate) fn calculate_effects(mut query: Query<EffectQuery>, time: Res<Time>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        for effector in &q.effectors.0 {
            *q.effect += effector.effect(*time, EntityStateRef::from_query(&q.entity))
        }
    })
}

pub(crate) fn calculate_sensors(mut query: Query<SensorQuery>, time: Res<Time>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        for effector in &mut q.sensors.0 {
            effector.sense(*time, EntityStateRef::from_query(&q.entity));
        }
    })
}

pub(crate) fn clear_effects(mut query: Query<&mut Effect>) {
    query.par_iter_mut().for_each_mut(|mut q| {
        *q = Effect::default();
    });
}

pub(crate) fn update_time(mut time: ResMut<Time>, config: Res<Config>) {
    time.0 += config.sub_dt;
}

pub(crate) fn integrate_pos(
    mut query: Query<(
        &mut Pos,
        &mut PrevPos,
        &mut Vel,
        &mut Effect,
        &mut Mass,
        &Fixed,
    )>,
    config: Res<Config>,
) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut pos, mut prev_pos, mut vel, effect, mass, fixed)| {
            if fixed.0 {
                return;
            }
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

pub(crate) fn integrate_att(
    mut query: Query<(
        &mut Att,
        &mut PrevAtt,
        &mut AngVel,
        &mut Effect,
        &mut Inertia,
        &mut InverseInertia,
        &Fixed,
    )>,
    config: Res<Config>,
) {
    query.par_iter_mut().for_each_mut(
        |(mut att, mut prev_att, mut ang_vel, effect, inertia, inverse_inertia, fixed)| {
            if fixed.0 {
                return;
            }
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

pub(crate) fn update_vel(
    mut query: Query<(&Pos, &PrevPos, &mut Vel, &Fixed)>,
    config: Res<Config>,
) {
    query
        .par_iter_mut()
        .for_each_mut(|(pos, prev_pos, mut vel, fixed)| {
            if fixed.0 {
                return;
            }
            vel.0 = body::calc_vel(pos.0, prev_pos.0, config.sub_dt);
        })
}

pub(crate) fn update_ang_vel(
    mut query: Query<(&Att, &PrevAtt, &mut AngVel, &Fixed)>,
    config: Res<Config>,
) {
    query
        .par_iter_mut()
        .for_each_mut(|(att, prev_att, mut ang_vel, fixed)| {
            if fixed.0 {
                return;
            }

            ang_vel.0 = body::calc_ang_vel(att.0, prev_att.0, config.sub_dt);
        })
}
