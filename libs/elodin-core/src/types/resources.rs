use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use bevy::prelude::{Deref, DerefMut, Resource};
use bevy_utils::Duration;
use nalgebra::Vector3;

use crate::spatial::SpatialMotion;

#[derive(Debug, Default)]
pub struct PhysFixed {
    pub(crate) timestep: Duration,
    pub(crate) overstep: Duration,
}

#[derive(Resource, Debug)]
pub struct PhysicsFixedTime(pub bevy::time::Time<PhysFixed>);

impl PhysicsFixedTime {
    pub fn accumulate(&mut self, delta: Duration) {
        self.0.context_mut().overstep += delta;
    }

    pub fn expend(&mut self) -> bool {
        let timestep = self.0.context().timestep;
        if let Some(new_value) = self.0.context_mut().overstep.checked_sub(timestep) {
            // reduce accumulated and increase elapsed by period
            self.0.context_mut().overstep = new_value;
            self.0.advance_by(timestep);
            true
        } else {
            // no more periods left in accumulated
            false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Resource)]
pub struct Time(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, Resource, Deref, DerefMut)]
pub struct Paused(pub bool);

#[derive(Debug, Resource)]
pub enum TickMode {
    FreeRun,
    Fixed,
    Lockstep(LockStepSignal),
}

#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct Config {
    pub dt: f64,
    pub sub_dt: f64,
    pub substep_count: usize,
    pub scale: f32,
    pub global_gravity: SpatialMotion,
}

impl Default for Config {
    fn default() -> Self {
        let dt = 1.0 / 60.0;
        let substep_count = 24;
        Self {
            dt,
            sub_dt: dt / substep_count as f64,
            substep_count,
            scale: 1.0,
            global_gravity: SpatialMotion::linear(Vector3::new(0., 9.81, 0.)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LockStepSignal(Arc<AtomicBool>);

impl LockStepSignal {
    pub fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn can_continue(&self) -> bool {
        self.0.swap(false, Ordering::Acquire)
    }
}
