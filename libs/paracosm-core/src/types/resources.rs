use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use bevy::prelude::{FixedTime, Resource};

#[derive(Debug, Resource)]
pub struct PhysicsFixedTime(pub FixedTime);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Resource)]
pub struct Time(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, Resource)]
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
