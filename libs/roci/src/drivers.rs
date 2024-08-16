use std::time::{Duration, Instant};

use tracing::warn;

use crate::{IntoSystem, System};

pub trait Driver {
    fn run(self);
}

pub trait DriverMode {
    type Input: DriverMode;
    type Output: DriverMode;
}

pub struct Hz<const HZ: usize>;

impl<const N: usize> DriverMode for Hz<N> {
    type Input = Self;
    type Output = Hz<N>;
}

pub struct Interrupt;

impl DriverMode for Interrupt {
    type Input = Self;
    type Output = Interrupt;
}

pub struct Upsample<const A: usize, const B: usize>;

impl<const A: usize, const B: usize> DriverMode for Upsample<A, B> {
    type Input = Hz<A>;
    type Output = Hz<B>;
}

pub struct OsSleepDriver<const HZ: usize, H: System> {
    system: H,
    start: Option<Instant>,
}

pub fn os_sleep_driver<const HZ: usize, P, S: IntoSystem<P>>(
    system: S,
) -> OsSleepDriver<HZ, S::System<Hz<HZ>>>
where
    S::System<Hz<HZ>>: System<Driver = Hz<HZ>>,
    <S::System<Hz<HZ>> as System>::World: Send,
{
    OsSleepDriver {
        system: system.into_system(),
        start: None,
    }
}

impl<const HZ: usize, H: System> OsSleepDriver<{ HZ }, H> {
    pub fn new(system: H) -> Self {
        Self {
            system,
            start: None,
        }
    }
}

impl<const HZ: usize, H> Driver for OsSleepDriver<{ HZ }, H>
where
    H: System<Driver = Hz<{ HZ }>>,
{
    fn run(mut self) {
        let mut world = self.system.init_world();
        loop {
            self.update(&mut world);
        }
    }
}

impl<H, const F: usize> System for OsSleepDriver<{ F }, H>
where
    H: System<Driver = Hz<F>>,
{
    type World = H::World;

    type Driver = Hz<{ F }>;

    fn update(&mut self, world: &mut Self::World) {
        let start = self.start.get_or_insert_with(Instant::now);
        let time_step = std::time::Duration::from_secs_f64(1.0 / F as f64);
        self.system.update(world);
        let elapsed = start.elapsed();
        let sleep_time = time_step.saturating_sub(elapsed);
        if sleep_time > Duration::ZERO {
            std::thread::sleep(sleep_time);
        } else {
            warn!("system::update took longer than sleep")
        }
        *start += time_step;
    }
}

pub struct LoopDriver<S: System> {
    system: S,
}

impl<S: System> Driver for LoopDriver<S> {
    fn run(mut self) {
        let mut world = self.system.init_world();
        loop {
            self.system.update(&mut world);
        }
    }
}

pub fn loop_driver(handler: impl System) -> LoopDriver<impl System> {
    LoopDriver { system: handler }
}
