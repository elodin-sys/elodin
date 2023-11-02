use super::{
    builder::{ConcreteSimFunc, Env, SimFunc},
    plugin::XpbdPlugin,
    types::{Config, LockStepSignal, PhysicsFixedTime, TickMode},
};
use bevy::{
    app::Plugins,
    prelude::{App, FixedTime},
};
use bevy_ecs::system::CommandQueue;
use std::{cell::RefCell, time::Duration};

pub struct SimRunner<'a> {
    sim_func: Box<dyn SimFunc<(), SimRunnerEnv> + 'a>,
    config: Config,
    lockstep: Option<LockStepSignal>,
    run_mode: RunMode,
}

impl<'a> SimRunner<'a> {
    pub fn new<T: 'a, F: SimFunc<T, SimRunnerEnv> + 'a>(sim_func: F) -> Self {
        SimRunner {
            sim_func: Box::new(ConcreteSimFunc::new(sim_func)),
            config: Config::default(),
            lockstep: None,
            run_mode: RunMode::RealTime,
        }
    }

    pub fn delta_t(mut self, dt: f64) -> Self {
        self.config.dt = dt;
        self.config.sub_dt = dt / self.config.substep_count as f64;
        self
    }

    pub fn substep_count(mut self, count: usize) -> Self {
        self.config.substep_count = count;
        self.config.sub_dt = self.config.dt / count as f64;
        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.config.scale = scale;
        self
    }

    pub fn run_mode(mut self, mode: RunMode) -> Self {
        self.run_mode = mode;
        self
    }

    pub fn lockstep(mut self, lockstep: impl Into<Option<LockStepSignal>>) -> Self {
        self.lockstep = lockstep.into();
        self
    }

    pub fn build(self) -> App {
        self.build_with_plugins(())
    }

    fn tick_mode(&mut self) -> TickMode {
        match self.run_mode {
            RunMode::FixedTicks(_)
            | RunMode::FixedTime(_)
            | RunMode::OneShot
            | RunMode::FreeRun => {
                if let Some(lockstep) = self.lockstep.take() {
                    TickMode::Lockstep(lockstep)
                } else {
                    TickMode::FreeRun
                }
            }
            RunMode::RealTime | RunMode::Scaled(_) => TickMode::Fixed,
        }
    }

    pub fn build_with_plugins<M>(mut self, plugins: impl Plugins<M>) -> App {
        let mut app = App::new();
        app.insert_resource(self.tick_mode());
        match self.run_mode {
            RunMode::FixedTicks(n) => {
                app.set_runner(move |mut app| {
                    for _ in 0..n {
                        app.update();
                    }
                });
            }
            RunMode::FixedTime(time) => {
                let n: usize = (time / self.config.dt) as usize;
                app.set_runner(move |mut app| {
                    for _ in 0..n {
                        app.update();
                    }
                });
            }
            RunMode::OneShot => {
                app.set_runner(|mut app| {
                    app.update();
                });
            }
            RunMode::RealTime => {
                let duration = Duration::from_secs_f64(self.config.dt);
                app.insert_resource(PhysicsFixedTime(FixedTime::new(duration)));
            }
            RunMode::Scaled(scale) => {
                let duration = Duration::from_secs_f64(self.config.dt / scale);
                app.insert_resource(PhysicsFixedTime(FixedTime::new(duration)));
            }

            RunMode::FreeRun => {}
        }
        app.insert_resource(bevy::time::Time::default());
        app.insert_resource(crate::Time(0.0))
            .insert_resource(self.config);
        app.add_plugins(XpbdPlugin);
        app.add_plugins(plugins);
        let mut env = SimRunnerEnv::new(app);
        self.sim_func.build(&mut env);
        let SimRunnerEnv {
            mut app,
            command_queue,
        } = env;
        let mut command_queue = command_queue.into_inner();
        command_queue.apply(&mut app.world);
        app
    }
}

pub trait IntoSimRunner<'a, T>: Sized {
    fn into_runner(self) -> SimRunner<'a>;

    fn delta_t(self, dt: f64) -> SimRunner<'a>;
    fn substep_count(self, count: usize) -> SimRunner<'a>;
    fn lockstep(self, lockstep: impl Into<Option<LockStepSignal>>) -> SimRunner<'a>;
    fn run_mode(self, mode: RunMode) -> SimRunner<'a>;
    fn scale(self, scale: f32) -> SimRunner<'a>;

    fn build_app(self) -> App;
}

impl<'a, T, F> IntoSimRunner<'a, T> for F
where
    T: 'a,
    F: SimFunc<T, SimRunnerEnv> + 'a,
{
    fn into_runner(self) -> SimRunner<'a> {
        SimRunner::new(self)
    }

    fn delta_t(self, dt: f64) -> SimRunner<'a> {
        self.into_runner().delta_t(dt)
    }

    fn substep_count(self, count: usize) -> SimRunner<'a> {
        self.into_runner().substep_count(count)
    }

    fn lockstep(self, lockstep: impl Into<Option<LockStepSignal>>) -> SimRunner<'a> {
        self.into_runner().lockstep(lockstep)
    }

    fn run_mode(self, mode: RunMode) -> SimRunner<'a> {
        self.into_runner().run_mode(mode)
    }

    fn build_app(self) -> App {
        self.into_runner().build()
    }

    fn scale(self, scale: f32) -> SimRunner<'a> {
        self.into_runner().scale(scale)
    }
}

impl<'a> IntoSimRunner<'a, ()> for SimRunner<'a> {
    fn into_runner(self) -> SimRunner<'a> {
        self
    }

    fn delta_t(self, dt: f64) -> SimRunner<'a> {
        SimRunner::delta_t(self, dt)
    }

    fn substep_count(self, count: usize) -> SimRunner<'a> {
        SimRunner::substep_count(self, count)
    }

    fn lockstep(self, lockstep: impl Into<Option<LockStepSignal>>) -> SimRunner<'a> {
        SimRunner::lockstep(self, lockstep)
    }

    fn run_mode(self, mode: RunMode) -> SimRunner<'a> {
        SimRunner::run_mode(self, mode)
    }

    fn build_app(self) -> App {
        SimRunner::build(self)
    }

    fn scale(self, scale: f32) -> SimRunner<'a> {
        self.scale(scale)
    }
}

pub struct SimRunnerEnv {
    pub app: App,
    pub command_queue: RefCell<CommandQueue>,
}

impl SimRunnerEnv {
    fn new(app: App) -> Self {
        Self {
            app,
            command_queue: Default::default(),
        }
    }
}

impl Env for SimRunnerEnv {
    type Param<'e> = &'e SimRunnerEnv;

    fn param(&mut self) -> Self::Param<'_> {
        self
    }
}

pub enum RunMode {
    FixedTicks(usize),
    FixedTime(f64),
    OneShot,
    FreeRun,
    RealTime,
    Scaled(f64),
}
