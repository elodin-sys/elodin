use super::{
    builder::{Env, SimFunc},
    components::{Config, LockStepSignal},
    plugin::XpbdPlugin,
};
use bevy::{app::Plugins, prelude::App};
use bevy_ecs::system::CommandQueue;
use std::{
    cell::RefCell,
    marker::PhantomData,
    time::{Duration, Instant},
};

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
            run_mode: RunMode::Default,
        }
    }

    pub fn delta_t(mut self, dt: f64) -> Self {
        self.config.dt = dt;
        self.config.sub_dt = dt / self.config.sub_dt;
        self
    }

    pub fn substep_count(mut self, count: usize) -> Self {
        self.config.substep_count = count;
        self.config.sub_dt = self.config.dt / self.config.sub_dt;
        self
    }

    pub fn lockstep(mut self, lockstep: impl Into<Option<LockStepSignal>>) -> Self {
        self.lockstep = lockstep.into();
        self
    }

    pub fn run_mode(mut self, mode: RunMode) -> Self {
        self.run_mode = mode;
        self
    }

    pub fn build(self) -> App {
        self.build_with_plugins(())
    }

    pub fn build_with_plugins<M>(mut self, plugins: impl Plugins<M>) -> App {
        let mut app = App::new();
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
                    for i in 0..n {
                        println!("update {i:?}");
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
                app.set_runner(move |mut app| {
                    let start = Instant::now();
                    app.update();
                    std::thread::sleep(duration - start.elapsed())
                });
            }
            RunMode::Default => {}
        }
        app.insert_resource(crate::Time(0.0))
            .insert_resource(self.config);

        app.add_plugins(plugins);
        app.add_plugins(XpbdPlugin {
            lockstep: self.lockstep,
        });
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

struct ConcreteSimFunc<F, T> {
    func: F,
    _phantom_data: PhantomData<T>,
}

impl<E, T> ConcreteSimFunc<E, T> {
    pub(crate) fn new(func: E) -> Self {
        Self {
            func,
            _phantom_data: PhantomData,
        }
    }
}
impl<F, T> SimFunc<(), SimRunnerEnv> for ConcreteSimFunc<F, T>
where
    F: for<'s> SimFunc<T, SimRunnerEnv>,
{
    fn build(&mut self, env: &mut SimRunnerEnv) {
        self.func.build(env)
    }
}

pub enum RunMode {
    FixedTicks(usize),
    FixedTime(f64),
    OneShot,
    RealTime,
    Default,
}
