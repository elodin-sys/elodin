use std::{collections::HashMap, panic::resume_unwind, path::PathBuf};

use bevy::prelude::App;

use crate::runner::IntoSimRunner;

#[derive(Default)]
pub struct JobSpec {
    tasks: Vec<Task>,
}

impl JobSpec {
    pub fn fn_task(mut self, func: impl FnOnce() + Send + Sync + 'static) -> Self {
        self.tasks.push(Task::RustFunc(Box::new(func)));
        self
    }

    pub fn sim<'a, T>(mut self, func: impl IntoSimRunner<'a, T>) -> Self {
        let runner = func.into_runner();
        //self.tasks.push(Task::Sim(runner.build())); // FIXME
        self
    }

    pub fn run(self) {
        let handles = self
            .tasks
            .into_iter()
            .map(|task| match task {
                Task::RustFunc(func) => std::thread::spawn(func),
                Task::Container(_) => todo!(),
                Task::Sim(mut app) => std::thread::spawn(move || app.run()),
            })
            .collect::<Vec<_>>();
        for handle in handles.into_iter() {
            if let Err(err) = handle.join() {
                resume_unwind(err)
            }
        }
    }
}

pub enum Task {
    RustFunc(Box<dyn FnOnce() + Send + Sync>),
    Container(Container),
    Sim(App),
}

pub struct Container {
    pub image_source: ImageSource,
    pub command: Option<String>,
    pub env: HashMap<String, String>,
}

pub enum ImageSource {
    Dockerfile(PathBuf),
    ImageName(String),
}

#[cfg(test)]
mod tests {
    use nalgebra::{vector, Vector3};

    use crate::{
        builder::{EntityBuilder, Free, XpbdBuilder},
        forces::gravity,
        spatial::{SpatialMotion, SpatialPos},
        types::LockStepSignal,
    };

    use super::*;

    #[test]
    fn test_basic_lockstep_sim_run() {
        fn sim(mut builder: XpbdBuilder<'_>) {
            builder.entity(
                EntityBuilder::default()
                    .mass(1.0)
                    .joint(
                        Free::default()
                            .pos(SpatialPos::linear(vector![0.0, 0.0, 1.0]))
                            .vel(SpatialMotion::linear(vector![1.0, 0.0, 0.0])),
                    )
                    .effector(gravity(1.0 / 6.649e-11, Vector3::zeros())),
            );
        }
        let lockstep = LockStepSignal::default();
        JobSpec::default()
            .sim(
                sim.lockstep(lockstep.clone())
                    .run_mode(crate::runner::RunMode::FixedTicks(100)),
            )
            .fn_task(move || {
                for _ in 0..100 {
                    lockstep.signal();
                }
            })
            .run()
    }

    #[test]
    fn test_basic_sim_run() {
        fn sim(mut builder: XpbdBuilder<'_>) {
            builder.entity(
                EntityBuilder::default()
                    .mass(1.0)
                    .joint(
                        Free::default()
                            .pos(SpatialPos::linear(vector![0.0, 0.0, 1.0]))
                            .vel(SpatialMotion::linear(vector![1.0, 0.0, 0.0])),
                    )
                    .effector(gravity(1.0 / 6.649e-11, Vector3::zeros())),
            );
        }
        JobSpec::default()
            .sim(sim.run_mode(crate::runner::RunMode::FixedTicks(1000)))
            .run()
    }
}
