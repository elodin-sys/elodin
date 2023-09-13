use std::{
    any::{type_name, Any},
    cell::RefCell,
    collections::HashMap,
    io,
    sync::Arc,
};

use bevy::reflect::TypePath;
use clap::Parser;
use rand::rngs::ThreadRng;
use rand_distr::Distribution as _;
use serde::{Deserialize, Serialize};

use crate::{
    runtime::JobSpec,
    xpbd::builder::{ConcreteSimFunc, Env, FromEnv, SimFunc},
};

#[derive(Default)]
pub struct MonteCarlo {
    job: Option<Arc<dyn SimFunc<(), MonteCarloEnv, JobSpec> + Sync + Send>>,
    spec: MonteCarloSpec,
}

#[derive(Deserialize, Serialize)]
pub struct MonteCarloSpec {
    vars: HashMap<String, DistributionSpec>,
    outputs: HashMap<String, OutputSpec>,
    count: usize,
    parallelism: usize,
}

#[derive(Deserialize, Serialize)]
enum OutputSpec {
    TimeSeries,
    AllValues,
    MaxValue,
    MinValue,
    Mean,
}

impl Default for MonteCarloSpec {
    fn default() -> Self {
        Self {
            vars: Default::default(),
            count: 100,
            parallelism: 4,
            outputs: Default::default(),
        }
    }
}

impl MonteCarlo {
    pub fn var<T: RandVar>(mut self, dist: DistributionSpec) -> Self {
        self.spec.vars.insert(T::type_path().to_string(), dist);
        self
    }

    pub fn job<T: Send + Sync + 'static>(
        mut self,
        job: impl SimFunc<T, MonteCarloEnv, JobSpec> + Sync + Send + 'static,
    ) -> Self {
        self.job = Some(Arc::new(ConcreteSimFunc::new(job)));
        self
    }

    pub fn run(mut self) {
        let args = MonteCarloArgs::parse();
        match args {
            MonteCarloArgs::Run { count, parallelism } => {
                if let Some(count) = count {
                    self.spec.count = count
                }
                if let Some(parallelism) = parallelism {
                    self.spec.parallelism = parallelism;
                }
                self.run_jobs()
            }
            MonteCarloArgs::Dump => {
                serde_json::to_writer(io::stdout(), &self.spec).unwrap();
            }
        }
    }

    pub fn run_jobs(self) {
        let vars: Arc<HashMap<String, Distribution>> = Arc::new(
            self.spec
                .vars
                .into_iter()
                .map(|(k, dist)| dist.try_into().map(|dist| (k, dist)))
                .collect::<Result<_, _>>()
                .unwrap(),
        );

        let mut handles = vec![];
        for _ in 0..self.spec.parallelism {
            let vars = vars.clone();
            let job = self.job.clone();
            let n = self.spec.count / self.spec.parallelism;
            handles.push(std::thread::spawn(move || {
                let mut env = MonteCarloEnv {
                    vars,
                    rng: ThreadRng::default().into(),
                };
                for i in 0..n {
                    let job_spec = if let Some(ref job) = job {
                        job.build(&mut env)
                    } else {
                        JobSpec::default()
                    };
                    job_spec.run();
                }
            }));
        }
        for handle in handles.into_iter() {
            handle.join().unwrap();
        }
    }
}

#[derive(Parser, Debug)]
enum MonteCarloArgs {
    Run {
        #[arg(short, long)]
        count: Option<usize>,
        #[arg(short = 'j', long)]
        parallelism: Option<usize>,
    },
    Dump,
}

pub struct MonteCarloEnv {
    vars: Arc<HashMap<String, Distribution>>,
    rng: RefCell<ThreadRng>,
}

#[derive(Serialize, Deserialize)]
pub enum DistributionSpec {
    Const(f64),
    Normal(Normal),
    Uniform(Uniform),
}

pub enum Output {}

pub enum Distribution {
    Const(f64),
    Normal(rand_distr::Normal<f64>),
    Uniform(rand_distr::Uniform<f64>),
}

impl TryInto<Distribution> for DistributionSpec {
    type Error = rand_distr::NormalError;

    fn try_into(self) -> Result<Distribution, Self::Error> {
        match self {
            DistributionSpec::Const(c) => Ok(Distribution::Const(c)),
            DistributionSpec::Normal(n) => {
                rand_distr::Normal::new(n.mean, n.std_dev).map(Distribution::Normal)
            }
            DistributionSpec::Uniform(u) => Ok(Distribution::Uniform(rand_distr::Uniform::new(
                u.start, u.end,
            ))),
        }
    }
}

impl Distribution {
    fn sample(&self, rng: &mut ThreadRng) -> f64 {
        match self {
            Distribution::Const(c) => *c,
            Distribution::Normal(n) => n.sample(rng),
            Distribution::Uniform(u) => u.sample(rng),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Uniform {
    start: f64,
    end: f64,
}

pub trait RandVar: Any + Sized + TypePath {
    fn from_float(float: f64) -> Self;
}

macro_rules! impl_prim_rand_var {
    ($prim: ty) => {
        impl RandVar for $prim {
            fn from_float(float: f64) -> Self {
                float as $prim
            }
        }
    };
}

impl RandVar for f64 {
    fn from_float(float: f64) -> Self {
        float
    }
}
impl_prim_rand_var!(f32);
impl_prim_rand_var!(u64);
impl_prim_rand_var!(i64);
impl_prim_rand_var!(u32);
impl_prim_rand_var!(i32);
impl_prim_rand_var!(u16);
impl_prim_rand_var!(i16);
impl_prim_rand_var!(u8);
impl_prim_rand_var!(i8);

impl<T: RandVar> FromEnv<MonteCarloEnv> for T {
    type Item<'a> = T where MonteCarloEnv: 'a;

    fn init(_env: &mut MonteCarloEnv) {}

    fn from_env(env: <MonteCarloEnv as crate::xpbd::builder::Env>::Param<'_>) -> Self::Item<'_> {
        let Some(dist) = env.vars.get(T::type_path()) else {
            panic!("no monte carlo var for type {}", type_name::<T>())
        };
        let f = dist.sample(&mut env.rng.borrow_mut());
        T::from_float(f)
    }
}

impl Env for MonteCarloEnv {
    type Param<'a> = &'a Self where Self: 'a;

    fn param(&mut self) -> Self::Param<'_> {
        self
    }
}

#[cfg(test)]
mod tests {}
