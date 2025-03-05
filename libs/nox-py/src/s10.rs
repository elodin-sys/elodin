#![allow(clippy::too_many_arguments)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use s10::{CargoRecipe, GroupRecipe, ProcessArgs, ProcessRecipe, Recipe as RustRecipe, SimRecipe};
use std::collections::HashMap;
use std::net::AddrParseError;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
#[pyclass(eq)]
pub enum Recipe {
    Cargo {
        name: String,
        path: PathBuf,
        package: Option<String>,
        bin: Option<String>,
        features: Vec<String>,
        args: Vec<String>,
        cwd: Option<String>,
        env: HashMap<String, String>,
        restart_policy: RestartPolicy,
    },
    Process {
        name: String,
        cmd: String,
        args: Vec<String>,
        cwd: Option<String>,
        env: HashMap<String, String>,
        restart_policy: RestartPolicy,
        no_watch: bool,
    },
    Group {
        name: String,
        recipes: Vec<Recipe>,
    },
    Sim {
        name: String,
        path: PathBuf,
        addr: String,
        optimize: bool,
    },
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartPolicy {
    Never,
    Instant,
}

impl Recipe {
    pub fn name(&self) -> String {
        match self {
            Recipe::Cargo { name, .. } => name.clone(),
            Recipe::Process { name, .. } => name.clone(),
            Recipe::Group { name, .. } => name.clone(),
            Recipe::Sim { name, .. } => name.clone(),
        }
    }

    pub fn to_rust(&self) -> Result<RustRecipe, AddrParseError> {
        match self {
            Recipe::Cargo {
                path,
                package,
                bin,
                features,
                args,
                cwd,
                env,
                restart_policy,
                ..
            } => Ok(RustRecipe::Cargo(CargoRecipe {
                path: path.clone(),
                package: package.clone(),
                bin: bin.clone(),
                features: features.clone(),
                process_args: ProcessArgs {
                    args: args.clone(),
                    cwd: cwd.clone(),
                    env: env.clone(),
                    restart_policy: match restart_policy {
                        RestartPolicy::Never => s10::RestartPolicy::Never,
                        RestartPolicy::Instant => s10::RestartPolicy::Instant,
                    },
                },
                destination: s10::Destination::Local,
            })),
            Recipe::Process {
                cmd,
                args,
                cwd,
                env,
                restart_policy,
                no_watch,
                ..
            } => Ok(RustRecipe::Process(ProcessRecipe {
                cmd: cmd.clone(),
                process_args: ProcessArgs {
                    args: args.clone(),
                    cwd: cwd.clone(),
                    env: env.clone(),
                    restart_policy: match restart_policy {
                        RestartPolicy::Never => s10::RestartPolicy::Never,
                        RestartPolicy::Instant => s10::RestartPolicy::Instant,
                    },
                },
                no_watch: *no_watch,
            })),
            Recipe::Group { recipes, .. } => Ok(RustRecipe::Group(GroupRecipe {
                refs: vec![],
                recipes: recipes
                    .iter()
                    .map(|recipe| Ok((recipe.name(), recipe.to_rust()?)))
                    .collect::<Result<HashMap<_, _>, _>>()?,
            })),
            Recipe::Sim {
                path,
                addr,
                optimize,
                ..
            } => Ok(RustRecipe::Sim(SimRecipe {
                path: path.clone(),
                addr: addr.parse()?,
                optimize: *optimize,
            })),
        }
    }
}

// We can't implement pymethods for Recipe directly since it doesn't implement PyClass
// Instead, we'll create a wrapper type that can be used from Python
#[pyclass]
#[derive(Clone)]
pub struct PyRecipe {
    inner: Recipe,
}

#[pymethods]
impl PyRecipe {
    #[new]
    #[pyo3(signature = (name, path=None, addr=None, optimize=None))]
    fn new(
        name: String,
        path: Option<String>,
        addr: Option<String>,
        optimize: Option<bool>,
    ) -> PyResult<Self> {
        let path = path.map(PathBuf::from).unwrap_or_default();
        let addr = addr.unwrap_or_else(|| "[::]:2240".to_string());
        let optimize = optimize.unwrap_or(false);

        let inner = Recipe::Sim {
            name,
            path,
            addr,
            optimize,
        };

        Ok(PyRecipe { inner })
    }

    pub fn to_json(&self) -> PyResult<String> {
        let recipe = self.inner.to_rust()?;
        serde_json::to_string(&recipe).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub fn name(&self) -> String {
        self.inner.name()
    }
}

pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child = PyModule::new(parent_module.py(), "s10")?;
    // Use our PyRecipe wrapper instead of the original Recipe
    child.add_class::<PyRecipe>()?;
    child.add_class::<RestartPolicy>()?;
    parent_module.add_submodule(&child)
}
