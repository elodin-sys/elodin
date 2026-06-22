#![allow(clippy::too_many_arguments)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use s10::{
    CargoRecipe, GroupRecipe, ProcessArgs, ProcessRecipe, ReadyProbe, Recipe as RustRecipe,
    SimRecipe,
};
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
        depends_on: Vec<String>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
    },
    Process {
        name: String,
        cmd: String,
        args: Vec<String>,
        cwd: Option<String>,
        env: HashMap<String, String>,
        restart_policy: RestartPolicy,
        no_watch: bool,
        depends_on: Vec<String>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
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
        env: HashMap<String, String>,
        depends_on: Vec<String>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
    },
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartPolicy {
    Never,
    Instant,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ready {
    inner: ReadyProbe,
}

#[pymethods]
impl Ready {
    #[staticmethod]
    fn tcp(addr: String) -> Self {
        // Stored verbatim so `${VAR:-default}` placeholders survive to spawn;
        // the address is parsed when the readiness probe runs.
        Self {
            inner: ReadyProbe::Tcp { addr },
        }
    }

    #[staticmethod]
    fn unix(path: String) -> Self {
        Self {
            inner: ReadyProbe::Unix {
                path: PathBuf::from(path),
            },
        }
    }

    #[staticmethod]
    fn file(path: String) -> Self {
        Self {
            inner: ReadyProbe::File {
                path: PathBuf::from(path),
            },
        }
    }

    #[staticmethod]
    fn log(pattern: String) -> Self {
        Self {
            inner: ReadyProbe::Log { pattern },
        }
    }

    #[staticmethod]
    fn delay(ms: u64) -> Self {
        Self {
            inner: ReadyProbe::Delay { ms },
        }
    }
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
                depends_on,
                ready,
                ready_timeout,
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
                    fail_on_error: false,
                    log_path: None,
                    depends_on: depends_on.clone(),
                    ready: ready.as_ref().map(|ready| ready.inner.clone()),
                    ready_timeout: ready_timeout.clone(),
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
                depends_on,
                ready,
                ready_timeout,
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
                    fail_on_error: false,
                    log_path: None,
                    depends_on: depends_on.clone(),
                    ready: ready.as_ref().map(|ready| ready.inner.clone()),
                    ready_timeout: ready_timeout.clone(),
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
                env,
                depends_on,
                ready,
                ready_timeout,
                ..
            } => Ok(RustRecipe::Sim(SimRecipe {
                path: path.clone(),
                addr: addr.parse()?,
                optimize: *optimize,
                env: env.clone(),
                log_path: None,
                depends_on: depends_on.clone(),
                ready: ready.as_ref().map(|ready| ready.inner.clone()),
                ready_timeout: ready_timeout.clone(),
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
    #[pyo3(signature = (name, path=None, addr=None, optimize=None, env=None, depends_on=None, ready=None, ready_timeout=None))]
    fn new(
        name: String,
        path: Option<String>,
        addr: Option<String>,
        optimize: Option<bool>,
        env: Option<HashMap<String, String>>,
        depends_on: Option<Vec<String>>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
    ) -> PyResult<Self> {
        let path = path.map(PathBuf::from).unwrap_or_default();
        let addr = addr.unwrap_or_else(|| "[::]:2240".to_string());
        let optimize = optimize.unwrap_or(false);

        let inner = Recipe::Sim {
            name,
            path,
            addr,
            optimize,
            env: env.unwrap_or_default(),
            depends_on: depends_on.unwrap_or_default(),
            ready,
            ready_timeout,
        };

        Ok(PyRecipe { inner })
    }

    /// Create a Cargo recipe to build and run a Rust project alongside the simulation.
    ///
    /// Args:
    ///     name: Display name for the recipe (used in logs)
    ///     path: Path to Cargo.toml or directory containing it
    ///     package: Package name (if workspace has multiple packages)
    ///     bin: Binary name (if package has multiple binaries)
    ///     args: Command-line arguments to pass to the binary
    ///     cwd: Working directory for the process
    #[staticmethod]
    #[pyo3(signature = (name, path, package=None, bin=None, args=None, cwd=None, env=None, restart_policy=None, depends_on=None, ready=None, ready_timeout=None))]
    fn cargo(
        name: String,
        path: String,
        package: Option<String>,
        bin: Option<String>,
        args: Option<Vec<String>>,
        cwd: Option<String>,
        env: Option<HashMap<String, String>>,
        restart_policy: Option<RestartPolicy>,
        depends_on: Option<Vec<String>>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
    ) -> PyResult<Self> {
        let inner = Recipe::Cargo {
            name,
            path: PathBuf::from(path),
            package,
            bin,
            features: vec![],
            args: args.unwrap_or_default(),
            cwd,
            env: env.unwrap_or_default(),
            restart_policy: restart_policy.unwrap_or(RestartPolicy::Never),
            depends_on: depends_on.unwrap_or_default(),
            ready,
            ready_timeout,
        };
        Ok(PyRecipe { inner })
    }

    /// Create a Process recipe to run an arbitrary command alongside the simulation.
    ///
    /// Args:
    ///     name: Display name for the recipe (used in logs)
    ///     cmd: Command to execute (path to executable or command name)
    ///     args: Command-line arguments to pass to the process
    ///     cwd: Working directory for the process
    #[staticmethod]
    #[pyo3(signature = (name, cmd, args=None, cwd=None, env=None, restart_policy=None, depends_on=None, ready=None, ready_timeout=None))]
    fn process(
        name: String,
        cmd: String,
        args: Option<Vec<String>>,
        cwd: Option<String>,
        env: Option<HashMap<String, String>>,
        restart_policy: Option<RestartPolicy>,
        depends_on: Option<Vec<String>>,
        ready: Option<Ready>,
        ready_timeout: Option<String>,
    ) -> PyResult<Self> {
        let inner = Recipe::Process {
            name,
            cmd,
            args: args.unwrap_or_default(),
            cwd,
            env: env.unwrap_or_default(),
            restart_policy: restart_policy.unwrap_or(RestartPolicy::Never),
            no_watch: true,
            depends_on: depends_on.unwrap_or_default(),
            ready,
            ready_timeout,
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
    child.add_class::<Ready>()?;
    parent_module.add_submodule(&child)
}
