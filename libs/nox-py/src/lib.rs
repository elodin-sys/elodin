use std::str::FromStr;
use std::sync::Arc;

use impeller2::types::ComponentId;
use numpy::PyUntypedArray;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod archetype;
pub mod component;
pub mod dyn_array;
pub mod entity;
pub mod error;
pub mod exec;
pub mod globals;
pub mod graph;
pub mod impeller2_server;
pub mod integrator;
pub mod linalg;
pub mod profile;
pub mod query;
pub mod s10;
pub mod six_dof;
pub mod spatial;
pub mod step_context;
pub mod system;
pub mod ukf;
pub mod utils;
pub mod world;
pub mod world_builder;

pub use archetype::*;
pub use component::*;
pub use entity::*;
pub use error::*;
pub use exec::*;
pub use graph::*;
pub use linalg::*;
pub use query::*;
pub use spatial::*;
pub use step_context::*;
pub use system::*;
pub use world::SystemExt as WorldSystemExt;
pub use world::{
    Buffers, Column, ColumnRef, DEFAULT_TIME_STEP, Entity, IntoSystemExt, TimeStep, World,
    WorldExt, WorldMetadata,
};
pub use world_builder::*;

pub use elodin_db::ComponentSchema;
pub use elodin_macros::{Archetype, Component};
pub use impeller2;
pub use impeller2_wkt;
pub use nox;

trait PyUntypedArrayExt {
    unsafe fn buf(&self, elem_size: usize) -> &[u8];
}

impl PyUntypedArrayExt for Bound<'_, PyUntypedArray> {
    unsafe fn buf(&self, elem_size: usize) -> &[u8] {
        use numpy::PyUntypedArrayMethods;
        unsafe {
            if !self.is_c_contiguous() {
                panic!("array must be c-style contiguous")
            }
            let len = self.shape().iter().product::<usize>() * elem_size;
            let obj = &*self.as_array_ptr();
            std::slice::from_raw_parts(obj.data as *const u8, len)
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    Rk4,
    SemiImplicit,
}

impl FromStr for Integrator {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rk4" => Ok(Integrator::Rk4),
            "semi-implicit" => Ok(Integrator::SemiImplicit),
            _ => Err(Error::PyO3(PyValueError::new_err("unknown integrator"))),
        }
    }
}

impl From<Integrator> for crate::integrator::Integrator {
    fn from(integrator: Integrator) -> Self {
        match integrator {
            Integrator::Rk4 => crate::integrator::Integrator::Rk4,
            Integrator::SemiImplicit => crate::integrator::Integrator::SemiImplicit,
        }
    }
}

#[pyfunction]
#[pyo3(name = "six_dof", signature = (time_step = None, sys = None, integrator = Integrator::Rk4))]
pub fn py_six_dof(
    time_step: Option<f64>,
    sys: Option<PySystem>,
    integrator: Integrator,
) -> PySystem {
    let integrator = integrator.into();
    let sys: Arc<dyn crate::system::System<Arg = (), Ret = ()> + Send + Sync> =
        if let Some(dt) = time_step {
            if let Some(sys) = sys {
                crate::six_dof::six_dof_with_dt(|| sys, dt, integrator)
            } else {
                crate::six_dof::six_dof_with_dt(|| (), dt, integrator)
            }
        } else if let Some(sys) = sys {
            crate::six_dof::six_dof(|| sys, integrator)
        } else {
            crate::six_dof::six_dof(|| (), integrator)
        };
    PySystem { inner: sys }
}

#[pyfunction]
pub fn _get_cache_dir() -> PyResult<String> {
    let directory = directories::ProjectDirs::from("systems", "elodin", "elodin-cli")
        .ok_or_else(|| PyErr::new::<PyOSError, _>("No project directory found"))?;

    let cache_dir_string = directory
        .cache_dir()
        .as_os_str()
        .to_str()
        .ok_or_else(|| PyErr::new::<PyOSError, _>("Cannot convert OsString to String"))?
        .to_string();

    Ok(cache_dir_string)
}

#[pymodule]
pub fn elodin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ComponentType>()?;
    m.add_class::<world_builder::WorldBuilder>()?;
    m.add_class::<PyExec>()?;
    m.add_class::<EntityId>()?;
    m.add_class::<SpatialTransform>()?;
    m.add_class::<SpatialForce>()?;
    m.add_class::<SpatialMotion>()?;
    m.add_class::<SpatialInertia>()?;
    m.add_class::<Quaternion>()?;
    m.add_class::<PrimitiveType>()?;
    m.add_class::<QueryInner>()?;
    m.add_class::<GraphQueryInner>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyComponent>()?;
    m.add_class::<Integrator>()?;
    m.add_class::<PyFnSystem>()?;
    m.add_class::<QueryMetadata>()?;
    m.add_class::<PySystemBuilder>()?;
    m.add_class::<PySystem>()?;
    m.add_class::<StepContext>()?;
    m.add_function(wrap_pyfunction!(py_six_dof, m)?)?;
    m.add_function(wrap_pyfunction!(skew, m)?)?;
    m.add_function(wrap_pyfunction!(_get_cache_dir, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    ukf::register(m)?;
    s10::register(m)?;
    env_logger::init();
    Ok(())
}
