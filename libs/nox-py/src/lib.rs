use std::collections::HashSet;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::{collections::BTreeMap, marker::PhantomData};

use impeller::{ComponentId, PolarsWorld};
use nox_ecs::{
    impeller,
    nox::{self, Noxpr},
    ErasedSystem,
};
use numpy::PyUntypedArray;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use tracing_subscriber::EnvFilter;

mod archetype;
mod asset;
mod component;
mod entity;
mod error;
mod exec;
mod graph;
mod impeller_client;
mod linalg;
mod query;
mod s10;
mod spatial;
mod system;
mod ukf;
mod well_known;
mod world_builder;

#[cfg(feature = "server")]
mod web_socket;

pub use archetype::*;
pub use asset::*;
pub use component::*;
pub use entity::*;
pub use error::*;
pub use exec::*;
pub use graph::*;
pub use linalg::*;
pub use query::*;
pub use spatial::*;
pub use system::*;
pub use well_known::*;
pub use world_builder::*;

trait PyUntypedArrayExt {
    unsafe fn buf(&self, elem_size: usize) -> &[u8];
}

impl PyUntypedArrayExt for PyUntypedArray {
    unsafe fn buf(&self, elem_size: usize) -> &[u8] {
        if !self.is_c_contiguous() {
            panic!("array must be c-style contiguous")
        }
        let len = self.shape().iter().product::<usize>() * elem_size;
        let obj = &*self.as_array_ptr();
        std::slice::from_raw_parts(obj.data as *const u8, len)
    }
}

#[pyclass]
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
            _ => Err(Error::PyErr(PyValueError::new_err("unknown integrator"))),
        }
    }
}

impl From<Integrator> for nox_ecs::Integrator {
    fn from(integrator: Integrator) -> Self {
        match integrator {
            Integrator::Rk4 => nox_ecs::Integrator::Rk4,
            Integrator::SemiImplicit => nox_ecs::Integrator::SemiImplicit,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (time_step = None, sys = None, integrator = Integrator::Rk4))]
pub fn six_dof(time_step: Option<f64>, sys: Option<System>, integrator: Integrator) -> System {
    let integrator = integrator.into();
    let sys: Arc<dyn nox_ecs::System<Arg = (), Ret = ()> + Send + Sync> =
        if let Some(dt) = time_step {
            if let Some(sys) = sys {
                nox_ecs::six_dof::six_dof_with_dt(|| sys, dt, integrator)
            } else {
                nox_ecs::six_dof::six_dof_with_dt(|| (), dt, integrator)
            }
        } else if let Some(sys) = sys {
            nox_ecs::six_dof::six_dof(|| sys, integrator)
        } else {
            nox_ecs::six_dof::six_dof(|| (), integrator)
        };
    System { inner: sys }
}

#[pyfunction]
pub fn read_batch_results(path: String) -> Result<(Vec<PyDataFrame>, Vec<String>), Error> {
    let sample_dirs = walkdir::WalkDir::new(path)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| entry.file_name() == "sample.json")
        .filter_map(|entry| entry.path().parent().map(Path::to_path_buf))
        .collect::<HashSet<_>>();
    let mut dfs = Vec::default();
    let mut ids = Vec::default();
    for sample_dir in sample_dirs {
        let dir_name = sample_dir.file_name().unwrap().to_string_lossy();
        ids.push(dir_name.to_string());
        let world = PolarsWorld::read_from_dir(sample_dir)?;
        let df = world.join_archetypes()?;
        dfs.push(PyDataFrame(df));
    }
    Ok((dfs, ids))
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
    m.add_class::<WorldBuilder>()?;
    m.add_class::<Exec>()?;
    m.add_class::<EntityId>()?;
    m.add_class::<SpatialTransform>()?;
    m.add_class::<SpatialForce>()?;
    m.add_class::<SpatialMotion>()?;
    m.add_class::<SpatialInertia>()?;
    m.add_class::<Quaternion>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Material>()?;
    m.add_class::<Handle>()?;
    m.add_class::<PrimitiveType>()?;
    m.add_class::<Metadata>()?;
    m.add_class::<QueryInner>()?;
    m.add_class::<GraphQueryInner>()?;
    m.add_class::<Edge>()?;
    m.add_class::<Component>()?;
    m.add_class::<impeller_client::Impeller>()?;
    m.add_class::<VectorArrow>()?;
    m.add_class::<BodyAxes>()?;
    m.add_class::<Color>()?;
    m.add_class::<Panel>()?;
    m.add_class::<Integrator>()?;
    m.add_class::<GraphEntity>()?;
    m.add_class::<Glb>()?;
    m.add_class::<Line3d>()?;
    m.add_class::<PyFnSystem>()?;
    m.add_class::<QueryMetadata>()?;
    m.add_class::<SystemBuilder>()?;
    m.add_class::<System>()?;
    m.add_function(wrap_pyfunction!(six_dof, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_results, m)?)?;
    m.add_function(wrap_pyfunction!(skew, m)?)?;
    m.add_function(wrap_pyfunction!(_get_cache_dir, m)?)?;
    ukf::register(m)?;
    s10::register(m)?;
    Ok(())
}
