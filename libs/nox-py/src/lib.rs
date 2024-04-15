use std::sync::Arc;
use std::{collections::BTreeMap, marker::PhantomData};

use conduit::well_known::GizmoType;
use nox_ecs::conduit::Asset;
use nox_ecs::{
    conduit,
    nox::{self, Noxpr},
    polars::PolarsWorld,
    ErasedSystem, System,
};
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use tracing_subscriber::EnvFilter;

mod archetype;
mod asset;
mod component;
mod conduit_client;
mod entity;
mod error;
mod exec;
mod graph;
mod pipeline_builder;
mod query;
mod sim_runner;
mod spatial;
mod system;
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
pub use pipeline_builder::*;
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
#[derive(Clone)]
pub struct Gizmo {
    inner: conduit::well_known::Gizmo,
}

#[pymethods]
impl Gizmo {
    #[staticmethod]
    fn vector(id: ComponentId, offset: usize, color: Color) -> Self {
        Self {
            inner: conduit::well_known::Gizmo {
                id: id.inner,
                ty: GizmoType::Vector {
                    range: offset..offset + 3,
                    color: color.inner,
                },
            },
        }
    }

    pub fn asset_id(&self) -> u64 {
        self.inner.asset_id().0
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}

#[pyclass]
pub struct Client {
    client: nox::Client,
}

#[pymethods]
impl Client {
    #[staticmethod]
    pub fn cpu() -> Result<Self, Error> {
        Ok(Self {
            client: nox::Client::cpu()?,
        })
    }
}

#[pyfunction]
pub fn six_dof(time_step: f64, sys: Option<PyObject>) -> RustSystem {
    let sys: Arc<dyn System<Arg = (), Ret = ()> + Send + Sync> = if let Some(sys) = sys {
        let sys = nox_ecs::six_dof::six_dof(|| PySystem { sys }, time_step);
        Arc::new(ErasedSystem::new(sys))
    } else {
        let sys = nox_ecs::six_dof::six_dof(|| (), time_step);
        Arc::new(ErasedSystem::new(sys))
    };
    RustSystem { inner: sys }
}

#[pyfunction]
pub fn advance_time(time_step: f64) -> RustSystem {
    let sys = nox_ecs::six_dof::advance_time(time_step);
    RustSystem {
        inner: Arc::new(ErasedSystem::new(sys)),
    }
}

#[pyfunction]
pub fn read_batch_results(path: String) -> Result<PyDataFrame, Error> {
    let world = PolarsWorld::read_from_dir(path)?;
    let df = world.join_archetypes()?;
    Ok(PyDataFrame(df))
}

#[pymodule]
pub fn elodin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ComponentType>()?;
    m.add_class::<ComponentId>()?;
    m.add_class::<PipelineBuilder>()?;
    m.add_class::<WorldBuilder>()?;
    m.add_class::<EntityId>()?;
    m.add_class::<Client>()?;
    m.add_class::<SpatialTransform>()?;
    m.add_class::<SpatialForce>()?;
    m.add_class::<SpatialMotion>()?;
    m.add_class::<SpatialInertia>()?;
    m.add_class::<Quaternion>()?;
    m.add_class::<RustSystem>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Material>()?;
    m.add_class::<Handle>()?;
    m.add_class::<Pbr>()?;
    m.add_class::<EntityMetadata>()?;
    m.add_class::<PrimitiveType>()?;
    m.add_class::<Metadata>()?;
    m.add_class::<QueryInner>()?;
    m.add_class::<GraphQueryInner>()?;
    m.add_class::<Edge>()?;
    m.add_class::<Component>()?;
    m.add_class::<conduit_client::Conduit>()?;
    m.add_class::<Gizmo>()?;
    m.add_class::<Color>()?;
    m.add_class::<Panel>()?;
    m.add_function(wrap_pyfunction!(six_dof, m)?)?;
    m.add_function(wrap_pyfunction!(advance_time, m)?)?;
    m.add_function(wrap_pyfunction!(read_batch_results, m)?)?;
    Ok(())
}
