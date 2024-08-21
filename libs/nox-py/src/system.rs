use crate::*;

use impeller::World;
use nox_ecs::{
    graph::{EdgeComponent, GraphQuery, TotalEdge},
    nox::{jax::JaxNoxprFn, NoxprComp, NoxprFn, NoxprNode, NoxprTy},
    CompiledSystem, Exec, ExecMetadata, SystemParam,
};
use pyo3::types::{IntoPyDict, PyBytes, PyTuple};
use std::{collections::HashMap, sync::Arc};

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyFnSystem {
    sys: PyObject,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    edge_ids: Vec<ComponentId>,
    #[allow(dead_code)]
    name: String,
}

#[pymethods]
impl PyFnSystem {
    #[new]
    fn new(
        sys: PyObject,
        input_ids: Vec<String>,
        output_ids: Vec<String>,
        edge_ids: Vec<String>,
        name: String,
    ) -> Self {
        Self {
            sys,
            input_ids: input_ids.iter().map(|x| ComponentId::new(x)).collect(),
            output_ids: output_ids.iter().map(|x| ComponentId::new(x)).collect(),
            edge_ids: edge_ids.iter().map(|x| ComponentId::new(x)).collect(),
            name,
        }
    }

    fn system(&self) -> System {
        System::new(self.clone())
    }
}

impl nox_ecs::System for PyFnSystem {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut nox_ecs::SystemBuilder) -> Result<(), nox_ecs::Error> {
        for &id in &self.input_ids {
            builder.init_with_column(id)?;
        }
        for &id in &self.output_ids {
            builder.init_with_column(id)?;
        }

        Ok(())
    }

    fn compile(&self, world: &World) -> Result<nox_ecs::CompiledSystem, nox_ecs::Error> {
        let sys = self.sys.clone();
        let mut input_ids = self.input_ids.clone();
        let output_ids = self.output_ids.clone();
        let builder = nox_ecs::SystemBuilder::new(world);
        let mut py_builder = SystemBuilder {
            total_edges: GraphQuery::<TotalEdge>::param(&builder)?.edges,
            ..Default::default()
        };
        for id in output_ids.iter() {
            if !input_ids.contains(id) {
                input_ids.push(*id);
            }
        }
        for (index, &id) in input_ids.iter().enumerate() {
            if py_builder.arg_map.contains_key(&id) {
                continue;
            }
            let col = builder
                .world
                .column_by_id(id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let arg_metadata = ArgMetadata {
                entity_map: col.entity_map(),
                len: col.len(),
                metadata: Metadata {
                    inner: col.metadata.clone(),
                },
            };
            py_builder.arg_map.insert(id, (arg_metadata, index));
        }
        for &id in &self.edge_ids {
            let col = builder.world.column_by_id(id).unwrap();
            let edges = col
                .iter()
                .map(move |(_, value)| nox_ecs::graph::Edge::from_value(value).unwrap())
                .collect();
            py_builder.edge_map.insert(id, edges);
        }
        let mut tys = self
            .output_ids
            .iter()
            .map(|id| NoxprTy::ArrayTy(builder.world.column_by_id(*id).unwrap().buffer_ty()));

        let ty = if self.output_ids.len() == 1 {
            tys.next().unwrap()
        } else {
            NoxprTy::Tuple(tys.collect())
        };
        let func = Python::with_gil(|py| {
            let func = sys.call1(py, (py_builder,))?;
            let jax = py.import_bound("jax").unwrap();
            let jit_args = [("keep_unused", true)].into_py_dict_bound(py);
            Ok::<_, pyo3::PyErr>(jax.call_method("jit", (func,), Some(&jit_args))?.into())
        })?;
        let func = NoxprFn::new(vec![], Noxpr::jax(func));
        Ok(CompiledSystem {
            computation: NoxprComp::new(func, ty),
            inputs: input_ids,
            outputs: output_ids,
        })
    }
}

pub trait CompiledSystemExt {
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<PyObject>, Error>;
    fn compile_hlo_module(&self, py: Python<'_>, world: &World) -> Result<Exec, Error>;
}

impl CompiledSystemExt for CompiledSystem {
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<PyObject>, Error> {
        let mut res = vec![];
        let mut visited_ids = HashSet::new();
        for id in self.inputs.iter().chain(self.outputs.iter()) {
            if visited_ids.contains(id) {
                continue;
            }
            let jnp = py.import_bound("jax.numpy")?;
            let col = world
                .column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let ty = col.metadata.component_type.clone();
            let elem_ty = ty.primitive_ty.element_type();
            let dtype = nox::jax::dtype(&elem_ty)?;
            let shape = PyTuple::new_bound(
                py,
                std::iter::once(col.len() as i64)
                    .chain(ty.shape.iter().copied())
                    .collect::<Vec<_>>(),
            );
            let arr = jnp.call_method1("zeros", (shape, dtype))?; // NOTE(sphw): this could be a huge bottleneck
            visited_ids.insert(id);
            res.push(arr.into());
        }
        Ok(res)
    }

    fn compile_hlo_module(&self, py: Python<'_>, world: &World) -> Result<Exec, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());
        // let NoxprNode::Jax(func) = dbg!(&*self.computation.func.inner.node) else {
        //     todo!()
        // };
        let input_arrays = self.arg_arrays(py, world)?;
        let py_code = "import jax
def build_expr(jit, args):
  xla = jit.lower(*args).compiler_ir('hlo')
  return xla";
        let fun: Py<PyAny> = PyModule::from_code_bound(py, py_code, "", "")?
            .getattr("build_expr")?
            .into();

        let comp = fun
            .call1(py, (func, input_arrays))?
            .extract::<PyObject>(py)?;
        let comp = comp.call_method0(py, "as_serialized_hlo_module_proto")?;
        let comp = comp
            .downcast_bound::<PyBytes>(py)
            .map_err(|_| Error::HloModuleNotBytes)?;
        let comp_bytes = comp.as_bytes();
        let hlo_module = nox::xla::HloModuleProto::parse_binary(comp_bytes)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let exec = Exec::new(
            ExecMetadata {
                arg_ids: self.inputs.clone(),
                ret_ids: self.outputs.clone(),
            },
            hlo_module,
        );

        Ok(exec)
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct SystemBuilder {
    arg_map: HashMap<ComponentId, (ArgMetadata, usize)>,
    pub edge_map: HashMap<ComponentId, Vec<nox_ecs::graph::Edge>>,
    pub total_edges: Vec<nox_ecs::graph::Edge>,
}

#[derive(Clone)]
pub struct ArgMetadata {
    pub entity_map: BTreeMap<impeller::EntityId, usize>,
    pub len: usize,
    pub metadata: Metadata,
}

impl SystemBuilder {
    pub fn get_var(&self, id: ComponentId) -> Option<(ArgMetadata, usize)> {
        self.arg_map.get(&id).cloned()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct System {
    pub inner: Arc<dyn nox_ecs::System<Arg = (), Ret = ()> + Send + Sync>,
}

impl System {
    pub fn new(sys: impl nox_ecs::System + Send + Sync + 'static) -> Self {
        let inner = Arc::new(ErasedSystem::new(sys));
        Self { inner }
    }
}

#[pymethods]
impl System {
    pub fn pipe(&self, other: System) -> System {
        let pipe = nox_ecs::Pipe::new(self.clone(), other);
        System::new(pipe)
    }

    pub fn __or__(&self, other: System) -> System {
        self.pipe(other)
    }
}

impl nox_ecs::System for System {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut nox_ecs::SystemBuilder) -> Result<(), nox_ecs::Error> {
        self.inner.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, nox_ecs::Error> {
        self.inner.compile(world)
    }
}

fn noxpr_to_callable(func: Arc<NoxprFn>) -> PyObject {
    if let NoxprNode::Jax(j) = &*func.inner.node {
        return j.clone();
    }
    let func = JaxNoxprFn {
        tracer: Default::default(),
        inner: func,
    };

    Python::with_gil(|py| func.into_py(py))
}
