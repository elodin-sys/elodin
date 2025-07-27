use crate::*;

use nox_ecs::utils::PrimTypeExt;
use nox_ecs::{
    CompiledSystem, Exec, ExecMetadata, SystemParam,
    graph::{EdgeComponent, GraphQuery, TotalEdge},
    nox::{NoxprComp, NoxprFn, NoxprNode, NoxprTy, jax::JaxNoxprFn},
};
use nox_ecs::{ComponentSchema, World};
use pyo3::IntoPyObjectExt;
use pyo3::types::{IntoPyDict, PyBytes};
use std::{collections::HashMap, sync::Arc};

#[pyclass]
#[derive(Debug)]
pub struct PyFnSystem {
    sys: Py<PyAny>,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    edge_ids: Vec<ComponentId>,
    #[allow(dead_code)]
    name: String,
}

impl Clone for PyFnSystem {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            sys: self.sys.clone_ref(py),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            edge_ids: self.edge_ids.clone(),
            name: self.name.clone(),
        })
    }
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
        let sys = Python::with_gil(|py| self.sys.clone_ref(py));
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
                schema: col.schema.clone(),
                component: Component {
                    name: col.metadata.name.to_string(),
                    ty: Some(col.schema.clone().into()),
                    metadata: col.metadata.metadata.clone(),
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
            let jax = py.import("jax").unwrap();
            // Clean up unused variable warning
            let _jit_args = [("keep_unused", true)].into_py_dict(py);
            // Get the jit function and evaluate it directly
            let jit_fn = jax.getattr("jit")?;
            Ok::<_, pyo3::PyErr>(jit_fn.call1((func,))?.into())
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
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<Py<PyAny>>, Error>;
    fn compile_hlo_module(&self, py: Python<'_>, world: &World) -> Result<Exec, Error>;
    fn compile_jax_module(&self, py: Python<'_>) -> Result<Py<PyAny>, Error>;
}

impl CompiledSystemExt for CompiledSystem {
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<Py<PyAny>>, Error> {
        let mut res = vec![];
        let mut visited_ids = HashSet::new();
        for id in self.inputs.iter().chain(self.outputs.iter()) {
            if visited_ids.contains(id) {
                continue;
            }
            let jnp = py.import("jax.numpy")?;
            let col = world
                .column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let elem_ty = col.schema.prim_type;
            let dtype = nox::jax::dtype(&elem_ty.to_element_type())?;
            // Build tuple shape manually instead of using PyTuple
            let shape_vec: Vec<_> = std::iter::once(col.len() as u64)
                .chain(col.schema.shape().iter().copied())
                .collect();
            // Use direct call pattern that takes standard args
            let arr = jnp.getattr("zeros")?.call((shape_vec, dtype), None)?; // NOTE(sphw): this could be a huge bottleneck
            visited_ids.insert(id);
            res.push(arr.into());
        }
        Ok(res)
    }

    fn compile_hlo_module(&self, py: Python<'_>, world: &World) -> Result<Exec, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());
        let input_arrays = self.arg_arrays(py, world)?;
        let py_code = "import jax
def build_expr(jit, args):
  xla = jit.lower(*args).compiler_ir('hlo')
  return xla";
        // We need to create a module directly and exec code in it
        let module = PyModule::new(py, "build_expr")?;
        let globals = module.dict();
        // Use std::ffi::CString for the string arguments to run
        let code_cstr = std::ffi::CString::new(py_code).unwrap();
        py.run(code_cstr.as_ref(), Some(&globals), None)?;
        let fun: Py<PyAny> = module.getattr("build_expr")?.into();

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

    fn compile_jax_module(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());

        let py_code = "
import jax
def build_expr(func):
    res = jax.jit(func)
    return res";

        // We need to create a module directly and exec code in it
        let module = PyModule::new(py, "build_expr")?;
        let globals = module.dict();
        // Use std::ffi::CString for the string arguments to run
        let code_cstr = std::ffi::CString::new(py_code).unwrap();
        py.run(code_cstr.as_ref(), Some(&globals), None)?;
        let fun: Py<PyAny> = module.getattr("build_expr")?.into();

        let comp = fun.call1(py, (func,))?;

        Ok(comp)
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
    pub entity_map: BTreeMap<impeller2::types::EntityId, usize>,
    pub len: usize,
    pub schema: ComponentSchema,
    pub component: Component,
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

fn noxpr_to_callable(func: Arc<NoxprFn>) -> Py<PyAny> {
    if let NoxprNode::Jax(j) = &*func.inner.node {
        return Python::with_gil(|py| j.clone_ref(py));
    }
    let func = JaxNoxprFn {
        tracer: Default::default(),
        inner: func,
    };

    Python::with_gil(|py| func.into_py_any(py).unwrap())
}
