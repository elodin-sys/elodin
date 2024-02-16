use std::{
    collections::{hash_map::Entry, BTreeMap},
    marker::PhantomData,
    ops::Deref,
};

use nox_ecs::{
    elodin_conduit, join_many,
    nox::{self, jax::JaxTracer, ArrayTy, Noxpr, NoxprNode, ScalarExt},
    ArchetypeId, ComponentArray, ErasedSystem, HostColumn, HostStore, Query, SharedWorld, System,
    Table, World,
};
use numpy::{ndarray::ArrayViewD, PyArray, PyUntypedArray};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyTuple},
};

mod spatial;
pub use spatial::*;

#[pyclass]
#[derive(Default)]
struct PipelineBuilder {
    builder: nox_ecs::PipelineBuilder,
}

#[pymethods]
impl PipelineBuilder {
    fn get_var(&mut self, id: ComponentId) -> Result<(ComponentArrayMetadata, PyObject), Error> {
        let var = self
            .builder
            .vars
            .get(&id.inner)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let var = var.borrow();
        let NoxprNode::Jax(buf) = var.buffer().deref() else {
            todo!()
        };
        let obj = Python::with_gil(|py| buf.to_object(py));
        let metadata = ComponentArrayMetadata {
            len: var.len,
            entity_map: var.entity_map.clone(),
        };
        Ok((metadata, obj))
    }

    fn set_var(
        &mut self,
        id: ComponentId,
        metadata: &ComponentArrayMetadata,
        value: PyObject,
    ) -> Result<(), Error> {
        let var = self
            .builder
            .vars
            .get(&id.inner)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let mut var = var.borrow_mut();
        let update_buffer = Noxpr::jax(value);
        if var.entity_map == metadata.entity_map {
            var.buffer = update_buffer;
        } else {
            let mut tracer = JaxTracer::new();
            let new_buf = nox_ecs::update_var(
                &var.entity_map,
                &metadata.entity_map,
                &var.buffer,
                &update_buffer,
            );
            var.buffer = Noxpr::jax(tracer.visit(&new_buf)?);
        }
        Ok(())
    }

    fn init_var(&mut self, id: ComponentId, ty: ComponentType) -> Result<(), Error> {
        let id = id.inner;
        if self.builder.param_ids.contains(&id) {
            return Ok(());
        }
        let column = self
            .builder
            .world
            .column_by_id(id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let ty: elodin_conduit::ComponentType = ty.into();
        let len = column.column.buffer.len();
        let shape = std::iter::once(len as i64)
            .chain(ty.dims().iter().copied())
            .collect();
        let op = Noxpr::parameter(
            self.builder.param_ops.len() as i64,
            ArrayTy {
                element_type: ty.element_type(),
                shape, // FIXME
            },
            format!("{:?}::{}", id, self.builder.param_ops.len()),
        );
        self.builder.param_ops.push(op.clone());
        self.builder.param_ids.push(id);

        Ok(())
    }

    fn var_arrays(&mut self, py: Python<'_>) -> Result<Vec<PyObject>, Error> {
        let mut res = vec![];
        for p in &self.builder.param_ops {
            let NoxprNode::Param(p) = p.deref() else {
                continue;
            };
            let jnp = py.import("jax.numpy")?;
            let dtype = nox::jax::dtype(&p.ty.element_type)?;
            let shape = PyTuple::new(py, p.ty.shape.iter().collect::<Vec<_>>());
            let arr = jnp.call_method1("zeros", (shape, dtype))?; // NOTE(sphw): this could be a huge bottleneck
            res.push(arr.into());
        }
        Ok(res)
    }

    fn inject_args(&mut self, args: Vec<PyObject>) -> Result<(), Error> {
        let builder = &mut self.builder;
        assert_eq!(args.len(), builder.param_ids.len());
        let nox_ecs::PipelineBuilder {
            vars,
            world,
            param_ids,
            ..
        } = builder;
        for (arg, id) in args.into_iter().zip(param_ids.iter()) {
            let column = world
                .column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let len = column.column.buffer.len();
            let array = ComponentArray {
                buffer: Noxpr::jax(arg),
                phantom_data: PhantomData,
                len,
                entity_map: column.entity_map.clone(),
            };
            vars.insert(*id, array.into());
        }
        Ok(())
    }

    fn ret_vars(&self, py: Python<'_>) -> Result<PyObject, Error> {
        let vars = self
            .builder
            .vars
            .values()
            .map(|var| {
                let var = var.borrow();
                var.buffer().to_jax()
            })
            .collect::<Result<Vec<_>, nox_ecs::nox::Error>>()?;
        Ok(PyTuple::new(py, vars).into())
    }
}

#[derive(Clone)]
#[pyclass]
pub struct ComponentId {
    inner: nox_ecs::ComponentId,
}

#[pymethods]
impl ComponentId {
    #[new]
    fn new(string: String) -> Self {
        Self {
            inner: nox_ecs::ComponentId::new(&string),
        }
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub enum ComponentType {
    // Primatives
    U8 = 0,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    F32,
    F64,

    // Variable Size
    String,
    Bytes,

    // Tensors
    Vector3F32,
    Vector3F64,
    Matrix3x3F32,
    Matrix3x3F64,
    QuaternionF32,
    QuaternionF64,
    SpatialPosF32,
    SpatialPosF64,
    SpatialMotionF32,
    SpatialMotionF64,

    // Msgs
    Filter,
}

impl From<ComponentType> for elodin_conduit::ComponentType {
    fn from(val: ComponentType) -> Self {
        match val {
            ComponentType::U8 => elodin_conduit::ComponentType::U8,
            ComponentType::U16 => elodin_conduit::ComponentType::U16,
            ComponentType::U32 => elodin_conduit::ComponentType::U32,
            ComponentType::U64 => elodin_conduit::ComponentType::U64,
            ComponentType::I8 => elodin_conduit::ComponentType::I8,
            ComponentType::I16 => elodin_conduit::ComponentType::I16,
            ComponentType::I32 => elodin_conduit::ComponentType::I32,
            ComponentType::I64 => elodin_conduit::ComponentType::I64,
            ComponentType::Bool => elodin_conduit::ComponentType::Bool,
            ComponentType::F32 => elodin_conduit::ComponentType::F32,
            ComponentType::F64 => elodin_conduit::ComponentType::F64,
            ComponentType::String => elodin_conduit::ComponentType::String,
            ComponentType::Bytes => elodin_conduit::ComponentType::Bytes,
            ComponentType::Vector3F32 => elodin_conduit::ComponentType::Vector3F32,
            ComponentType::Vector3F64 => elodin_conduit::ComponentType::Vector3F64,
            ComponentType::Matrix3x3F32 => elodin_conduit::ComponentType::Matrix3x3F32,
            ComponentType::Matrix3x3F64 => elodin_conduit::ComponentType::Matrix3x3F64,
            ComponentType::QuaternionF32 => elodin_conduit::ComponentType::QuaternionF32,
            ComponentType::QuaternionF64 => elodin_conduit::ComponentType::QuaternionF64,
            ComponentType::SpatialPosF32 => elodin_conduit::ComponentType::SpatialPosF32,
            ComponentType::SpatialPosF64 => elodin_conduit::ComponentType::SpatialPosF64,
            ComponentType::SpatialMotionF32 => elodin_conduit::ComponentType::SpatialMotionF32,
            ComponentType::SpatialMotionF64 => elodin_conduit::ComponentType::SpatialMotionF64,
            ComponentType::Filter => elodin_conduit::ComponentType::Filter,
        }
    }
}

#[pyclass]
#[derive(Default)]
pub struct WorldBuilder {
    world: World<HostStore>,
}

impl WorldBuilder {
    fn get_or_insert_archetype(
        &mut self,
        py: Python<'_>,
        archetype: &PyObject,
    ) -> Result<&mut Table<HostStore>, Error> {
        let archetype_id = archetype
            .call_method0(py, "archetype_id")?
            .extract::<u64>(py)?;
        let archetype_id = ArchetypeId::new(archetype_id.into());

        match self.world.archetypes.entry(archetype_id) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let archetype_id = archetype
                    .call_method0(py, "archetype_id")?
                    .extract::<u64>(py)?;
                let archetype_id = ArchetypeId::new(archetype_id.into());
                let datas = archetype
                    .call_method0(py, "component_data")?
                    .extract::<Vec<PyObject>>(py)?;
                let component_ids = datas
                    .iter()
                    .map(|data| {
                        let id = data.getattr(py, "id")?.extract::<ComponentId>(py)?;
                        Ok(id.inner)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let columns = datas
                    .iter()
                    .map(|data| {
                        let id = data.getattr(py, "id")?.extract::<ComponentId>(py)?;
                        let ty = data.getattr(py, "type")?.extract::<ComponentType>(py)?;
                        Ok((
                            id.inner,
                            nox_ecs::Column::<HostStore>::new(HostColumn::new(ty.into(), id.inner)),
                        ))
                    })
                    .collect::<Result<_, Error>>()?;
                for id in component_ids {
                    self.world.component_map.insert(id, archetype_id);
                }
                let table = Table {
                    columns,
                    entity_buffer: HostColumn::new(
                        elodin_conduit::ComponentType::U64,
                        nox_ecs::ComponentId::new("entity_id"),
                    ),
                    entity_map: BTreeMap::default(),
                };
                Ok(entry.insert(table))
            }
        }
    }
}

#[pymethods]
impl WorldBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&mut self, py: Python<'_>, archetype: PyObject) -> Result<EntityId, Error> {
        let entity_id = EntityId::rand();
        self.spawn_with_entity_id(py, archetype, entity_id)
    }

    pub fn spawn_with_entity_id(
        &mut self,
        py: Python<'_>,
        archetype: PyObject,
        entity_id: EntityId,
    ) -> Result<EntityId, Error> {
        let entity_id = entity_id.inner;
        let table = self.get_or_insert_archetype(py, &archetype)?;
        table
            .entity_map
            .insert(entity_id, table.entity_buffer.len());
        table
            .entity_buffer
            .push((table.entity_buffer.len() as u64).constant());

        let datas = archetype
            .call_method0(py, "component_data")?
            .extract::<Vec<PyObject>>(py)?;
        let component_ids = datas.iter().map(|data| {
            let id = data.getattr(py, "id")?.extract::<ComponentId>(py)?;
            Ok::<_, Error>(id.inner)
        });
        let arrays = archetype.call_method0(py, "arrays")?;
        let arrays = arrays.extract::<Vec<&numpy::PyUntypedArray>>(py)?;
        for (arr, id) in arrays.iter().zip(component_ids) {
            let id = id?;
            let col = table
                .columns
                .get_mut(&id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let ty = col.buffer.component_type();
            let size = ty.element_type().element_size_in_bytes();
            let buf = unsafe {
                if !arr.is_c_contiguous() {
                    panic!("array must be c-style contiguous")
                }
                let len = arr.shape().iter().product::<usize>() * size;
                let obj = &*arr.as_array_ptr();
                std::slice::from_raw_parts(obj.data as *const u8, len)
            };
            col.buffer.push_raw(buf);
        }
        Ok(EntityId { inner: entity_id })
    }

    pub fn run(&mut self, py: Python<'_>, sys: PyObject) -> Result<(), Error> {
        // skip `python3 <name of script>`
        let args: Vec<_> = std::env::args_os().skip(2).collect();
        let mut pargs = pico_args::Arguments::from_vec(args);
        let cmd = pargs.subcommand().map_err(|_| Error::UnexpectedInput)?;

        match cmd.as_deref() {
            Some("monte-carlo") => {}
            Some(cmd) => return Err(Error::UnknownCommand(cmd.to_string())),
            None => todo!("spawn TCP server, and run simulation locally"),
        }

        let build_dir: String = pargs
            .value_from_str("--build-dir")
            .map_err(|err| Error::MissingArg(err.to_string()))?;
        let build_dir = std::path::PathBuf::from(build_dir);

        let exec = self.build(py, sys)?.exec;
        exec.write_to_dir(build_dir)?;
        Ok(())
    }

    // TODO: reuse run() in build() after proper world serialization
    pub fn build(&mut self, py: Python<'_>, sys: PyObject) -> Result<Exec, Error> {
        let world = std::mem::take(&mut self.world);
        let builder = nox_ecs::PipelineBuilder::from_world(world);
        let builder = PipelineBuilder { builder };
        let py_code = "import jax
def build_expr(builder, sys):
    sys.init(builder)
    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)
        return builder.ret_vars()
    xla = jax.xla_computation(lambda a: call(a, builder))(builder.var_arrays())
    return xla";

        let fun: Py<PyAny> = PyModule::from_code(py, py_code, "", "")?
            .getattr("build_expr")?
            .into();
        let builder = PyCell::new(py, builder)?;
        let comp = fun
            .call1(py, (builder.borrow_mut(), sys))?
            .extract::<PyObject>(py)?;
        let comp = comp.call_method0(py, "as_serialized_hlo_module_proto")?;
        let comp = comp
            .downcast::<PyBytes>(py)
            .map_err(|_| Error::HloModuleNotBytes)?;
        let comp_bytes = comp.as_bytes();
        let hlo_module = nox::xla::HloModuleProto::parse_binary(comp_bytes)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let builder = builder.replace(PipelineBuilder::default());
        let builder = builder.builder;
        let ret_ids = builder.vars.keys().copied().collect::<Vec<_>>();
        let exec = nox_ecs::WorldExec {
            world: SharedWorld {
                host: builder.world,
                ..Default::default()
            },
            tick_exec: nox_ecs::Exec {
                exec: Default::default(),
                metadata: nox_ecs::ExecMetadata {
                    arg_ids: builder.param_ids,
                    ret_ids,
                },
                hlo_module,
            },
            startup_exec: None,
            history: nox_ecs::history::History::default(),
        };

        Ok(Exec { exec })
    }
}

struct PySystem {
    sys: PyObject,
}

impl System for PySystem {
    type Arg = ();

    type Ret = ();

    fn init_builder(
        &self,
        in_builder: &mut nox_ecs::PipelineBuilder,
    ) -> Result<(), nox_ecs::Error> {
        let builder = std::mem::take(in_builder);
        let builder = PipelineBuilder { builder };
        let builder = Python::with_gil(move |py| {
            let builder = PyCell::new(py, builder)?;
            self.sys.call_method1(py, "init", (builder.borrow_mut(),))?;
            Ok::<_, Error>(builder.replace(PipelineBuilder::default()))
        })
        .unwrap();
        *in_builder = builder.builder;
        Ok(())
    }

    fn add_to_builder(
        &self,
        in_builder: &mut nox_ecs::PipelineBuilder,
    ) -> Result<(), nox_ecs::Error> {
        let builder = std::mem::take(in_builder);
        let builder = PipelineBuilder { builder };
        let builder = Python::with_gil(move |py| {
            let builder = PyCell::new(py, builder)?;
            self.sys.call_method1(py, "call", (builder.borrow_mut(),))?;
            Ok::<_, Error>(builder.replace(PipelineBuilder::default()))
        })
        .unwrap();
        *in_builder = builder.builder;
        Ok(())
    }
}

#[pyclass]
pub struct Exec {
    exec: nox_ecs::WorldExec,
}

#[pymethods]
impl Exec {
    pub fn run(&mut self, client: &Client) -> Result<(), Error> {
        Python::with_gil(|_| self.exec.run(&client.client).map_err(Error::from))
    }

    fn column_array(
        this_cell: &PyCell<Self>,
        id: ComponentId,
    ) -> Result<&'_ numpy::PyUntypedArray, Error> {
        let mut this = this_cell.borrow_mut();
        let column = this.exec.column(id.inner)?;
        let dyn_array = column
            .column
            .buffer
            .dyn_ndarray()
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        fn untyped_pyarray<'py, T: numpy::Element + 'static>(
            view: &ArrayViewD<'_, T>,
            container: &'py PyAny,
        ) -> &'py PyUntypedArray {
            // # Safety
            // This is one of those things that I'm like 75% sure is safe enough,
            // but also close to 100% sure it breaks Rust's rules.
            // There are essentially two safety guarantees that we want to keep
            // when doing weird borrow stuff, ensure you aren't creating a reference
            // to free-ed / uninitialized/unaligned memory and to ensure that you are not
            // accidentally creating aliasing. We know we aren't doing the first one here,
            // because `Exec` is guaranteed to stay around as long as the `PyCell` is around.
            // We are technically breaking the 2nd rule, BUT, I think the way we are doing it is also ok.
            // What can happen is that we call `column_array`, then we call `run`, then we call `column_array` again.
            // We still have an outstanding reference to the array, and calling `column_array` again will cause the contents to be overwritten.
            // In most languages, this would cause all sorts of problems, but thankfully, in Python, we have the GIL to save the day.
            // While `exec` is run, the GIL is taken, so no one can access the old array result.
            // We never re-alloc the underlying buffer because lengths are immutable during execution.
            unsafe {
                let arr = PyArray::borrow_from_array(view, container);
                arr.as_untyped()
            }
        }
        match dyn_array {
            nox_ecs::DynArrayView::F64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::F32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U16(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U8(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I16(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I8(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::Bool(f) => Ok(untyped_pyarray(&f, this_cell)),
        }
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

#[derive(Clone)]
#[pyclass]
pub struct EntityId {
    inner: elodin_conduit::EntityId,
}

#[pymethods]
impl EntityId {
    #[staticmethod]
    pub fn rand() -> Self {
        EntityId {
            inner: elodin_conduit::EntityId::rand(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Nox(#[from] nox::Error),
    #[error("{0}")]
    NoxEcs(#[from] nox_ecs::Error),
    #[error("{0}")]
    PyErr(#[from] PyErr),
    #[error("hlo module was not PyBytes")]
    HloModuleNotBytes,
    #[error("unexpected input")]
    UnexpectedInput,
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("{0}")]
    MissingArg(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::NoxEcs(nox_ecs::Error::ComponentNotFound) => {
                PyValueError::new_err("component not found")
            }
            Error::NoxEcs(nox_ecs::Error::ValueSizeMismatch) => {
                PyValueError::new_err("value size mismatch")
            }
            Error::PyErr(err) => err,
            err => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ComponentArrayMetadata {
    len: usize,
    entity_map: BTreeMap<elodin_conduit::EntityId, usize>,
}

#[pymethods]
impl ComponentArrayMetadata {
    fn join(
        &self,
        expr: PyObject,
        other_metadata: ComponentArrayMetadata,
        other_exprs: Vec<PyObject>,
    ) -> Result<(ComponentArrayMetadata, Vec<PyObject>), Error> {
        let arr = ComponentArray {
            buffer: Noxpr::jax(expr),
            len: self.len,
            entity_map: self.entity_map.clone(),
            phantom_data: PhantomData::<()>,
        };
        let other_exprs = other_exprs.into_iter().map(Noxpr::jax).collect();
        let query = Query {
            exprs: other_exprs,
            entity_map: other_metadata.entity_map,
            len: other_metadata.len,
            phantom_data: PhantomData::<()>,
        };
        let out = join_many(query, &arr);
        let mut tracer = JaxTracer::new();
        let exprs = out
            .exprs
            .into_iter()
            .map(|expr| tracer.visit(&expr))
            .collect::<Result<Vec<_>, _>>()?;
        let metadata = ComponentArrayMetadata {
            len: out.len,
            entity_map: out.entity_map,
        };
        Ok((metadata, exprs))
    }
}

#[pyclass]
pub struct RustSystem {
    inner: Box<dyn System<Arg = (), Ret = ()> + Send + Sync>,
}

#[pymethods]
impl RustSystem {
    fn init(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.inner.init_builder(&mut builder.builder)?;
        Ok(())
    }
    fn call(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.inner.add_to_builder(&mut builder.builder)?;
        Ok(())
    }
}

#[pyfunction]
pub fn six_dof(time_step: f64, sys: Option<PyObject>) -> RustSystem {
    let sys: Box<dyn System<Arg = (), Ret = ()> + Send + Sync> = if let Some(sys) = sys {
        let sys = nox_ecs::six_dof::six_dof(|| PySystem { sys }, time_step);
        Box::new(ErasedSystem::new(sys))
    } else {
        let sys = nox_ecs::six_dof::six_dof(|| (), time_step);
        Box::new(ErasedSystem::new(sys))
    };
    RustSystem { inner: sys }
}

#[pyfunction]
// TODO: remove after https://github.com/PyO3/maturin/issues/368 is resolved
fn run_cli(_py: Python) -> PyResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ::elodin::Cli::from_args(&args).run();
    Ok(())
}

#[pymodule]
pub fn elodin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ComponentType>()?;
    m.add_class::<ComponentId>()?;
    m.add_class::<PipelineBuilder>()?;
    m.add_class::<WorldBuilder>()?;
    m.add_class::<EntityId>()?;
    m.add_class::<Client>()?;
    m.add_class::<ComponentArrayMetadata>()?;
    m.add_class::<SpatialTransform>()?;
    m.add_class::<SpatialForce>()?;
    m.add_class::<SpatialMotion>()?;
    m.add_class::<SpatialInertia>()?;
    m.add_class::<Quaternion>()?;
    m.add_class::<RustSystem>()?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(six_dof, m)?)?;
    Ok(())
}
