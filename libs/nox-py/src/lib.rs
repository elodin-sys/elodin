use core::fmt;
use std::{
    collections::{hash_map::Entry, BTreeMap},
    marker::PhantomData,
    net::SocketAddr,
    ops::Deref,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use clap::Parser;
use nox_ecs::{
    conduit, join_many,
    nox::{self, ArrayTy, Noxpr, NoxprNode, NoxprTy, ScalarExt},
    spawn_tcp_server, ArchetypeId, ComponentArray, ErasedSystem, HostColumn, HostStore,
    SharedWorld, System, Table, World,
};
use nox_ecs::{
    conduit::{Asset, TagValue},
    join_query,
};
use numpy::{ndarray::ArrayViewD, PyArray, PyArray1, PyReadonlyArray1, PyUntypedArray};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyTuple},
};

mod conduit_client;
mod graph;
mod spatial;

pub use graph::*;
pub use spatial::*;

#[pyclass]
#[derive(Default)]
struct PipelineBuilder {
    builder: nox_ecs::PipelineBuilder,
}

#[pymethods]
impl PipelineBuilder {
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
        let ty: conduit::ComponentType = ty.into();
        let len = column.column.buffer.len();
        let shape = std::iter::once(len as i64)
            .chain(ty.shape.iter().map(|x| *x as i64))
            .collect();
        let op = Noxpr::parameter(
            self.builder.param_ops.len() as i64,
            NoxprTy::ArrayTy(ArrayTy {
                element_type: ty.primitive_ty.element_type(),
                shape, // FIXME
            }),
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
            let NoxprTy::ArrayTy(ty) = &p.ty else {
                unreachable!()
            };
            let dtype = nox::jax::dtype(&ty.element_type)?;
            let shape = PyTuple::new(py, ty.shape.iter().collect::<Vec<_>>());
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

#[derive(Clone, Copy, Debug)]
#[pyclass]
pub struct ComponentId {
    inner: conduit::ComponentId,
}

#[pymethods]
impl ComponentId {
    #[new]
    fn new(py: Python<'_>, inner: PyObject) -> Result<Self, Error> {
        if let Ok(s) = inner.extract::<String>(py) {
            Ok(Self {
                inner: conduit::ComponentId::new(&s),
            })
        } else if let Ok(s) = inner.extract::<u64>(py) {
            Ok(Self {
                inner: conduit::ComponentId(s),
            })
        } else {
            Err(Error::UnexpectedInput)
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentType {
    #[pyo3(get, set)]
    pub ty: PrimitiveType,
    #[pyo3(get, set)]
    pub shape: Py<PyArray1<usize>>,
}

#[pymethods]
impl ComponentType {
    #[new]
    fn new(ty: PrimitiveType, shape: numpy::PyArrayLike1<usize>) -> Self {
        let py_readonly: &PyReadonlyArray1<usize> = shape.deref();
        let py_array: &PyArray1<usize> = py_readonly.deref();
        let shape = py_array.to_owned();
        Self { ty, shape }
    }

    #[classattr]
    #[pyo3(name = "SpatialPosF64")]
    fn spatial_pos_f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![7]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "SpatialMotionF64")]
    fn spatial_motion_f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![6]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "U64")]
    fn u64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::U64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "F32")]
    fn f32(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::F32,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "F64")]
    fn f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "Edge")]
    fn edge(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![2]).to_owned();
        Self {
            ty: PrimitiveType::U64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "Quaternion")]
    fn quaternion(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![4]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }
}

impl From<ComponentType> for conduit::ComponentType {
    fn from(val: ComponentType) -> Self {
        Python::with_gil(|py| {
            let shape = val.shape.as_ref(py);
            let shape = shape.to_vec().unwrap().into();
            conduit::ComponentType {
                primitive_ty: val.ty.into(),
                shape,
            }
        })
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    F64,
    F32,
    U64,
    U32,
    U16,
    U8,
    I64,
    I32,
    I16,
    I8,
    Bool,
}

impl From<PrimitiveType> for conduit::PrimitiveTy {
    fn from(val: PrimitiveType) -> Self {
        match val {
            PrimitiveType::F64 => conduit::PrimitiveTy::F64,
            PrimitiveType::F32 => conduit::PrimitiveTy::F32,
            PrimitiveType::U64 => conduit::PrimitiveTy::U64,
            PrimitiveType::U32 => conduit::PrimitiveTy::U32,
            PrimitiveType::U16 => conduit::PrimitiveTy::U16,
            PrimitiveType::U8 => conduit::PrimitiveTy::U8,
            PrimitiveType::I64 => conduit::PrimitiveTy::I64,
            PrimitiveType::I32 => conduit::PrimitiveTy::I32,
            PrimitiveType::I16 => conduit::PrimitiveTy::I16,
            PrimitiveType::I8 => conduit::PrimitiveTy::I8,
            PrimitiveType::Bool => conduit::PrimitiveTy::Bool,
        }
    }
}

#[pyclass(subclass)]
#[derive(Default)]
pub struct WorldBuilder {
    world: World<HostStore>,
}

impl WorldBuilder {
    fn get_or_insert_archetype(
        &mut self,
        archetype: &Archetype,
    ) -> Result<&mut Table<HostStore>, Error> {
        let archetype_id = archetype.archetype_id;
        match self.world.archetypes.entry(archetype_id) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let columns = archetype
                    .component_datas
                    .iter()
                    .map(
                        |ComponentData {
                             id,
                             ty,
                             asset,
                             name,
                             ..
                         }| {
                            let name = name.to_owned().unwrap_or_default();
                            let mut col = nox_ecs::Column::<HostStore>::new(
                                HostColumn::new(ty.clone().into(), id.inner),
                                conduit::Metadata {
                                    component_id: id.inner,
                                    component_type: ty.clone().into(),
                                    tags: std::iter::once((
                                        "name".to_string(),
                                        TagValue::String(name),
                                    ))
                                    .collect(),
                                },
                            );
                            col.buffer.asset = *asset;
                            Ok((id.inner, col))
                        },
                    )
                    .collect::<Result<_, Error>>()?;
                for id in &archetype.component_ids {
                    self.world.component_map.insert(id.inner, archetype_id);
                }
                let table = Table {
                    columns,
                    entity_buffer: HostColumn::new(
                        conduit::ComponentType::u64(),
                        conduit::ComponentId::new("entity_id"),
                    ),
                    entity_map: BTreeMap::default(),
                };
                Ok(entry.insert(table))
            }
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyBufBytes {
    bytes: bytes::Bytes,
}

pub struct PyAsset {
    object: PyObject,
}

impl PyAsset {
    pub fn try_new(py: Python<'_>, object: PyObject) -> Result<Self, Error> {
        let _ = object.getattr(py, "asset_id")?;
        let _ = object.getattr(py, "bytes")?;
        Ok(Self { object })
    }
}

impl PyAsset {
    fn asset_id(&self) -> conduit::AssetId {
        Python::with_gil(|py| {
            let id: u64 = self
                .object
                .call_method0(py, "asset_id")
                .unwrap()
                .extract(py)
                .unwrap();
            conduit::AssetId(id)
        })
    }

    fn bytes(&self) -> Result<bytes::Bytes, Error> {
        Python::with_gil(|py| {
            let bytes: PyBufBytes = self.object.call_method0(py, "bytes")?.extract(py)?;
            Ok(bytes.bytes)
        })
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Args {
    Build {
        #[arg(long)]
        dir: PathBuf,
    },
    Repl {
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
    },
    Run {
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
        #[arg(long)]
        no_repl: bool,
    },
}

#[pymethods]
impl WorldBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(mut slf: PyRefMut<'_, Self>, archetype: Archetype<'_>) -> Result<Entity, Error> {
        let entity_id = EntityId {
            inner: conduit::EntityId(slf.world.entity_len),
        };

        slf.spawn_with_entity_id(archetype, entity_id.clone())?;
        let world = slf.into();
        Ok(Entity {
            id: entity_id,
            world,
        })
    }

    pub fn spawn_with_entity_id(
        &mut self,
        archetype: Archetype,
        entity_id: EntityId,
    ) -> Result<EntityId, Error> {
        let entity_id = entity_id.inner;
        let table = self.get_or_insert_archetype(&archetype)?;
        table
            .entity_map
            .insert(entity_id, table.entity_buffer.len());
        table.entity_buffer.push(entity_id.0.constant());
        for (arr, id) in archetype
            .arrays
            .iter()
            .zip(archetype.component_ids.into_iter())
        {
            let col = table
                .columns
                .get_mut(&id.inner)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let ty = col.buffer.component_type();
            let size = ty.primitive_ty.element_type().element_size_in_bytes();
            let buf = unsafe { arr.buf(size) };
            col.buffer.push_raw(buf);
        }
        self.world.entity_len += 1;
        Ok(EntityId { inner: entity_id })
    }

    fn insert_asset(&mut self, py: Python<'_>, asset: PyObject) -> Result<Handle, Error> {
        let asset = PyAsset::try_new(py, asset).unwrap();
        let inner = self
            .world
            .assets
            .insert_bytes(asset.asset_id(), asset.bytes()?);
        Ok(Handle { inner })
    }

    pub fn run(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        time_step: Option<f64>,
        client: Option<&Client>,
    ) -> Result<Option<String>, Error> {
        tracing_subscriber::fmt::init();
        // skip `python3 <name of script>`
        let args = std::env::args_os().skip(2);
        let args = Args::parse_from(args);
        match args {
            Args::Build { dir } => {
                let exec = self.build(py, sys, time_step)?.exec;
                exec.write_to_dir(dir)?;
                Ok(None)
            }
            Args::Repl { addr } => Ok(Some(addr.to_string())),
            Args::Run { addr, no_repl } => {
                let exec = self.build(py, sys, time_step)?.exec;
                let client = match client {
                    Some(c) => c.client.clone(),
                    None => nox::Client::cpu()?,
                };
                if no_repl {
                    let ppid = std::os::unix::process::parent_id();
                    spawn_tcp_server(addr, exec, &client, || {
                        let sig_err = py.check_signals().is_err();
                        let current_ppid = std::os::unix::process::parent_id();
                        let ppid_changed = ppid != current_ppid;
                        sig_err || ppid_changed
                    })?;
                    Ok(None)
                } else {
                    std::thread::spawn(move || {
                        spawn_tcp_server(addr, exec, &client, || false).unwrap()
                    });
                    Ok(Some(addr.to_string()))
                }
            }
        }
    }

    pub fn build(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        time_step: Option<f64>,
    ) -> Result<Exec, Error> {
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
    var_array = builder.var_arrays()
    xla = jax.xla_computation(lambda a: call(a, builder))(var_array)
    return xla";

        if let Some(ts) = time_step {
            let ts = Duration::from_secs_f64(ts);
            // 4ms (~240 ticks/sec) is the minimum time step
            if ts <= Duration::from_millis(4) {
                return Err(Error::InvalidTimeStep(ts));
            }
        }

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
        let time_step = time_step.map(Duration::from_secs_f64);
        let exec = nox_ecs::WorldExec {
            world: SharedWorld {
                host: builder.world,
                ..Default::default()
            },
            tick_exec: nox_ecs::Exec {
                exec: Default::default(),
                metadata: nox_ecs::ExecMetadata {
                    time_step,
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

#[derive(Clone)]
#[pyclass(get_all, set_all)]
pub struct ComponentData {
    id: ComponentId,
    ty: ComponentType,
    asset: bool,
    from_expr: PyObject,
    name: Option<String>,
}

#[pymethods]
impl ComponentData {
    #[new]
    pub fn new(
        id: ComponentId,
        ty: ComponentType,
        asset: bool,
        from_expr: PyObject,
        name: Option<String>,
    ) -> Self {
        Self {
            id,
            ty,
            asset,
            from_expr,
            name,
        }
    }

    pub fn to_metadata(&self) -> Metadata {
        Metadata::new(self.id, self.ty.clone(), self.name.clone())
    }
}

pub struct Archetype<'py> {
    component_datas: Vec<ComponentData>,
    component_ids: Vec<ComponentId>,
    arrays: Vec<&'py PyUntypedArray>,
    archetype_id: ArchetypeId,
}

impl<'s> FromPyObject<'s> for Archetype<'s> {
    fn extract(archetype: &'s PyAny) -> PyResult<Self> {
        let archetype_id = archetype.call_method0("archetype_id")?.extract::<u64>()?;
        let archetype_id = ArchetypeId::new(archetype_id.into());
        let component_datas = archetype
            .call_method0("component_data")?
            .extract::<Vec<ComponentData>>()?;
        let component_ids = component_datas
            .iter()
            .map(|data| data.id)
            .collect::<Vec<_>>();
        let arrays = archetype.call_method0("arrays")?;
        let arrays = arrays.extract::<Vec<&numpy::PyUntypedArray>>()?;
        Ok(Self {
            component_datas,
            component_ids,
            arrays,
            archetype_id,
        })
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
    inner: conduit::EntityId,
}

#[pymethods]
impl EntityId {
    #[new]
    fn new(id: u64) -> Self {
        EntityId { inner: id.into() }
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.0.fmt(f)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Entity {
    id: EntityId,
    world: Py<WorldBuilder>,
}

#[pymethods]
impl Entity {
    pub fn id(&self) -> EntityId {
        self.id.clone()
    }

    pub fn insert(&mut self, py: Python<'_>, archetype: Archetype<'_>) -> Result<Self, Error> {
        let mut world = self.world.borrow_mut(py);
        world.spawn_with_entity_id(archetype, self.id())?;
        Ok(self.clone())
    }

    pub fn metadata(&mut self, py: Python<'_>, metadata: EntityMetadata) -> Self {
        let mut world = self.world.borrow_mut(py);
        let metadata = world.world.insert_asset(metadata.inner);
        world.world.spawn_with_id(metadata, self.id.inner);
        self.clone()
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
    #[error("invalid time step: {0:?}")]
    InvalidTimeStep(std::time::Duration),
    #[error("conduit error {0}")]
    Conduit(#[from] conduit::Error),
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

#[derive(Clone)]
#[pyclass]
pub struct Handle {
    inner: nox_ecs::Handle<()>,
}

#[pymethods]
impl Handle {
    fn asarray(&self) -> Result<PyObject, Error> {
        Ok(nox::NoxprScalarExt::constant(self.inner.id).to_jax()?)
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = nox::NoxprScalarExt::constant(self.inner.id).to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, _jax: PyObject) -> Self {
        todo!()
    }

    #[staticmethod]
    fn from_array(_arr: PyObject) -> Self {
        todo!()
    }
}
#[pyclass]
#[derive(Clone)]
pub struct Pbr {
    inner: conduit::well_known::Pbr,
}

#[pymethods]
impl Pbr {
    #[new]
    fn new(mesh: Mesh, material: Material) -> Self {
        Self {
            inner: conduit::well_known::Pbr::Bundle {
                mesh: mesh.inner,
                material: material.inner,
            },
        }
    }

    #[staticmethod]
    fn from_url(url: String) -> Result<Self, Error> {
        let inner = conduit::well_known::Pbr::Url(url);
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_path(path: PathBuf) -> Result<Self, Error> {
        let inner = conduit::well_known::Pbr::path(path)?;
        Ok(Self { inner })
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
#[derive(Clone)]
pub struct Mesh {
    inner: conduit::well_known::Mesh,
}

#[pymethods]
impl Mesh {
    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[staticmethod]
    pub fn cuboid(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: conduit::well_known::Mesh::cuboid(x, y, z),
        }
    }

    #[staticmethod]
    pub fn sphere(radius: f32) -> Self {
        Self {
            inner: conduit::well_known::Mesh::sphere(radius, 36, 18),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Material {
    inner: conduit::well_known::Material,
}

#[pymethods]
impl Material {
    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[staticmethod]
    fn color(r: f32, g: f32, b: f32) -> Self {
        Material {
            inner: conduit::well_known::Material::color(r, g, b),
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Color {
    inner: conduit::well_known::Color,
}

#[pymethods]
impl Color {
    #[new]
    fn new(r: f32, g: f32, b: f32) -> Self {
        Color {
            inner: conduit::well_known::Color::rgb(r, g, b),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct EntityMetadata {
    inner: conduit::well_known::EntityMetadata,
}

#[pymethods]
impl EntityMetadata {
    #[new]
    fn new(name: String, color: Option<Color>) -> Self {
        let color = color.unwrap_or(Color::new(1.0, 1.0, 1.0));
        Self {
            inner: conduit::well_known::EntityMetadata {
                name,
                color: color.inner,
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
#[derive(Clone)]
pub struct QueryInner {
    query: nox_ecs::Query<()>,
    metadata: Vec<Metadata>,
}

#[pymethods]
impl QueryInner {
    #[staticmethod]
    fn from_builder(
        builder: &mut PipelineBuilder,
        component_ids: Vec<ComponentId>,
    ) -> Result<QueryInner, Error> {
        let metadata = component_ids
            .iter()
            .map(|id| {
                builder
                    .builder
                    .world
                    .column_by_id(id.inner)
                    .map(|c| Metadata {
                        inner: Arc::new(c.column.metadata.clone()),
                    })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::NoxEcs(nox_ecs::Error::ComponentNotFound))?;
        let query = component_ids
            .iter()
            .copied()
            .map(|id| {
                builder
                    .builder
                    .vars
                    .get(&id.inner)
                    .ok_or(nox_ecs::Error::ComponentNotFound)
            })
            .try_fold(None, |mut query, a| {
                let a = a?;
                if query.is_some() {
                    query = Some(join_many(query.take().unwrap(), &*a.borrow()));
                } else {
                    let a = a.borrow().clone();
                    let q: nox_ecs::Query<()> = a.into();
                    query = Some(q);
                }
                Ok::<_, Error>(query)
            })?
            .expect("query must not be empty");
        Ok(Self { query, metadata })
    }

    fn map(&self, new_buf: PyObject, metadata: Metadata) -> QueryInner {
        let expr = Noxpr::jax(new_buf);
        QueryInner {
            query: nox_ecs::Query {
                exprs: vec![expr],
                entity_map: self.query.entity_map.clone(),
                len: self.query.len,
                phantom_data: PhantomData,
            },
            metadata: vec![metadata],
        }
    }

    fn arrays(&self) -> Result<Vec<PyObject>, Error> {
        self.query
            .exprs
            .iter()
            .map(|e| e.to_jax().map_err(Error::from))
            .collect()
    }

    fn insert_into_builder(&self, builder: &mut PipelineBuilder) {
        self.query.insert_into_builder_erased(
            &mut builder.builder,
            self.metadata.iter().map(|m| m.inner.component_id),
        );
    }

    fn join_query(&self, other: &QueryInner) -> QueryInner {
        let query = join_query(self.query.clone(), other.query.clone());
        let metadata = self
            .metadata
            .iter()
            .cloned()
            .chain(other.metadata.iter().cloned())
            .collect();
        QueryInner { query, metadata }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Metadata {
    inner: Arc<conduit::Metadata>,
}

#[pymethods]
impl Metadata {
    #[new]
    fn new(component_id: ComponentId, ty: ComponentType, name: Option<String>) -> Self {
        let inner = Arc::new(conduit::Metadata {
            component_id: component_id.inner,
            component_type: ty.into(),
            tags: name
                .into_iter()
                .map(|n| ("name".to_string(), TagValue::String(n)))
                .collect(),
        });
        Metadata { inner }
    }
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
    m.add_class::<ComponentData>()?;
    m.add_class::<conduit_client::Conduit>()?;
    m.add_function(wrap_pyfunction!(six_dof, m)?)?;
    Ok(())
}
