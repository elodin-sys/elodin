use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use nox_ecs::{
    elodin_conduit::{self},
    join_many,
    nox::{self, jax::JaxTracer, ArrayTy, Noxpr, NoxprNode, ScalarExt},
    ArchetypeId, ComponentArray, HostColumn, HostStore, Query, Table, World,
};
use numpy::{ndarray::ArrayViewD, PyArray, PyUntypedArray};
use parking_lot::Mutex;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyTuple},
};

#[pyclass]
#[derive(Clone)]
struct PipelineBuilder {
    builder: Arc<Mutex<nox_ecs::PipelineBuilder>>,
}

#[pymethods]
impl PipelineBuilder {
    fn get_var(&self, id: ComponentId) -> Result<(ComponentArrayMetadata, PyObject), Error> {
        let builder = self.builder.lock();
        let var = builder
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
        &self,
        id: ComponentId,
        metadata: &ComponentArrayMetadata,
        value: PyObject,
    ) -> Result<(), Error> {
        let builder = self.builder.lock();
        let var = builder
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
        let mut builder = self.builder.lock();
        let id = id.inner;
        if builder.param_ids.contains(&id) {
            return Ok(());
        }
        let column = builder
            .world
            .column_by_id(id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let ty: elodin_conduit::ComponentType = ty.into();
        let len = column.column.buffer.len();
        let shape = std::iter::once(len as i64)
            .chain(ty.dims().iter().copied())
            .collect();
        let op = Noxpr::parameter(
            builder.param_ops.len() as i64,
            ArrayTy {
                element_type: ty.element_type(),
                shape, // FIXME
            },
            format!("{:?}::{}", id, builder.param_ops.len()),
        );
        builder.param_ops.push(op.clone());
        builder.param_ids.push(id);

        Ok(())
    }

    fn var_arrays(&self, py: Python<'_>) -> Result<Vec<PyObject>, Error> {
        let builder = self.builder.lock();
        let mut res = vec![];
        for p in &builder.param_ops {
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
        let mut builder = self.builder.lock();
        assert_eq!(args.len(), builder.param_ids.len());
        let nox_ecs::PipelineBuilder {
            vars,
            world,
            param_ids,
            ..
        } = builder.deref_mut();
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
        let builder = self.builder.lock();
        let vars = builder
            .vars
            .values()
            .map(|var| {
                let var = var.borrow();
                let NoxprNode::Jax(buf) = var.buffer().deref() else {
                    todo!()
                };
                buf.clone()
            })
            .collect::<Vec<_>>();
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
        if let Some(id) = self
            .world
            .archetype_id_map
            .get(&ArchetypeId::Raw(archetype_id))
        {
            Ok(&mut self.world.archetypes[*id])
        } else {
            self.insert_archetype(py, archetype)
        }
    }
    fn insert_archetype(
        &mut self,
        py: Python<'_>,
        archetype: &PyObject,
    ) -> Result<&mut Table<HostStore>, Error> {
        let archetype_id = archetype
            .call_method0(py, "archetype_id")?
            .extract::<u64>(py)?;
        let archetype_id = ArchetypeId::Raw(archetype_id);
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
                    nox_ecs::Column::<HostStore>::new(HostColumn::from_ty(ty.into())),
                ))
            })
            .collect::<Result<_, Error>>()?;
        let archetype_index = self.world.archetypes.len();
        self.world.archetypes.push(Table {
            columns,
            entity_buffer: HostColumn::from_ty(elodin_conduit::ComponentType::U64),
            entity_map: BTreeMap::default(),
        });
        for id in component_ids {
            self.world.component_map.insert(id, archetype_index);
        }
        self.world
            .archetype_id_map
            .insert(archetype_id, archetype_index);
        Ok(&mut self.world.archetypes[archetype_index])
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

    pub fn build(&mut self, py: Python<'_>, sys: PyObject, client: &Client) -> Result<Exec, Error> {
        let world = std::mem::take(&mut self.world);
        let builder = nox_ecs::PipelineBuilder::from_world(world);
        let builder = PipelineBuilder {
            builder: Arc::new(Mutex::new(builder)),
        };
        let py_code = "import jax
def build_expr(builder, sys):
    sys.init(builder)
    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)
        return builder.ret_vars()
    xla = jax.xla_computation(lambda a: call(a, builder))(builder.var_arrays())
    return (builder, xla)";

        let fun: Py<PyAny> = PyModule::from_code(py, py_code, "", "")?
            .getattr("build_expr")?
            .into();
        let (builder, comp) = fun
            .call1(py, (builder, sys))?
            .extract::<(PyObject, PyObject)>(py)?;
        let builder = builder.extract::<PipelineBuilder>(py)?;
        let comp = comp.call_method0(py, "as_serialized_hlo_module_proto")?;
        let comp = comp
            .downcast::<PyBytes>(py)
            .map_err(|_| Error::HloModuleNotBytes)?;
        let hlo_module = nox::xla::HloModuleProto::parse_binary(comp.as_bytes())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let comp = hlo_module.computation();
        let exec = client.client.0.compile(&comp).map_err(|err| {
            PyValueError::new_err(format!("failed to compile computation {:?}", err))
        })?;
        let builder = std::mem::take(&mut *builder.builder.lock());

        let ret_ids = builder.vars.keys().copied().collect::<Vec<_>>();
        let world = builder
            .world
            .copy_to_client(&client.client)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to copy world to client {:?}", err))
            })?;
        let exec = nox_ecs::Exec {
            client_world: world,
            arg_ids: builder.param_ids,
            ret_ids,
            exec,
            host_world: builder.world,
            loaded_components: HashSet::default(),
            dirty_components: HashSet::default(),
        };

        Ok(Exec { exec })
    }
}

#[pyclass]
pub struct Exec {
    exec: nox_ecs::Exec,
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

#[pymodule]
pub fn nox_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ComponentType>()?;
    m.add_class::<ComponentId>()?;
    m.add_class::<PipelineBuilder>()?;
    m.add_class::<WorldBuilder>()?;
    m.add_class::<EntityId>()?;
    m.add_class::<Client>()?;
    m.add_class::<ComponentArrayMetadata>()?;
    Ok(())
}
