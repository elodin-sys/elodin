//! Prepared component handles and `pre_commit` hooks, which read and write world columns between
//! the tick and the DB commit instead of going through the DB.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use elodin_db::DB;
use impeller::types::{ComponentId, PacketId, PrimType, Timestamp};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use stellarator::util::CancelToken;

use crate::World;
use crate::entity::EntityId;
use crate::native_hook::HookReport;
use crate::witness::Channels;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
}

/// One component of a fixed list of entities, resolved once. A callback sees its rows, in the
/// order the entities were given, through an array refilled from the world when it is viewed.
#[pyclass(frozen)]
pub struct ComponentHandle {
    component_id: ComponentId,
    #[pyo3(get)]
    name: String,
    access: Access,
    entities: Vec<EntityId>,
    /// Row of each entity in the component's column.
    pub(crate) rows: Vec<usize>,
    pub(crate) row_bytes: usize,
    #[pyo3(get)]
    topology_generation: u64,
    array: Py<PyUntypedArray>,
}

#[pymethods]
impl ComponentHandle {
    #[getter(component_id)]
    fn py_component_id(&self) -> u64 {
        self.component_id.0
    }

    #[getter]
    fn access(&self) -> &'static str {
        match self.access {
            Access::Read => "read",
            Access::Write => "write",
        }
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.array.bind(py).getattr("dtype").map(Bound::unbind)
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.array.bind(py).getattr("shape").map(Bound::unbind)
    }

    #[getter]
    fn rows(&self) -> Vec<EntityId> {
        self.entities.clone()
    }
}

pub(crate) fn numpy_dtype(prim_type: PrimType) -> &'static str {
    match prim_type {
        PrimType::U8 => "uint8",
        PrimType::U16 => "uint16",
        PrimType::U32 => "uint32",
        PrimType::U64 => "uint64",
        PrimType::I8 => "int8",
        PrimType::I16 => "int16",
        PrimType::I32 => "int32",
        PrimType::I64 => "int64",
        PrimType::Bool => "bool",
        PrimType::F32 => "float32",
        PrimType::F64 => "float64",
    }
}

fn array_data(array: &Bound<'_, PyUntypedArray>) -> *mut u8 {
    // Safety: a bound array's object is alive and initialized.
    unsafe { (*array.as_array_ptr()).data.cast() }
}

impl ComponentHandle {
    /// Resolves `entities`' rows of component `name`. Missing rows, and rows listed twice for
    /// writing, are errors.
    pub fn prepare(
        py: Python<'_>,
        world: &World,
        name: &str,
        entities: Vec<EntityId>,
        access: &str,
        topology_generation: u64,
    ) -> PyResult<Self> {
        let access = match access {
            "read" => Access::Read,
            "write" => Access::Write,
            _ => return Err(PyValueError::new_err("access must be 'read' or 'write'")),
        };
        let component_id = ComponentId::new(name);
        let (Some((schema, _)), Some(column)) = (
            world.metadata.component_map.get(&component_id),
            world.host.get(&component_id),
        ) else {
            return Err(PyValueError::new_err(format!(
                "no component named {name:?}"
            )));
        };
        let row_of: HashMap<u64, usize> = column
            .entity_ids
            .as_chunks::<8>()
            .0
            .iter()
            .enumerate()
            .map(|(row, id)| (u64::from_le_bytes(*id), row))
            .collect();
        let mut rows = Vec::with_capacity(entities.len());
        let mut written = HashSet::new();
        for entity in &entities {
            let id = entity.inner.0;
            let Some(&row) = row_of.get(&id) else {
                return Err(PyValueError::new_err(format!(
                    "entity {id} has no {name:?} component"
                )));
            };
            if access == Access::Write && !written.insert(row) {
                return Err(PyValueError::new_err(format!(
                    "entity {id} is listed twice for writing {name:?}"
                )));
            }
            rows.push(row);
        }
        let shape: Vec<usize> = std::iter::once(rows.len())
            .chain(schema.shape().iter().map(|&dim| dim as usize))
            .collect();
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("zeros", (shape, numpy_dtype(schema.prim_type)))?;
        if access == Access::Read {
            array.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
        }
        Ok(Self {
            component_id,
            name: name.to_string(),
            access,
            entities,
            rows,
            row_bytes: schema.size(),
            topology_generation,
            array: array.downcast_into::<PyUntypedArray>()?.unbind(),
        })
    }

    pub fn component_id(&self) -> ComponentId {
        self.component_id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn entities(&self) -> &[EntityId] {
        &self.entities
    }

    pub fn topology_generation(&self) -> u64 {
        self.topology_generation
    }

    pub fn writes(&self) -> bool {
        self.access == Access::Write
    }

    /// Whether `self` resolves the same rows of the same component as `other`, with the same
    /// access.
    pub fn replaces(&self, other: &ComponentHandle) -> bool {
        self.component_id == other.component_id
            && self.rows == other.rows
            && self.access == other.access
    }

    /// Copies the rows from the world into the array; returns the bytes copied.
    fn fill(&self, py: Python<'_>, world: &World) -> usize {
        let column = &world.host[&self.component_id].buffer;
        let array = array_data(self.array.bind(py));
        for (i, &row) in self.rows.iter().enumerate() {
            let src = &column[row * self.row_bytes..][..self.row_bytes];
            // Safety: the array was created with one `row_bytes` row per entry of `rows`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    array.add(i * self.row_bytes),
                    src.len(),
                )
            };
        }
        self.rows.len() * self.row_bytes
    }

    /// Copies the array into the world's rows; returns the bytes copied.
    fn store(&self, py: Python<'_>, world: &mut World) -> usize {
        let column = &mut world
            .host
            .get_mut(&self.component_id)
            .expect("prepared column exists")
            .buffer;
        let array = array_data(self.array.bind(py));
        for (i, &row) in self.rows.iter().enumerate() {
            let dst = &mut column[row * self.row_bytes..][..self.row_bytes];
            // Safety: the array was created with one `row_bytes` row per entry of `rows`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    array.add(i * self.row_bytes),
                    dst.as_mut_ptr(),
                    dst.len(),
                )
            };
        }
        self.rows.len() * self.row_bytes
    }
}

/// Handles over the same number of rows, viewed together.
#[pyclass(frozen)]
pub struct BatchHandle {
    /// Unique per process, so witness channels and native hooks can name the batch.
    pub(crate) id: u64,
    pub(crate) fields: Vec<(String, Py<ComponentHandle>)>,
}

impl BatchHandle {
    pub fn rows(&self) -> usize {
        self.fields.first().map_or(0, |(_, h)| h.get().rows.len())
    }

    pub fn new(fields: Vec<(String, Py<ComponentHandle>)>) -> PyResult<Self> {
        let rows: HashSet<usize> = fields.iter().map(|(_, h)| h.get().rows.len()).collect();
        if rows.len() > 1 {
            let counts: Vec<String> = fields
                .iter()
                .map(|(name, h)| format!("{name}: {}", h.get().rows.len()))
                .collect();
            return Err(PyValueError::new_err(format!(
                "batch fields must have the same number of rows ({})",
                counts.join(", ")
            )));
        }
        Ok(Self {
            id: crate::witness::next_batch_id(),
            fields,
        })
    }
}

#[pymethods]
impl BatchHandle {
    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        for (name, handle) in &self.fields {
            fields.set_item(name, handle.clone_ref(py))?;
        }
        Ok(fields)
    }
}

/// A batch's views by field name, as attributes or items.
#[pyclass]
pub struct BatchView {
    fields: Py<PyDict>,
}

#[pymethods]
impl BatchView {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let field = self.fields.bind(py).get_item(name)?;
        field
            .map(Bound::unbind)
            .ok_or_else(|| PyAttributeError::new_err(name.to_string()))
    }

    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let field = self.fields.bind(py).get_item(name)?;
        field
            .map(Bound::unbind)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(&self, _exc_type: PyObject, _exc: PyObject, _traceback: PyObject) -> bool {
        false
    }
}

/// How `WorldStepContext.read_at` picks a sample.
#[derive(Clone, Copy, PartialEq)]
enum Selection {
    /// Only a sample at the requested timestamp.
    Exact,
    /// The newest sample at or before it.
    AtOrBefore,
}

impl Selection {
    fn parse(selection: &str) -> PyResult<Self> {
        match selection {
            "exact" => Ok(Self::Exact),
            "at_or_before" => Ok(Self::AtOrBefore),
            _ => Err(PyValueError::new_err(
                "selection must be 'exact' or 'at_or_before'",
            )),
        }
    }

    fn accepts(self, selected: Timestamp, requested: Timestamp) -> bool {
        self == Self::AtOrBefore || selected == requested
    }
}

/// An `entity.component` pair of the world, which has DB history unless it is transient.
struct Pair {
    id: ComponentId,
    dtype: &'static str,
    shape: Vec<usize>,
    size: usize,
    external_control: bool,
}

impl Pair {
    fn resolve(world: &World, name: &str) -> PyResult<Self> {
        let found = world.host.iter().find_map(|(component_id, column)| {
            let (schema, metadata) = world.metadata.component_map.get(component_id)?;
            let entity = name
                .strip_suffix(metadata.name.as_str())?
                .strip_suffix('.')?;
            let mut ids = column.entity_ids.as_chunks::<8>().0.iter();
            ids.any(|id| {
                let id = impeller::types::EntityId(u64::from_le_bytes(*id));
                world
                    .metadata
                    .entity_metadata
                    .get(&id)
                    .is_some_and(|entity_metadata| entity_metadata.name == entity)
            })
            .then_some((schema, metadata))
        });
        let Some((schema, metadata)) = found else {
            return Err(PyValueError::new_err(format!(
                "no entity component named {name:?}"
            )));
        };
        if metadata.is_transient() {
            return Err(PyValueError::new_err(format!(
                "{name:?} is transient, so it has no DB history"
            )));
        }
        Ok(Self {
            id: ComponentId::new(name),
            dtype: numpy_dtype(schema.prim_type),
            shape: schema.shape().iter().map(|&dim| dim as usize).collect(),
            size: schema.size(),
            external_control: metadata
                .metadata
                .get("external_control")
                .is_some_and(|value| value == "true"),
        })
    }
}

enum HistorySource {
    Component(Pair),
    Message(PacketId),
}

/// The committed DB history of an entity's component or of a message log, resolved once, which
/// `pre_commit` callbacks read with `WorldStepContext.read_at`.
#[pyclass(frozen)]
pub struct HistoryHandle {
    #[pyo3(get)]
    name: String,
    source: HistorySource,
}

impl HistoryHandle {
    pub fn component(world: &World, name: &str) -> PyResult<Self> {
        Ok(Self {
            name: name.to_string(),
            source: HistorySource::Component(Pair::resolve(world, name)?),
        })
    }

    pub fn message(name: &str) -> Self {
        Self {
            name: name.to_string(),
            source: HistorySource::Message(impeller::types::msg_id(name)),
        }
    }

    /// The committed sample `selection` picks at `timestamp`, with its own timestamp.
    fn select(
        &self,
        db: &DB,
        timestamp: Timestamp,
        selection: Selection,
    ) -> Option<(Timestamp, Vec<u8>)> {
        db.with_state(|state| match &self.source {
            HistorySource::Component(pair) => {
                let series = &state.get_component(pair.id)?.time_series;
                let (selected, value) = series.get_at_or_before(timestamp)?;
                selection
                    .accepts(selected, timestamp)
                    .then(|| (selected, value.to_vec()))
            }
            HistorySource::Message(id) => {
                let log = state.get_msg_log(*id)?;
                let index = log.timestamps().partition_point(|&t| t <= timestamp);
                let (selected, value) = log.get_index(index.checked_sub(1)?)?;
                selection
                    .accepts(selected, timestamp)
                    .then(|| (selected, value.into_owned()))
            }
        })
    }
}

/// Components, each marked `external_control`, that `pre_commit` callbacks append to at their
/// own timestamps with `WorldStepContext.defer_write`.
#[pyclass(frozen)]
pub struct TimestampedWriter {
    /// Each value's key, and the name and pair it is appended to.
    fields: Vec<(String, Arc<str>, Pair)>,
}

impl TimestampedWriter {
    pub fn new(world: &World, fields: Vec<(String, String)>) -> PyResult<Self> {
        let fields = fields
            .into_iter()
            .map(|(key, name)| {
                let pair = Pair::resolve(world, &name)?;
                if !pair.external_control {
                    return Err(PyValueError::new_err(format!(
                        "{name:?} must be marked external_control=\"true\", so the simulation never writes it"
                    )));
                }
                Ok((key, Arc::from(name), pair))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self { fields })
    }
}

/// Values a `pre_commit` callback appends at `timestamp` once the cycle's world commit landed.
#[derive(Debug)]
pub struct DeferredWrite {
    pub timestamp: Timestamp,
    /// The name, pair and value of each component written.
    pub rows: Vec<(Arc<str>, ComponentId, Vec<u8>)>,
}

/// Passed to a `pre_commit` callback. Views read the world's current values, and writes to a
/// write handle's view reach the world, and the same commit, when the callback returns. History
/// reads see only committed DB samples, and deferred writes land after the commit.
#[pyclass(unsendable)]
pub struct WorldStepContext {
    #[pyo3(get)]
    tick: u64,
    #[pyo3(get)]
    timestamp: i64,
    /// Null once the callback returns.
    world: Cell<*mut World>,
    channels: Cell<*mut Channels>,
    viewed: RefCell<Vec<Py<ComponentHandle>>>,
    recorded: RefCell<HashSet<ComponentId>>,
    deferred: RefCell<Vec<DeferredWrite>>,
    /// None in a replay.
    db: Option<Arc<DB>>,
    counters: Arc<CallbackCounters>,
    recipe_cancel_token: Option<CancelToken>,
}

#[pymethods]
impl WorldStepContext {
    /// The handle's rows as an array, read-only for a read handle.
    fn view(&self, py: Python<'_>, handle: Py<ComponentHandle>) -> PyResult<PyObject> {
        let world = self.world.get();
        if world.is_null() {
            return Err(PyRuntimeError::new_err(
                "views end when the pre_commit callback returns",
            ));
        }
        let mut viewed = self.viewed.borrow_mut();
        if !viewed.iter().any(|seen| seen.is(&handle)) {
            // Safety: `world` is non-null only while the tick loop lends it to this callback.
            let bytes = handle.get().fill(py, unsafe { &*world });
            self.counters.views.fetch_add(1, Ordering::Relaxed);
            self.counters
                .bytes_copied
                .fetch_add(bytes as u64, Ordering::Relaxed);
            viewed.push(handle.clone_ref(py));
        }
        Ok(handle.get().array.clone_ref(py).into_any())
    }

    /// Views every field of `batch`.
    fn batch(&self, py: Python<'_>, batch: &BatchHandle) -> PyResult<BatchView> {
        let fields = PyDict::new(py);
        for (name, handle) in &batch.fields {
            fields.set_item(name, self.view(py, handle.clone_ref(py))?)?;
        }
        Ok(BatchView {
            fields: fields.unbind(),
        })
    }

    /// Records the handle's component, recorded with `Recording.manual()`, at this commit.
    fn record(&self, handle: &ComponentHandle) -> PyResult<()> {
        let world = self.world.get();
        if world.is_null() {
            return Err(PyRuntimeError::new_err(
                "records end when the pre_commit callback returns",
            ));
        }
        // Safety: as in `view`.
        let metadata = &unsafe { &*world }.metadata.component_map[&handle.component_id].1;
        if metadata
            .metadata
            .get("recording")
            .is_none_or(|r| r != "manual")
        {
            return Err(PyValueError::new_err(format!(
                "{:?} is not recorded manually",
                handle.name
            )));
        }
        self.recorded.borrow_mut().insert(handle.component_id);
        Ok(())
    }

    /// The committed sample of `history` that `selection` picks at `timestamp`: its timestamp and
    /// a copy of its value, or `None`.
    fn read_at(
        &self,
        py: Python<'_>,
        history: &HistoryHandle,
        timestamp: i64,
        selection: &str,
    ) -> PyResult<Option<(i64, PyObject)>> {
        self.ensure_open("history reads")?;
        let selection = Selection::parse(selection)?;
        let Some(db) = &self.db else {
            return Err(PyRuntimeError::new_err(
                "history reads need the DB, which a replay does not have",
            ));
        };
        self.counters
            .history_queries
            .fetch_add(1, Ordering::Relaxed);
        let Some((selected, value)) = history.select(db, Timestamp(timestamp), selection) else {
            self.counters.history_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        self.counters
            .history_bytes
            .fetch_add(value.len() as u64, Ordering::Relaxed);
        let array = PyArray1::from_vec(py, value).into_any();
        let array = match &history.source {
            HistorySource::Component(pair) => array
                .call_method1("view", (pair.dtype,))?
                .call_method1("reshape", (pair.shape.clone(),))?,
            HistorySource::Message(_) => array,
        };
        Ok(Some((selected.0, array.unbind())))
    }

    /// Appends `values`, one per key of `writer`, at `timestamp` once this cycle's world commit
    /// lands. Writes keep their call order, across callbacks too.
    fn defer_write(
        &self,
        py: Python<'_>,
        writer: &TimestampedWriter,
        timestamp: i64,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        self.ensure_open("deferred writes")?;
        if values.len() != writer.fields.len() {
            let keys: Vec<&str> = writer.fields.iter().map(|(key, ..)| key.as_str()).collect();
            return Err(PyValueError::new_err(format!(
                "values must have exactly the keys {keys:?}"
            )));
        }
        let numpy = py.import("numpy")?;
        let rows = writer
            .fields
            .iter()
            .map(|(key, name, pair)| {
                let value = values
                    .get_item(key)?
                    .ok_or_else(|| PyKeyError::new_err(key.clone()))?;
                let array = numpy.call_method1("ascontiguousarray", (value, pair.dtype))?;
                let bytes = array.call_method0("tobytes")?;
                let bytes = bytes.downcast::<PyBytes>()?.as_bytes();
                if bytes.len() != pair.size {
                    return Err(PyValueError::new_err(format!(
                        "{key:?} must have the shape {:?}",
                        pair.shape
                    )));
                }
                Ok((name.clone(), pair.id, bytes.to_vec()))
            })
            .collect::<PyResult<_>>()?;
        self.deferred.borrow_mut().push(DeferredWrite {
            timestamp: Timestamp(timestamp),
            rows,
        });
        Ok(())
    }

    /// Fills witness channel `name` with the bytes of `data`: every row of its batch, in order.
    /// Does nothing when the run records no witness.
    fn witness_bytes(&self, py: Python<'_>, name: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_open("witness channels")?;
        // Safety: `channels` is non-null while the world is, only while the tick loop lends it.
        let channels = unsafe { &mut *self.channels.get() };
        let Some(index) = channels.find(name).map_err(PyKeyError::new_err)? else {
            return Ok(());
        };
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("ascontiguousarray", (data,))?;
        let array = array.downcast_into::<PyUntypedArray>()?;
        let len = array.shape().iter().product::<usize>() * array.dtype().itemsize();
        // Safety: a C-contiguous array holds `len` bytes at its data pointer.
        let bytes = unsafe { std::slice::from_raw_parts(array_data(&array), len) };
        channels.write(index, bytes).map_err(PyValueError::new_err)
    }

    /// The recorded bytes of external channel `name`, as `uint8` rows, in a replay that has them;
    /// otherwise `None`.
    fn witness_replay(&self, py: Python<'_>, name: &str) -> PyResult<Option<PyObject>> {
        self.ensure_open("witness channels")?;
        // Safety: as in `witness_bytes`.
        let channels = unsafe { &*self.channels.get() };
        let Some(index) = channels.find(name).map_err(PyKeyError::new_err)? else {
            return Ok(None);
        };
        let Some(bytes) = channels.replayed(index) else {
            return Ok(None);
        };
        let def = &channels.defs()[index];
        let array = PyArray1::from_slice(py, bytes)
            .call_method1("reshape", ((def.rows, def.row_bytes),))?;
        Ok(Some(array.unbind()))
    }

    /// Gracefully terminate all s10-managed recipes; see `StepContext.stop_recipes`.
    fn stop_recipes(&self) {
        if let Some(token) = &self.recipe_cancel_token {
            token.cancel();
        }
    }
}

impl WorldStepContext {
    fn ensure_open(&self, what: &str) -> PyResult<()> {
        if self.world.get().is_null() {
            return Err(PyRuntimeError::new_err(format!(
                "{what} end when the pre_commit callback returns"
            )));
        }
        Ok(())
    }

    /// Ends the callback: copies the written views into `world`.
    fn finish(&self, py: Python<'_>, world: &mut World) -> HookWrites {
        self.world.set(std::ptr::null_mut());
        self.channels.set(std::ptr::null_mut());
        let mut written = HashSet::new();
        for handle in self.viewed.borrow_mut().drain(..) {
            let handle = handle.get();
            if handle.writes() {
                let bytes = handle.store(py, world);
                self.counters
                    .bytes_copied
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                written.insert(handle.component_id);
            }
        }
        HookWrites {
            written,
            recorded: self.recorded.take(),
            deferred: self.deferred.take(),
        }
    }
}

/// What a `pre_commit` hook did for the commit that follows it.
#[derive(Debug, Default)]
pub struct HookWrites {
    /// Components it wrote, which the commit then compares.
    pub written: HashSet<ComponentId>,
    /// Manually recorded components it asked the commit to record.
    pub recorded: HashSet<ComponentId>,
    /// Writes to append after the commit, in call order.
    pub deferred: Vec<DeferredWrite>,
}

/// Runs after the tick and before the commit.
pub trait PreCommit {
    /// Names the hook in failures.
    fn name(&self) -> &str;

    /// Runs the hook for the cycle ending at `tick`. It may fill `channels`.
    fn step(
        &mut self,
        tick: u64,
        timestamp: Timestamp,
        world: &mut World,
        channels: &mut Channels,
    ) -> Result<HookWrites, String>;

    /// The rows each call owns, for a hook that splits its rows across calls.
    fn partitions(&self) -> Vec<(u64, u64)> {
        Vec::new()
    }

    /// What the hook measured, for the run receipt.
    fn report(&self) -> Option<HookReport> {
        None
    }

    /// Forgets what the hook measured so far, as the loop does when warmup ends.
    fn reset_report(&mut self) {}
}

/// A Python `pre_commit(tick, ctx: WorldStepContext)` callback.
pub struct PythonPreCommit {
    pub func: PyObject,
    pub name: String,
    /// None in a replay.
    pub db: Option<Arc<DB>>,
    pub counters: Arc<CallbackCounters>,
    pub recipe_cancel_token: Option<CancelToken>,
}

impl PreCommit for PythonPreCommit {
    fn name(&self) -> &str {
        &self.name
    }

    fn step(
        &mut self,
        tick: u64,
        timestamp: Timestamp,
        world: &mut World,
        channels: &mut Channels,
    ) -> Result<HookWrites, String> {
        Python::with_gil(|py| {
            let ctx = WorldStepContext {
                tick,
                timestamp: timestamp.0,
                world: Cell::new(world),
                channels: Cell::new(channels),
                viewed: RefCell::default(),
                recorded: RefCell::default(),
                deferred: RefCell::default(),
                db: self.db.clone(),
                counters: self.counters.clone(),
                recipe_cancel_token: self.recipe_cancel_token.clone(),
            };
            let ctx = Py::new(py, ctx).map_err(|err| err.to_string())?;
            let result = self.func.call1(py, (tick, ctx.clone_ref(py)));
            let written = ctx.borrow(py).finish(py, world);
            result.map(|_| written).map_err(|err| {
                err.print(py);
                err.to_string()
            })
        })
    }
}

/// What the step callbacks did, for the run receipt.
#[derive(Default)]
pub struct CallbackCounters {
    pub prepared_handles: AtomicU64,
    pub validation_failures: AtomicU64,
    pub views: AtomicU64,
    pub bytes_copied: AtomicU64,
    /// Calls on a `StepContext`, which goes through the DB.
    pub legacy_operations: AtomicU64,
    pub history_queries: AtomicU64,
    /// Queries that selected no sample.
    pub history_misses: AtomicU64,
    pub history_bytes: AtomicU64,
    /// `defer_write` calls appended.
    pub deferred_batches: AtomicU64,
    /// Component values appended, one per writer key.
    pub deferred_rows: AtomicU64,
    pub deferred_bytes: AtomicU64,
    pub deferred_failures: AtomicU64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CallbackReport {
    pub prepared_handles: u64,
    pub validation_failures: u64,
    pub views: u64,
    /// Every view is a copy: rows into it when viewed and, for writes, back out.
    pub bytes_copied: u64,
    pub legacy_operations: u64,
    pub history_queries: u64,
    pub history_misses: u64,
    pub history_bytes: u64,
    pub deferred_batches: u64,
    pub deferred_rows: u64,
    pub deferred_bytes: u64,
    pub deferred_failures: u64,
    pub hooks: Vec<HookReport>,
}

impl CallbackCounters {
    pub fn report(&self) -> CallbackReport {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        CallbackReport {
            prepared_handles: load(&self.prepared_handles),
            validation_failures: load(&self.validation_failures),
            views: load(&self.views),
            bytes_copied: load(&self.bytes_copied),
            legacy_operations: load(&self.legacy_operations),
            history_queries: load(&self.history_queries),
            history_misses: load(&self.history_misses),
            history_bytes: load(&self.history_bytes),
            deferred_batches: load(&self.deferred_batches),
            deferred_rows: load(&self.deferred_rows),
            deferred_bytes: load(&self.deferred_bytes),
            deferred_failures: load(&self.deferred_failures),
            hooks: Vec::new(),
        }
    }

    /// Restarts the per-cycle counts, as the loop does when warmup ends.
    pub fn reset_activity(&self) {
        for counter in [
            &self.views,
            &self.bytes_copied,
            &self.legacy_operations,
            &self.history_queries,
            &self.history_misses,
            &self.history_bytes,
            &self.deferred_batches,
            &self.deferred_rows,
            &self.deferred_bytes,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_history_selects_exact_or_at_or_before() {
        let dir = tempfile::tempdir().unwrap();
        let db = DB::create(dir.path().join("db")).unwrap();
        for (at, value) in [(32, 1u8), (48, 2)] {
            let id = impeller::types::msg_id("camera.frame");
            db.push_msg(Timestamp(at), id, &[value]).unwrap();
        }
        let history = HistoryHandle::message("camera.frame");
        let select = |at, selection| history.select(&db, Timestamp(at), selection);

        assert_eq!(select(48, Selection::Exact), Some((Timestamp(48), vec![2])));
        assert_eq!(select(40, Selection::Exact), None);
        assert_eq!(
            select(40, Selection::AtOrBefore),
            Some((Timestamp(32), vec![1]))
        );
        assert_eq!(select(31, Selection::AtOrBefore), None);
        let missing = HistoryHandle::message("camera.depth");
        assert_eq!(
            missing.select(&db, Timestamp(48), Selection::AtOrBefore),
            None
        );
    }
}
