//! Prepared component handles and `pre_commit` hooks, which read and write world columns between
//! the tick and the DB commit instead of going through the DB.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use impeller::types::{ComponentId, PrimType, Timestamp};
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use stellarator::util::CancelToken;

use crate::World;
use crate::entity::EntityId;
use crate::native_hook::HookReport;

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

fn numpy_dtype(prim_type: PrimType) -> &'static str {
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
    pub(crate) fields: Vec<(String, Py<ComponentHandle>)>,
}

impl BatchHandle {
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
        Ok(Self { fields })
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

/// Passed to a `pre_commit` callback. Views read the world's current values, and writes to a
/// write handle's view reach the world, and the same commit, when the callback returns.
#[pyclass(unsendable)]
pub struct WorldStepContext {
    #[pyo3(get)]
    tick: u64,
    #[pyo3(get)]
    timestamp: i64,
    /// Null once the callback returns.
    world: Cell<*mut World>,
    viewed: RefCell<Vec<Py<ComponentHandle>>>,
    recorded: RefCell<HashSet<ComponentId>>,
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

    /// Gracefully terminate all s10-managed recipes; see `StepContext.stop_recipes`.
    fn stop_recipes(&self) {
        if let Some(token) = &self.recipe_cancel_token {
            token.cancel();
        }
    }
}

impl WorldStepContext {
    /// Ends the callback: copies the written views into `world`.
    fn finish(&self, py: Python<'_>, world: &mut World) -> HookWrites {
        self.world.set(std::ptr::null_mut());
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
}

/// Runs after the tick and before the commit.
pub trait PreCommit {
    fn step(
        &mut self,
        tick: u64,
        timestamp: Timestamp,
        world: &mut World,
    ) -> Result<HookWrites, String>;

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
    pub counters: Arc<CallbackCounters>,
    pub recipe_cancel_token: Option<CancelToken>,
}

impl PreCommit for PythonPreCommit {
    fn step(
        &mut self,
        tick: u64,
        timestamp: Timestamp,
        world: &mut World,
    ) -> Result<HookWrites, String> {
        Python::with_gil(|py| {
            let ctx = WorldStepContext {
                tick,
                timestamp: timestamp.0,
                world: Cell::new(world),
                viewed: RefCell::default(),
                recorded: RefCell::default(),
                counters: self.counters.clone(),
                recipe_cancel_token: self.recipe_cancel_token.clone(),
            };
            let ctx = Py::new(py, ctx).map_err(|err| err.to_string())?;
            let result = self.func.call1(py, (tick, ctx.clone_ref(py)));
            let written = ctx.borrow(py).finish(py, world);
            result.map(|_| written).map_err(|err| {
                err.print(py);
                format!("pre_commit failed at tick {tick}: {err}")
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
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CallbackReport {
    pub prepared_handles: u64,
    pub validation_failures: u64,
    pub views: u64,
    /// Every view is a copy: rows into it when viewed and, for writes, back out.
    pub bytes_copied: u64,
    pub legacy_operations: u64,
    pub hooks: Vec<HookReport>,
}

impl CallbackCounters {
    pub fn report(&self) -> CallbackReport {
        CallbackReport {
            prepared_handles: self.prepared_handles.load(Ordering::Relaxed),
            validation_failures: self.validation_failures.load(Ordering::Relaxed),
            views: self.views.load(Ordering::Relaxed),
            bytes_copied: self.bytes_copied.load(Ordering::Relaxed),
            legacy_operations: self.legacy_operations.load(Ordering::Relaxed),
            hooks: Vec::new(),
        }
    }

    /// Restarts the per-cycle counts, as the loop does when warmup ends.
    pub fn reset_activity(&self) {
        self.views.store(0, Ordering::Relaxed);
        self.bytes_copied.store(0, Ordering::Relaxed);
        self.legacy_operations.store(0, Ordering::Relaxed);
    }
}
