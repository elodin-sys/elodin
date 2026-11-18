//! Native `pre_commit` hooks: a C function that gets a prepared batch's rows as plain arrays,
//! runs without the GIL, and can split the rows across pool workers.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use cranelift_mlir::pool::{self, Task};
use impeller::types::{ComponentId, Timestamp};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::World;
use crate::prepared::{BatchHandle, HookWrites, PreCommit};
use crate::tick_metrics::{PhaseStats, PhaseStatsSnapshot};

/// The version of the layout below, which a hook can check in `ElodinStep::abi_version`.
pub const ABI_VERSION: u32 = 1;

const ERROR_CAPACITY: usize = 512;

/// One field of the batch: `ElodinStep::rows` rows of `row_bytes` bytes, back to back.
#[repr(C)]
pub struct ElodinColumn {
    pub name: *const c_char,
    pub data: *mut u8,
    pub row_bytes: u64,
    /// Nonzero for a write field. A call may write only those, and only its own rows.
    pub writable: u32,
}

/// What one call of a hook sees: the whole batch, of which it owns rows `row_start..row_end`.
#[repr(C)]
pub struct ElodinStep {
    pub abi_version: u32,
    pub tick: u64,
    pub timestamp: i64,
    pub rows: u64,
    pub row_start: u64,
    pub row_end: u64,
    pub columns: *const ElodinColumn,
    pub column_count: u64,
    /// What the init symbol stored, or null.
    pub state: *mut c_void,
    /// Where a failing call writes a NUL-terminated message.
    pub error: *mut c_char,
    pub error_capacity: u64,
}

type StepFn = unsafe extern "C" fn(*const ElodinStep) -> i32;
type InitFn = unsafe extern "C" fn(*const ElodinStep, *mut *mut c_void) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void);

#[derive(Clone, Copy)]
struct Symbols {
    step: StepFn,
    init: Option<InitFn>,
    destroy: Option<DestroyFn>,
}

/// A batch field's rows in its component's column.
#[derive(Clone)]
struct Field {
    name: CString,
    component_id: ComponentId,
    rows: Vec<usize>,
    row_bytes: usize,
    writes: bool,
    /// The first row when the rows are consecutive, so calls can use the column in place.
    first_row: Option<usize>,
}

/// A `pre_commit` hook in a shared library, over the rows of a prepared batch.
#[pyclass(frozen)]
pub struct NativeStepHook {
    library: Arc<libloading::Library>,
    symbol: String,
    symbols: Symbols,
    fields: Vec<Field>,
    workers: usize,
}

#[pymethods]
impl NativeStepHook {
    /// Loads `symbol`, and `{symbol}_init` and `{symbol}_destroy` if present, from the shared
    /// library at `path`. Each tick calls `symbol` once per worker, each on a share of the rows.
    #[staticmethod]
    #[pyo3(signature = (path, input, symbol, workers = 1))]
    fn load(path: &str, input: &BatchHandle, symbol: &str, workers: usize) -> PyResult<Self> {
        // Safety: loading runs the library's initializers; the caller chose to trust it.
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|err| PyValueError::new_err(format!("cannot load {path:?}: {err}")))?;
        // Safety: the symbols are taken to have the documented hook ABI. The copied pointers stay
        // valid because this hook and each of its runners keep the library loaded.
        let step = unsafe { library.get::<StepFn>(symbol.as_bytes()) }
            .map_err(|err| PyValueError::new_err(format!("{path:?} has no {symbol:?}: {err}")))?;
        let symbols = Symbols {
            step: *step,
            init: unsafe { library.get::<InitFn>(format!("{symbol}_init").as_bytes()) }
                .ok()
                .map(|init| *init),
            destroy: unsafe { library.get::<DestroyFn>(format!("{symbol}_destroy").as_bytes()) }
                .ok()
                .map(|destroy| *destroy),
        };
        let fields = input
            .fields
            .iter()
            .map(|(name, handle)| {
                let handle = handle.get();
                let consecutive = handle.rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
                Ok(Field {
                    name: CString::new(name.as_str())?,
                    component_id: handle.component_id(),
                    rows: handle.rows.clone(),
                    row_bytes: handle.row_bytes,
                    writes: handle.writes(),
                    first_row: handle.rows.first().copied().filter(|_| consecutive),
                })
            })
            .collect::<Result<Vec<_>, std::ffi::NulError>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self {
            library: Arc::new(library),
            symbol: symbol.to_string(),
            symbols,
            fields,
            workers: workers.max(1),
        })
    }
}

impl NativeStepHook {
    /// The hook as one run's `pre_commit`.
    pub fn runner(&self) -> Box<dyn PreCommit + Send> {
        Box::new(NativeRunner::new(
            self.symbol.clone(),
            Some(self.library.clone()),
            self.symbols,
            self.fields.clone(),
            self.workers,
        ))
    }
}

/// One worker's share of a tick.
struct Call {
    step: ElodinStep,
    func: StepFn,
    status: i32,
    started: Instant,
    took: Duration,
    error: Box<[c_char; ERROR_CAPACITY]>,
}

extern "C" fn run_call(ctx: *mut u8) {
    // Safety: `ctx` is one of the dispatching runner's `calls`, which only this task touches
    // until the dispatch returns; the hook follows the documented ABI.
    let call = unsafe { &mut *ctx.cast::<Call>() };
    call.started = Instant::now();
    call.status = unsafe { (call.func)(&call.step) };
    call.took = call.started.elapsed();
}

fn message(error: &[c_char]) -> String {
    // Safety: `c_char` and `u8` have the same size and alignment.
    let bytes = unsafe { std::slice::from_raw_parts(error.as_ptr().cast::<u8>(), error.len()) };
    match CStr::from_bytes_until_nul(bytes) {
        Ok(message) if !message.is_empty() => message.to_string_lossy().into_owned(),
        _ => "no message".to_string(),
    }
}

struct NativeRunner {
    symbol: String,
    _library: Option<Arc<libloading::Library>>,
    symbols: Symbols,
    fields: Vec<Field>,
    columns: Vec<ElodinColumn>,
    /// The gathered rows of each field whose rows are not consecutive.
    staging: Vec<Vec<u8>>,
    calls: Vec<Call>,
    tasks: Vec<Task>,
    estimate: AtomicU64,
    /// What init stored, once it ran.
    state: Option<*mut c_void>,
    stats: HookStats,
}

// Safety: the pointers address the runner's own buffers, or the hook's state, which the hook
// contract lets the runner's thread and the calls it dispatches use.
unsafe impl Send for NativeRunner {}

impl NativeRunner {
    fn new(
        symbol: String,
        library: Option<Arc<libloading::Library>>,
        symbols: Symbols,
        fields: Vec<Field>,
        workers: usize,
    ) -> Self {
        let rows = fields.first().map_or(0, |field| field.rows.len());
        let columns = fields
            .iter()
            .map(|field| ElodinColumn {
                name: field.name.as_ptr(),
                data: std::ptr::null_mut(),
                row_bytes: field.row_bytes as u64,
                writable: field.writes.into(),
            })
            .collect::<Vec<_>>();
        let staging = fields
            .iter()
            .map(|field| match field.first_row {
                Some(_) => Vec::new(),
                None => vec![0; field.rows.len() * field.row_bytes],
            })
            .collect();
        let partitions = workers.min(rows).max(1);
        let mut calls = (0..partitions)
            .map(|i| Call {
                step: ElodinStep {
                    abi_version: ABI_VERSION,
                    tick: 0,
                    timestamp: 0,
                    rows: rows as u64,
                    row_start: (rows * i / partitions) as u64,
                    row_end: (rows * (i + 1) / partitions) as u64,
                    columns: columns.as_ptr(),
                    column_count: columns.len() as u64,
                    state: std::ptr::null_mut(),
                    error: std::ptr::null_mut(),
                    error_capacity: ERROR_CAPACITY as u64,
                },
                func: symbols.step,
                status: 0,
                started: Instant::now(),
                took: Duration::ZERO,
                error: Box::new([0; ERROR_CAPACITY]),
            })
            .collect::<Vec<_>>();
        let tasks = calls
            .iter_mut()
            .map(|call| {
                call.step.error = call.error.as_mut_ptr();
                Task::new(run_call, (call as *mut Call).cast())
            })
            .collect::<Vec<_>>();
        pool::ensure_workers(partitions - 1);
        Self {
            symbol,
            _library: library,
            symbols,
            fields,
            columns,
            staging,
            calls,
            tasks,
            estimate: AtomicU64::new(0),
            state: None,
            stats: HookStats::default(),
        }
    }

    /// Runs the init symbol, if any, over every row; returns the state it stored.
    fn init(&self, tick: u64, timestamp: Timestamp) -> Result<*mut c_void, String> {
        let mut state = std::ptr::null_mut();
        let Some(init) = self.symbols.init else {
            return Ok(state);
        };
        let mut error = [0; ERROR_CAPACITY];
        let mut step = ElodinStep {
            tick,
            timestamp: timestamp.0,
            row_start: 0,
            error: error.as_mut_ptr(),
            ..self.calls[0].step
        };
        step.row_end = step.rows;
        // Safety: `step` and `state` outlive the call; the hook follows the documented ABI.
        match unsafe { init(&step, &mut state) } {
            0 => Ok(state),
            _ => Err(format!("{}_init failed: {}", self.symbol, message(&error))),
        }
    }
}

impl PreCommit for NativeRunner {
    fn name(&self) -> &str {
        &self.symbol
    }

    fn step(
        &mut self,
        tick: u64,
        timestamp: Timestamp,
        world: &mut World,
    ) -> Result<HookWrites, String> {
        for ((field, column), staging) in self
            .fields
            .iter()
            .zip(&mut self.columns)
            .zip(&mut self.staging)
        {
            let buffer = &mut world
                .host
                .get_mut(&field.component_id)
                .expect("prepared column exists")
                .buffer;
            let size = field.row_bytes;
            column.data = match field.first_row {
                Some(row) => buffer[row * size..].as_mut_ptr(),
                None => {
                    for (i, &row) in field.rows.iter().enumerate() {
                        staging[i * size..][..size].copy_from_slice(&buffer[row * size..][..size]);
                    }
                    staging.as_mut_ptr()
                }
            };
        }
        let state = match self.state {
            Some(state) => state,
            None => *self.state.insert(self.init(tick, timestamp)?),
        };
        for call in &mut self.calls {
            call.step.tick = tick;
            call.step.timestamp = timestamp.0;
            call.step.state = state;
            call.error[0] = 0;
        }
        let dispatched = Instant::now();
        match self.calls.as_mut_slice() {
            [call] => run_call((call as *mut Call).cast()),
            // Safety: the tasks point into `self.calls`, which outlive the dispatch.
            _ => unsafe { pool::run(&self.estimate, self.tasks.as_ptr(), self.tasks.len()) },
        }
        self.stats
            .observe(dispatched, dispatched.elapsed(), &self.calls);
        if let Some(call) = self.calls.iter().find(|call| call.status != 0) {
            return Err(format!(
                "{} returned {}: {}",
                self.symbol,
                call.status,
                message(&call.error[..])
            ));
        }
        for (field, staging) in self.fields.iter().zip(&self.staging) {
            if !field.writes || field.first_row.is_some() {
                continue;
            }
            let buffer = &mut world
                .host
                .get_mut(&field.component_id)
                .expect("prepared column exists")
                .buffer;
            let size = field.row_bytes;
            for (i, &row) in field.rows.iter().enumerate() {
                buffer[row * size..][..size].copy_from_slice(&staging[i * size..][..size]);
            }
        }
        let written = self.fields.iter().filter(|field| field.writes);
        Ok(HookWrites {
            written: written.map(|field| field.component_id).collect(),
            ..HookWrites::default()
        })
    }

    fn report(&self) -> Option<HookReport> {
        Some(self.stats.report(&self.symbol, self.calls.len()))
    }

    fn reset_report(&mut self) {
        self.stats = HookStats::default();
    }
}

impl Drop for NativeRunner {
    fn drop(&mut self) {
        if let (Some(state), Some(destroy)) = (self.state, self.symbols.destroy) {
            // Safety: `state` came from init, and a runner is dropped once.
            unsafe { destroy(state) };
        }
    }
}

/// Per call: how long the last partition to start waited, and how long the slowest one took.
#[derive(Default)]
struct HookStats {
    queue: PhaseStats,
    queue_samples: Vec<u32>,
    response: PhaseStats,
    response_samples: Vec<u32>,
    busy: Duration,
    capacity: Duration,
}

impl HookStats {
    fn observe(&mut self, dispatched: Instant, wall: Duration, calls: &[Call]) {
        let queue = calls
            .iter()
            .map(|call| call.started.saturating_duration_since(dispatched));
        let queue = queue.max().unwrap_or_default();
        let response = calls.iter().map(|call| call.took).max().unwrap_or_default();
        self.queue.observe(queue);
        self.queue_samples
            .push(queue.as_nanos().min(u32::MAX as u128) as u32);
        self.response.observe(response);
        self.response_samples
            .push(response.as_nanos().min(u32::MAX as u128) as u32);
        self.busy += calls.iter().map(|call| call.took).sum::<Duration>();
        self.capacity += wall * calls.len() as u32;
    }

    fn report(&self, symbol: &str, workers: usize) -> HookReport {
        HookReport {
            symbol: symbol.to_string(),
            workers,
            queue: PhaseStatsSnapshot::new(&self.queue, Some(self.queue_samples.clone())),
            response: PhaseStatsSnapshot::new(&self.response, Some(self.response_samples.clone())),
            occupancy: self.busy.as_secs_f64() / self.capacity.as_secs_f64().max(f64::MIN_POSITIVE),
        }
    }
}

/// What a native hook did over the measured cycles.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HookReport {
    pub symbol: String,
    pub workers: usize,
    /// Per call, from dispatch to the start of the last partition to start.
    pub queue: PhaseStatsSnapshot,
    /// Per call, the slowest partition.
    pub response: PhaseStatsSnapshot,
    /// Partition time over workers times call wall time.
    pub occupancy: f64,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Writes twice each `input` row into the same `output` row; fails at tick 3.
    unsafe extern "C" fn double(step: *const ElodinStep) -> i32 {
        let step = unsafe { &*step };
        if step.tick == 3 {
            let message = c"tick 3";
            unsafe { std::ptr::copy_nonoverlapping(message.as_ptr(), step.error, 7) };
            return 7;
        }
        let columns = unsafe { std::slice::from_raw_parts(step.columns, 2) };
        let (input, output) = (columns[0].data.cast::<f64>(), columns[1].data.cast::<f64>());
        for row in step.row_start as usize..step.row_end as usize {
            unsafe { *output.add(row) = 2.0 * *input.add(row) };
        }
        0
    }

    fn field(name: &str, rows: &[usize], writes: bool) -> Field {
        let consecutive = rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
        Field {
            name: CString::new(name).unwrap(),
            component_id: ComponentId::new(name),
            rows: rows.to_vec(),
            row_bytes: 8,
            writes,
            first_row: rows.first().copied().filter(|_| consecutive),
        }
    }

    fn column(world: &World, name: &str) -> Vec<f64> {
        let buffer = &world.host[&ComponentId::new(name)].buffer;
        buffer
            .chunks(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn calls_share_the_rows_and_write_back_gathered_ones() {
        for (output_rows, workers) in [([1, 2, 3], 1), ([1, 2, 3], 2), ([3, 0, 1], 3)] {
            let mut world = World::default();
            for (name, values) in [
                ("input", [10.0, 11.0, 12.0, 13.0, 14.0]),
                ("output", [0.0; 5]),
            ] {
                let buffer = values.iter().flat_map(|v: &f64| v.to_le_bytes()).collect();
                let column = crate::world::Column {
                    buffer,
                    entity_ids: vec![],
                };
                world.host.insert(ComponentId::new(name), column);
            }
            let fields = vec![
                field("input", &[4, 0, 2], false),
                field("output", &output_rows, true),
            ];
            let symbols = Symbols {
                step: double,
                init: None,
                destroy: None,
            };
            let mut hook = NativeRunner::new("double".into(), None, symbols, fields, workers);

            let writes = hook.step(1, Timestamp(0), &mut world).unwrap();
            assert_eq!(writes.written, HashSet::from([ComponentId::new("output")]));
            let output = column(&world, "output");
            let doubled: Vec<f64> = output_rows.iter().map(|&row| output[row]).collect();
            assert_eq!(
                doubled,
                [28.0, 20.0, 24.0],
                "rows {output_rows:?}, {workers} workers"
            );

            let err = hook.step(3, Timestamp(0), &mut world).unwrap_err();
            assert!(err.ends_with("double returned 7: tick 3"), "{err}");
        }
    }
}
