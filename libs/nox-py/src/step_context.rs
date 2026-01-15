//! StepContext provides a way for pre_step and post_step callbacks to read and write
//! component data directly to the database without needing a separate TCP connection.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use elodin_db::DB;
use impeller2::types::{ComponentId, PrimType, Timestamp};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use stellarator::util::CancelToken;

use crate::Error;

/// Helper function to convert a byte buffer to a numpy array based on primitive type.
fn buf_to_numpy_array<'py>(py: Python<'py>, buf: &[u8], prim_type: PrimType) -> Bound<'py, PyAny> {
    match prim_type {
        PrimType::F64 => {
            let data: Vec<f64> = buf
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::F32 => {
            let data: Vec<f32> = buf
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::I64 => {
            let data: Vec<i64> = buf
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::I32 => {
            let data: Vec<i32> = buf
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::U64 => {
            let data: Vec<u64> = buf
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::U32 => {
            let data: Vec<u32> = buf
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::Bool => {
            let data: Vec<bool> = buf.iter().map(|&b| b != 0).collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::U8 => {
            let data: Vec<u8> = buf.to_vec();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::U16 => {
            let data: Vec<u16> = buf
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::I8 => {
            let data: Vec<i8> = buf.iter().map(|&b| b as i8).collect();
            PyArray1::from_vec(py, data).into_any()
        }
        PrimType::I16 => {
            let data: Vec<i16> = buf
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            PyArray1::from_vec(py, data).into_any()
        }
    }
}

/// Context object passed to pre_step and post_step callbacks, providing direct DB access.
///
/// This enables SITL workflows to read and write component data (like sensor readings
/// and motor commands) directly to the database within the same process, avoiding the
/// overhead of a separate TCP connection.
#[pyclass]
pub struct StepContext {
    db: Arc<DB>,
    /// Current timestamp (uses AtomicI64 for thread-safe interior mutability so truncate() can reset it)
    timestamp: AtomicI64,
    tick: u64,
    /// Shared tick counter that can be reset by truncate()
    tick_counter: Arc<AtomicU64>,
    /// Start timestamp for the simulation (used to reset timestamp after truncate)
    start_timestamp: Timestamp,
    /// Optional cancel token to gracefully terminate s10-managed recipes
    recipe_cancel_token: Option<CancelToken>,
}

impl StepContext {
    /// Create a new StepContext with access to the database and shared tick counter.
    pub fn new(
        db: Arc<DB>,
        tick_counter: Arc<AtomicU64>,
        timestamp: Timestamp,
        tick: u64,
        start_timestamp: Timestamp,
        recipe_cancel_token: Option<CancelToken>,
    ) -> Self {
        Self {
            db,
            timestamp: AtomicI64::new(timestamp.0),
            tick,
            tick_counter,
            start_timestamp,
            recipe_cancel_token,
        }
    }
}

#[pymethods]
impl StepContext {
    /// Write component data to the database.
    ///
    /// Args:
    ///     pair_name: The full component name in "entity.component" format
    ///                (e.g., "drone.motor_command")
    ///     data: NumPy array containing the component data to write
    ///     timestamp: Optional timestamp (microseconds since epoch) to write at.
    ///                If None, uses the current simulation timestamp.
    ///
    /// Raises:
    ///     RuntimeError: If the component doesn't exist in the database
    ///     ValueError: If the data size doesn't match the component schema
    ///
    /// Note:
    ///     Timestamps must be monotonically increasing per component. Writing with
    ///     a timestamp less than the last write will raise an error (TimeTravel).
    #[pyo3(signature = (pair_name, data, timestamp=None))]
    fn write_component(
        &self,
        pair_name: &str,
        data: &Bound<'_, PyUntypedArray>,
        timestamp: Option<i64>,
    ) -> Result<(), Error> {
        let pair_id = ComponentId::new(pair_name);

        // Use provided timestamp or fall back to current simulation timestamp
        let ts = timestamp.unwrap_or_else(|| self.timestamp.load(Ordering::SeqCst));

        // Get the data buffer from the numpy array
        let buf = unsafe {
            if !data.is_c_contiguous() {
                return Err(Error::PyErr(pyo3::exceptions::PyValueError::new_err(
                    "array must be c-style contiguous",
                )));
            }
            let obj = &*data.as_array_ptr();
            let shape = data.shape();
            let elem_size = data.dtype().itemsize();
            let len = shape.iter().product::<usize>() * elem_size;
            std::slice::from_raw_parts(obj.data as *const u8, len)
        };

        self.db.with_state(|state| {
            let component = state.get_component(pair_id).ok_or_else(|| {
                Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "component '{}' not found in database",
                    pair_name
                )))
            })?;

            // Validate buffer size matches schema
            let expected_size = component.schema.size();
            if buf.len() != expected_size {
                return Err(Error::PyErr(pyo3::exceptions::PyValueError::new_err(
                    format!(
                        "data size mismatch: expected {} bytes, got {} bytes",
                        expected_size,
                        buf.len()
                    ),
                )));
            }

            component
                .time_series
                .push_buf(Timestamp(ts), buf)
                .map_err(Error::from)
        })
    }

    /// Read the latest component data from the database as a numpy array.
    ///
    /// Args:
    ///     pair_name: The full component name in "entity.component" format
    ///                (e.g., "drone.accel", "drone.gyro")
    ///
    /// Returns:
    ///     NumPy array containing the component data (dtype matches component schema)
    ///
    /// Raises:
    ///     RuntimeError: If the component doesn't exist or has no data
    fn read_component<'py>(
        &self,
        py: Python<'py>,
        pair_name: &str,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let pair_id = ComponentId::new(pair_name);

        self.db.with_state(|state| {
            let component = state.get_component(pair_id).ok_or_else(|| {
                Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "component '{}' not found in database",
                    pair_name
                )))
            })?;

            let (_, buf) = component.time_series.latest().ok_or_else(|| {
                Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "component '{}' has no data",
                    pair_name
                )))
            })?;

            // Determine the numpy dtype based on the schema's primitive type
            let prim_type = component.schema.prim_type;
            let shape = component.schema.shape();

            // Convert bytes to numpy array based on primitive type
            // Note: We return a 1D array regardless of the component shape for simplicity.
            // The Python side can reshape if needed.
            let _ = shape; // Mark as intentionally unused
            match prim_type {
                PrimType::F64 => {
                    let data: Vec<f64> = buf
                        .chunks_exact(8)
                        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::F32 => {
                    let data: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::I64 => {
                    let data: Vec<i64> = buf
                        .chunks_exact(8)
                        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::I32 => {
                    let data: Vec<i32> = buf
                        .chunks_exact(4)
                        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::U64 => {
                    let data: Vec<u64> = buf
                        .chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::U32 => {
                    let data: Vec<u32> = buf
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::Bool => {
                    let data: Vec<bool> = buf.iter().map(|&b| b != 0).collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::U8 => {
                    let data: Vec<u8> = buf.to_vec();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::U16 => {
                    let data: Vec<u16> = buf
                        .chunks_exact(2)
                        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::I8 => {
                    let data: Vec<i8> = buf.iter().map(|&b| b as i8).collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
                PrimType::I16 => {
                    let data: Vec<i16> = buf
                        .chunks_exact(2)
                        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(PyArray1::from_vec(py, data).into_any())
                }
            }
        })
    }

    /// Current simulation tick count.
    #[getter]
    fn tick(&self) -> u64 {
        self.tick
    }

    /// Current simulation timestamp (microseconds since epoch).
    #[getter]
    fn timestamp(&self) -> i64 {
        self.timestamp.load(Ordering::SeqCst)
    }

    /// Truncate all component data and message logs in the database, resetting the tick counter to 0.
    ///
    /// This clears all stored time-series data while preserving component schemas and metadata.
    /// The simulation tick will be reset to 0, effectively starting fresh.
    ///
    /// Use this to control the freshness of the database and ensure reliable data from a known tick.
    ///
    /// After truncate(), any subsequent write_component() calls in the same callback will write
    /// at the start timestamp (tick 0), preventing TimeTravel errors on the next tick.
    fn truncate(&self) {
        self.db.truncate();
        self.tick_counter.store(0, Ordering::SeqCst);
        // Reset timestamp to start_timestamp (tick 0) so any subsequent write_component()
        // calls in this callback write at the correct timestamp
        self.timestamp
            .store(self.start_timestamp.0, Ordering::SeqCst);
    }

    /// Gracefully terminate all s10-managed recipes (external processes).
    ///
    /// This signals all processes managed by s10 (registered via `world.recipe()`) to shut down
    /// gracefully. On Unix systems, processes receive SIGTERM and have approximately 2 seconds
    /// to clean up before being force-killed (SIGKILL). On Windows, processes are terminated
    /// immediately.
    ///
    /// Use this to ensure clean shutdown of external processes (like Betaflight SITL) before
    /// the simulation exits, preventing memory corruption or resource leaks.
    ///
    /// This is a no-op if no recipes were registered or if running with `--no-s10`.
    ///
    /// Example:
    ///     ```python
    ///     def post_step(tick: int, ctx: el.StepContext):
    ///         if tick >= MAX_TICKS - 1:
    ///             ctx.stop_recipes()  # Gracefully stop external processes
    ///     ```
    fn stop_recipes(&self) {
        if let Some(token) = &self.recipe_cancel_token {
            token.cancel();
        }
    }

    /// Perform multiple component reads and writes in a single DB operation.
    ///
    /// This is more efficient than calling read_component/write_component multiple
    /// times, as it only acquires the database lock once for all operations.
    ///
    /// Args:
    ///     reads: List of component names to read (e.g., ["drone.accel", "drone.gyro"])
    ///     writes: Dict mapping component names to numpy arrays to write
    ///             (e.g., {"drone.motor_command": motors_array})
    ///     write_timestamps: Optional dict mapping component names to timestamps
    ///                       (microseconds since epoch). Components not in this dict
    ///                       use the current simulation timestamp.
    ///
    /// Returns:
    ///     Dict mapping read component names to their numpy array values
    ///
    /// Raises:
    ///     RuntimeError: If any component doesn't exist or has no data
    ///     ValueError: If any write data size doesn't match the component schema
    ///
    /// Note:
    ///     Timestamps must be monotonically increasing per component. Writing with
    ///     a timestamp less than the last write will raise an error (TimeTravel).
    #[pyo3(signature = (reads=vec![], writes=None, write_timestamps=None))]
    fn component_batch_operation<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<String>,
        writes: Option<&Bound<'py, PyDict>>,
        write_timestamps: Option<&Bound<'py, PyDict>>,
    ) -> Result<Bound<'py, PyDict>, Error> {
        // Pre-parse all component IDs outside the lock
        let read_ids: Vec<(String, ComponentId)> = reads
            .iter()
            .map(|name| (name.clone(), ComponentId::new(name)))
            .collect();

        // Pre-extract timestamps from Python dict outside the lock
        let timestamps: HashMap<String, i64> = if let Some(ts_dict) = write_timestamps {
            let mut ts_map = HashMap::new();
            for (key, value) in ts_dict.iter() {
                let name: String = key.extract().map_err(|e| {
                    Error::PyErr(pyo3::exceptions::PyValueError::new_err(format!(
                        "write_timestamps key must be a string: {}",
                        e
                    )))
                })?;
                let ts: i64 = value.extract().map_err(|e| {
                    Error::PyErr(pyo3::exceptions::PyValueError::new_err(format!(
                        "write_timestamps value for '{}' must be an integer: {}",
                        name, e
                    )))
                })?;
                ts_map.insert(name, ts);
            }
            ts_map
        } else {
            HashMap::new()
        };

        // Get the default timestamp once
        let default_ts = self.timestamp.load(Ordering::SeqCst);

        // Pre-extract write data from Python dict outside the lock
        let write_data: Vec<(String, ComponentId, Vec<u8>)> = if let Some(writes_dict) = writes {
            let mut data = Vec::new();
            for (key, value) in writes_dict.iter() {
                let name: String = key.extract().map_err(|e| {
                    Error::PyErr(pyo3::exceptions::PyValueError::new_err(format!(
                        "write key must be a string: {}",
                        e
                    )))
                })?;
                let pair_id = ComponentId::new(&name);

                // Extract numpy array data
                let array = value.downcast::<PyUntypedArray>().map_err(|_| {
                    Error::PyErr(pyo3::exceptions::PyValueError::new_err(format!(
                        "write value for '{}' must be a numpy array",
                        name
                    )))
                })?;

                let buf = unsafe {
                    if !array.is_c_contiguous() {
                        return Err(Error::PyErr(pyo3::exceptions::PyValueError::new_err(
                            format!("array for '{}' must be c-style contiguous", name),
                        )));
                    }
                    let obj = &*array.as_array_ptr();
                    let shape = array.shape();
                    let elem_size = array.dtype().itemsize();
                    let len = shape.iter().product::<usize>() * elem_size;
                    std::slice::from_raw_parts(obj.data as *const u8, len).to_vec()
                };

                data.push((name, pair_id, buf));
            }
            data
        } else {
            Vec::new()
        };

        // Single lock acquisition for all operations
        let read_results: HashMap<String, (Vec<u8>, PrimType)> =
            self.db.with_state(|state| -> Result<_, Error> {
                // Process all writes first
                for (name, pair_id, buf) in &write_data {
                    let component = state.get_component(*pair_id).ok_or_else(|| {
                        Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "component '{}' not found in database",
                            name
                        )))
                    })?;

                    // Validate buffer size matches schema
                    let expected_size = component.schema.size();
                    if buf.len() != expected_size {
                        return Err(Error::PyErr(pyo3::exceptions::PyValueError::new_err(
                            format!(
                                "data size mismatch for '{}': expected {} bytes, got {} bytes",
                                name,
                                expected_size,
                                buf.len()
                            ),
                        )));
                    }

                    // Use per-component timestamp if provided, otherwise use default
                    let ts = timestamps.get(name).copied().unwrap_or(default_ts);
                    component
                        .time_series
                        .push_buf(Timestamp(ts), buf)
                        .map_err(Error::from)?;
                }

                // Process all reads
                let mut results = HashMap::new();
                for (name, pair_id) in &read_ids {
                    let component = state.get_component(*pair_id).ok_or_else(|| {
                        Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "component '{}' not found in database",
                            name
                        )))
                    })?;

                    let (_, buf) = component.time_series.latest().ok_or_else(|| {
                        Error::PyErr(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "component '{}' has no data",
                            name
                        )))
                    })?;

                    let prim_type = component.schema.prim_type;
                    results.insert(name.clone(), (buf.to_vec(), prim_type));
                }

                Ok(results)
            })?;

        // Convert results to Python dict outside the lock
        let result_dict = PyDict::new(py);
        for (name, (buf, prim_type)) in read_results {
            let array = buf_to_numpy_array(py, &buf, prim_type);
            result_dict.set_item(name, array)?;
        }

        Ok(result_dict)
    }
}
