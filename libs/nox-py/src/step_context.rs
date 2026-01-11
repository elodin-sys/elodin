//! StepContext provides a way for pre_step and post_step callbacks to read and write
//! component data directly to the database without needing a separate TCP connection.

use std::collections::HashMap;
use std::sync::Arc;

use elodin_db::DB;
use impeller2::types::{ComponentId, PrimType, Timestamp};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    timestamp: Timestamp,
    tick: u64,
}

impl StepContext {
    /// Create a new StepContext with access to the database.
    pub fn new(db: Arc<DB>, timestamp: Timestamp, tick: u64) -> Self {
        Self {
            db,
            timestamp,
            tick,
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
    ///
    /// Raises:
    ///     RuntimeError: If the component doesn't exist in the database
    ///     ValueError: If the data size doesn't match the component schema
    fn write_component(
        &self,
        pair_name: &str,
        data: &Bound<'_, PyUntypedArray>,
    ) -> Result<(), Error> {
        let pair_id = ComponentId::new(pair_name);

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
                .push_buf(self.timestamp, buf)
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

    /// Current simulation timestamp (nanoseconds since epoch).
    #[getter]
    fn timestamp(&self) -> i64 {
        self.timestamp.0
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
    ///
    /// Returns:
    ///     Dict mapping read component names to their numpy array values
    ///
    /// Raises:
    ///     RuntimeError: If any component doesn't exist or has no data
    ///     ValueError: If any write data size doesn't match the component schema
    #[pyo3(signature = (reads=vec![], writes=None))]
    fn component_batch_operation<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<String>,
        writes: Option<&Bound<'py, PyDict>>,
    ) -> Result<Bound<'py, PyDict>, Error> {
        // Pre-parse all component IDs outside the lock
        let read_ids: Vec<(String, ComponentId)> = reads
            .iter()
            .map(|name| (name.clone(), ComponentId::new(name)))
            .collect();

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

                    component
                        .time_series
                        .push_buf(self.timestamp, buf)
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
