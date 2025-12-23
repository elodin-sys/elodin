//! PostStepContext provides a way for post_step callbacks to read and write component data
//! directly to the database without needing a separate TCP connection.

use std::sync::Arc;

use elodin_db::DB;
use impeller2::types::{ComponentId, PrimType, Timestamp};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::Error;

/// Context object passed to post_step callbacks, providing direct DB write access.
///
/// This enables SITL workflows to write component data (like motor commands from
/// Betaflight) back to the database within the same process, avoiding the overhead
/// of a separate TCP connection.
#[pyclass]
pub struct PostStepContext {
    db: Arc<DB>,
    timestamp: Timestamp,
    tick: u64,
}

impl PostStepContext {
    /// Create a new PostStepContext with access to the database.
    pub fn new(db: Arc<DB>, timestamp: Timestamp, tick: u64) -> Self {
        Self { db, timestamp, tick }
    }
}

#[pymethods]
impl PostStepContext {
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
}

