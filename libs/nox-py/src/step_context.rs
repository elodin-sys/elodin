//! StepContext provides a way for pre_step and post_step callbacks to read and write
//! component data directly to the database without needing a separate TCP connection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Background worker that pushes rendered frames to the DB without blocking the
/// simulation tick. Uses a bounded channel so the sim never blocks — if the
/// worker is behind, the oldest unsent frame is dropped (editor skips a frame).
pub struct FrameDbWriter {
    tx: std::sync::mpsc::SyncSender<(impeller2::types::PacketId, Timestamp, Vec<u8>)>,
}

impl FrameDbWriter {
    pub fn new(db: Arc<DB>) -> Self {
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<(impeller2::types::PacketId, Timestamp, Vec<u8>)>(8);
        std::thread::Builder::new()
            .name("frame-db-writer".into())
            .spawn(move || {
                while let Ok((msg_id, ts, data)) = rx.recv() {
                    let mut result = db.push_msg(ts, msg_id, &data);
                    // Single writer per msg_id (this thread) makes truncate-then-push atomic for that log.
                    if matches!(result.as_ref(), Err(elodin_db::Error::MapOverflow)) {
                        db.truncate_msg_log(msg_id);
                        result = db.push_msg(ts, msg_id, &data);
                    }
                    if let Err(e) = result {
                        tracing::warn!("Background DB push failed: {e}");
                    }
                }
            })
            .expect("failed to spawn frame-db-writer thread");
        Self { tx }
    }

    pub fn push(&self, msg_id: impeller2::types::PacketId, ts: Timestamp, data: Vec<u8>) {
        let _ = self.tx.try_send((msg_id, ts, data));
    }
}

/// Shared frame DB writer, created lazily alongside the render client.
pub type SharedFrameDbWriter = Arc<Mutex<Option<FrameDbWriter>>>;

use elodin_db::DB;
use elodin_db::render_bridge::RenderBridgeClient;
use impeller2::types::{ComponentId, PrimType, Timestamp};
use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use stellarator::util::CancelToken;

use crate::Error;

/// Shared persistent render bridge client.
/// Created lazily on first render_camera() call, reused across all ticks.
pub type SharedRenderClient = Arc<Mutex<Option<RenderBridgeClient>>>;

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
    /// Shared persistent render bridge client (created lazily on first render call)
    render_client: SharedRenderClient,
    /// Background writer that pushes rendered frames to DB without blocking the tick
    frame_db_writer: SharedFrameDbWriter,
}

impl StepContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db: Arc<DB>,
        tick_counter: Arc<AtomicU64>,
        timestamp: Timestamp,
        tick: u64,
        start_timestamp: Timestamp,
        recipe_cancel_token: Option<CancelToken>,
        render_client: SharedRenderClient,
        frame_db_writer: SharedFrameDbWriter,
    ) -> Self {
        Self {
            db,
            timestamp: AtomicI64::new(timestamp.0),
            tick,
            tick_counter,
            start_timestamp,
            recipe_cancel_token,
            render_client,
            frame_db_writer,
        }
    }

    /// Internal implementation of render_cameras that handles the persistent client.
    /// Returns the rendered frames (also written to DB) so callers can use them without a separate read_msg.
    fn render_cameras_impl(
        &self,
        camera_names: &[&str],
    ) -> Result<Vec<elodin_db::render_bridge::RenderedFrame>, Error> {
        if camera_names.is_empty() {
            return Ok(vec![]);
        }

        let total_start = Instant::now();
        let timestamp = Timestamp(self.timestamp.load(Ordering::SeqCst));

        // Get or create the persistent render client.
        let mut guard = self.render_client.lock().map_err(|_| {
            Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(
                "Render client lock poisoned",
            ))
        })?;

        // Lazily connect on first use.
        if guard.is_none() {
            let client = RenderBridgeClient::connect(Duration::from_secs(30))
                .map_err(|e| Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(e)))?;
            *guard = Some(client);
        }

        let client = guard.as_mut().unwrap();

        // Send batch render request (UDS round-trip).
        let uds_start = Instant::now();
        let frames = client
            .render_cameras(camera_names, timestamp)
            .map_err(|e| Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(e)))?;
        let uds_ms = uds_start.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            render_uds_round_trip_ms = uds_ms,
            frame_count = frames.len()
        );

        // Push frames to DB via the background worker thread. The worker uses a
        // bounded channel so we never block. We need to clone frame data since the
        // caller consumes it for the numpy return value.
        {
            let mut writer_guard = self.frame_db_writer.lock().map_err(|_| {
                Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(
                    "Frame DB writer lock poisoned",
                ))
            })?;
            if writer_guard.is_none() {
                *writer_guard = Some(FrameDbWriter::new(self.db.clone()));
            }
            let writer = writer_guard.as_ref().unwrap();
            for frame in &frames {
                let msg_id = impeller2::types::msg_id(&frame.camera_name);
                writer.push(msg_id, frame.timestamp, frame.data.clone());
            }
        }

        tracing::debug!(total_render_cameras_ms = total_start.elapsed().as_secs_f64() * 1000.0);

        Ok(frames)
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
                return Err(Error::PyO3(pyo3::exceptions::PyValueError::new_err(
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
                Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "component '{}' not found in database",
                    pair_name
                )))
            })?;

            // Validate buffer size matches schema
            let expected_size = component.schema.size();
            if buf.len() != expected_size {
                return Err(Error::PyO3(pyo3::exceptions::PyValueError::new_err(
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
                Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "component '{}' not found in database",
                    pair_name
                )))
            })?;

            let (_, buf) = component.time_series.latest().ok_or_else(|| {
                Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
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

    /// Read the latest message payload from the database as a numpy byte array.
    ///
    /// This is used to read sensor camera frames that were written by the
    /// editor/headless renderer as messages (via MsgWithTimestamp).
    ///
    /// Args:
    ///     msg_name: The message name (e.g., "ball.downward_cam")
    ///
    /// Returns:
    ///     NumPy uint8 array containing the message payload, or None if no message exists
    fn read_msg<'py>(
        &self,
        py: Python<'py>,
        msg_name: &str,
    ) -> Result<Option<Bound<'py, PyAny>>, Error> {
        let msg_id = impeller2::types::msg_id(msg_name);

        self.db.with_state(|state| {
            if let Some(msg_log) = state.get_msg_log(msg_id) {
                if let Some((_timestamp, buf)) = msg_log.latest() {
                    let data: Vec<u8> = buf.to_vec();
                    Ok(Some(numpy::PyArray1::from_vec(py, data).into_any()))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
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

    /// Trigger the headless Bevy renderer to render a sensor camera and block
    /// until the frame is ready. Returns the frame bytes directly (and writes to DB for editor).
    ///
    /// Returns a numpy uint8 array of the RGBA frame, or None if no frame was produced.
    /// Also writes the frame to the database so `read_msg(camera_name)` and the editor can use it.
    ///
    /// Args:
    ///     camera_name: The sensor camera name in "entity.camera" format
    ///                  (e.g., "drone.scene_cam")
    ///
    /// Returns:
    ///     Optional numpy array (uint8) of the frame bytes (RGBA), or None.
    ///
    /// Raises:
    ///     RuntimeError: If no render bridge is available (not running under
    ///                   `elodin run` or `elodin editor`), if the bridge is
    ///                   disconnected, or if the renderer does not respond
    ///                   within 5 seconds.
    fn render_camera<'py>(
        &self,
        py: Python<'py>,
        camera_name: &str,
    ) -> Result<Option<Bound<'py, PyAny>>, Error> {
        let frames = self.render_cameras_impl(&[camera_name])?;
        Ok(frames
            .into_iter()
            .next()
            .map(|f| numpy::PyArray1::from_vec(py, f.data).into_any()))
    }

    /// Render multiple sensor cameras in a single batch request.
    ///
    /// This is more efficient than calling render_camera() multiple times when
    /// you have multiple cameras, as it uses a single round-trip to the render
    /// server and renders all cameras in one GPU frame.
    ///
    /// After this call returns, `read_msg(camera_name)` for each camera is
    /// guaranteed to contain the rendered frame at the current simulation timestamp.
    ///
    /// Args:
    ///     camera_names: List of sensor camera names in "entity.camera" format
    ///                   (e.g., ["drone.scene_cam", "drone.thermal_cam"])
    ///
    /// Raises:
    ///     RuntimeError: If no render bridge is available, if the bridge is
    ///                   disconnected, or if the renderer does not respond.
    fn render_cameras(&self, camera_names: &Bound<'_, PyList>) -> Result<(), Error> {
        let names: Vec<String> = camera_names.extract().map_err(|e| {
            Error::PyO3(pyo3::exceptions::PyValueError::new_err(format!(
                "camera_names must be a list of strings: {e}"
            )))
        })?;
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.render_cameras_impl(&name_refs).map(|_| ())
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
                    Error::PyO3(pyo3::exceptions::PyValueError::new_err(format!(
                        "write_timestamps key must be a string: {}",
                        e
                    )))
                })?;
                let ts: i64 = value.extract().map_err(|e| {
                    Error::PyO3(pyo3::exceptions::PyValueError::new_err(format!(
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
                    Error::PyO3(pyo3::exceptions::PyValueError::new_err(format!(
                        "write key must be a string: {}",
                        e
                    )))
                })?;
                let pair_id = ComponentId::new(&name);

                // Extract numpy array data
                let array = value.downcast::<PyUntypedArray>().map_err(|_| {
                    Error::PyO3(pyo3::exceptions::PyValueError::new_err(format!(
                        "write value for '{}' must be a numpy array",
                        name
                    )))
                })?;

                let buf = unsafe {
                    if !array.is_c_contiguous() {
                        return Err(Error::PyO3(pyo3::exceptions::PyValueError::new_err(
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
                        Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "component '{}' not found in database",
                            name
                        )))
                    })?;

                    // Validate buffer size matches schema
                    let expected_size = component.schema.size();
                    if buf.len() != expected_size {
                        return Err(Error::PyO3(pyo3::exceptions::PyValueError::new_err(
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
                        Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "component '{}' not found in database",
                            name
                        )))
                    })?;

                    let (_, buf) = component.time_series.latest().ok_or_else(|| {
                        Error::PyO3(pyo3::exceptions::PyRuntimeError::new_err(format!(
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
