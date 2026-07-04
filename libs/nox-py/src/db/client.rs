//! TCP client for Elodin-DB: discovery, latest-value subscription, historical
//! time-series reads, and SQL (Arrow IPC) queries.
//!
//! One-shot operations (`components`, `time_series`, `sql`) open a fresh
//! connection on a temporary stellarator executor and release the GIL while
//! blocked. The latest-value path runs a persistent subscription on its own
//! stellarator thread with exponential-backoff reconnect (ported from
//! `fsw/udp_component_broadcast/impeller_py`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, Timestamp};
use impeller2_stellar::Client as StellarClient;
use impeller2_wkt::{
    ArrowIPC, DumpMetadata, DumpMetadataResp, DumpSchema, DumpSchemaResp, GetSchema,
    GetTimeSeries, SQLQuery, Stream, StreamBehavior, StreamReply,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use stellarator::struc_con::{Joinable, Thread, stellar};

use super::format_prim_type;

/// Connection state of the latest-value subscription.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// A single component sample.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentData {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub values: Vec<f64>,
    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl ComponentData {
    fn __repr__(&self) -> String {
        format!(
            "ComponentData(name='{}', timestamp={}, values={:?}, shape={:?})",
            self.name, self.timestamp, self.values, self.shape
        )
    }
}

/// Component schema + metadata from discovery.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub prim_type: String,
    #[pyo3(get)]
    pub shape: Vec<u64>,
    #[pyo3(get)]
    pub element_names: Vec<String>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl ComponentInfo {
    fn __repr__(&self) -> String {
        format!(
            "ComponentInfo(name='{}', type='{}', shape={:?}, element_names={:?})",
            self.name, self.prim_type, self.shape, self.element_names
        )
    }
}

struct ComponentValue {
    name: String,
    values: Vec<f64>,
    shape: Vec<usize>,
    timestamp: i64,
}

/// Python client for Elodin-DB.
#[pyclass]
pub struct Client {
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    running: Arc<Mutex<bool>>,
    recv_thread: Option<Thread<Option<()>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
}

#[pymethods]
impl Client {
    #[new]
    fn new(addr: &str) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        Ok(Self {
            addr,
            latest_values: Arc::new(Mutex::new(HashMap::new())),
            components: Arc::new(Mutex::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
            recv_thread: None,
            connection_state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
        })
    }

    #[getter]
    fn addr(&self) -> String {
        self.addr.to_string()
    }

    /// Discover all components registered in the database.
    fn components(&mut self, py: Python<'_>) -> PyResult<HashMap<String, ComponentInfo>> {
        let addr = self.addr;
        let result = py.allow_threads(move || {
            super::block_on(move || async move { discover_components(addr).await })
        });
        match result {
            Ok((info_map, id_map)) => {
                if let Ok(mut components) = self.components.lock() {
                    for (id, name) in id_map {
                        components.insert(id, name);
                    }
                }
                Ok(info_map)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "failed to discover components: {e}"
            ))),
        }
    }

    /// Register a component name for the latest-value subscription.
    fn track(&mut self, name: &str) {
        let comp_id = ComponentId::new(name);
        if let Ok(mut components) = self.components.lock() {
            components.insert(comp_id, name.to_string());
        }
    }

    /// Latest known sample for `name`, or None. Starts the real-time
    /// subscription thread on first use (values arrive asynchronously, so the
    /// first calls may return None).
    fn latest(&mut self, name: &str) -> PyResult<Option<ComponentData>> {
        if self.recv_thread.is_none() {
            self.track(name);
            self.start_subscription();
        } else {
            self.track(name);
        }
        let values = self
            .latest_values
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("lock error: {e}")))?;
        Ok(values.get(name).map(|v| ComponentData {
            name: v.name.clone(),
            timestamp: v.timestamp,
            values: v.values.clone(),
            shape: v.shape.clone(),
        }))
    }

    /// Connection state of the latest-value subscription
    /// ("Disconnected" | "Connecting" | "Connected" | "Reconnecting").
    fn state(&self) -> String {
        self.connection_state
            .lock()
            .map(|s| format!("{:?}", *s))
            .unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Historical time-series read.
    ///
    /// Returns `(timestamps_le_i64_bytes, data_bytes, prim_type, shape)`;
    /// the Python wrapper reassembles numpy arrays with zero extra copies.
    #[pyo3(signature = (name, start_us, stop_us, limit = None))]
    fn time_series<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        start_us: i64,
        stop_us: i64,
        limit: Option<usize>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, String, Vec<u64>)> {
        let addr = self.addr;
        let comp_id = ComponentId::new(name);
        let result = py.allow_threads(move || {
            super::block_on(move || async move {
                let mut client = StellarClient::connect(addr).await?;
                let schema = client.request(&GetSchema { component_id: comp_id }).await?;
                let prim = schema.0.prim_type();
                let dims = schema.0.dim().to_vec();
                let series = client
                    .request(&GetTimeSeries {
                        id: [7, 7],
                        range: Timestamp(start_us)..Timestamp(stop_us),
                        component_id: comp_id,
                        limit,
                    })
                    .await?;
                let timestamps = series
                    .timestamps()
                    .map_err(impeller2_stellar::Error::from)?;
                let mut ts_bytes = Vec::with_capacity(timestamps.len() * 8);
                for t in timestamps {
                    ts_bytes.extend_from_slice(&t.0.to_le_bytes());
                }
                let data = series.data().map_err(impeller2_stellar::Error::from)?.to_vec();
                Ok::<_, impeller2_stellar::Error>((ts_bytes, data, prim, dims))
            })
        });
        match result {
            Ok((ts, data, prim, dims)) => Ok((
                PyBytes::new(py, &ts),
                PyBytes::new(py, &data),
                format_prim_type(prim),
                dims,
            )),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "time_series({name:?}) failed: {e}"
            ))),
        }
    }

    /// Run a SQL query; returns the raw Arrow IPC stream bytes, one entry per
    /// record batch. The Python wrapper turns these into a pyarrow Table.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let addr = self.addr;
        let query = query.to_string();
        let result: Result<Vec<Vec<u8>>, impeller2_stellar::Error> =
            py.allow_threads(move || {
                super::block_on(move || async move {
                    let mut client = StellarClient::connect(addr).await?;
                    let mut stream = client.stream(&SQLQuery(query)).await?;
                    let mut batches = Vec::new();
                    loop {
                        let msg: ArrowIPC = stream.next().await?;
                        let Some(batch) = msg.batch else {
                            break;
                        };
                        batches.push(batch.into_owned());
                    }
                    Ok(batches)
                })
            });
        match result {
            Ok(batches) => Ok(batches.iter().map(|b| PyBytes::new(py, b)).collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!("sql query failed: {e}"))),
        }
    }

    /// Stop the subscription thread (if running).
    fn close(&mut self, py: Python<'_>) {
        if let Ok(mut running) = self.running.lock() {
            *running = false;
        }
        if let Some(handle) = self.recv_thread.take() {
            py.allow_threads(|| {
                super::block_on(move || async move {
                    let _ = handle.cancel().await;
                });
            });
        }
        if let Ok(mut state) = self.connection_state.lock() {
            *state = ConnectionState::Disconnected;
        }
    }
}

impl Client {
    fn start_subscription(&mut self) {
        if let Ok(mut running) = self.running.lock() {
            *running = true;
        }
        let addr = self.addr;
        let latest_values = Arc::clone(&self.latest_values);
        let running = Arc::clone(&self.running);
        let components = Arc::clone(&self.components);
        let connection_state = Arc::clone(&self.connection_state);
        let handle = stellar(move || async move {
            subscribe_with_reconnect(addr, latest_values, running, components, connection_state)
                .await;
        });
        self.recv_thread = Some(handle);
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Ok(mut running) = self.running.lock() {
            *running = false;
        }
        if let Some(handle) = self.recv_thread.take() {
            super::block_on(move || async move {
                let _ = handle.cancel().await;
            });
        }
    }
}

async fn discover_components(
    addr: SocketAddr,
) -> Result<
    (HashMap<String, ComponentInfo>, HashMap<ComponentId, String>),
    impeller2_stellar::Error,
> {
    let mut client = StellarClient::connect(addr).await?;
    let metadata_resp: DumpMetadataResp = client.request(&DumpMetadata).await?;
    let schema_resp: DumpSchemaResp = client.request(&DumpSchema).await?;

    let mut info_map = HashMap::new();
    let mut id_map = HashMap::new();
    for metadata in metadata_resp.component_metadata {
        let name = metadata.name.clone();
        let schema = schema_resp.schemas.get(&metadata.component_id);
        let prim_type = schema
            .map(|s| format_prim_type(s.prim_type()))
            .unwrap_or_else(|| "unknown".to_string());
        let shape = schema.map(|s| s.dim().to_vec()).unwrap_or_default();
        let element_names: Vec<String> = metadata
            .metadata
            .get("element_names")
            .map(|names| {
                names
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        info_map.insert(
            name.clone(),
            ComponentInfo {
                name: name.clone(),
                prim_type,
                shape,
                element_names,
                metadata: metadata.metadata.clone(),
            },
        );
        id_map.insert(metadata.component_id, name);
    }
    Ok((info_map, id_map))
}

async fn subscribe_with_reconnect(
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    running: Arc<Mutex<bool>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
) {
    let mut backoff = std::time::Duration::from_millis(100);
    let max_backoff = std::time::Duration::from_secs(5);

    loop {
        let should_run = running.lock().map(|r| *r).unwrap_or(false);
        if !should_run {
            break;
        }
        if let Ok(mut state) = connection_state.lock() {
            *state = ConnectionState::Connecting;
        }
        match subscribe_once(
            addr,
            Arc::clone(&latest_values),
            Arc::clone(&running),
            Arc::clone(&components),
            Arc::clone(&connection_state),
        )
        .await
        {
            Ok(()) => break,
            Err(e) => {
                if let Ok(mut state) = connection_state.lock() {
                    *state = ConnectionState::Reconnecting;
                }
                let should_run = running.lock().map(|r| *r).unwrap_or(false);
                if !should_run {
                    break;
                }
                tracing::debug!("subscription error: {e}; retrying in {backoff:?}");
                stellarator::sleep(backoff).await;
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Disconnected;
    }
}

async fn subscribe_once(
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    running: Arc<Mutex<bool>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
) -> Result<(), impeller2_stellar::Error> {
    let mut client = StellarClient::connect(addr).await?;
    let stream = Stream {
        behavior: StreamBehavior::RealTime,
        id: 1,
    };
    let mut sub_stream = client.stream(&stream).await?;
    let mut registry = HashMapRegistry::default();
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Connected;
    }

    loop {
        let should_run = running.lock().map(|r| *r).unwrap_or(false);
        if !should_run {
            return Ok(());
        }
        let timeout = stellarator::sleep(std::time::Duration::from_millis(100));
        futures_lite::future::or(
            async {
                match sub_stream.next().await {
                    Ok(StreamReply::Table(table)) => {
                        let mut extractor = ValueExtractor {
                            components: &components,
                            latest_values: &latest_values,
                        };
                        if let Err(e) = table.sink(&registry, &mut extractor) {
                            tracing::debug!("failed to process table: {e}");
                        }
                    }
                    Ok(StreamReply::VTable(vtable_msg)) => {
                        registry.map.insert(vtable_msg.id, vtable_msg.vtable);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!("stream error: {e}");
                    }
                }
            },
            timeout,
        )
        .await;
    }
}

struct ValueExtractor<'a> {
    components: &'a Arc<Mutex<HashMap<ComponentId, String>>>,
    latest_values: &'a Arc<Mutex<HashMap<String, ComponentValue>>>,
}

impl Decomponentize for ValueExtractor<'_> {
    type Error = std::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        let name = match self.components.lock() {
            Ok(map) => match map.get(&component_id) {
                Some(n) => n.clone(),
                None => return Ok(()),
            },
            Err(_) => return Ok(()),
        };
        macro_rules! to_f64 {
            ($array:expr) => {
                (
                    $array.buf().iter().map(|&v| v as f64).collect(),
                    vec![$array.buf().len()],
                )
            };
        }
        let (values, shape): (Vec<f64>, Vec<usize>) = match value {
            ComponentView::F64(array) => (array.buf().to_vec(), vec![array.buf().len()]),
            ComponentView::F32(array) => to_f64!(array),
            ComponentView::U64(array) => to_f64!(array),
            ComponentView::I64(array) => to_f64!(array),
            ComponentView::U32(array) => to_f64!(array),
            ComponentView::I32(array) => to_f64!(array),
            ComponentView::U16(array) => to_f64!(array),
            ComponentView::I16(array) => to_f64!(array),
            ComponentView::U8(array) => to_f64!(array),
            ComponentView::I8(array) => to_f64!(array),
            ComponentView::Bool(array) => (
                array
                    .buf()
                    .iter()
                    .map(|&v| if v { 1.0 } else { 0.0 })
                    .collect(),
                vec![array.buf().len()],
            ),
        };
        let ts = timestamp.map(|t| t.0).unwrap_or(0);
        if let Ok(mut map) = self.latest_values.lock() {
            map.insert(
                name.clone(),
                ComponentValue {
                    name,
                    values,
                    shape,
                    timestamp: ts,
                },
            );
        }
        Ok(())
    }
}
