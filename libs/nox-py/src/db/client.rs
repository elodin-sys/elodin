//! TCP client for Elodin-DB: discovery, latest-value subscription, historical
//! time-series reads, and SQL (Arrow IPC) queries.
//!
//! Request/response operations (`components`, `time_series`, `sql`,
//! `earliest_timestamp`) are serviced by one persistent connection owned by a
//! dedicated stellarator thread; Python callers enqueue a request and block
//! (GIL released) on a reply channel. The latest-value path runs a persistent
//! subscription on a second stellarator thread with exponential-backoff
//! reconnect; shutdown is event-driven (a closed `WaitQueue` raced against
//! each connection attempt), never a timeout raced against an individual
//! packet read — cancelling a length-delimited read mid-packet would desync
//! the stream.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex};

use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, LenPacket, PrimType, Timestamp, msg_id};
use impeller2_stellar::Client as StellarClient;
use impeller2_wkt::{
    ArrowIPC, DumpMetadata, DumpMetadataResp, DumpSchema, DumpSchemaResp, EarliestTimestamp,
    GetEarliestTimestamp, GetMsgs, GetSchema, GetTimeSeries, MsgBatch, MsgMetadata, SQLQuery,
    SetMsgMetadata, Stream, StreamBehavior, StreamReply,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use stellarator::struc_con::{Joinable, Thread, stellar};
use stellarator::sync::WaitQueue;

use super::format_prim_type;

/// Packet id used for one-shot `GetTimeSeries` requests. Arbitrary but fixed;
/// replies are matched by request id, not by this packet id.
const TIME_SERIES_PACKET_ID: [u8; 2] = [7, 7];

/// Stream id for the latest-value subscription.
const SUBSCRIPTION_STREAM_ID: u64 = 1;

/// Samples fetched per `GetTimeSeries` request when paginating large ranges.
const TIME_SERIES_CHUNK: usize = 65536;

/// Connection state of the latest-value subscription.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
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

/// Latest sample for one component, stored as raw little-endian bytes so no
/// precision or shape information is lost (u64/i64 do not fit in f64).
struct ComponentValue {
    prim: PrimType,
    shape: Vec<usize>,
    data: Vec<u8>,
    timestamp: i64,
}

/// `(timestamps_le_i64_bytes, data_bytes, prim_type, shape)` as handed to the
/// Python wrapper by [`Client::time_series`].
type TimeSeriesParts<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, String, Vec<u64>);

/// `(timestamp_us, data_bytes, prim_type, shape)` as handed to the Python
/// wrapper by [`Client::latest`].
type LatestParts<'py> = (i64, Bound<'py, PyBytes>, String, Vec<usize>);

// ── request worker ───────────────────────────────────────────────────────────

enum DbRequest {
    Components,
    TimeSeries {
        component_id: ComponentId,
        start: i64,
        stop: i64,
        limit: Option<usize>,
        chunk: usize,
    },
    Sql(String),
    EarliestTimestamp,
    RegisterMsg {
        msg_id: [u8; 2],
        name: String,
    },
    SendMsg {
        msg_id: [u8; 2],
        timestamp: i64,
        payload: Vec<u8>,
    },
    GetMsgs {
        msg_id: [u8; 2],
        start: i64,
        stop: i64,
        limit: Option<usize>,
    },
}

enum DbResponse {
    Components {
        info: HashMap<String, ComponentInfo>,
        ids: HashMap<ComponentId, String>,
    },
    TimeSeries {
        timestamps: Vec<u8>,
        data: Vec<u8>,
        prim: PrimType,
        dims: Vec<u64>,
    },
    Sql(Vec<Vec<u8>>),
    Timestamp(i64),
    Unit,
    Msgs(Vec<(i64, Vec<u8>)>),
}

struct PendingRequest {
    req: DbRequest,
    reply: SyncSender<Result<DbResponse, String>>,
}

/// Unbounded command queue: sync producers (Python threads, each blocked on
/// its reply), one async consumer (the request worker). Depth is bounded by
/// the number of concurrently blocked callers.
struct RequestQueue {
    inner: Mutex<RequestQueueInner>,
    wq: WaitQueue,
}

struct RequestQueueInner {
    requests: VecDeque<PendingRequest>,
    closed: bool,
}

impl RequestQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(RequestQueueInner {
                requests: VecDeque::new(),
                closed: false,
            }),
            wq: WaitQueue::new(),
        }
    }

    /// Push a request; `Err` if the queue is closed.
    fn push(&self, req: PendingRequest) -> Result<(), ()> {
        {
            let Ok(mut inner) = self.inner.lock() else {
                return Err(());
            };
            if inner.closed {
                return Err(());
            }
            inner.requests.push_back(req);
        }
        self.wq.wake();
        Ok(())
    }

    /// Async pop; `None` once the queue is closed and drained.
    async fn pop(&self) -> Option<PendingRequest> {
        loop {
            {
                let Ok(mut inner) = self.inner.lock() else {
                    return None;
                };
                if let Some(req) = inner.requests.pop_front() {
                    return Some(req);
                }
                if inner.closed {
                    return None;
                }
            }
            let wait = self.wq.wait_for(|| {
                let Ok(inner) = self.inner.lock() else {
                    return true;
                };
                !inner.requests.is_empty() || inner.closed
            });
            if wait.await.is_err() {
                return None;
            }
        }
    }

    fn close(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.closed = true;
        }
        self.wq.wake();
    }
}

async fn request_worker_loop(addr: SocketAddr, queue: Arc<RequestQueue>) {
    let mut client: Option<StellarClient> = None;
    while let Some(pending) = queue.pop().await {
        if client.is_none() {
            match StellarClient::connect(addr).await {
                Ok(c) => client = Some(c),
                Err(e) => {
                    let _ = pending.reply.send(Err(e.to_string()));
                    continue;
                }
            }
        }
        let c = client.as_mut().unwrap();
        let result = execute_request(c, pending.req).await;
        if result.is_err() {
            // Drop the (possibly desynced) connection; reconnect on demand.
            client = None;
        }
        let _ = pending.reply.send(result.map_err(|e| e.to_string()));
    }
}

async fn execute_request(
    c: &mut StellarClient,
    req: DbRequest,
) -> Result<DbResponse, impeller2_stellar::Error> {
    match req {
        DbRequest::Components => {
            let metadata_resp: DumpMetadataResp = c.request(&DumpMetadata).await?;
            let schema_resp: DumpSchemaResp = c.request(&DumpSchema).await?;
            let mut info = HashMap::new();
            let mut ids = HashMap::new();
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
                info.insert(
                    name.clone(),
                    ComponentInfo {
                        name: name.clone(),
                        prim_type,
                        shape,
                        element_names,
                        metadata: metadata.metadata.clone(),
                    },
                );
                ids.insert(metadata.component_id, name);
            }
            Ok(DbResponse::Components { info, ids })
        }
        DbRequest::TimeSeries {
            component_id,
            start,
            stop,
            limit,
            chunk,
        } => {
            let schema = c.request(&GetSchema { component_id }).await?;
            let prim = schema.0.prim_type();
            let dims = schema.0.dim().to_vec();
            let sample_len: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
            let sample_bytes = sample_len * prim.size();
            let (timestamps, data) =
                paginated_time_series(c, component_id, start, stop, limit, chunk, sample_bytes)
                    .await?;
            Ok(DbResponse::TimeSeries {
                timestamps,
                data,
                prim,
                dims,
            })
        }
        DbRequest::Sql(query) => {
            let mut stream = c.stream(&SQLQuery(query)).await?;
            let mut batches = Vec::new();
            loop {
                let msg: ArrowIPC = stream.next().await?;
                let Some(batch) = msg.batch else {
                    break;
                };
                batches.push(batch.into_owned());
            }
            Ok(DbResponse::Sql(batches))
        }
        DbRequest::EarliestTimestamp => {
            let ts: EarliestTimestamp = c.request(&GetEarliestTimestamp).await?;
            Ok(DbResponse::Timestamp(ts.0.0))
        }
        DbRequest::RegisterMsg { msg_id, name } => {
            let metadata = MsgMetadata {
                name,
                // Payloads are opaque bytes on the wire; producers choose the
                // encoding (the Python wrapper offers a JSON convenience).
                schema: <Vec<u8> as postcard_schema::Schema>::SCHEMA.into(),
                metadata: Default::default(),
            };
            let (result, _) = c
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata,
                })
                .await;
            result?;
            Ok(DbResponse::Unit)
        }
        DbRequest::SendMsg {
            msg_id,
            timestamp,
            payload,
        } => {
            let mut pkt =
                LenPacket::msg_with_timestamp(msg_id, Timestamp(timestamp), payload.len());
            pkt.extend_from_slice(&payload);
            let (result, _) = c.send(pkt).await;
            result?;
            Ok(DbResponse::Unit)
        }
        DbRequest::GetMsgs {
            msg_id,
            start,
            stop,
            limit,
        } => {
            let batch: MsgBatch = c
                .request(&GetMsgs {
                    msg_id,
                    range: Timestamp(start)..Timestamp(stop),
                    limit,
                })
                .await?;
            Ok(DbResponse::Msgs(
                batch.data.into_iter().map(|(t, b)| (t.0, b)).collect(),
            ))
        }
    }
}

/// Fetch `[start, stop)` in chunks of at most `chunk` samples per request.
///
/// Successive requests restart at the last seen timestamp (inclusive) and skip
/// the samples already consumed, so runs of identical timestamps that straddle
/// a chunk boundary are not lost.
async fn paginated_time_series(
    c: &mut StellarClient,
    component_id: ComponentId,
    start: i64,
    stop: i64,
    limit: Option<usize>,
    chunk: usize,
    sample_bytes: usize,
) -> Result<(Vec<u8>, Vec<u8>), impeller2_stellar::Error> {
    let mut ts_out: Vec<u8> = Vec::new();
    let mut data_out: Vec<u8> = Vec::new();
    let mut cursor = start;
    let mut skip = 0usize; // samples at `cursor` already consumed
    let mut total = 0usize;
    let mut first_request = true;

    loop {
        let chunk_limit = match limit {
            Some(l) if l - total < chunk => l - total,
            _ => chunk,
        };
        if chunk_limit == 0 {
            break;
        }
        let series = match c
            .request(&GetTimeSeries {
                id: TIME_SERIES_PACKET_ID,
                range: Timestamp(cursor)..Timestamp(stop),
                component_id,
                limit: Some(chunk_limit + skip),
            })
            .await
        {
            Ok(series) => series,
            // The first request's errors (unknown component, empty range)
            // propagate; on later pages a server-side "out of bounds" just
            // means the remaining range is empty.
            Err(e @ impeller2_stellar::Error::Response(_)) if !first_request => {
                tracing::trace!("time series pagination finished: {e}");
                break;
            }
            Err(e) => return Err(e),
        };
        first_request = false;
        let timestamps = series
            .timestamps()
            .map_err(impeller2_stellar::Error::from)?;
        let data = series.data().map_err(impeller2_stellar::Error::from)?;
        let n = timestamps.len();
        if n <= skip {
            break;
        }
        for t in &timestamps[skip..] {
            ts_out.extend_from_slice(&t.0.to_le_bytes());
        }
        data_out.extend_from_slice(&data[skip * sample_bytes..n * sample_bytes]);
        let got = n - skip;
        total += got;
        if n < chunk_limit + skip {
            break; // the server returned everything left in the range
        }
        let last = timestamps[n - 1].0;
        // Restart at `last` (inclusive) and skip the samples with that
        // timestamp we already emitted. The trailing run over the whole
        // chunk is exactly that count: when the cursor advanced, the run
        // lies fully within this chunk; when it did not (a run longer than
        // the chunk), the run covers the previously skipped samples too.
        skip = timestamps.iter().rev().take_while(|t| t.0 == last).count();
        cursor = last;
    }
    Ok((ts_out, data_out))
}

// ── client ───────────────────────────────────────────────────────────────────

/// Python client for Elodin-DB.
#[pyclass]
pub struct Client {
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    /// Closing this wait queue signals the subscription thread to exit.
    shutdown: Arc<WaitQueue>,
    recv_thread: Option<Thread<Option<()>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
    requests: Arc<RequestQueue>,
    request_thread: Option<Thread<Option<()>>>,
}

impl Client {
    /// Enqueue a request and block (GIL released by callers via
    /// `py.allow_threads`) until the worker replies.
    fn request_blocking(&self, req: DbRequest) -> PyResult<DbResponse> {
        let (reply_tx, reply_rx) = sync_channel(1);
        self.requests
            .push(PendingRequest {
                req,
                reply: reply_tx,
            })
            .map_err(|_| PyRuntimeError::new_err("client is closed"))?;
        match reply_rx.recv() {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
            Err(_) => Err(PyRuntimeError::new_err("request worker stopped")),
        }
    }
}

#[pymethods]
impl Client {
    #[new]
    fn new(addr: &str) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        let requests = Arc::new(RequestQueue::new());
        let worker_queue = requests.clone();
        let request_thread = stellar(move || async move {
            request_worker_loop(addr, worker_queue).await;
        });
        Ok(Self {
            addr,
            latest_values: Arc::new(Mutex::new(HashMap::new())),
            components: Arc::new(Mutex::new(HashMap::new())),
            shutdown: Arc::new(WaitQueue::new()),
            recv_thread: None,
            connection_state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
            requests,
            request_thread: Some(request_thread),
        })
    }

    #[getter]
    fn addr(&self) -> String {
        self.addr.to_string()
    }

    /// Discover all components registered in the database.
    fn components(&mut self, py: Python<'_>) -> PyResult<HashMap<String, ComponentInfo>> {
        let resp = py.allow_threads(|| self.request_blocking(DbRequest::Components))?;
        let DbResponse::Components { info, ids } = resp else {
            return Err(PyRuntimeError::new_err("unexpected response"));
        };
        if let Ok(mut components) = self.components.lock() {
            for (id, name) in ids {
                components.insert(id, name);
            }
        }
        Ok(info)
    }

    /// Earliest data timestamp in the database (microseconds).
    fn earliest_timestamp(&self, py: Python<'_>) -> PyResult<i64> {
        let resp = py.allow_threads(|| self.request_blocking(DbRequest::EarliestTimestamp))?;
        let DbResponse::Timestamp(ts) = resp else {
            return Err(PyRuntimeError::new_err("unexpected response"));
        };
        Ok(ts)
    }

    /// Register a component name for the latest-value subscription.
    fn track(&mut self, name: &str) {
        let comp_id = ComponentId::new(name);
        if let Ok(mut components) = self.components.lock() {
            components.insert(comp_id, name.to_string());
        }
    }

    /// Latest known sample for `name` as `(timestamp_us, bytes, prim, shape)`,
    /// or None. Starts the real-time subscription thread on first use (values
    /// arrive asynchronously, so the first calls may return None).
    fn latest<'py>(&mut self, py: Python<'py>, name: &str) -> PyResult<Option<LatestParts<'py>>> {
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
        Ok(values.get(name).map(|v| {
            (
                v.timestamp,
                PyBytes::new(py, &v.data),
                format_prim_type(v.prim),
                v.shape.clone(),
            )
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

    /// Historical time-series read (paginated internally).
    ///
    /// Returns `(timestamps_le_i64_bytes, data_bytes, prim_type, shape)`;
    /// the Python wrapper reassembles numpy arrays without extra copies.
    /// `chunk` overrides the pagination page size (mainly for tests).
    #[pyo3(signature = (name, start_us, stop_us, limit = None, chunk = None))]
    fn time_series<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        start_us: i64,
        stop_us: i64,
        limit: Option<usize>,
        chunk: Option<usize>,
    ) -> PyResult<TimeSeriesParts<'py>> {
        let req = DbRequest::TimeSeries {
            component_id: ComponentId::new(name),
            start: start_us,
            stop: stop_us,
            limit,
            chunk: chunk.unwrap_or(TIME_SERIES_CHUNK).max(1),
        };
        let resp = py
            .allow_threads(|| self.request_blocking(req))
            .map_err(|e| PyRuntimeError::new_err(format!("time_series({name:?}) failed: {e}")))?;
        let DbResponse::TimeSeries {
            timestamps,
            data,
            prim,
            dims,
        } = resp
        else {
            return Err(PyRuntimeError::new_err("unexpected response"));
        };
        Ok((
            PyBytes::new(py, &timestamps),
            PyBytes::new(py, &data),
            format_prim_type(prim),
            dims,
        ))
    }

    /// Register message-log metadata for `name` (idempotent server-side).
    fn register_msg(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let req = DbRequest::RegisterMsg {
            msg_id: msg_id(name),
            name: name.to_string(),
        };
        py.allow_threads(|| self.request_blocking(req))?;
        Ok(())
    }

    /// Append one message to the log named `name`.
    fn send_msg(
        &self,
        py: Python<'_>,
        name: &str,
        payload: Vec<u8>,
        timestamp_us: i64,
    ) -> PyResult<()> {
        let req = DbRequest::SendMsg {
            msg_id: msg_id(name),
            timestamp: timestamp_us,
            payload,
        };
        py.allow_threads(|| self.request_blocking(req))?;
        Ok(())
    }

    /// Historical messages of `name` in `[start_us, stop_us)` as
    /// `[(timestamp_us, payload_bytes)]`.
    #[pyo3(signature = (name, start_us, stop_us, limit = None))]
    fn get_msgs<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        start_us: i64,
        stop_us: i64,
        limit: Option<usize>,
    ) -> PyResult<Vec<(i64, Bound<'py, PyBytes>)>> {
        let req = DbRequest::GetMsgs {
            msg_id: msg_id(name),
            start: start_us,
            stop: stop_us,
            limit,
        };
        let resp = py
            .allow_threads(|| self.request_blocking(req))
            .map_err(|e| PyRuntimeError::new_err(format!("get_msgs({name:?}) failed: {e}")))?;
        let DbResponse::Msgs(msgs) = resp else {
            return Err(PyRuntimeError::new_err("unexpected response"));
        };
        Ok(msgs
            .into_iter()
            .map(|(t, b)| (t, PyBytes::new(py, &b)))
            .collect())
    }

    /// Run a SQL query; returns the raw Arrow IPC stream bytes, one entry per
    /// record batch. The Python wrapper turns these into a pyarrow Table.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let req = DbRequest::Sql(query.to_string());
        let resp = py
            .allow_threads(|| self.request_blocking(req))
            .map_err(|e| PyRuntimeError::new_err(format!("sql query failed: {e}")))?;
        let DbResponse::Sql(batches) = resp else {
            return Err(PyRuntimeError::new_err("unexpected response"));
        };
        Ok(batches.iter().map(|b| PyBytes::new(py, b)).collect())
    }

    /// Stop the subscription and request-worker threads.
    fn close(&mut self, py: Python<'_>) {
        self.shutdown.close();
        self.requests.close();
        let recv_thread = self.recv_thread.take();
        let request_thread = self.request_thread.take();
        py.allow_threads(|| {
            super::block_on(move || async move {
                if let Some(handle) = recv_thread {
                    let _ = handle.join().await;
                }
                if let Some(handle) = request_thread {
                    let _ = handle.join().await;
                }
            });
        });
        if let Ok(mut state) = self.connection_state.lock() {
            *state = ConnectionState::Disconnected;
        }
    }
}

impl Client {
    fn start_subscription(&mut self) {
        let addr = self.addr;
        let latest_values = Arc::clone(&self.latest_values);
        let components = Arc::clone(&self.components);
        let connection_state = Arc::clone(&self.connection_state);
        let shutdown = Arc::clone(&self.shutdown);
        let handle = stellar(move || async move {
            subscribe_with_reconnect(addr, latest_values, components, connection_state, shutdown)
                .await;
        });
        self.recv_thread = Some(handle);
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.shutdown.close();
        self.requests.close();
        let recv_thread = self.recv_thread.take();
        let request_thread = self.request_thread.take();
        if recv_thread.is_some() || request_thread.is_some() {
            super::block_on(move || async move {
                if let Some(handle) = recv_thread {
                    let _ = handle.join().await;
                }
                if let Some(handle) = request_thread {
                    let _ = handle.join().await;
                }
            });
        }
    }
}

// ── latest-value subscription ────────────────────────────────────────────────

/// Outcome of racing a connection's packet loop against the shutdown signal.
enum SubscribeOutcome {
    /// The shutdown queue was closed; exit the reconnect loop.
    Shutdown,
    /// The connection failed; reconnect after backoff.
    ConnectionError(impeller2_stellar::Error),
}

async fn subscribe_with_reconnect(
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
    shutdown: Arc<WaitQueue>,
) {
    let mut backoff = std::time::Duration::from_millis(100);
    let max_backoff = std::time::Duration::from_secs(5);

    loop {
        if let Ok(mut state) = connection_state.lock() {
            *state = ConnectionState::Connecting;
        }
        // Race the whole connection (connect + packet loop) against shutdown.
        // On shutdown the connection future is dropped wholesale, which is
        // safe: the connection is abandoned, not resumed mid-packet.
        let outcome = futures_lite::future::or(
            async {
                let err = subscribe_once(
                    addr,
                    Arc::clone(&latest_values),
                    Arc::clone(&components),
                    Arc::clone(&connection_state),
                )
                .await;
                SubscribeOutcome::ConnectionError(err)
            },
            async {
                let _ = shutdown.wait().await;
                SubscribeOutcome::Shutdown
            },
        )
        .await;

        match outcome {
            SubscribeOutcome::Shutdown => break,
            SubscribeOutcome::ConnectionError(e) => {
                if let Ok(mut state) = connection_state.lock() {
                    *state = ConnectionState::Reconnecting;
                }
                tracing::debug!("subscription error: {e}; retrying in {backoff:?}");
                // Back off, but leave immediately if shutdown arrives.
                let slept = futures_lite::future::or(
                    async {
                        stellarator::sleep(backoff).await;
                        true
                    },
                    async {
                        let _ = shutdown.wait().await;
                        false
                    },
                )
                .await;
                if !slept {
                    break;
                }
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Disconnected;
    }
}

/// Runs one subscription connection until it fails. Only returns on error;
/// shutdown is handled by the caller dropping this future.
async fn subscribe_once(
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
) -> impeller2_stellar::Error {
    let mut client = match StellarClient::connect(addr).await {
        Ok(c) => c,
        Err(e) => return e,
    };
    let stream = Stream {
        // Batched: one wake + one Table packet per DB update instead of one
        // task per component server-side.
        behavior: StreamBehavior::RealTimeBatched,
        id: SUBSCRIPTION_STREAM_ID,
    };
    let mut sub_stream = match client.stream(&stream).await {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mut registry = HashMapRegistry::default();
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Connected;
    }

    loop {
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
            Err(e) => return e,
        }
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
        // Skip (and pay no copy cost for) untracked components.
        let name = match self.components.lock() {
            Ok(map) => match map.get(&component_id) {
                Some(n) => n.clone(),
                None => return Ok(()),
            },
            Err(_) => return Ok(()),
        };
        let ts = timestamp.map(|t| t.0).unwrap_or(0);
        if let Ok(mut map) = self.latest_values.lock() {
            map.insert(
                name,
                ComponentValue {
                    prim: value.prim_type(),
                    shape: value.shape().to_vec(),
                    data: value.as_bytes().to_vec(),
                    timestamp: ts,
                },
            );
        }
        Ok(())
    }
}
