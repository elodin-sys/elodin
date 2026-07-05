//! Live / replay component streams.
//!
//! One `StreamSub` owns a dedicated stellarator thread running a single
//! stream connection (real-time batched, or fixed-rate replay). Decoded rows
//! flow to Python through a bounded queue with backpressure: the async
//! producer waits for space instead of dropping, so replay streams deliver
//! every tick. Shutdown is event-driven (`WaitQueue` raced around the whole
//! connection), consistent with `client.rs`.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::net::SocketAddr;
use std::sync::{Arc, Condvar, Mutex};

use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, MsgBuf, PacketId, PrimType, Timestamp};
use impeller2_stellar::Client as StellarClient;
use impeller2_wkt::{
    FixedRateBehavior, InitialTimestamp, Stream, StreamBehavior, StreamReply, TimestampedMsgStream,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use stellarator::struc_con::{Joinable, Thread, stellar};
use stellarator::sync::WaitQueue;

use super::format_prim_type;

/// One decoded component sample within a row.
struct RowValue {
    name: String,
    prim: PrimType,
    shape: Vec<usize>,
    data: Vec<u8>,
    /// This component's own sample timestamp (a batched stream carries each
    /// component's latest value, which may be older than the row's newest).
    timestamp: i64,
}

/// One stream row: the newest timestamp in the packet + the requested
/// components present in it.
struct RowData {
    timestamp: i64,
    values: Vec<RowValue>,
}

/// `(row_timestamp_us, [(name, bytes, prim, shape, timestamp_us)])` handed
/// to Python.
type PyRow<'py> = (
    i64,
    Vec<(String, Bound<'py, PyBytes>, String, Vec<usize>, i64)>,
);

/// Bounded item queue: async producer with backpressure, sync (Python-thread)
/// consumer.
struct BoundedQueue<T> {
    inner: Mutex<BoundedQueueInner<T>>,
    /// Wakes the Python consumer waiting in `pop_blocking`.
    consumer_cv: Condvar,
    /// Wakes the async producer waiting for queue space.
    producer_wq: WaitQueue,
    maxlen: usize,
}

struct BoundedQueueInner<T> {
    items: VecDeque<T>,
    closed: bool,
}

impl<T> BoundedQueue<T> {
    fn new(maxlen: usize) -> Self {
        Self {
            inner: Mutex::new(BoundedQueueInner {
                items: VecDeque::new(),
                closed: false,
            }),
            consumer_cv: Condvar::new(),
            producer_wq: WaitQueue::new(),
            maxlen: maxlen.max(1),
        }
    }

    /// Async push with backpressure; `false` once the queue is closed.
    async fn push(&self, item: T) -> bool {
        let mut item = Some(item);
        loop {
            {
                let Ok(mut inner) = self.inner.lock() else {
                    return false;
                };
                if inner.closed {
                    return false;
                }
                if inner.items.len() < self.maxlen {
                    inner
                        .items
                        .push_back(item.take().expect("item consumed twice"));
                    self.consumer_cv.notify_one();
                    return true;
                }
            }
            let wait = self.producer_wq.wait_for(|| {
                let Ok(inner) = self.inner.lock() else {
                    return true;
                };
                inner.items.len() < self.maxlen || inner.closed
            });
            if wait.await.is_err() {
                return false;
            }
        }
    }

    /// Blocking pop with timeout; `None` on timeout or once closed and empty
    /// (disambiguate with `is_closed`).
    fn pop_blocking(&self, timeout: std::time::Duration) -> Option<T> {
        let Ok(mut inner) = self.inner.lock() else {
            return None;
        };
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(item) = inner.items.pop_front() {
                self.producer_wq.wake();
                return Some(item);
            }
            if inner.closed {
                return None;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let (guard, _) = self.consumer_cv.wait_timeout(inner, deadline - now).ok()?;
            inner = guard;
        }
    }

    fn is_closed(&self) -> bool {
        self.inner.lock().map(|i| i.closed).unwrap_or(true)
    }

    fn close(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.closed = true;
        }
        self.consumer_cv.notify_all();
        self.producer_wq.wake();
    }
}

fn random_stream_id() -> u64 {
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

/// A live or replay component stream subscription.
#[pyclass]
pub struct StreamSub {
    queue: Arc<BoundedQueue<RowData>>,
    shutdown: Arc<WaitQueue>,
    thread: Option<Thread<Option<()>>>,
}

#[pymethods]
impl StreamSub {
    /// `rate_hz = None` → real-time (batched) stream of new data;
    /// `rate_hz = Some(hz)` → fixed-rate replay from `initial`
    /// ("earliest" | "latest" | "manual" with `initial_us`).
    #[new]
    #[pyo3(signature = (addr, names, rate_hz = None, initial = "earliest", initial_us = None, maxlen = 1024))]
    fn new(
        addr: &str,
        names: Vec<String>,
        rate_hz: Option<f64>,
        initial: &str,
        initial_us: Option<i64>,
        maxlen: usize,
    ) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        if names.is_empty() {
            return Err(PyValueError::new_err("stream needs at least one component"));
        }
        let behavior = match rate_hz {
            None => StreamBehavior::RealTimeBatched,
            Some(hz) => {
                if !(hz.is_finite() && hz > 0.0) {
                    return Err(PyValueError::new_err(format!("invalid rate_hz {hz}")));
                }
                let initial_timestamp = match initial {
                    "earliest" => InitialTimestamp::Earliest,
                    "latest" => InitialTimestamp::Latest,
                    "manual" => {
                        InitialTimestamp::Manual(Timestamp(initial_us.ok_or_else(|| {
                            PyValueError::new_err("manual start requires initial_us")
                        })?))
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown initial timestamp {other:?}"
                        )));
                    }
                };
                StreamBehavior::FixedRate(FixedRateBehavior {
                    initial_timestamp,
                    timestep: (1e9 / hz) as u64,
                    frequency: hz.round().max(1.0) as u64,
                })
            }
        };
        let wanted: HashMap<ComponentId, String> = names
            .iter()
            .map(|n| (ComponentId::new(n), n.clone()))
            .collect();
        let queue = Arc::new(BoundedQueue::new(maxlen));
        let shutdown = Arc::new(WaitQueue::new());
        let thread_queue = queue.clone();
        let thread_shutdown = shutdown.clone();
        let thread = stellar(move || async move {
            stream_thread(addr, behavior, wanted, thread_queue, thread_shutdown).await;
        });
        Ok(Self {
            queue,
            shutdown,
            thread: Some(thread),
        })
    }

    /// Next row as `(timestamp_us, [(name, bytes, prim, shape)])`, or None on
    /// timeout / stream end (check `is_closed`).
    fn next_row<'py>(&self, py: Python<'py>, timeout_ms: u64) -> Option<PyRow<'py>> {
        let queue = self.queue.clone();
        let row = py.allow_threads(move || {
            queue.pop_blocking(std::time::Duration::from_millis(timeout_ms))
        })?;
        Some((
            row.timestamp,
            row.values
                .into_iter()
                .map(|v| {
                    (
                        v.name,
                        PyBytes::new(py, &v.data),
                        format_prim_type(v.prim),
                        v.shape,
                        v.timestamp,
                    )
                })
                .collect(),
        ))
    }

    /// True once the stream has ended (closed, or the connection failed).
    fn is_closed(&self) -> bool {
        self.queue.is_closed()
    }

    /// Stop the stream and join its thread.
    fn close(&mut self, py: Python<'_>) {
        self.shutdown.close();
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            py.allow_threads(|| {
                super::block_on(move || async move {
                    let _ = handle.join().await;
                });
            });
        }
    }
}

impl Drop for StreamSub {
    fn drop(&mut self) {
        self.shutdown.close();
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            super::block_on(move || async move {
                let _ = handle.join().await;
            });
        }
    }
}

async fn stream_thread(
    addr: SocketAddr,
    behavior: StreamBehavior,
    wanted: HashMap<ComponentId, String>,
    queue: Arc<BoundedQueue<RowData>>,
    shutdown: Arc<WaitQueue>,
) {
    // Race the whole connection against shutdown (safe: the connection is
    // abandoned wholesale, never resumed mid-packet).
    futures_lite::future::or(
        async {
            if let Err(e) = run_stream(addr, behavior, &wanted, &queue).await {
                tracing::debug!("stream ended: {e}");
            }
        },
        async {
            let _ = shutdown.wait().await;
        },
    )
    .await;
    queue.close();
}

async fn run_stream(
    addr: SocketAddr,
    behavior: StreamBehavior,
    wanted: &HashMap<ComponentId, String>,
    queue: &Arc<BoundedQueue<RowData>>,
) -> Result<(), impeller2_stellar::Error> {
    let mut client = StellarClient::connect(addr).await?;
    let stream = Stream {
        behavior,
        id: random_stream_id(),
    };
    let mut sub = client.stream(&stream).await?;
    let mut registry = HashMapRegistry::default();

    loop {
        match sub.next().await? {
            StreamReply::Table(table) => {
                let mut extractor = RowExtractor {
                    wanted,
                    row: RowData {
                        timestamp: 0,
                        values: Vec::new(),
                    },
                };
                if let Err(e) = table.sink(&registry, &mut extractor) {
                    tracing::debug!("failed to process table: {e}");
                    continue;
                }
                if extractor.row.values.is_empty() {
                    continue;
                }
                if !queue.push(extractor.row).await {
                    return Ok(()); // consumer closed the stream
                }
            }
            StreamReply::VTable(vtable_msg) => {
                registry.map.insert(vtable_msg.id, vtable_msg.vtable);
            }
            _ => {}
        }
    }
}

// ── message-log stream ───────────────────────────────────────────────────────

/// One received message: `(timestamp_us, payload)`.
type MsgItem = (i64, Vec<u8>);

/// A live message-log stream subscription (new messages only).
#[pyclass]
pub struct MsgStreamSub {
    queue: Arc<BoundedQueue<MsgItem>>,
    shutdown: Arc<WaitQueue>,
    thread: Option<Thread<Option<()>>>,
}

#[pymethods]
impl MsgStreamSub {
    #[new]
    #[pyo3(signature = (addr, name, maxlen = 1024))]
    fn new(addr: &str, name: &str, maxlen: usize) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        let msg_id = impeller2::types::msg_id(name);
        let queue = Arc::new(BoundedQueue::new(maxlen));
        let shutdown = Arc::new(WaitQueue::new());
        let thread_queue = queue.clone();
        let thread_shutdown = shutdown.clone();
        let thread = stellar(move || async move {
            futures_lite::future::or(
                async {
                    if let Err(e) = run_msg_stream(addr, msg_id, &thread_queue).await {
                        tracing::debug!("msg stream ended: {e}");
                    }
                },
                async {
                    let _ = thread_shutdown.wait().await;
                },
            )
            .await;
            thread_queue.close();
        });
        Ok(Self {
            queue,
            shutdown,
            thread: Some(thread),
        })
    }

    /// Next message as `(timestamp_us, payload_bytes)`, or None on timeout /
    /// stream end (check `is_closed`).
    fn next_msg<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: u64,
    ) -> Option<(i64, Bound<'py, PyBytes>)> {
        let queue = self.queue.clone();
        let (ts, payload) = py.allow_threads(move || {
            queue.pop_blocking(std::time::Duration::from_millis(timeout_ms))
        })?;
        Some((ts, PyBytes::new(py, &payload)))
    }

    /// True once the stream has ended (closed, or the connection failed).
    fn is_closed(&self) -> bool {
        self.queue.is_closed()
    }

    /// Stop the stream and join its thread.
    fn close(&mut self, py: Python<'_>) {
        self.shutdown.close();
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            py.allow_threads(|| {
                super::block_on(move || async move {
                    let _ = handle.join().await;
                });
            });
        }
    }
}

impl Drop for MsgStreamSub {
    fn drop(&mut self) {
        self.shutdown.close();
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            super::block_on(move || async move {
                let _ = handle.join().await;
            });
        }
    }
}

async fn run_msg_stream(
    addr: SocketAddr,
    msg_id: PacketId,
    queue: &Arc<BoundedQueue<MsgItem>>,
) -> Result<(), impeller2_stellar::Error> {
    let mut client = StellarClient::connect(addr).await?;
    let mut sub = client.stream(&TimestampedMsgStream { msg_id }).await?;
    loop {
        let msg: MsgBuf<_> = sub.next().await?;
        let ts = msg.timestamp.map(|t| t.0).unwrap_or(0);
        if !queue.push((ts, msg.buf.to_vec())).await {
            return Ok(()); // consumer closed the stream
        }
    }
}

struct RowExtractor<'a> {
    wanted: &'a HashMap<ComponentId, String>,
    row: RowData,
}

impl Decomponentize for RowExtractor<'_> {
    type Error = std::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        let Some(name) = self.wanted.get(&component_id) else {
            return Ok(());
        };
        let ts = timestamp.map(|t| t.0).unwrap_or(0);
        self.row.timestamp = self.row.timestamp.max(ts);
        self.row.values.push(RowValue {
            name: name.clone(),
            prim: value.prim_type(),
            shape: value.shape().to_vec(),
            data: value.as_bytes().to_vec(),
            timestamp: ts,
        });
        Ok(())
    }
}
