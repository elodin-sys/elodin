//! Batched telemetry writer.
//!
//! One `TableWriter` owns: one TCP connection (on a dedicated stellarator
//! thread), one vtable whose fields share a leading `i64` microsecond
//! timestamp, and one bounded row queue with a configurable overflow policy.
//! Every `write_row` emits exactly one `Table` packet. On reconnect the
//! component metadata + vtable handshake is replayed automatically.
//!
//! A reader task on the same executor drains the connection's receive side so
//! server rejections (`ErrorResponse`, e.g. schema mismatches) surface via
//! `last_error` instead of silently filling TCP buffers.

use std::collections::VecDeque;
use std::hash::{BuildHasher, Hasher};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use impeller2::types::{ComponentId, LenPacket, Msg, OwnedPacket, PrimType};
use impeller2::vtable::builder::{
    component, raw_field, raw_table, schema, timestamp, timestamp_ns, vtable,
};
use impeller2_stellar::{PacketSink, PacketStream};
use impeller2_wkt::{ErrorResponse, SetComponentMetadata, VTableMsg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use stellarator::io::{OwnedWriter, SplitExt};
use stellarator::net::TcpStream;
use stellarator::sync::WaitQueue;

use super::parse_prim_type;

/// One field as passed from Python:
/// `(name, prim_type, dims, element_names, offset, size)`.
type PyFieldSpec = (String, String, Vec<u64>, Option<String>, usize, usize);

/// How long a blocking `write_row` waits for the writer thread to hand the
/// row to the socket before reporting a timeout.
const BLOCKING_WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Allocate a per-process-unique vtable id. The counter starts at a random
/// point in the full 16-bit space so independent processes writing to the same
/// database do not deterministically collide in its global vtable registry.
fn next_vtable_id() -> [u8; 2] {
    static NEXT_VTABLE_ID: OnceLock<AtomicU16> = OnceLock::new();
    let counter = NEXT_VTABLE_ID.get_or_init(|| {
        // RandomState is seeded with fresh process-level entropy.
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        AtomicU16::new(seed as u16)
    });
    counter.fetch_add(1, Ordering::Relaxed).to_le_bytes()
}

/// Overflow policy for `write_row_nowait` when the queue is full.
#[derive(Clone, Copy, PartialEq)]
enum QueuePolicy {
    /// Shed the oldest queued row to make room for the incoming one.
    DropOldest,
    /// Discard the incoming row.
    DropNewest,
}

#[derive(Clone)]
pub(crate) struct FieldSpec {
    pub name: String,
    pub prim: PrimType,
    pub dims: Vec<u64>,
    pub element_names: Option<String>,
    pub offset: usize,
    pub size: usize,
}

struct Registration {
    vtable_id: [u8; 2],
    fields: Vec<FieldSpec>,
    /// Leading i64 is nanoseconds (`timestamp_ns` vtable op) instead of the
    /// default microseconds.
    timestamp_ns: bool,
}

struct RowCmd {
    data: Vec<u8>,
    /// Present for blocking writes; the sender reports the send result.
    result_tx: Option<SyncSender<Result<(), String>>>,
}

/// Bounded MPSC row queue: sync producers (Python threads), one async
/// consumer (the writer thread's stellarator executor).
struct RowQueue {
    inner: Mutex<RowQueueInner>,
    /// Wakes the async consumer; `wake()` stores a wakeup if none is waiting.
    wq: WaitQueue,
    maxlen: usize,
    policy: QueuePolicy,
}

struct RowQueueInner {
    rows: VecDeque<RowCmd>,
    closed: bool,
}

impl RowQueue {
    fn new(maxlen: usize, policy: QueuePolicy) -> Self {
        Self {
            inner: Mutex::new(RowQueueInner {
                rows: VecDeque::new(),
                closed: false,
            }),
            wq: WaitQueue::new(),
            maxlen,
            policy,
        }
    }

    /// Non-blocking push; returns the number of rows shed (0 or 1).
    fn push_nowait(&self, cmd: RowCmd) -> u64 {
        let mut shed = 0;
        {
            let Ok(mut inner) = self.inner.lock() else {
                return 1;
            };
            if inner.closed {
                return 1;
            }
            if inner.rows.len() >= self.maxlen {
                match self.policy {
                    QueuePolicy::DropOldest => {
                        // Never shed a blocking row (its producer is waiting on
                        // the result); shed the oldest fire-and-forget row.
                        match inner.rows.iter().position(|r| r.result_tx.is_none()) {
                            Some(idx) => {
                                inner.rows.remove(idx);
                                shed = 1;
                            }
                            None => return 1, // all queued rows are blocking
                        }
                    }
                    QueuePolicy::DropNewest => {
                        return 1;
                    }
                }
            }
            inner.rows.push_back(cmd);
        }
        self.wq.wake();
        shed
    }

    /// Push for blocking writes. Bypasses `maxlen`: the caller rendezvouses on
    /// the row's result, so queue growth is bounded by the number of
    /// concurrently blocking producer threads.
    fn push_blocking(&self, cmd: RowCmd) -> Result<(), ()> {
        {
            let Ok(mut inner) = self.inner.lock() else {
                return Err(());
            };
            if inner.closed {
                return Err(());
            }
            inner.rows.push_back(cmd);
        }
        self.wq.wake();
        Ok(())
    }

    /// Async pop; `None` once the queue is closed and drained.
    async fn pop(&self) -> Option<RowCmd> {
        loop {
            {
                let Ok(mut inner) = self.inner.lock() else {
                    return None;
                };
                if let Some(cmd) = inner.rows.pop_front() {
                    return Some(cmd);
                }
                if inner.closed {
                    return None;
                }
            }
            let wait = self.wq.wait_for(|| {
                let Ok(inner) = self.inner.lock() else {
                    return true;
                };
                !inner.rows.is_empty() || inner.closed
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

/// Connection state of a writer, stored as an atomic for lock-free reads.
mod writer_state {
    pub const DISCONNECTED: u8 = 0;
    pub const CONNECTED: u8 = 1;

    pub fn name(v: u8) -> &'static str {
        match v {
            CONNECTED => "Connected",
            _ => "Disconnected",
        }
    }
}

/// Batched writer for a fixed set of components sharing a timestamp.
#[pyclass]
pub struct TableWriter {
    queue: Arc<RowQueue>,
    thread: Option<std::thread::JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
    last_error: Arc<Mutex<Option<String>>>,
    state: Arc<AtomicU8>,
    #[pyo3(get)]
    row_size: usize,
}

#[pymethods]
impl TableWriter {
    /// `fields`: `(name, prim_type, dims, element_names, offset, size)` per
    /// component. Offsets are relative to the packet payload start; offset 0..8
    /// is reserved for the shared `i64` timestamp (`timestamp_unit`:
    /// microseconds by default, or nanoseconds).
    #[new]
    #[pyo3(signature = (addr, fields, maxlen = 1024, queue = "drop-oldest", timestamp_unit = "us"))]
    fn new(
        addr: &str,
        fields: Vec<PyFieldSpec>,
        maxlen: usize,
        queue: &str,
        timestamp_unit: &str,
    ) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        let policy = match queue {
            "drop-oldest" => QueuePolicy::DropOldest,
            "drop-newest" => QueuePolicy::DropNewest,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown queue policy {other:?} (expected \"drop-oldest\" or \"drop-newest\")"
                )));
            }
        };
        let ts_ns = match timestamp_unit {
            "us" => false,
            "ns" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown timestamp unit {other:?} (expected \"us\" or \"ns\")"
                )));
            }
        };
        if fields.is_empty() {
            return Err(PyValueError::new_err(
                "table_writer needs at least one field",
            ));
        }
        let mut specs = Vec::with_capacity(fields.len());
        let mut row_size = 8usize; // leading i64 timestamp
        for (name, prim, dims, element_names, offset, size) in fields {
            let prim = parse_prim_type(&prim)?;
            if offset < 8 {
                return Err(PyValueError::new_err(format!(
                    "field {name:?} offset {offset} overlaps the timestamp (bytes 0..8)"
                )));
            }
            row_size = row_size.max(offset + size);
            specs.push(FieldSpec {
                name,
                prim,
                dims,
                element_names,
                offset,
                size,
            });
        }
        let reg = Registration {
            vtable_id: next_vtable_id(),
            fields: specs,
            timestamp_ns: ts_ns,
        };
        let dropped = Arc::new(AtomicU64::new(0));
        let last_error = Arc::new(Mutex::new(None));
        let state = Arc::new(AtomicU8::new(writer_state::DISCONNECTED));
        let row_queue = Arc::new(RowQueue::new(maxlen.max(1), policy));

        let ctx = WriterCtx {
            queue: row_queue.clone(),
            dropped: dropped.clone(),
            last_error: last_error.clone(),
            state: state.clone(),
        };
        let thread = std::thread::Builder::new()
            .name("elodin-db-writer".into())
            .spawn(move || {
                stellarator::run(|| writer_loop(addr, reg, ctx));
            })
            .map_err(|e| PyRuntimeError::new_err(format!("failed to spawn writer thread: {e}")))?;
        Ok(Self {
            queue: row_queue,
            thread: Some(thread),
            dropped,
            last_error,
            state,
            row_size,
        })
    }

    /// Rows dropped by `write_row_nowait` (queue full or writer stopped) plus
    /// rows lost to connection errors.
    #[getter]
    fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// The most recent transport error or database `ErrorResponse` (e.g. a
    /// schema mismatch for an already-registered component), or None.
    #[getter]
    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|e| e.clone())
    }

    /// Connection state: "Connected" | "Disconnected".
    fn state(&self) -> String {
        writer_state::name(self.state.load(Ordering::Relaxed)).to_string()
    }

    /// Blocking write: waits for the row to be handed to the OS socket.
    /// Raises on connection failure.
    fn write_row(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<()> {
        self.check_row(&data)?;
        let queue = self.queue.clone();
        py.allow_threads(move || {
            let (result_tx, result_rx) = sync_channel(1);
            queue
                .push_blocking(RowCmd {
                    data,
                    result_tx: Some(result_tx),
                })
                .map_err(|_| PyRuntimeError::new_err("writer is closed"))?;
            match result_rx.recv_timeout(BLOCKING_WRITE_TIMEOUT) {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(format!("write failed: {e}"))),
                Err(_) => Err(PyRuntimeError::new_err("write timed out")),
            }
        })
    }

    /// Non-blocking write: enqueue and return immediately. On overflow the
    /// queue policy decides which row is shed (counted in `dropped`); this
    /// method never raises for transport reasons.
    fn write_row_nowait(&self, data: Vec<u8>) -> PyResult<()> {
        self.check_row(&data)?;
        let shed = self.queue.push_nowait(RowCmd {
            data,
            result_tx: None,
        });
        if shed > 0 {
            self.dropped.fetch_add(shed, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Close the writer and join its thread.
    fn close(&mut self, py: Python<'_>) {
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            py.allow_threads(|| {
                let _ = handle.join();
            });
        }
    }
}

impl TableWriter {
    fn check_row(&self, data: &[u8]) -> PyResult<()> {
        if data.len() != self.row_size {
            return Err(PyValueError::new_err(format!(
                "row must be exactly {} bytes (got {})",
                self.row_size,
                data.len()
            )));
        }
        Ok(())
    }
}

impl Drop for TableWriter {
    fn drop(&mut self) {
        self.queue.close();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Shared handles between the Python-facing `TableWriter` and its thread.
struct WriterCtx {
    queue: Arc<RowQueue>,
    dropped: Arc<AtomicU64>,
    last_error: Arc<Mutex<Option<String>>>,
    state: Arc<AtomicU8>,
}

/// One live connection: the send half plus a reader task draining the receive
/// half for `ErrorResponse`s. Dropping it cancels the reader task.
struct Conn {
    sink: PacketSink<OwnedWriter<TcpStream>>,
    _reader: stellarator::JoinHandleDropGuard<()>,
    registered: bool,
}

async fn writer_loop(addr: SocketAddr, reg: Registration, ctx: WriterCtx) {
    let mut conn: Option<Conn> = None;

    loop {
        let Some(cmd) = ctx.queue.pop().await else {
            break; // closed and drained
        };
        let result = send_row(addr, &mut conn, &reg, &cmd.data, &ctx).await;
        if let Err(ref e) = result {
            conn = None;
            ctx.state
                .store(writer_state::DISCONNECTED, Ordering::Relaxed);
            record_error(&ctx.last_error, &e.to_string());
            if cmd.result_tx.is_none() {
                ctx.dropped.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("dropped row: {e}");
            }
        }
        if let Some(result_tx) = cmd.result_tx {
            let _ = result_tx.send(result.map_err(|e| e.to_string()));
        }
    }
    ctx.state
        .store(writer_state::DISCONNECTED, Ordering::Relaxed);
}

fn record_error(last_error: &Mutex<Option<String>>, msg: &str) {
    if let Ok(mut slot) = last_error.lock() {
        *slot = Some(msg.to_string());
    }
}

async fn connect(addr: SocketAddr, ctx: &WriterCtx) -> Result<Conn, impeller2_stellar::Error> {
    let stream = TcpStream::connect(addr).await?;
    let (reader, writer) = stream.split();
    let last_error = ctx.last_error.clone();
    let reader_task = stellarator::spawn(async move {
        let mut rx = PacketStream::new(reader);
        let mut buf = vec![0u8; 1024];
        // Exits when the connection closes; the writer loop reconnects.
        while let Ok(pkt) = rx.next_grow(buf).await {
            if let OwnedPacket::Msg(m) = &pkt
                && m.id == ErrorResponse::ID
                && let Ok(err) = m.parse::<ErrorResponse>()
            {
                tracing::debug!("elodin-db rejected a packet: {}", err.description);
                record_error(&last_error, &err.description);
            }
            buf = pkt.into_buf().into_inner();
        }
    })
    .drop_guard();
    Ok(Conn {
        sink: PacketSink::new(writer),
        _reader: reader_task,
        registered: false,
    })
}

async fn send_row(
    addr: SocketAddr,
    conn: &mut Option<Conn>,
    reg: &Registration,
    data: &[u8],
    ctx: &WriterCtx,
) -> Result<(), impeller2_stellar::Error> {
    if conn.is_none() {
        *conn = Some(connect(addr, ctx).await?);
        ctx.state.store(writer_state::CONNECTED, Ordering::Relaxed);
    }
    let c = conn.as_mut().unwrap();

    if !c.registered {
        for field in &reg.fields {
            let comp_id = ComponentId::new(&field.name);
            let mut msg = SetComponentMetadata::new(comp_id, &field.name);
            if let Some(names) = &field.element_names {
                msg = msg.metadata(
                    [("element_names".to_string(), names.clone())]
                        .into_iter()
                        .collect(),
                );
            }
            let (result, _) = c.sink.send(&msg).await;
            result?;
        }
        let time_field = raw_table(0, 8);
        // Collect the field builders BEFORE calling vtable(): the builder
        // deduplicates shared ops by Arc address, so all ops must stay alive
        // for the whole build (see VTableBuilder::keepalive).
        let fields: Vec<_> = reg
            .fields
            .iter()
            .map(|field| {
                let comp = component(ComponentId::new(&field.name));
                let ts_op = if reg.timestamp_ns {
                    timestamp_ns(time_field.clone(), comp)
                } else {
                    timestamp(time_field.clone(), comp)
                };
                raw_field(
                    field.offset as u16,
                    field.size as u16,
                    schema(field.prim, &field.dims, ts_op),
                )
            })
            .collect();
        let vtable_def = vtable(fields);
        let vtable_msg = VTableMsg {
            id: reg.vtable_id,
            vtable: vtable_def,
        };
        let (result, _) = c.sink.send(&vtable_msg).await;
        result?;
        c.registered = true;
    }

    let mut packet = LenPacket::table(reg.vtable_id, data.len());
    packet.extend_from_slice(data);
    let (result, _) = c.sink.send(packet).await;
    result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(v: u8) -> RowCmd {
        RowCmd {
            data: vec![v],
            result_tx: None,
        }
    }

    fn drain(q: &RowQueue) -> Vec<u8> {
        let mut inner = q.inner.lock().unwrap();
        inner.rows.drain(..).map(|c| c.data[0]).collect()
    }

    #[test]
    fn drop_oldest_sheds_front() {
        let q = RowQueue::new(2, QueuePolicy::DropOldest);
        assert_eq!(q.push_nowait(cmd(1)), 0);
        assert_eq!(q.push_nowait(cmd(2)), 0);
        assert_eq!(q.push_nowait(cmd(3)), 1); // sheds row 1
        assert_eq!(drain(&q), vec![2, 3]);
    }

    #[test]
    fn drop_newest_discards_incoming() {
        let q = RowQueue::new(2, QueuePolicy::DropNewest);
        assert_eq!(q.push_nowait(cmd(1)), 0);
        assert_eq!(q.push_nowait(cmd(2)), 0);
        assert_eq!(q.push_nowait(cmd(3)), 1); // discards row 3
        assert_eq!(drain(&q), vec![1, 2]);
    }

    #[test]
    fn closed_queue_drops_all() {
        let q = RowQueue::new(2, QueuePolicy::DropOldest);
        q.close();
        assert_eq!(q.push_nowait(cmd(1)), 1);
        assert!(q.push_blocking(cmd(2)).is_err());
    }

    #[test]
    fn vtable_ids_are_unique_in_process() {
        let a = next_vtable_id();
        let b = next_vtable_id();
        assert_ne!(a, b);
    }
}
