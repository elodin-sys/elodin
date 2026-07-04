//! Batched telemetry writer.
//!
//! One `TableWriter` owns: one TCP connection (on a dedicated stellarator
//! thread), one vtable whose fields share a leading `i64` microsecond
//! timestamp, and one bounded command queue. Every `write_row` emits exactly
//! one `Table` packet. On reconnect the component metadata + vtable handshake
//! is replayed automatically.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use impeller2::types::{ComponentId, LenPacket, PrimType};
use impeller2::vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::{SetComponentMetadata, VTableMsg};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::parse_prim_type;

/// Globally unique vtable ids for writers in this process. Starts above the
/// range used by sims/examples to avoid collisions on shared connections.
static NEXT_VTABLE_ID: AtomicU8 = AtomicU8::new(100);

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
}

struct RowCmd {
    data: Vec<u8>,
    /// Present for blocking writes; the sender reports the send result.
    result_tx: Option<SyncSender<Result<(), String>>>,
}

/// Batched writer for a fixed set of components sharing a timestamp.
#[pyclass]
pub struct TableWriter {
    tx: Option<SyncSender<RowCmd>>,
    thread: Option<std::thread::JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
    #[pyo3(get)]
    row_size: usize,
}

#[pymethods]
impl TableWriter {
    /// `fields`: `(name, prim_type, dims, element_names, offset, size)` per
    /// component. Offsets are relative to the packet payload start; offset 0..8
    /// is reserved for the shared `i64` microsecond timestamp.
    #[new]
    #[pyo3(signature = (addr, fields, maxlen = 1024))]
    fn new(
        addr: &str,
        fields: Vec<(String, String, Vec<u64>, Option<String>, usize, usize)>,
        maxlen: usize,
    ) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        if fields.is_empty() {
            return Err(PyValueError::new_err("table_writer needs at least one field"));
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
        let vtable_id = [NEXT_VTABLE_ID.fetch_add(1, Ordering::Relaxed), 1];
        let reg = Registration {
            vtable_id,
            fields: specs,
        };
        let dropped = Arc::new(AtomicU64::new(0));
        let (tx, rx) = sync_channel::<RowCmd>(maxlen.max(1));
        let dropped_for_thread = dropped.clone();
        let thread = std::thread::Builder::new()
            .name("elodin-db-writer".into())
            .spawn(move || {
                stellarator::run(|| writer_loop(addr, rx, reg, dropped_for_thread));
            })
            .map_err(|e| PyRuntimeError::new_err(format!("failed to spawn writer thread: {e}")))?;
        Ok(Self {
            tx: Some(tx),
            thread: Some(thread),
            dropped,
            row_size,
        })
    }

    /// Rows dropped by `write_row_nowait` (queue full or writer stopped) plus
    /// rows lost to connection errors.
    #[getter]
    fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Blocking write: waits for queue space and for the row to be handed to
    /// the OS socket. Raises on connection failure.
    fn write_row(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<()> {
        self.check_row(&data)?;
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?
            .clone();
        py.allow_threads(move || {
            let (result_tx, result_rx) = sync_channel(1);
            tx.send(RowCmd {
                data,
                result_tx: Some(result_tx),
            })
            .map_err(|_| PyRuntimeError::new_err("writer thread stopped"))?;
            match result_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(format!("write failed: {e}"))),
                Err(_) => Err(PyRuntimeError::new_err("write timed out")),
            }
        })
    }

    /// Non-blocking write: enqueue and return immediately. If the queue is
    /// full or the writer is stopped the row is dropped (counted in
    /// `dropped`); this method never raises for transport reasons.
    fn write_row_nowait(&self, data: Vec<u8>) -> PyResult<()> {
        self.check_row(&data)?;
        let Some(tx) = self.tx.as_ref() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        match tx.try_send(RowCmd {
            data,
            result_tx: None,
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Close the writer and join its thread.
    fn close(&mut self, py: Python<'_>) {
        drop(self.tx.take());
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
        drop(self.tx.take());
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

async fn writer_loop(
    addr: SocketAddr,
    rx: Receiver<RowCmd>,
    reg: Registration,
    dropped: Arc<AtomicU64>,
) {
    let mut client: Option<Client> = None;
    let mut registered = false;

    loop {
        let Ok(cmd) = rx.recv() else {
            // channel closed -> writer dropped, exit
            break;
        };
        let result = send_row(addr, &mut client, &mut registered, &reg, &cmd.data).await;
        if let Err(ref e) = result {
            client = None;
            registered = false;
            if cmd.result_tx.is_none() {
                dropped.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("dropped row: {e}");
            }
        }
        if let Some(result_tx) = cmd.result_tx {
            let _ = result_tx.send(result.map_err(|e| e.to_string()));
        }
    }
}

async fn send_row(
    addr: SocketAddr,
    client: &mut Option<Client>,
    registered: &mut bool,
    reg: &Registration,
    data: &[u8],
) -> Result<(), impeller2_stellar::Error> {
    if client.is_none() {
        let c = Client::connect(addr).await?;
        *client = Some(c);
        *registered = false;
    }
    let c = client.as_mut().unwrap();

    if !*registered {
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
            let (result, _) = c.send(&msg).await;
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
                raw_field(
                    field.offset as u16,
                    field.size as u16,
                    schema(
                        field.prim,
                        &field.dims,
                        timestamp(time_field.clone(), component(ComponentId::new(&field.name))),
                    ),
                )
            })
            .collect();
        let vtable_def = vtable(fields);
        let vtable_msg = VTableMsg {
            id: reg.vtable_id,
            vtable: vtable_def,
        };
        let (result, _) = c.send(&vtable_msg).await;
        result?;
        *registered = true;
    }

    let mut packet = LenPacket::table(reg.vtable_id, data.len());
    packet.extend_from_slice(data);
    let (result, _) = c.send(packet).await;
    result?;
    Ok(())
}
