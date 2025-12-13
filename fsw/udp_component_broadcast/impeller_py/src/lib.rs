//! Python bindings for Elodin-DB impeller2 protocol
//!
//! Provides a simple Python API for subscribing to and sending component data
//! to/from Elodin-DB instances.

use anyhow::{Context, Result};
use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, LenPacket, PrimType, Timestamp};
use impeller2::vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::{
    DumpMetadata, DumpMetadataResp, DumpSchema, DumpSchemaResp, SetComponentMetadata, Stream,
    StreamBehavior, StreamReply, VTableMsg,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use stellarator::struc_con::{Joinable, Thread, stellar};
use tracing::{debug, info, warn};

/// Command sent to the sender thread
struct SendCommand {
    vtable_id: [u8; 2],
    comp_id: ComponentId,
    component_name: String,
    values: Vec<f64>,
    timestamp_us: i64,
    num_values: usize,
    /// Channel to send result back
    result_tx: std::sync::mpsc::SyncSender<Result<()>>,
}

/// Connection state for resilience tracking
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// Component data returned to Python
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

/// Component info from discovery
#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub prim_type: String,
    #[pyo3(get)]
    pub shape: Vec<u64>,
}

#[pymethods]
impl ComponentInfo {
    fn __repr__(&self) -> String {
        format!(
            "ComponentInfo(name='{}', type='{}', shape={:?})",
            self.name, self.prim_type, self.shape
        )
    }
}

/// Internal state for component values
struct ComponentValue {
    name: String,
    values: Vec<f64>,
    shape: Vec<usize>,
    timestamp: i64,
}

/// Python client for Elodin-DB
#[pyclass]
pub struct ImpellerClient {
    addr: SocketAddr,
    connected: bool,
    /// Latest component values (thread-safe)
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    /// Discovered components (ComponentId -> name)
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    /// Flag to stop background thread
    running: Arc<Mutex<bool>>,
    /// Background stellarator thread handle for receiving
    recv_thread: Option<Thread<Option<()>>>,
    /// VTable IDs registered for sending
    send_vtables: HashMap<String, [u8; 2]>,
    /// Next VTable ID for sending
    next_vtable_id: u8,
    /// Connection state for subscription
    connection_state: Arc<Mutex<ConnectionState>>,
    /// Sender thread command channel
    sender_tx: Option<Sender<SendCommand>>,
    /// Sender thread handle
    sender_thread: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl ImpellerClient {
    /// Create a new client for the given address (e.g., "127.0.0.1:2240")
    #[new]
    fn new(addr: &str) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid address: {}", e)))?;

        Ok(Self {
            addr,
            connected: false,
            latest_values: Arc::new(Mutex::new(HashMap::new())),
            components: Arc::new(Mutex::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
            recv_thread: None,
            send_vtables: HashMap::new(),
            next_vtable_id: 100, // Start at 100 to avoid conflicts
            connection_state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
            sender_tx: None,
            sender_thread: None,
        })
    }

    /// Register a component name to track (computes the ComponentId from the name)
    fn track_component(&mut self, name: &str) -> PyResult<()> {
        let comp_id = ComponentId::new(name);
        if let Ok(mut components) = self.components.lock() {
            components.insert(comp_id, name.to_string());
            info!("Tracking component '{}' (id={:?})", name, comp_id);
        }
        Ok(())
    }

    /// Connect to the database (initializes client state and sender thread)
    fn connect(&mut self) -> PyResult<()> {
        // Start the sender thread with its own stellarator executor
        let (tx, rx) = mpsc::channel::<SendCommand>();
        let addr = self.addr;

        let sender_thread = std::thread::spawn(move || {
            run_sender_thread(addr, rx);
        });

        self.sender_tx = Some(tx);
        self.sender_thread = Some(sender_thread);
        self.connected = true;
        info!("Initialized client for {}", self.addr);
        Ok(())
    }

    /// Disconnect from the database
    fn disconnect(&mut self) {
        // Stop background thread
        if let Ok(mut running) = self.running.lock() {
            *running = false;
        }

        // Wait for recv thread to finish (with timeout)
        if let Some(handle) = self.recv_thread.take() {
            // Cancel the stellarator thread - use stellar() to run the cancel
            let _ = stellar(|| async move {
                let _ = handle.cancel().await;
            });
        }

        // Stop sender thread by dropping the channel
        drop(self.sender_tx.take());

        // Wait for sender thread to finish
        if let Some(handle) = self.sender_thread.take() {
            let _ = handle.join();
        }

        self.connected = false;
        info!("Disconnected from Elodin-DB");
    }

    /// Get current connection state
    fn get_connection_state(&self) -> String {
        self.connection_state
            .lock()
            .map(|s| format!("{:?}", *s))
            .unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Check if subscription is connected
    fn is_connected(&self) -> bool {
        self.connection_state
            .lock()
            .map(|s| *s == ConnectionState::Connected)
            .unwrap_or(false)
    }

    /// Discover all components registered in the database (with retry)
    fn discover_components(&mut self) -> PyResult<HashMap<String, ComponentInfo>> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected"));
        }

        let addr = self.addr;

        // Run discovery in a stellarator thread and wait for result
        let result = stellarator::run(|| async move { discover_components_async(addr).await });

        match result {
            Ok((info_map, id_map)) => {
                // Merge discovered components with any already tracked
                if let Ok(mut components) = self.components.lock() {
                    for (id, name) in id_map {
                        components.insert(id, name);
                    }
                }
                Ok(info_map)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to discover components: {}",
                e
            ))),
        }
    }

    /// Subscribe to real-time component updates with automatic reconnection
    fn subscribe_realtime(&mut self) -> PyResult<()> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected"));
        }

        // Mark as running
        if let Ok(mut running) = self.running.lock() {
            *running = true;
        }

        let addr = self.addr;
        let latest_values = Arc::clone(&self.latest_values);
        let running = Arc::clone(&self.running);
        let components = Arc::clone(&self.components);
        let connection_state = Arc::clone(&self.connection_state);

        // Spawn stellarator thread for receiving with reconnection logic
        let handle = stellar(move || async move {
            subscribe_with_reconnect(addr, latest_values, running, components, connection_state)
                .await;
        });

        self.recv_thread = Some(handle);
        info!("Started real-time subscription");
        Ok(())
    }

    /// Get the latest data for a component by name
    fn get_latest(&self, component_name: &str) -> PyResult<Option<ComponentData>> {
        let values = self
            .latest_values
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        Ok(values.get(component_name).map(|v| ComponentData {
            name: v.name.clone(),
            timestamp: v.timestamp,
            values: v.values.clone(),
            shape: v.shape.clone(),
        }))
    }

    /// Send component data to the database (uses persistent connection)
    fn send_component(&mut self, name: &str, values: Vec<f64>, timestamp_us: i64) -> PyResult<()> {
        self.send_component_fast(name, values, timestamp_us)
    }

    /// Send component data using the sender thread
    fn send_component_fast(
        &mut self,
        name: &str,
        values: Vec<f64>,
        timestamp_us: i64,
    ) -> PyResult<()> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected"));
        }

        // Get vtable_id first (requires mutable borrow)
        let vtable_id = self.get_or_create_vtable_id(name);
        let comp_id = ComponentId::new(name);
        let num_values = values.len();

        // Now get sender_tx (immutable borrow)
        let sender_tx = self
            .sender_tx
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Sender thread not running"))?;

        // Create a channel for the result
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);

        // Send command to sender thread
        let cmd = SendCommand {
            vtable_id,
            comp_id,
            component_name: name.to_string(),
            values,
            timestamp_us,
            num_values,
            result_tx,
        };

        sender_tx
            .send(cmd)
            .map_err(|_| PyRuntimeError::new_err("Sender thread died"))?;

        // Wait for result (with timeout)
        match result_rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(PyRuntimeError::new_err(format!(
                "Failed to send component: {}",
                e
            ))),
            Err(_) => Err(PyRuntimeError::new_err("Send timeout")),
        }
    }

    /// Reset the sender connection (useful after DB restart)
    /// This restarts the sender thread with a fresh connection
    fn reset_sender(&mut self) -> PyResult<()> {
        // Stop existing sender thread
        drop(self.sender_tx.take());
        if let Some(handle) = self.sender_thread.take() {
            let _ = handle.join();
        }

        // Start a new sender thread
        let (tx, rx) = mpsc::channel::<SendCommand>();
        let addr = self.addr;

        let sender_thread = std::thread::spawn(move || {
            run_sender_thread(addr, rx);
        });

        self.sender_tx = Some(tx);
        self.sender_thread = Some(sender_thread);

        info!("Reset sender connection");
        Ok(())
    }

    /// List all component names we're tracking
    fn list_components(&self) -> Vec<String> {
        self.components
            .lock()
            .map(|c| c.values().cloned().collect())
            .unwrap_or_default()
    }
}

impl ImpellerClient {
    fn get_or_create_vtable_id(&mut self, name: &str) -> [u8; 2] {
        if let Some(&id) = self.send_vtables.get(name) {
            return id;
        }

        let id = [self.next_vtable_id, 0];
        self.next_vtable_id = self.next_vtable_id.wrapping_add(1);
        self.send_vtables.insert(name.to_string(), id);
        id
    }
}

impl Drop for ImpellerClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// Discover components from the database
async fn discover_components_async(
    addr: SocketAddr,
) -> Result<(HashMap<String, ComponentInfo>, HashMap<ComponentId, String>)> {
    debug!("Discovering components from {}", addr);

    let mut client = Client::connect(addr)
        .await
        .context("Failed to connect for discovery")?;

    // Request metadata
    let metadata_resp: DumpMetadataResp = client
        .request(&DumpMetadata)
        .await
        .context("Failed to get metadata")?;

    // Request schemas
    let schema_resp: DumpSchemaResp = client
        .request(&DumpSchema)
        .await
        .context("Failed to get schema")?;

    let mut info_map = HashMap::new();
    let mut id_map = HashMap::new();

    for metadata in metadata_resp.component_metadata {
        let name = metadata.name.clone();

        // Find schema for this component
        let prim_type = schema_resp
            .schemas
            .iter()
            .find(|(id, _)| **id == metadata.component_id)
            .map(|(_, s)| format_prim_type(s.prim_type()))
            .unwrap_or_else(|| "unknown".to_string());

        let shape = schema_resp
            .schemas
            .iter()
            .find(|(id, _)| **id == metadata.component_id)
            .map(|(_, s)| s.dim().to_vec())
            .unwrap_or_default();

        info_map.insert(
            name.clone(),
            ComponentInfo {
                name: name.clone(),
                prim_type,
                shape,
            },
        );

        id_map.insert(metadata.component_id, name);
    }

    info!("Discovered {} components", info_map.len());

    Ok((info_map, id_map))
}

fn format_prim_type(prim_type: PrimType) -> String {
    match prim_type {
        PrimType::U8 => "u8",
        PrimType::U16 => "u16",
        PrimType::U32 => "u32",
        PrimType::U64 => "u64",
        PrimType::I8 => "i8",
        PrimType::I16 => "i16",
        PrimType::I32 => "i32",
        PrimType::I64 => "i64",
        PrimType::Bool => "bool",
        PrimType::F32 => "f32",
        PrimType::F64 => "f64",
    }
    .to_string()
}

/// Subscribe to real-time updates with automatic reconnection
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
        // Check if we should stop
        let should_run = running.lock().map(|r| *r).unwrap_or(false);
        if !should_run {
            info!("Stopping subscription loop (shutdown requested)");
            break;
        }

        // Update connection state
        if let Ok(mut state) = connection_state.lock() {
            *state = ConnectionState::Connecting;
        }

        // Try to connect and subscribe
        match subscribe_and_receive_once(
            addr,
            Arc::clone(&latest_values),
            Arc::clone(&running),
            Arc::clone(&components),
            Arc::clone(&connection_state),
        )
        .await
        {
            Ok(()) => {
                // Normal exit (running flag set to false)
                break;
            }
            Err(e) => {
                // Update state to reconnecting
                if let Ok(mut state) = connection_state.lock() {
                    *state = ConnectionState::Reconnecting;
                }

                // Check if we should stop before sleeping
                let should_run = running.lock().map(|r| *r).unwrap_or(false);
                if !should_run {
                    break;
                }

                warn!("Subscription error: {}. Retrying in {:?}...", e, backoff);
                stellarator::sleep(backoff).await;

                // Exponential backoff with cap
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }

    // Final state update
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Disconnected;
    }
}

/// Single subscription attempt
async fn subscribe_and_receive_once(
    addr: SocketAddr,
    latest_values: Arc<Mutex<HashMap<String, ComponentValue>>>,
    running: Arc<Mutex<bool>>,
    components: Arc<Mutex<HashMap<ComponentId, String>>>,
    connection_state: Arc<Mutex<ConnectionState>>,
) -> Result<()> {
    // Connect
    debug!("Attempting to connect to {}", addr);
    let mut client = Client::connect(addr)
        .await
        .with_context(|| format!("Failed to connect to {}", addr))?;

    info!("TCP connection established to {}", addr);

    // Subscribe to real-time stream
    let stream = Stream {
        behavior: StreamBehavior::RealTime,
        id: 1,
    };

    debug!("Subscribing to real-time stream on {}", addr);
    let mut sub_stream = client
        .stream(&stream)
        .await
        .context("Failed to create stream subscription")?;

    info!("Stream subscription established");

    // VTable registry
    let mut registry = HashMapRegistry::default();

    // Update connection state to connected
    if let Ok(mut state) = connection_state.lock() {
        *state = ConnectionState::Connected;
    }

    info!("Subscribed to real-time stream");

    // Reset backoff on successful connection (for next failure)

    loop {
        // Check if we should stop
        let should_run = running.lock().map(|r| *r).unwrap_or(false);
        if !should_run {
            info!("Stopping subscription loop");
            return Ok(());
        }

        // Process next packet
        // Use a short timeout via select with sleep to check running flag periodically
        let timeout = stellarator::sleep(std::time::Duration::from_millis(100));

        futures_lite::future::or(
            async {
                match sub_stream.next().await {
                    Ok(reply) => match reply {
                        StreamReply::Table(table) => {
                            // Extract components from table
                            let mut extractor = ValueExtractor {
                                components: &components,
                                latest_values: &latest_values,
                            };

                            if let Err(e) = table.sink(&registry, &mut extractor) {
                                debug!("Failed to process table: {}", e);
                            }
                        }
                        StreamReply::VTable(vtable_msg) => {
                            debug!("Received VTable {:?}", vtable_msg.id);
                            registry.map.insert(vtable_msg.id, vtable_msg.vtable);
                        }
                        StreamReply::Timestamp(_) => {
                            // Timestamp messages are informational, ignore them
                        }
                    },
                    Err(e) => {
                        debug!("Stream error: {}", e);
                    }
                }
            },
            timeout,
        )
        .await;
    }
}

/// Extracts component values from tables
struct ValueExtractor<'a> {
    components: &'a Arc<Mutex<HashMap<ComponentId, String>>>,
    latest_values: &'a Arc<Mutex<HashMap<String, ComponentValue>>>,
}

impl Decomponentize for ValueExtractor<'_> {
    type Error = anyhow::Error;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        // Get component name from our tracked components
        let name = match self.components.lock() {
            Ok(map) => match map.get(&component_id) {
                Some(n) => n.clone(),
                None => return Ok(()), // Unknown component, skip
            },
            Err(_) => return Ok(()), // Lock error, skip
        };

        // Convert values to f64
        let (values, shape) = match value {
            ComponentView::F64(array) => (array.buf().to_vec(), vec![array.buf().len()]),
            ComponentView::F32(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::U64(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::I64(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::U32(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::I32(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::U16(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::I16(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::U8(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
            ComponentView::I8(array) => (
                array.buf().iter().map(|&v| v as f64).collect(),
                vec![array.buf().len()],
            ),
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

        // Store value
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

/// Run the sender thread with its own stellarator executor
///
/// This thread maintains a persistent connection to the DB and processes
/// send commands from the channel. The stellarator executor stays alive
/// for the lifetime of the thread.
fn run_sender_thread(addr: SocketAddr, rx: Receiver<SendCommand>) {
    // Use stellar() to create a thread with a persistent stellarator executor
    let handle = stellar(move || async move { sender_thread_loop(addr, rx).await });

    // Wait for the thread to complete (when channel is closed)
    // The result is wrapped in Option due to how stellar() works
    let _ = stellarator::struc_con::thread(|_| {
        futures_lite::future::block_on(async {
            let _ = handle.join().await;
        })
    });
}

/// Main loop for the sender thread
async fn sender_thread_loop(addr: SocketAddr, rx: Receiver<SendCommand>) {
    let mut client: Option<Client> = None;
    let mut sent_vtables: HashSet<[u8; 2]> = HashSet::new();

    loop {
        // Try to receive a command (blocking in a way that plays nice with async)
        let cmd = match rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => {
                // Channel closed, exit
                debug!("Sender thread: channel closed, exiting");
                break;
            }
        };

        // Process the send command
        let result = send_one_component(
            addr,
            &mut client,
            &mut sent_vtables,
            cmd.vtable_id,
            cmd.comp_id,
            &cmd.component_name,
            cmd.values,
            cmd.timestamp_us,
            cmd.num_values,
        )
        .await;

        // Send result back
        let _ = cmd.result_tx.send(result);
    }

    debug!("Sender thread exiting");
}

/// Send a single component, managing connection and vtable state
#[allow(clippy::too_many_arguments)]
async fn send_one_component(
    addr: SocketAddr,
    client: &mut Option<Client>,
    sent_vtables: &mut HashSet<[u8; 2]>,
    vtable_id: [u8; 2],
    comp_id: ComponentId,
    component_name: &str,
    values: Vec<f64>,
    timestamp_us: i64,
    num_values: usize,
) -> Result<()> {
    // Ensure we have a connection
    if client.is_none() {
        debug!("Sender thread: connecting to {}", addr);
        let c = Client::connect(addr)
            .await
            .with_context(|| format!("Failed to connect to {}", addr))?;
        *client = Some(c);
        sent_vtables.clear(); // New connection needs fresh vtables
        info!("Sender thread: connected to {}", addr);
    }

    let c = client.as_mut().unwrap();

    // Send VTable and component metadata if needed
    if !sent_vtables.contains(&vtable_id) {
        // First, register the component metadata (name mapping)
        // This is required for new components to be discoverable in the database
        let metadata_msg = SetComponentMetadata::new(comp_id, component_name);
        let (result, _) = c.send(&metadata_msg).await;
        if let Err(e) = result {
            // Connection broken, clear it
            *client = None;
            sent_vtables.clear();
            return Err(anyhow::anyhow!("Failed to send component metadata: {}", e));
        }
        info!("Registered component metadata for '{}'", component_name);

        // Then send the VTable definition
        let time_field = raw_table(0, 8);
        let data_size = num_values * 8;

        let vtable_def = vtable(vec![raw_field(
            8,
            data_size as u16,
            schema(
                PrimType::F64,
                &[num_values as u64],
                timestamp(time_field, component(comp_id)),
            ),
        )]);

        let vtable_msg = VTableMsg {
            id: vtable_id,
            vtable: vtable_def,
        };

        let (result, _) = c.send(&vtable_msg).await;
        if let Err(e) = result {
            // Connection broken, clear it
            *client = None;
            sent_vtables.clear();
            return Err(anyhow::anyhow!("Failed to send VTable: {}", e));
        }

        sent_vtables.insert(vtable_id);
        info!("Sent VTable definition for '{}'", component_name);
    }

    // Build and send data packet
    let data_size = num_values * 8;
    let packet_size = 8 + data_size;
    let mut packet = LenPacket::table(vtable_id, packet_size);
    packet.extend_aligned(&timestamp_us.to_le_bytes());
    for value in &values {
        packet.extend_aligned(&value.to_le_bytes());
    }

    let (result, _) = c.send(packet).await;
    if let Err(e) = result {
        // Connection broken, clear it
        *client = None;
        sent_vtables.clear();
        return Err(anyhow::anyhow!("Failed to send data: {}", e));
    }

    Ok(())
}

/// Python module definition
#[pymodule]
fn impeller_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init();

    m.add_class::<ImpellerClient>()?;
    m.add_class::<ComponentData>()?;
    m.add_class::<ComponentInfo>()?;
    Ok(())
}
