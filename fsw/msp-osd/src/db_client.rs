use anyhow::{Context, Result};
use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::schema::Schema;
use impeller2::types::{ComponentId, ComponentView, PrimType, Timestamp};
use impeller2_stellar::Client;
use impeller2_wkt::{DumpMetadata, DumpMetadataResp, DumpSchema, DumpSchemaResp, Stream, StreamBehavior, StreamReply};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::telemetry::{SystemStatus, TelemetryProcessor};

pub struct DbClient {
    addr: SocketAddr,
    component_names: Vec<String>,
    vtable_registry: HashMapRegistry,
}

impl DbClient {
    pub fn new(addr: SocketAddr, component_names: Vec<String>) -> Self {
        Self {
            addr,
            component_names,
            vtable_registry: HashMapRegistry::default(),
        }
    }

    pub async fn connect_and_stream(
        mut self,
        telemetry_processor: Arc<TelemetryProcessor>,
        update_tx: async_channel::Sender<()>,
    ) -> Result<()> {
        loop {
            match self.run_stream_loop(telemetry_processor.clone(), update_tx.clone()).await {
                Ok(()) => {
                    info!("Stream ended normally, will reconnect...");
                }
                Err(e) => {
                    warn!("Stream error: {}, will retry...", e);
                }
            }

            // Update status to indicate reconnection
            telemetry_processor
                .update_status(SystemStatus::Initializing)
                .await;

            // Wait before retrying
            stellarator::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn run_stream_loop(
        &mut self,
        telemetry_processor: Arc<TelemetryProcessor>,
        update_tx: async_channel::Sender<()>,
    ) -> Result<()> {
        info!("Attempting to connect to database at {}", self.addr);
        
        // Connect to database
        let mut client = self.connect_with_retry().await?;

        // Update connection status
        telemetry_processor.set_db_connected(true).await;
        telemetry_processor
            .update_status(SystemStatus::Ready)
            .await;

        // Discover components
        let _components = self.discover_components(&mut client).await?;

        // Subscribe to real-time stream
        let stream = Stream {
            behavior: StreamBehavior::RealTime,
            id: 1,
        };

        // Create stream subscription
        let mut sub_stream = client
            .stream(&stream)
            .await
            .context("Failed to create stream subscription")?;

        info!("Subscribed to real-time telemetry stream");

        // Process incoming telemetry
        loop {
            match sub_stream.next().await {
                Ok(reply) => match reply {
                    StreamReply::Table(table) => {
                        debug!("Received table data, ID: {:?}", table.id);
                        
                        // Create extractor for this table
                        let mut extractor = TelemetryExtractor::new(
                            &_components,
                            telemetry_processor.clone(),
                        );
                        
                        // Try to decomponentize the table
                        match table.sink(&self.vtable_registry, &mut extractor) {
                            Ok(Ok(())) => {
                                // Successfully processed table
                                telemetry_processor.increment_update_count().await;
                                let _ = update_tx.send(()).await;
                            }
                            Ok(Err(e)) => {
                                debug!("Error in extractor: {:?}", e);
                            }
                            Err(e) => {
                                debug!("VTable not found for packet ID {:?}: {}", table.id, e);
                            }
                        }
                    }
                    StreamReply::VTable(vtable_msg) => {
                        info!("Received VTable message, ID: {:?}", vtable_msg.id);
                        self.vtable_registry.map.insert(vtable_msg.id, vtable_msg.vtable);
                    }
                },
                Err(e) => {
                    warn!("Stream error: {}", e);
                    telemetry_processor.set_db_connected(false).await;
                    break;
                }
            }
        }

        Ok(())
    }

    async fn connect_with_retry(&self) -> Result<Client> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match Client::connect(self.addr).await {
                Ok(client) => {
                    info!("Connected to database at {} (attempt #{})", self.addr, attempt);
                    return Ok(client);
                }
                Err(e) => {
                    if attempt == 1 {
                        info!("Waiting for database at {} - {}", self.addr, e);
                    } else if attempt % 10 == 0 {
                        info!(
                            "Still waiting for database at {} (attempt #{}) - {}",
                            self.addr, attempt, e
                        );
                    }
                    stellarator::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    async fn discover_components(&self, client: &mut Client) -> Result<HashMap<ComponentId, ComponentInfo>> {
        let mut components = HashMap::new();

        // Request metadata dump
        let dump_metadata = DumpMetadata;
        let metadata_resp: DumpMetadataResp = client
            .request(&dump_metadata)
            .await
            .context("Failed to dump metadata")?;

        // Request schema dump
        let dump_schema = DumpSchema;
        let schema_resp: DumpSchemaResp = client
            .request(&dump_schema)
            .await
            .context("Failed to dump schemas")?;

        // Process component metadata
        for metadata in metadata_resp.component_metadata {
            // Check if this component is one we're interested in
            if self.component_names.contains(&metadata.name) {
                // Find matching schema
                if let Some(schema) = schema_resp
                    .schemas
                    .iter()
                    .find(|(id, _)| **id == metadata.component_id)
                    .map(|(_, s)| s.clone())
                {
                    info!(
                        "Found component: {} (ID: {}, Type: {:?})",
                        metadata.name, metadata.component_id, format_schema(&schema)
                    );
                    
                    components.insert(
                        metadata.component_id,
                        ComponentInfo {
                            id: metadata.component_id,
                            name: metadata.name.clone(),
                            schema,
                        },
                    );
                }
            }
        }

        if components.is_empty() {
            warn!(
                "No matching components found. Looking for: {:?}",
                self.component_names
            );
        } else {
            info!("Discovered {} relevant components", components.len());
        }

        Ok(components)
    }
}

#[derive(Debug, Clone)]
struct ComponentInfo {
    id: ComponentId,
    name: String,
    schema: Schema<Vec<u64>>,
}

/// Format a schema for display/logging
fn format_schema(schema: &Schema<Vec<u64>>) -> String {
    let prim_type = format_prim_type(schema.prim_type());
    let shape = schema.dim();

    if shape.is_empty() {
        prim_type.to_string()
    } else {
        format!(
            "{}[{}]",
            prim_type,
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Format primitive type for display
fn format_prim_type(prim_type: PrimType) -> &'static str {
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
}

/// Helper to extract telemetry values from packets
struct TelemetryExtractor {
    components: HashMap<ComponentId, ComponentInfo>,
    telemetry_processor: Arc<TelemetryProcessor>,
}

impl TelemetryExtractor {
    fn new(
        components: &HashMap<ComponentId, ComponentInfo>,
        telemetry_processor: Arc<TelemetryProcessor>,
    ) -> Self {
        Self {
            components: components.clone(),
            telemetry_processor,
        }
    }
}

impl Decomponentize for TelemetryExtractor {
    type Error = anyhow::Error;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        component_view: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        // Get component info
        if let Some(component_info) = self.components.get(&component_id) {
            let component_name = &component_info.name;
            
            // Extract values as f64 vector
            let values: Vec<f64> = match component_view {
                ComponentView::F64(array) => array.buf().to_vec(),
                ComponentView::F32(array) => array.buf().iter().map(|&v| v as f64).collect(),
                _ => return Ok(()), // Skip non-float components for now
            };
            
            // Process based on component name
            match component_name.as_str() {
                name if name.ends_with(".world_pos") && values.len() >= 7 => {
                    // world_pos is [quat.w, quat.x, quat.y, quat.z, pos.x, pos.y, pos.z]
                    debug!("Extracting position from {}: [{}, {}, {}]", name, values[4], values[5], values[6]);
                    let proc = self.telemetry_processor.clone();
                    let (x, y, z) = (values[4], values[5], values[6]);
                    stellarator::spawn(async move {
                        proc.update_position(x, y, z).await;
                    });
                }
                name if name.ends_with(".world_vel") && values.len() >= 6 => {
                    // world_vel is [ang_vel.x, ang_vel.y, ang_vel.z, lin_vel.x, lin_vel.y, lin_vel.z]
                    debug!("Extracting velocity from {}: [{}, {}, {}]", name, values[3], values[4], values[5]);
                    let proc = self.telemetry_processor.clone();
                    let (vx, vy, vz) = (values[3], values[4], values[5]);
                    stellarator::spawn(async move {
                        proc.update_velocity(vx, vy, vz).await;
                    });
                }
                name if name.ends_with(".gyro") && values.len() >= 3 => {
                    debug!("Extracting gyro from {}: [{}, {}, {}]", name, values[0], values[1], values[2]);
                    let proc = self.telemetry_processor.clone();
                    let (x, y, z) = (values[0] as f32, values[1] as f32, values[2] as f32);
                    stellarator::spawn(async move {
                        proc.update_gyro(x, y, z).await;
                    });
                }
                name if name.ends_with(".accel") && values.len() >= 3 => {
                    debug!("Extracting accel from {}: [{}, {}, {}]", name, values[0], values[1], values[2]);
                    let proc = self.telemetry_processor.clone();
                    let (x, y, z) = (values[0] as f32, values[1] as f32, values[2] as f32);
                    stellarator::spawn(async move {
                        proc.update_accel(x, y, z).await;
                    });
                }
                name if name.ends_with(".magnetometer") && values.len() >= 3 => {
                    debug!("Extracting magnetometer from {}: [{}, {}, {}]", name, values[0], values[1], values[2]);
                    let proc = self.telemetry_processor.clone();
                    let (x, y, z) = (values[0] as f32, values[1] as f32, values[2] as f32);
                    stellarator::spawn(async move {
                        proc.update_magnetometer(x, y, z).await;
                    });
                }
                _ => {
                    // Log other components for debugging
                    debug!("Skipping component: {} (len={})", component_name, values.len());
                }
            }
        }

        Ok(())
    }
}
