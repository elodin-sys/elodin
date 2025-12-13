use anyhow::{Context, Result};
use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, Timestamp};
use impeller2_stellar::Client;
use impeller2_wkt::{
    DumpMetadata, DumpMetadataResp, FixedRateBehavior, InitialTimestamp, Stream, StreamBehavior,
    StreamReply,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::config::InputMappings;
use crate::telemetry::{SystemStatus, TelemetryProcessor};

/// Component info discovered from the database
#[derive(Debug, Clone)]
struct ComponentInfo {
    name: String,
}

/// Database client that streams telemetry based on configured input mappings
pub struct DbClient {
    addr: SocketAddr,
    mappings: InputMappings,
    vtable_registry: HashMapRegistry,
    replay: bool,
}

impl DbClient {
    pub fn new(addr: SocketAddr, mappings: InputMappings, replay: bool) -> Self {
        Self {
            addr,
            mappings,
            vtable_registry: HashMapRegistry::default(),
            replay,
        }
    }

    /// Get the unique list of component names required by the mappings
    fn required_components(&self) -> Vec<String> {
        let mut components = vec![
            self.mappings.position.component.clone(),
            self.mappings.orientation.component.clone(),
            self.mappings.velocity.component.clone(),
        ];
        // Add target component if configured
        if let Some(target) = &self.mappings.target {
            components.push(target.component.clone());
        }
        components.sort();
        components.dedup();
        components
    }

    pub async fn connect_and_stream(
        mut self,
        telemetry_processor: Arc<TelemetryProcessor>,
        update_tx: async_channel::Sender<()>,
    ) -> Result<()> {
        loop {
            match self
                .run_stream_loop(telemetry_processor.clone(), update_tx.clone())
                .await
            {
                Ok(()) => {
                    info!("Stream ended normally, will reconnect...");
                }
                Err(e) => {
                    warn!("Stream error: {}, will retry...", e);
                }
            }

            telemetry_processor
                .update_status(SystemStatus::Initializing)
                .await;
            stellarator::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn run_stream_loop(
        &mut self,
        telemetry_processor: Arc<TelemetryProcessor>,
        update_tx: async_channel::Sender<()>,
    ) -> Result<()> {
        info!("Attempting to connect to database at {}", self.addr);

        let mut client = self.connect_with_retry().await?;

        telemetry_processor.set_db_connected(true).await;
        telemetry_processor.update_status(SystemStatus::Ready).await;

        // Discover components and build ID -> name mapping
        let components = self.discover_components(&mut client).await?;

        // Select stream behavior based on replay mode
        let behavior = if self.replay {
            // Use 60Hz playback with matching timestep for 1x speed
            // The DB finds the nearest sample for each timestamp, so this works
            // regardless of the original recording rate
            const PLAYBACK_HZ: u64 = 60;
            StreamBehavior::FixedRate(FixedRateBehavior {
                timestep: 1_000_000_000 / PLAYBACK_HZ, // nanoseconds per tick
                frequency: PLAYBACK_HZ,
                initial_timestamp: InitialTimestamp::Earliest,
            })
        } else {
            StreamBehavior::RealTime
        };

        let stream = Stream { behavior, id: 1 };

        let mut sub_stream = client
            .stream(&stream)
            .await
            .context("Failed to create stream subscription")?;

        if self.replay {
            info!("Subscribed to replay stream (from earliest timestamp)");
        } else {
            info!("Subscribed to real-time telemetry stream");
        }

        // Process incoming telemetry
        loop {
            match sub_stream.next().await {
                Ok(reply) => match reply {
                    StreamReply::Table(table) => {
                        debug!("Received table data, ID: {:?}", table.id);

                        let mut extractor = TelemetryExtractor::new(
                            &components,
                            &self.mappings,
                            telemetry_processor.clone(),
                        );

                        match table.sink(&self.vtable_registry, &mut extractor) {
                            Ok(Ok(())) => {
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
                        self.vtable_registry
                            .map
                            .insert(vtable_msg.id, vtable_msg.vtable);
                    }
                    StreamReply::Timestamp(ts) => {
                        debug!("Stream timestamp: {:?}", ts.timestamp);
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
                    info!(
                        "Connected to database at {} (attempt #{})",
                        self.addr, attempt
                    );
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

    async fn discover_components(
        &self,
        client: &mut Client,
    ) -> Result<HashMap<ComponentId, ComponentInfo>> {
        let mut components = HashMap::new();
        let required = self.required_components();

        info!("Looking for components: {:?}", required);

        // Request metadata dump
        let dump_metadata = DumpMetadata;
        let metadata_resp: DumpMetadataResp = client
            .request(&dump_metadata)
            .await
            .context("Failed to dump metadata")?;

        // Process component metadata
        for metadata in metadata_resp.component_metadata {
            if required.contains(&metadata.name) {
                info!(
                    "Found component: {} (ID: {})",
                    metadata.name, metadata.component_id
                );

                components.insert(
                    metadata.component_id,
                    ComponentInfo {
                        name: metadata.name.clone(),
                    },
                );
            }
        }

        if components.len() < required.len() {
            let found: Vec<_> = components.values().map(|c| c.name.clone()).collect();
            let missing: Vec<_> = required
                .iter()
                .filter(|r| !found.contains(r))
                .cloned()
                .collect();
            warn!("Missing components: {:?}", missing);
        } else {
            info!("Discovered all {} required components", components.len());
        }

        Ok(components)
    }
}

/// Extracts telemetry values based on configured input mappings
struct TelemetryExtractor {
    components: HashMap<ComponentId, ComponentInfo>,
    mappings: InputMappings,
    telemetry_processor: Arc<TelemetryProcessor>,
}

impl TelemetryExtractor {
    fn new(
        components: &HashMap<ComponentId, ComponentInfo>,
        mappings: &InputMappings,
        telemetry_processor: Arc<TelemetryProcessor>,
    ) -> Self {
        Self {
            components: components.clone(),
            mappings: mappings.clone(),
            telemetry_processor,
        }
    }

    /// Extract a value at the given index from an f64 array
    fn extract_f64(values: &[f64], idx: usize) -> Option<f64> {
        values.get(idx).copied()
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
        // Get component name for this ID
        let component_name = match self.components.get(&component_id) {
            Some(info) => &info.name,
            None => return Ok(()), // Unknown component, skip
        };

        // Extract values as f64 vector
        let values: Vec<f64> = match component_view {
            ComponentView::F64(array) => array.buf().to_vec(),
            ComponentView::F32(array) => array.buf().iter().map(|&v| v as f64).collect(),
            _ => return Ok(()), // Skip non-float components
        };

        // Check if this component is used for position
        if component_name == &self.mappings.position.component {
            let m = &self.mappings.position;
            if let (Some(x), Some(y), Some(z)) = (
                Self::extract_f64(&values, m.x),
                Self::extract_f64(&values, m.y),
                Self::extract_f64(&values, m.z),
            ) {
                debug!("Position: x={:.3}, y={:.3}, z={:.3}", x, y, z);
                let proc = self.telemetry_processor.clone();
                stellarator::spawn(async move {
                    proc.update_position(x, y, z).await;
                });
            }
        }

        // Check if this component is used for orientation
        // Elodin stores quaternions as [x, y, z, w] (scalar w is last)
        if component_name == &self.mappings.orientation.component {
            let m = &self.mappings.orientation;
            if let (Some(qx), Some(qy), Some(qz), Some(qw)) = (
                Self::extract_f64(&values, m.qx),
                Self::extract_f64(&values, m.qy),
                Self::extract_f64(&values, m.qz),
                Self::extract_f64(&values, m.qw),
            ) {
                debug!(
                    "Orientation: qx={:.3}, qy={:.3}, qz={:.3}, qw={:.3}",
                    qx, qy, qz, qw
                );
                let proc = self.telemetry_processor.clone();
                stellarator::spawn(async move {
                    proc.update_orientation(qx, qy, qz, qw).await;
                });
            }
        }

        // Check if this component is used for velocity
        if component_name == &self.mappings.velocity.component {
            let m = &self.mappings.velocity;
            if let (Some(vx), Some(vy), Some(vz)) = (
                Self::extract_f64(&values, m.x),
                Self::extract_f64(&values, m.y),
                Self::extract_f64(&values, m.z),
            ) {
                debug!("Velocity: vx={:.3}, vy={:.3}, vz={:.3}", vx, vy, vz);
                let proc = self.telemetry_processor.clone();
                stellarator::spawn(async move {
                    proc.update_velocity(vx, vy, vz).await;
                });
            }
        }

        // Check if this component is used for target position
        if let Some(target_mapping) = &self.mappings.target {
            if component_name == &target_mapping.component {
                if let (Some(x), Some(y), Some(z)) = (
                    Self::extract_f64(&values, target_mapping.x),
                    Self::extract_f64(&values, target_mapping.y),
                    Self::extract_f64(&values, target_mapping.z),
                ) {
                    debug!("Target position: x={:.3}, y={:.3}, z={:.3}", x, y, z);
                    let proc = self.telemetry_processor.clone();
                    stellarator::spawn(async move {
                        proc.update_target_position(x, y, z).await;
                    });
                }
            }
        }

        Ok(())
    }
}
