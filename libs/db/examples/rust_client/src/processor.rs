use anyhow::Result;
use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, Timestamp};
use impeller2_stellar::SubStream;
use impeller2_wkt::{StreamReply, VTableMsg};
use std::collections::HashMap;
use std::time::Instant;
use stellarator::buf::Slice;
use tracing::{debug, info};

use crate::discovery::DiscoveredComponent;
use crate::tui::TelemetryRow;

/// Stores telemetry values
#[derive(Debug, Clone)]
struct TelemetryValue {
    name: String,
    values: Vec<f64>,
    unit: String,
}

/// Processes incoming telemetry packets
pub struct TelemetryProcessor {
    components: HashMap<ComponentId, DiscoveredComponent>,
    packet_count: usize,
    latest_values: HashMap<ComponentId, TelemetryValue>,
    last_timestamp: Option<Timestamp>,
    /// Registry of VTables for decomponentizing packets
    vtable_registry: HashMapRegistry,
    /// Channel to send telemetry to the TUI
    telemetry_sender: Option<async_channel::Sender<TelemetryRow>>,
}

impl TelemetryProcessor {
    pub fn new(components: HashMap<ComponentId, DiscoveredComponent>) -> Self {
        Self {
            components,
            packet_count: 0,
            latest_values: HashMap::new(),
            last_timestamp: None,
            vtable_registry: HashMapRegistry::default(),
            telemetry_sender: None,
        }
    }

    /// Set the channel sender for telemetry data
    pub fn set_telemetry_sender(&mut self, sender: async_channel::Sender<TelemetryRow>) {
        self.telemetry_sender = Some(sender);
    }

    /// Process incoming packets from a subscription stream
    pub async fn process_sub_stream(
        &mut self,
        stream: &mut SubStream<'_, StreamReply<Slice<Vec<u8>>>>,
    ) -> Result<()> {
        loop {
            // Wait for packets (blocking)
            match stream.next().await {
                Ok(reply) => match reply {
                    StreamReply::Table(table) => {
                        self.handle_table(table).await?;
                    }
                    StreamReply::VTable(vtable_msg) => {
                        self.handle_vtable(vtable_msg);
                    }
                },
                Err(e) => {
                    debug!("Stream ended: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a VTable message
    fn handle_vtable(&mut self, vtable_msg: VTableMsg) {
        debug!("Received VTable with ID: {:?}", vtable_msg.id);
        info!("Storing VTable for packet ID: {:?}", vtable_msg.id);

        // Store the VTable in our registry
        self.vtable_registry
            .map
            .insert(vtable_msg.id, vtable_msg.vtable);
    }

    /// Handle a table packet  
    async fn handle_table(
        &mut self,
        table: impeller2::types::OwnedTable<Slice<Vec<u8>>>,
    ) -> Result<()> {
        self.packet_count += 1;

        // Create extractor for this packet
        let mut extractor = TelemetryExtractor::new(&self.components, &mut self.latest_values);

        // Use the VTable registry to decomponentize the table
        match table.sink(&self.vtable_registry, &mut extractor) {
            Ok(Ok(())) => {
                // Update timestamp from latest values if available
                if let Some(_first_value) = self.latest_values.values().next() {
                    self.last_timestamp = Some(Timestamp(self.packet_count as i64 * 1000));
                }

                // Send telemetry data to TUI via channel
                if let Some(sender) = &self.telemetry_sender {
                    for value in self.latest_values.values() {
                        let row = TelemetryRow {
                            _timestamp: Instant::now(),
                            component_name: value.name.clone(),
                            values: value.values.clone(),
                            unit: value.unit.clone(),
                            is_waiting: false, // This is real data
                        };
                        let _ = sender.send(row).await;
                    }
                }
            }
            Ok(Err(e)) => {
                debug!("Error in extractor: {:?}", e);
            }
            Err(e) => {
                if self.packet_count == 1 {
                    info!("Waiting for VTable definitions...");
                }
                debug!("VTable not found for packet ID {:?}: {}", table.id, e);
            }
        }

        debug!(
            "Processed table packet #{} (ID: {:?})",
            self.packet_count, table.id
        );

        Ok(())
    }
}

/// Helper to extract telemetry values from packets
struct TelemetryExtractor<'a> {
    components: &'a HashMap<ComponentId, DiscoveredComponent>,
    latest_values: &'a mut HashMap<ComponentId, TelemetryValue>,
}

impl<'a> TelemetryExtractor<'a> {
    fn new(
        components: &'a HashMap<ComponentId, DiscoveredComponent>,
        latest_values: &'a mut HashMap<ComponentId, TelemetryValue>,
    ) -> Self {
        Self {
            components,
            latest_values,
        }
    }
}

impl Decomponentize for TelemetryExtractor<'_> {
    type Error = anyhow::Error;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        // Find the component definition
        if let Some(component) = self.components.get(&component_id) {
            // Extract values based on the component view type
            let values = match value {
                ComponentView::F64(array) => array.buf().to_vec(),
                ComponentView::F32(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::U64(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::I64(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::U32(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::I32(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::U16(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::I16(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::U8(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::I8(array) => array.buf().iter().map(|&v| v as f64).collect(),
                ComponentView::Bool(array) => array
                    .buf()
                    .iter()
                    .map(|&v| if v { 1.0 } else { 0.0 })
                    .collect(),
            };

            // Extract unit from metadata
            let unit = component.metadata.get("unit").cloned().unwrap_or_default();

            // Store the latest values
            self.latest_values.insert(
                component_id,
                TelemetryValue {
                    name: component.name.clone(),
                    values,
                    unit,
                },
            );
        } else {
            debug!("Unknown component ID: {:?}", component_id);
        }

        Ok(())
    }
}
