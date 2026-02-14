//! Control sender for BDX jet simulation
//!
//! Sends ControlCommands [elevator, aileron, rudder, throttle] to elodin-db
//! following the pattern from libs/db/examples/rust_client/src/control.rs

use anyhow::Result;
use impeller2::types::{ComponentId, LenPacket, PrimType};
use impeller2::vtable::builder::{component, raw_field, schema, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::VTableMsg;
use std::time::{Duration, Instant};
use tracing::info;

use crate::input::ControlInput;

/// Control sender for BDX jet commands
pub struct ControlSender {
    /// VTable ID for control commands
    vtable_id: [u8; 2],
    /// Component ID for bdx.control_commands
    component_id: ComponentId,
    /// Last time we sent an update
    last_send_time: Instant,
}

impl ControlSender {
    pub fn new() -> Self {
        Self {
            vtable_id: [2, 0], // Use a different VTable ID than simulation
            component_id: ComponentId::new("bdx.control_commands"),
            last_send_time: Instant::now(),
        }
    }

    /// Send the VTable definition for control commands
    ///
    /// ControlCommands is a 4-element f64 array: [elevator, aileron, rudder, throttle]
    ///
    /// No explicit timestamp is included in the vtable; the DB will apply its
    /// own sim-aligned implicit timestamp (`apply_implicit_timestamp`) when
    /// ingesting rows from this table.  This keeps the controller's writes on
    /// the same time base as the simulation and avoids timeline flicker in the
    /// editor.
    pub async fn send_vtable(&self, client: &mut Client) -> Result<()> {
        // Layout: f64 x 4 = 32 bytes (no timestamp field)
        let vtable = vtable(vec![raw_field(
            0,  // offset: start of packet
            32, // 4 * 8 bytes for f64[4]
            schema(
                PrimType::F64,
                &[4], // Shape: 4-element array
                component(self.component_id),
            ),
        )]);

        let vtable_msg = VTableMsg {
            id: self.vtable_id,
            vtable,
        };

        let (result, _) = client.send(&vtable_msg).await;
        result?;
        info!("Sent VTable definition for bdx.control_commands");
        Ok(())
    }

    /// Send control update
    pub async fn send_control(&mut self, client: &mut Client, input: ControlInput) -> Result<()> {
        // Rate limit to ~60Hz
        let now = Instant::now();
        if now.duration_since(self.last_send_time) < Duration::from_millis(16) {
            return Ok(());
        }
        self.last_send_time = now;

        let values = input.as_array();

        // Build packet: 4 x f64 = 32 bytes (no explicit timestamp)
        let mut packet = LenPacket::table(self.vtable_id, 32);
        for value in values {
            packet.extend_aligned(&value.to_le_bytes());
        }

        // Send the packet
        let (result, _) = client.send(packet).await;
        result?;

        Ok(())
    }
}

impl Default for ControlSender {
    fn default() -> Self {
        Self::new()
    }
}
