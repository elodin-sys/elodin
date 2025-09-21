use anyhow::Result;
use impeller2::types::{ComponentId, LenPacket, PrimType, Timestamp};
use impeller2::vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::VTableMsg;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Control module for sending commands to the simulation
pub struct ControlSender {
    trim_vtable_id: [u8; 2],
    trim_component_id: ComponentId,
    start_time: Instant,
    last_send_time: Instant,
}

impl ControlSender {
    pub fn new() -> Self {
        Self {
            trim_vtable_id: [3, 0],  // Use a unique VTable ID for control
            // Back to the original component name
            trim_component_id: ComponentId::new("rocket.fin_control_trim"),
            start_time: Instant::now(),
            last_send_time: Instant::now(),
        }
    }
    
    /// Send the VTable definition for trim control
    pub async fn send_vtable(&self, client: &mut Client) -> Result<()> {
        // Create VTable with timestamp field and f64 value
        let time_field = raw_table(0, 8);  // First 8 bytes for timestamp
        let vtable = vtable(vec![
            raw_field(8, 8, schema(PrimType::F64, &[], timestamp(time_field, component(self.trim_component_id)))),
        ]);
        
        let vtable_msg = VTableMsg {
            id: self.trim_vtable_id,
            vtable,
        };
        
        let (result, _) = client.send(&vtable_msg).await;
        result?;
        info!("Sent VTable definition for fin control trim");
        Ok(())
    }
    
    /// Generate and send a sinusoidal trim value
    pub async fn send_trim_update(&mut self, client: &mut Client) -> Result<()> {
        // Only send updates at ~60Hz to avoid overwhelming the database
        let now = Instant::now();
        if now.duration_since(self.last_send_time) < Duration::from_millis(16) {
            return Ok(());
        }
        self.last_send_time = now;
        
        // Calculate time since start in seconds
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        // Generate sinusoidal value: amplitude 2.0, period 4 seconds
        // This gives us a nice visible oscillation
        let frequency = 0.25; // 0.25 Hz = 4 second period
        let amplitude = 10.0;  // ±2 degrees
        let trim_value = amplitude * (2.0 * std::f64::consts::PI * frequency * elapsed).sin();
        
        // Use current timestamp - the simulation now skips writing back fin_control_trim
        let timestamp = Timestamp::now();
        
        // Build the packet with timestamp and value
        let mut packet = LenPacket::table(self.trim_vtable_id, 16); // 8 bytes timestamp + 8 bytes f64
        packet.extend_aligned(&timestamp.0.to_le_bytes());
        packet.extend_aligned(&trim_value.to_le_bytes());
        
        // Send the packet
        let (result, _) = client.send(packet).await;
        result?;
        
        debug!(
            "Sent trim control: {:.3}° at time {:.2}s (timestamp: {})",
            trim_value, elapsed, timestamp.0
        );
        
        // Log periodically for visibility
        if elapsed as u64 % 2 == 0 && (elapsed * 10.0) as u64 % 10 == 0 {
            info!("Trim control oscillating: {:.3}° (t={:.1}s)", trim_value, elapsed);
        }
        
        Ok(())
    }
}

/// Run the control loop that sends sinusoidal trim commands
pub async fn run_control_loop(client: &mut Client) -> Result<()> {
    info!("Starting sinusoidal trim control (±2° @ 0.25Hz)");
    
    let mut controller = ControlSender::new();
    
    // Send the VTable definition once
    controller.send_vtable(client).await?;
    
    // Give the database a moment to process the VTable
    stellarator::sleep(Duration::from_millis(100)).await;
    
    info!("Beginning trim control oscillation...");
    
    // Continuously send trim updates
    loop {
        controller.send_trim_update(client).await?;
        
        // Small delay to control update rate (~60Hz)
        stellarator::sleep(Duration::from_millis(10)).await;
    }
}
