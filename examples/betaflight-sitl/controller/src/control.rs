//! Elodin DB transport for semantic pilot input.

use anyhow::Result;
use impeller2::types::{ComponentId, LenPacket, PrimType};
use impeller2::vtable::builder::{component, raw_field, schema, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::VTableMsg;

use crate::input::ControlInput;

pub struct ControlSender {
    vtable_id: [u8; 2],
    component_id: ComponentId,
    heartbeat: u64,
}

impl ControlSender {
    pub fn new() -> Self {
        Self {
            vtable_id: [3, 0],
            component_id: ComponentId::new("drone.manual_control"),
            heartbeat: 0,
        }
    }

    pub async fn send_vtable(&self, client: &mut Client) -> Result<()> {
        // roll, pitch, throttle, yaw, armed, angle_mode, heartbeat
        let table = vtable(vec![raw_field(
            0,
            (7 * size_of::<f64>()) as u16,
            schema(PrimType::F64, &[7], component(self.component_id)),
        )]);
        let (result, _) = client
            .send(&VTableMsg {
                id: self.vtable_id,
                vtable: table,
            })
            .await;
        result?;
        Ok(())
    }

    pub async fn send(&mut self, client: &mut Client, input: ControlInput) -> Result<()> {
        self.heartbeat = self.heartbeat.wrapping_add(1);
        let mut packet = LenPacket::table(self.vtable_id, 7 * size_of::<f64>());
        for value in input.as_array().into_iter().chain([self.heartbeat as f64]) {
            packet.extend_aligned(&value.to_le_bytes());
        }
        let (result, _) = client.send(packet).await;
        result?;
        Ok(())
    }
}
