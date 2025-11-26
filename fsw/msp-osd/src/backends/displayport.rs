use crate::backends::Backend;
use crate::config::SerialConfig;
use crate::osd_grid::OsdGrid;
use anyhow::Result;
use async_trait::async_trait;
use serialport::SerialPort;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, trace};

// MSP command IDs
const MSP_DISPLAYPORT: u8 = 182;

// DisplayPort subcommands
const MSP_DP_HEARTBEAT: u8 = 0;
const MSP_DP_CLEAR_SCREEN: u8 = 2;
const MSP_DP_WRITE_STRING: u8 = 3;
const MSP_DP_DRAW_SCREEN: u8 = 4;
const MSP_DP_OPTIONS: u8 = 5;

pub struct DisplayPortBackend {
    port: Arc<Mutex<Box<dyn SerialPort>>>,
    last_heartbeat: std::time::Instant,
}

impl DisplayPortBackend {
    pub fn new(config: &SerialConfig) -> Result<Self> {
        let port = serialport::new(&config.port, config.baud)
            .timeout(Duration::from_millis(100))
            .open()?;

        tracing::info!(
            "Opened serial port {} at {} baud",
            config.port,
            config.baud
        );

        Ok(Self {
            port: Arc::new(Mutex::new(port)),
            last_heartbeat: std::time::Instant::now(),
        })
    }

    /// Encode an MSP v1 packet
    fn encode_msp(cmd: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(6 + payload.len());
        out.push(b'$');
        out.push(b'M');
        out.push(b'>'); // Direction: FC -> OSD
        
        let size = payload.len() as u8;
        out.push(size);
        out.push(cmd);

        let mut cksum = size ^ cmd;
        for &b in payload {
            out.push(b);
            cksum ^= b;
        }
        out.push(cksum);
        
        trace!("MSP packet: cmd={}, size={}, checksum={:02x}", cmd, size, cksum);
        out
    }

    fn send_heartbeat(&self) -> Result<()> {
        let payload = [MSP_DP_HEARTBEAT];
        let packet = Self::encode_msp(MSP_DISPLAYPORT, &payload);
        
        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        port.flush()?;
        
        debug!("Sent MSP_DP_HEARTBEAT");
        Ok(())
    }

    fn send_clear_screen(&self) -> Result<()> {
        let payload = [MSP_DP_CLEAR_SCREEN];
        let packet = Self::encode_msp(MSP_DISPLAYPORT, &payload);
        
        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        
        trace!("Sent MSP_DP_CLEAR_SCREEN");
        Ok(())
    }

    fn send_write_string(&self, row: u8, col: u8, text: &str) -> Result<()> {
        let mut payload = Vec::with_capacity(4 + text.len() + 1);
        payload.push(MSP_DP_WRITE_STRING);
        payload.push(row);
        payload.push(col);
        payload.push(0); // Attribute: 0 = default font, no blink

        // Add text bytes
        payload.extend_from_slice(text.as_bytes());
        payload.push(0); // Null terminator

        let packet = Self::encode_msp(MSP_DISPLAYPORT, &payload);
        
        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        
        trace!("Sent MSP_DP_WRITE_STRING: row={}, col={}, text='{}'", row, col, text);
        Ok(())
    }

    fn send_draw_screen(&self) -> Result<()> {
        let payload = [MSP_DP_DRAW_SCREEN];
        let packet = Self::encode_msp(MSP_DISPLAYPORT, &payload);
        
        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        port.flush()?;
        
        trace!("Sent MSP_DP_DRAW_SCREEN");
        Ok(())
    }

    fn send_options(&self, font_index: u8) -> Result<()> {
        let payload = [MSP_DP_OPTIONS, font_index];
        let packet = Self::encode_msp(MSP_DISPLAYPORT, &payload);
        
        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        
        debug!("Sent MSP_DP_OPTIONS with font_index={}", font_index);
        Ok(())
    }
}

#[async_trait]
impl Backend for DisplayPortBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    async fn init(&mut self) -> Result<()> {
        // Send initial options to set HD font (if available)
        self.send_options(0)?; // 0 = default font
        
        // Send initial heartbeat
        self.send_heartbeat()?;
        
        Ok(())
    }

    async fn render(&mut self, grid: &OsdGrid) -> Result<()> {
        // Send heartbeat every 500ms
        if self.last_heartbeat.elapsed() > Duration::from_millis(500) {
            self.send_heartbeat()?;
            self.last_heartbeat = std::time::Instant::now();
        }

        // Clear screen
        self.send_clear_screen()?;

        // Send each non-empty line
        for (row, text) in grid.non_empty_lines() {
            // Find the first non-space character to optimize
            if let Some(start_col) = text.chars().position(|c| c != ' ') {
                let trimmed = text[start_col..].trim_end();
                if !trimmed.is_empty() {
                    self.send_write_string(row, start_col as u8, trimmed)?;
                }
            }
        }

        // Draw/present the frame
        self.send_draw_screen()?;

        Ok(())
    }

    async fn cleanup(&mut self) -> Result<()> {
        // Send a clear screen before closing
        self.send_clear_screen()?;
        self.send_draw_screen()?;
        Ok(())
    }
}
