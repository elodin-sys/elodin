use crate::backends::Backend;
use crate::config::SerialConfig;
use crate::osd_grid::OsdGrid;
use anyhow::Result;
use async_trait::async_trait;
use serialport::SerialPort;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, trace};

// MSP v1 command IDs
const MSP_DISPLAYPORT: u8 = 182;

// DisplayPort subcommands
const MSP_DP_HEARTBEAT: u8 = 0;
const MSP_DP_CLEAR_SCREEN: u8 = 2;
const MSP_DP_WRITE_STRING: u8 = 3;
const MSP_DP_DRAW_SCREEN: u8 = 4;
#[allow(dead_code)]
const MSP_DP_OPTIONS: u8 = 5;

// MSP v2 command IDs
const MSP2_COMMON_SET_RECORDING: u16 = 0x3005;

pub struct DisplayPortBackend {
    port: Arc<Mutex<Box<dyn SerialPort>>>,
    last_heartbeat: std::time::Instant,
    /// Used in async trait impl (false positive dead_code warning due to async_trait macro)
    #[allow(dead_code)]
    auto_record: bool,
}

/// CRC8-DVB-S2 checksum algorithm used by MSP v2
fn crc8_dvb_s2(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0xD5;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

impl DisplayPortBackend {
    pub fn new(config: &SerialConfig) -> Result<Self> {
        let port = serialport::new(&config.port, config.baud)
            .timeout(Duration::from_millis(100))
            .open()?;

        tracing::info!("Opened serial port {} at {} baud", config.port, config.baud);
        if config.auto_record {
            tracing::info!("Auto-record enabled: will start VTX recording on init");
        }

        Ok(Self {
            port: Arc::new(Mutex::new(port)),
            last_heartbeat: std::time::Instant::now(),
            auto_record: config.auto_record,
        })
    }

    /// Encode an MSP v1 packet
    fn encode_msp_v1(cmd: u8, payload: &[u8]) -> Vec<u8> {
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

        trace!(
            "MSPv1 packet: cmd={}, size={}, checksum={:02x}",
            cmd,
            size,
            cksum
        );
        out
    }

    /// Encode an MSP v2 packet
    /// MSPv2 format: $X< [flags:1] [cmd:2 LE] [size:2 LE] [payload:N] [crc8:1]
    fn encode_msp_v2(cmd: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + payload.len());
        out.push(b'$');
        out.push(b'X');
        out.push(b'<'); // Direction: Request to device

        let flags: u8 = 0;
        out.push(flags);
        out.extend_from_slice(&cmd.to_le_bytes()); // Command ID (little-endian)
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes()); // Size (little-endian)
        out.extend_from_slice(payload);

        // CRC8 over flags, cmd, size, and payload (bytes 3 onwards, excluding header)
        let crc = crc8_dvb_s2(&out[3..]);
        out.push(crc);

        trace!(
            "MSPv2 packet: cmd=0x{:04x}, size={}, crc={:02x}",
            cmd,
            payload.len(),
            crc
        );
        out
    }

    fn send_heartbeat(&self) -> Result<()> {
        let payload = [MSP_DP_HEARTBEAT];
        let packet = Self::encode_msp_v1(MSP_DISPLAYPORT, &payload);

        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        port.flush()?;

        debug!("Sent MSP_DP_HEARTBEAT");
        Ok(())
    }

    fn send_clear_screen(&self) -> Result<()> {
        let payload = [MSP_DP_CLEAR_SCREEN];
        let packet = Self::encode_msp_v1(MSP_DISPLAYPORT, &payload);

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

        let packet = Self::encode_msp_v1(MSP_DISPLAYPORT, &payload);

        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;

        trace!(
            "Sent MSP_DP_WRITE_STRING: row={}, col={}, text='{}'",
            row,
            col,
            text
        );
        Ok(())
    }

    fn send_draw_screen(&self) -> Result<()> {
        let payload = [MSP_DP_DRAW_SCREEN];
        let packet = Self::encode_msp_v1(MSP_DISPLAYPORT, &payload);

        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        port.flush()?;

        trace!("Sent MSP_DP_DRAW_SCREEN");
        Ok(())
    }

    #[allow(dead_code)]
    fn send_options(&self, font_index: u8) -> Result<()> {
        let payload = [MSP_DP_OPTIONS, font_index];
        let packet = Self::encode_msp_v1(MSP_DISPLAYPORT, &payload);

        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;

        debug!("Sent MSP_DP_OPTIONS with font_index={}", font_index);
        Ok(())
    }

    /// Send MSP2_COMMON_SET_RECORDING command to start or stop VTX recording
    /// Uses MSPv2 protocol (Walksnail Avatar compatible)
    fn send_recording_command(&self, start: bool) -> Result<()> {
        let payload = [if start { 0x01 } else { 0x00 }];
        let packet = Self::encode_msp_v2(MSP2_COMMON_SET_RECORDING, &payload);

        let mut port = self.port.lock().unwrap();
        port.write_all(&packet)?;
        port.flush()?;

        info!(
            "Sent MSP2_COMMON_SET_RECORDING: {}",
            if start { "START" } else { "STOP" }
        );
        Ok(())
    }

    /// Start VTX recording (convenience method)
    /// Used in async trait impl (false positive dead_code warning due to async_trait macro)
    #[allow(dead_code)]
    pub fn start_recording(&self) -> Result<()> {
        self.send_recording_command(true)
    }

    /// Stop VTX recording (convenience method)
    #[allow(dead_code)]
    pub fn stop_recording(&self) -> Result<()> {
        self.send_recording_command(false)
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

        // Start recording if auto_record is enabled
        if self.auto_record {
            self.start_recording()?;
        }

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
