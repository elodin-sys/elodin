use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use impeller2::types::Timestamp;

const ENV_KEY: &str = "ELODIN_RENDER_BRIDGE_SOCK";

// ---------------------------------------------------------------------------
// Batch render request (multiple cameras in one request)
// ---------------------------------------------------------------------------

/// A batch render request for one or more cameras at a given timestamp.
pub struct BatchRenderRequest {
    pub camera_names: Vec<String>,
    pub timestamp: Timestamp,
}

// ---------------------------------------------------------------------------
// Server side
// ---------------------------------------------------------------------------

pub struct RenderBridgeServer {
    listener: UnixListener,
    path: PathBuf,
    /// Persistent client connection (accepted once, reused for all requests)
    client: Mutex<Option<BufReader<UnixStream>>>,
}

impl RenderBridgeServer {
    pub fn bind() -> std::io::Result<Self> {
        let path = PathBuf::from(std::env::var(ENV_KEY).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "ELODIN_RENDER_BRIDGE_SOCK not set — render-server must be started via s10",
            )
        })?);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(false)?;
        Ok(Self {
            listener,
            path,
            client: Mutex::new(None),
        })
    }

    /// Accept a persistent client connection (blocking).
    /// Call this once during server startup after warm-up.
    pub fn accept_client(&self) -> std::io::Result<()> {
        let (stream, _) = self.listener.accept()?;
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(None)?;
        let reader = BufReader::new(stream);
        *self.client.lock().unwrap() = Some(reader);
        Ok(())
    }

    /// Check if a client is connected.
    pub fn has_client(&self) -> bool {
        self.client.lock().unwrap().is_some()
    }

    /// Read the next batch render request from the persistent client connection.
    /// Blocks until a request arrives or the connection closes.
    /// Returns None if the connection was closed.
    pub fn recv_batch(&self) -> Option<BatchRenderRequest> {
        let mut guard = self.client.lock().unwrap();
        let reader = guard.as_mut()?;

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                *guard = None;
                return None;
            }
            Ok(_) => {}
            Err(_) => {
                *guard = None;
                return None;
            }
        }

        let line = line.trim_end();
        let mut parts = line.splitn(3, ' ');
        let cmd = parts.next()?;

        match cmd {
            "RENDER" => {
                let camera_name = parts.next()?.to_string();
                let timestamp: i64 = parts.next()?.parse().ok()?;
                Some(BatchRenderRequest {
                    camera_names: vec![camera_name],
                    timestamp: Timestamp(timestamp),
                })
            }
            "RENDER_BATCH" => {
                let count: usize = parts.next()?.parse().ok()?;
                let timestamp: i64 = parts.next()?.parse().ok()?;
                let mut camera_names = Vec::with_capacity(count);
                for _ in 0..count {
                    let mut cam_line = String::new();
                    if reader.read_line(&mut cam_line).ok()? == 0 {
                        return None;
                    }
                    camera_names.push(cam_line.trim_end().to_string());
                }
                Some(BatchRenderRequest {
                    camera_names,
                    timestamp: Timestamp(timestamp),
                })
            }
            _ => None,
        }
    }

    /// Send batch response with multiple frames back to the client.
    /// Format: "FRAMES {count} {timestamp}\n" followed by "{camera_name} {len}\n{bytes}" for each frame.
    pub fn respond_batch(&self, timestamp: Timestamp, frames: &[(String, Vec<u8>)]) {
        let guard = self.client.lock().unwrap();
        let Some(reader) = guard.as_ref() else {
            return;
        };
        let stream = reader.get_ref();
        let mut writer = BufWriter::new(stream);

        let _ = writeln!(writer, "FRAMES {} {}", frames.len(), timestamp.0);
        for (camera_name, frame_bytes) in frames {
            let _ = writeln!(writer, "{} {}", camera_name, frame_bytes.len());
            let _ = writer.write_all(frame_bytes);
        }
        let _ = writer.flush();
    }

    /// Send an empty response (no frames rendered).
    pub fn respond_empty(&self) {
        let guard = self.client.lock().unwrap();
        let Some(reader) = guard.as_ref() else {
            return;
        };
        let stream = reader.get_ref();
        let mut writer = BufWriter::new(stream);
        let _ = writeln!(writer, "FRAMES 0 0");
        let _ = writer.flush();
    }
}

impl Drop for RenderBridgeServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Client side
// ---------------------------------------------------------------------------

/// Rendered frame returned from the render bridge.
pub struct RenderedFrame {
    pub camera_name: String,
    pub timestamp: Timestamp,
    pub data: Vec<u8>,
}

/// Persistent client connection to the render bridge server.
/// Created once and reused for all render requests to avoid per-request
/// connection overhead.
pub struct RenderBridgeClient {
    reader: BufReader<UnixStream>,
    /// Reused for reading frame data to avoid per-frame allocation.
    frame_buffer: Vec<u8>,
}

impl RenderBridgeClient {
    /// Connect to the render bridge server. Blocks until connection succeeds or timeout.
    pub fn connect(timeout: Duration) -> Result<Self, String> {
        let sock_path = std::env::var(ENV_KEY).map_err(|_| {
            "No render bridge available. Use 'elodin run' or 'elodin editor' for sensor camera rendering.".to_string()
        })?;

        let deadline = Instant::now() + timeout;

        let stream = loop {
            match UnixStream::connect(Path::new(&sock_path)) {
                Ok(s) => break s,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(format!("Failed to connect to render bridge: {e}")),
            }
        };

        stream
            .set_nonblocking(false)
            .map_err(|e| format!("Failed to set blocking mode: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| format!("Failed to set read timeout: {e}"))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| format!("Failed to set write timeout: {e}"))?;

        Ok(Self {
            reader: BufReader::new(stream),
            frame_buffer: Vec::new(),
        })
    }

    /// Render a single camera (convenience wrapper around render_cameras_blocking).
    pub fn render_camera(
        &mut self,
        camera_name: &str,
        timestamp: Timestamp,
    ) -> Result<Option<RenderedFrame>, String> {
        let frames = self.render_cameras(&[camera_name], timestamp)?;
        Ok(frames.into_iter().next())
    }

    /// Render multiple cameras in a single batch request.
    /// Returns a Vec of rendered frames (may be fewer than requested if some cameras fail).
    pub fn render_cameras(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
    ) -> Result<Vec<RenderedFrame>, String> {
        if camera_names.is_empty() {
            return Ok(vec![]);
        }

        let stream = self.reader.get_mut();

        if camera_names.len() == 1 {
            writeln!(stream, "RENDER {} {}", camera_names[0], timestamp.0)
                .map_err(|e| format!("Failed to send render request: {e}"))?;
        } else {
            writeln!(
                stream,
                "RENDER_BATCH {} {}",
                camera_names.len(),
                timestamp.0
            )
            .map_err(|e| format!("Failed to send batch request: {e}"))?;
            for name in camera_names {
                writeln!(stream, "{}", name)
                    .map_err(|e| format!("Failed to send camera name: {e}"))?;
            }
        }
        stream
            .flush()
            .map_err(|e| format!("Failed to flush: {e}"))?;

        let mut response_line = String::new();
        self.reader
            .read_line(&mut response_line)
            .map_err(|e| format!("Render timeout or read error: {e}"))?;

        let response_line = response_line.trim_end();

        // Parse "FRAMES {count} {timestamp}"
        let mut parts = response_line.splitn(3, ' ');
        let cmd = parts.next().unwrap_or("");
        if cmd != "FRAMES" {
            return Err(format!("Unexpected response: {response_line}"));
        }

        let count: usize = parts
            .next()
            .ok_or("Missing frame count in response")?
            .parse()
            .map_err(|_| "Invalid frame count in response")?;
        let resp_timestamp: i64 = parts
            .next()
            .ok_or("Missing timestamp in response")?
            .parse()
            .map_err(|_| "Invalid timestamp in response")?;

        let mut frames = Vec::with_capacity(count);
        for _ in 0..count {
            let mut frame_header = String::new();
            self.reader
                .read_line(&mut frame_header)
                .map_err(|e| format!("Failed to read frame header: {e}"))?;

            let frame_header = frame_header.trim_end();
            let mut header_parts = frame_header.rsplitn(2, ' ');
            let frame_len: usize = header_parts
                .next()
                .ok_or("Missing frame length")?
                .parse()
                .map_err(|_| "Invalid frame length")?;
            let camera_name = header_parts
                .next()
                .ok_or("Missing camera name")?
                .to_string();

            self.frame_buffer.resize(frame_len, 0);
            self.reader
                .read_exact(&mut self.frame_buffer[..frame_len])
                .map_err(|e| format!("Failed to read frame data: {e}"))?;

            frames.push(RenderedFrame {
                camera_name,
                timestamp: Timestamp(resp_timestamp),
                data: self.frame_buffer.clone(),
            });
        }

        Ok(frames)
    }
}

/// Legacy function: connects fresh for each call. Prefer using RenderBridgeClient
/// for better performance with multiple render calls.
pub fn render_camera_blocking(
    camera_name: &str,
    timestamp: Timestamp,
    timeout: Duration,
) -> Result<Option<RenderedFrame>, String> {
    let mut client = RenderBridgeClient::connect(timeout)?;
    client.render_camera(camera_name, timestamp)
}
