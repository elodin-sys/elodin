use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use impeller2::types::Timestamp;

const ENV_KEY: &str = "ELODIN_RENDER_BRIDGE_SOCK";

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------------------
// Batch render request (multiple cameras in one request)
// ---------------------------------------------------------------------------

/// A batch render request for one or more cameras at a given timestamp.
pub struct BatchRenderRequest {
    pub camera_names: Vec<String>,
    pub timestamp: Timestamp,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderBridgeRespondMetrics {
    pub response_header_write_ms: f64,
    pub frame_header_write_ms: f64,
    pub frame_bytes_write_ms: f64,
    pub flush_ms: f64,
    pub frame_count: usize,
    pub total_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderBridgeClientMetrics {
    pub send_request_ms: f64,
    pub response_header_read_ms: f64,
    pub frame_header_read_ms: f64,
    pub frame_data_read_ms: f64,
    pub on_frame_ms: f64,
    pub frame_count: usize,
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// Server side
// ---------------------------------------------------------------------------

pub struct RenderBridgeServer {
    listener: UnixListener,
    path: PathBuf,
    /// Persistent client connection (accepted once, reused for all requests).
    /// The stream is cloned so reads and writes use independent buffered wrappers,
    /// preventing protocol desynchronization from shared BufReader/BufWriter state.
    client: Mutex<Option<(BufReader<UnixStream>, BufWriter<UnixStream>)>>,
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
        let write_stream = stream.try_clone()?;
        let reader = BufReader::new(stream);
        let writer = BufWriter::new(write_stream);
        *self.client.lock().unwrap() = Some((reader, writer));
        Ok(())
    }

    /// Check if a client is connected.
    pub fn has_client(&self) -> bool {
        self.client.lock().unwrap().is_some()
    }

    /// Read the next batch render request from the persistent client connection.
    /// Blocks until a request arrives or the connection closes.
    /// Returns None only if the connection was closed (EOF or I/O error).
    /// Malformed or unknown command lines are logged and skipped; the next line is read.
    pub fn recv_batch(&self) -> Option<BatchRenderRequest> {
        let mut guard = self.client.lock().unwrap();
        let (reader, _) = guard.as_mut()?;

        loop {
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
            let Some(cmd) = parts.next() else {
                tracing::warn!("Render bridge: empty command line, skipping");
                continue;
            };

            match cmd {
                "RENDER" => {
                    let Some(camera_name) = parts.next().map(String::from) else {
                        tracing::warn!("Render bridge: RENDER missing camera name, skipping");
                        continue;
                    };
                    let Some(timestamp_str) = parts.next() else {
                        tracing::warn!("Render bridge: RENDER missing timestamp, skipping");
                        continue;
                    };
                    let Ok(timestamp) = timestamp_str.parse::<i64>() else {
                        tracing::warn!(
                            "Render bridge: RENDER invalid timestamp '{timestamp_str}', skipping"
                        );
                        continue;
                    };
                    return Some(BatchRenderRequest {
                        camera_names: vec![camera_name],
                        timestamp: Timestamp(timestamp),
                    });
                }
                "RENDER_BATCH" => {
                    let Some(count_str) = parts.next() else {
                        tracing::warn!("Render bridge: RENDER_BATCH missing count, skipping");
                        continue;
                    };
                    let Ok(count) = count_str.parse::<usize>() else {
                        tracing::warn!(
                            "Render bridge: RENDER_BATCH invalid count '{count_str}', skipping"
                        );
                        continue;
                    };
                    let Some(timestamp_str) = parts.next() else {
                        tracing::warn!("Render bridge: RENDER_BATCH missing timestamp, skipping");
                        continue;
                    };
                    let Ok(timestamp) = timestamp_str.parse::<i64>() else {
                        tracing::warn!(
                            "Render bridge: RENDER_BATCH invalid timestamp '{timestamp_str}', skipping"
                        );
                        continue;
                    };
                    let mut camera_names = Vec::with_capacity(count);
                    for _ in 0..count {
                        let mut cam_line = String::new();
                        match reader.read_line(&mut cam_line) {
                            Ok(0) => {
                                *guard = None;
                                return None;
                            }
                            Ok(_) => camera_names.push(cam_line.trim_end().to_string()),
                            Err(_) => {
                                *guard = None;
                                return None;
                            }
                        }
                    }
                    return Some(BatchRenderRequest {
                        camera_names,
                        timestamp: Timestamp(timestamp),
                    });
                }
                _ => {
                    tracing::warn!("Render bridge: unknown command '{cmd}', skipping");
                    continue;
                }
            }
        }
    }

    /// Send batch response with multiple frames back to the client.
    /// Format: "FRAMES {count} {timestamp}\n" followed by "{camera_name} {len}\n{bytes}" for each frame.
    /// On write error, disconnects the client and returns the error.
    pub fn respond_batch(
        &self,
        timestamp: Timestamp,
        frames: &[(String, Vec<u8>)],
    ) -> std::io::Result<RenderBridgeRespondMetrics> {
        let mut guard = self.client.lock().unwrap();
        let Some((_, writer)) = guard.as_mut() else {
            return Ok(RenderBridgeRespondMetrics::default());
        };

        let mut metrics = RenderBridgeRespondMetrics {
            frame_count: frames.len(),
            ..Default::default()
        };

        let response_header_start = Instant::now();
        if let Err(e) = writeln!(writer, "FRAMES {} {}", frames.len(), timestamp.0) {
            *guard = None;
            return Err(e);
        }
        metrics.response_header_write_ms = elapsed_ms(response_header_start);
        for (camera_name, frame_bytes) in frames {
            let frame_header_start = Instant::now();
            if let Err(e) = writeln!(writer, "{} {}", camera_name, frame_bytes.len()) {
                *guard = None;
                return Err(e);
            }
            metrics.frame_header_write_ms += elapsed_ms(frame_header_start);
            let frame_bytes_start = Instant::now();
            if let Err(e) = writer.write_all(frame_bytes) {
                *guard = None;
                return Err(e);
            }
            metrics.frame_bytes_write_ms += elapsed_ms(frame_bytes_start);
            metrics.total_bytes += frame_bytes.len();
        }
        let flush_start = Instant::now();
        if let Err(e) = writer.flush() {
            *guard = None;
            return Err(e);
        }
        metrics.flush_ms = elapsed_ms(flush_start);
        Ok(metrics)
    }

    /// Send an empty response (no frames rendered).
    /// On write error, disconnects the client and returns the error.
    pub fn respond_empty(&self) -> std::io::Result<()> {
        let mut guard = self.client.lock().unwrap();
        let Some((_, writer)) = guard.as_mut() else {
            return Ok(());
        };
        if let Err(e) = writeln!(writer, "FRAMES 0 0") {
            *guard = None;
            return Err(e);
        }
        if let Err(e) = writer.flush() {
            *guard = None;
            return Err(e);
        }
        Ok(())
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
    writer: BufWriter<UnixStream>,
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

        let write_stream = stream
            .try_clone()
            .map_err(|e| format!("Failed to clone stream for writer: {e}"))?;

        Ok(Self {
            reader: BufReader::new(stream),
            writer: BufWriter::new(write_stream),
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

    fn send_render_request(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
    ) -> Result<f64, String> {
        let send_request_start = Instant::now();
        if camera_names.len() == 1 {
            writeln!(self.writer, "RENDER {} {}", camera_names[0], timestamp.0)
                .map_err(|e| format!("Failed to send render request: {e}"))?;
        } else {
            writeln!(
                self.writer,
                "RENDER_BATCH {} {}",
                camera_names.len(),
                timestamp.0
            )
            .map_err(|e| format!("Failed to send batch request: {e}"))?;
            for name in camera_names {
                writeln!(self.writer, "{}", name)
                    .map_err(|e| format!("Failed to send camera name: {e}"))?;
            }
        }
        self.writer
            .flush()
            .map_err(|e| format!("Failed to flush: {e}"))?;

        Ok(elapsed_ms(send_request_start))
    }

    fn render_cameras_buffered_into<F>(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
        mut on_frame: F,
    ) -> Result<RenderBridgeClientMetrics, String>
    where
        F: FnMut(String, Timestamp, &mut Vec<u8>) -> Result<(), String>,
    {
        let mut metrics = RenderBridgeClientMetrics::default();
        if camera_names.is_empty() {
            return Ok(metrics);
        }

        metrics.send_request_ms = self.send_render_request(camera_names, timestamp)?;

        let response_header_start = Instant::now();
        let mut response_line = String::new();
        self.reader
            .read_line(&mut response_line)
            .map_err(|e| format!("Render timeout or read error: {e}"))?;
        metrics.response_header_read_ms = elapsed_ms(response_header_start);

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

        for _ in 0..count {
            let frame_header_start = Instant::now();
            let mut frame_header = String::new();
            self.reader
                .read_line(&mut frame_header)
                .map_err(|e| format!("Failed to read frame header: {e}"))?;
            metrics.frame_header_read_ms += elapsed_ms(frame_header_start);

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
            let frame_data_start = Instant::now();
            self.reader
                .read_exact(&mut self.frame_buffer[..frame_len])
                .map_err(|e| format!("Failed to read frame data: {e}"))?;
            metrics.frame_data_read_ms += elapsed_ms(frame_data_start);
            metrics.frame_count += 1;
            metrics.total_bytes += frame_len;

            let on_frame_start = Instant::now();
            on_frame(
                camera_name,
                Timestamp(resp_timestamp),
                &mut self.frame_buffer,
            )?;
            metrics.on_frame_ms += elapsed_ms(on_frame_start);
        }

        Ok(metrics)
    }

    /// Render multiple cameras and stream each decoded frame to a callback.
    /// This avoids building an intermediate Vec when the caller only needs to
    /// consume or forward the frame bytes.
    pub fn render_cameras_into<F>(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
        mut on_frame: F,
    ) -> Result<RenderBridgeClientMetrics, String>
    where
        F: FnMut(&str, Timestamp, &[u8]) -> Result<(), String>,
    {
        self.render_cameras_buffered_into(
            camera_names,
            timestamp,
            |camera_name, frame_timestamp, buffer| {
                on_frame(&camera_name, frame_timestamp, &buffer[..])
            },
        )
    }

    /// Render multiple cameras and move each decoded frame into the callback.
    /// This avoids a second copy when the caller needs to take ownership of the
    /// frame bytes immediately (for example, handing them to the DB writer).
    pub fn render_cameras_owned_into<F>(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
        mut on_frame: F,
    ) -> Result<RenderBridgeClientMetrics, String>
    where
        F: FnMut(String, Timestamp, Vec<u8>) -> Result<(), String>,
    {
        self.render_cameras_buffered_into(
            camera_names,
            timestamp,
            |camera_name, frame_timestamp, buffer| {
                on_frame(camera_name, frame_timestamp, std::mem::take(buffer))
            },
        )
    }

    /// Render multiple cameras in a single batch request.
    /// Returns a Vec of rendered frames (may be fewer than requested if some cameras fail)
    /// plus timing metrics for the client-side transport path.
    pub fn render_cameras_with_metrics(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
    ) -> Result<(Vec<RenderedFrame>, RenderBridgeClientMetrics), String> {
        let mut frames = Vec::new();
        let metrics = self.render_cameras_into(
            camera_names,
            timestamp,
            |camera_name, frame_timestamp, data| {
                frames.push(RenderedFrame {
                    camera_name: camera_name.to_string(),
                    timestamp: frame_timestamp,
                    data: data.to_vec(),
                });
                Ok(())
            },
        )?;

        Ok((frames, metrics))
    }

    /// Render multiple cameras in a single batch request.
    /// Returns a Vec of rendered frames (may be fewer than requested if some cameras fail).
    pub fn render_cameras(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
    ) -> Result<Vec<RenderedFrame>, String> {
        let (frames, _metrics) = self.render_cameras_with_metrics(camera_names, timestamp)?;
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
