use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use impeller2::types::Timestamp;

const ENV_KEY: &str = "ELODIN_RENDER_BRIDGE_SOCK";

/// 2 MB — large enough to hold a full 640×480 RGBA frame (~1.2 MB) plus headers
/// in a single BufWriter/BufReader, and as the kernel socket buffer so that
/// `write_all` can push an entire frame without blocking on the reader.
const IO_BUF_CAPACITY: usize = 2 * 1024 * 1024;

fn tune_socket_buffers(stream: UnixStream) -> UnixStream {
    let sock = socket2::Socket::from(stream);
    let _ = sock.set_send_buffer_size(IO_BUF_CAPACITY);
    let _ = sock.set_recv_buffer_size(IO_BUF_CAPACITY);
    UnixStream::from(sock)
}

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
    /// Persistent client connection (accepted once, reused for all requests).
    /// The stream is cloned so reads and writes use independent buffered wrappers,
    /// preventing protocol desynchronization from shared BufReader/BufWriter state.
    client: Mutex<Option<(BufReader<UnixStream>, BufWriter<UnixStream>)>>,
}

// ---------------------------------------------------------------------------
// Standalone protocol helpers (used by the IPC thread directly on raw streams)
// ---------------------------------------------------------------------------

/// Read a batch render request from a buffered reader.
/// Returns `None` on EOF or I/O error (connection closed).
/// Malformed lines are logged and skipped.
pub fn read_batch_request_from(reader: &mut impl BufRead) -> Option<BatchRenderRequest> {
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return None,
            Ok(_) => {}
            Err(_) => return None,
        }

        let line = line.trim_end();
        let mut parts = line.splitn(3, ' ');
        let Some(cmd) = parts.next() else {
            continue;
        };

        match cmd {
            "RENDER" => {
                let Some(camera_name) = parts.next().map(String::from) else {
                    continue;
                };
                let Some(timestamp_str) = parts.next() else {
                    continue;
                };
                let Ok(timestamp) = timestamp_str.parse::<i64>() else {
                    continue;
                };
                return Some(BatchRenderRequest {
                    camera_names: vec![camera_name],
                    timestamp: Timestamp(timestamp),
                });
            }
            "RENDER_BATCH" => {
                let Some(count_str) = parts.next() else {
                    continue;
                };
                let Ok(count) = count_str.parse::<usize>() else {
                    continue;
                };
                let Some(timestamp_str) = parts.next() else {
                    continue;
                };
                let Ok(timestamp) = timestamp_str.parse::<i64>() else {
                    continue;
                };
                let mut camera_names = Vec::with_capacity(count);
                for _ in 0..count {
                    let mut cam_line = String::new();
                    match reader.read_line(&mut cam_line) {
                        Ok(0) => return None,
                        Ok(_) => camera_names.push(cam_line.trim_end().to_string()),
                        Err(_) => return None,
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

/// Write a batch response (FRAMES header + per-frame data) to a writer.
pub fn write_batch_response_to(
    writer: &mut impl Write,
    timestamp: Timestamp,
    frames: &[(String, Vec<u8>)],
) -> std::io::Result<()> {
    writeln!(writer, "FRAMES {} {}", frames.len(), timestamp.0)?;
    for (camera_name, frame_bytes) in frames {
        writeln!(writer, "{} {}", camera_name, frame_bytes.len())?;
        writer.write_all(frame_bytes)?;
    }
    writer.flush()
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
        let stream = tune_socket_buffers(stream);
        let write_stream = stream.try_clone()?;
        let reader = BufReader::with_capacity(IO_BUF_CAPACITY, stream);
        let writer = BufWriter::with_capacity(IO_BUF_CAPACITY, write_stream);
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
    pub fn respond_batch(
        &self,
        timestamp: Timestamp,
        frames: &[(String, Vec<u8>)],
    ) -> std::io::Result<()> {
        let mut guard = self.client.lock().unwrap();
        let Some((_, writer)) = guard.as_mut() else {
            return Ok(());
        };

        if let Err(e) = writeln!(writer, "FRAMES {} {}", frames.len(), timestamp.0) {
            *guard = None;
            return Err(e);
        }
        for (camera_name, frame_bytes) in frames {
            if let Err(e) = writeln!(writer, "{} {}", camera_name, frame_bytes.len()) {
                *guard = None;
                return Err(e);
            }
            if let Err(e) = writer.write_all(frame_bytes) {
                *guard = None;
                return Err(e);
            }
        }
        if let Err(e) = writer.flush() {
            *guard = None;
            return Err(e);
        }
        Ok(())
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

        let stream = tune_socket_buffers(stream);
        let write_stream = stream
            .try_clone()
            .map_err(|e| format!("Failed to clone stream for writer: {e}"))?;

        Ok(Self {
            reader: BufReader::with_capacity(IO_BUF_CAPACITY, stream),
            writer: BufWriter::with_capacity(IO_BUF_CAPACITY, write_stream),
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
    ) -> Result<(), String> {
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
        Ok(())
    }

    fn read_frames_into(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
        mut on_frame: impl FnMut(String, Timestamp, &mut Vec<u8>) -> Result<(), String>,
    ) -> Result<(), String> {
        if camera_names.is_empty() {
            return Ok(());
        }

        self.send_render_request(camera_names, timestamp)?;

        let mut response_line = String::new();
        self.reader
            .read_line(&mut response_line)
            .map_err(|e| format!("Render timeout or read error: {e}"))?;

        let response_line = response_line.trim_end();
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

            on_frame(
                camera_name,
                Timestamp(resp_timestamp),
                &mut self.frame_buffer,
            )?;
        }

        Ok(())
    }

    /// Render multiple cameras in a single batch request.
    pub fn render_cameras(
        &mut self,
        camera_names: &[&str],
        timestamp: Timestamp,
    ) -> Result<Vec<RenderedFrame>, String> {
        let mut frames = Vec::new();
        self.read_frames_into(camera_names, timestamp, |name, ts, buf| {
            frames.push(RenderedFrame {
                camera_name: name,
                timestamp: ts,
                data: buf.clone(),
            });
            Ok(())
        })?;
        Ok(frames)
    }
}
