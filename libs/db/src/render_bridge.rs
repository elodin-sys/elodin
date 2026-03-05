use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use impeller2::types::Timestamp;

const ENV_KEY: &str = "ELODIN_RENDER_BRIDGE_SOCK";

pub struct RenderRequest {
    pub camera_name: String,
    pub timestamp: Timestamp,
}

pub struct RenderBridgeServer {
    listener: UnixListener,
    path: PathBuf,
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
        listener.set_nonblocking(true)?;
        Ok(Self { listener, path })
    }

    pub fn try_recv(&self) -> Option<(RenderRequest, UnixStream)> {
        let (stream, _) = self.listener.accept().ok()?;
        stream.set_nonblocking(false).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .ok()?;
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let line = line.trim_end();
        let mut parts = line.splitn(3, ' ');
        let cmd = parts.next()?;
        if cmd != "RENDER" {
            return None;
        }
        let camera_name = parts.next()?.to_string();
        let timestamp: i64 = parts.next()?.parse().ok()?;
        Some((
            RenderRequest {
                camera_name,
                timestamp: Timestamp(timestamp),
            },
            reader.into_inner(),
        ))
    }

    /// Send a response with the rendered frame bytes back to the client.
    /// The client will write them to its local DB.
    pub fn respond_with_frame(
        mut stream: UnixStream,
        camera_name: &str,
        timestamp: Timestamp,
        frame_bytes: &[u8],
    ) {
        let _ = writeln!(
            stream,
            "FRAME {} {} {}",
            camera_name,
            timestamp.0,
            frame_bytes.len()
        );
        let _ = stream.write_all(frame_bytes);
        let _ = stream.flush();
    }

    /// Send an OK response with no frame (e.g., when camera is unknown).
    pub fn respond_empty(mut stream: UnixStream) {
        let _ = writeln!(stream, "OK");
        let _ = stream.flush();
    }
}

impl Drop for RenderBridgeServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Rendered frame returned from the render bridge.
pub struct RenderedFrame {
    pub camera_name: String,
    pub timestamp: Timestamp,
    pub data: Vec<u8>,
}

/// Client side: connects to the render bridge UDS and sends a blocking render
/// request. Returns the rendered frame bytes directly over the socket,
/// bypassing TCP to avoid deadlocking the stellarator async runtime.
pub fn render_camera_blocking(
    camera_name: &str,
    timestamp: Timestamp,
    timeout: Duration,
) -> Result<Option<RenderedFrame>, String> {
    let sock_path = std::env::var(ENV_KEY).map_err(|_| {
        "No render bridge available. Use 'elodin run' or 'elodin editor' for sensor camera rendering.".to_string()
    })?;

    let deadline = Instant::now() + timeout;

    let stream = loop {
        match UnixStream::connect(Path::new(&sock_path)) {
            Ok(s) => break s,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(format!("Failed to connect to render bridge: {e}")),
        }
    };

    let remaining = deadline.saturating_duration_since(Instant::now());
    let read_timeout = remaining.max(Duration::from_secs(1));
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(|e| format!("Failed to set timeout: {e}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| format!("Failed to set timeout: {e}"))?;

    let mut stream = stream;
    writeln!(stream, "RENDER {} {}", camera_name, timestamp.0)
        .map_err(|e| format!("Failed to send render request: {e}"))?;
    stream
        .flush()
        .map_err(|e| format!("Failed to flush: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|e| format!("Render timeout or read error: {e}"))?;

    let response_line = response_line.trim_end();

    if response_line == "OK" {
        return Ok(None);
    }

    // Parse "FRAME {camera_name} {timestamp} {length}"
    let mut parts = response_line.splitn(4, ' ');
    let cmd = parts.next().unwrap_or("");
    if cmd != "FRAME" {
        return Err(format!("Unexpected response: {response_line}"));
    }
    let resp_camera = parts
        .next()
        .ok_or("Missing camera name in response")?
        .to_string();
    let resp_timestamp: i64 = parts
        .next()
        .ok_or("Missing timestamp in response")?
        .parse()
        .map_err(|_| "Invalid timestamp in response")?;
    let frame_len: usize = parts
        .next()
        .ok_or("Missing frame length in response")?
        .parse()
        .map_err(|_| "Invalid frame length in response")?;

    let mut frame_data = vec![0u8; frame_len];
    reader
        .read_exact(&mut frame_data)
        .map_err(|e| format!("Failed to read frame data: {e}"))?;

    Ok(Some(RenderedFrame {
        camera_name: resp_camera,
        timestamp: Timestamp(resp_timestamp),
        data: frame_data,
    }))
}
