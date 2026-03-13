# Render Bridge

A lightweight Unix socket IPC layer for headless sensor camera rendering in Elodin.

## Overview

Render Bridge provides the communication channel between Elodin simulations and the headless Bevy render server used for sensor camera image generation. When a simulation needs rendered camera frames (e.g., for vision-based navigation), it sends requests over a Unix domain socket and receives raw image data back.

This crate is intentionally minimal — it depends only on `impeller2` (for `Timestamp`) and `tracing`, keeping it decoupled from the heavy database and editor dependency trees.

## Architecture

```
┌────────────────────────┐         Unix Socket         ┌────────────────────────┐
│   Simulation (nox-py)  │◄──────────────────────────►│  Headless Render Server │
│                        │                             │  (elodin-editor)       │
│  RenderBridgeClient    │   RENDER / RENDER_BATCH     │  RenderBridgeServer    │
│  - render_camera()     │──────────────────────────►  │  - recv_batch()        │
│  - render_cameras()    │                             │  - respond_batch()     │
│                        │   FRAMES {count} {ts}       │                        │
│                        │◄──────────────────────────  │                        │
└────────────────────────┘                             └────────────────────────┘
```

The socket path is communicated via the `ELODIN_RENDER_BRIDGE_SOCK` environment variable, set by `s10` when orchestrating the simulation and render server processes together.

## Protocol

The wire protocol is a simple line-based text format over a Unix stream socket:

**Single camera request:**
```
RENDER {camera_name} {timestamp}\n
```

**Batch request (multiple cameras):**
```
RENDER_BATCH {count} {timestamp}\n
{camera_name_1}\n
{camera_name_2}\n
...
```

**Response:**
```
FRAMES {count} {timestamp}\n
{camera_name} {byte_length}\n
{raw_image_bytes}
...
```

## Usage

### Server (in the headless render app)

```rust
use render_bridge::RenderBridgeServer;

let server = RenderBridgeServer::bind()?;
server.accept_client()?;

loop {
    let Some(request) = server.recv_batch() else { break };

    // Render the requested cameras...
    let frames: Vec<(String, Vec<u8>)> = render_cameras(&request);

    server.respond_batch(request.timestamp, &frames)?;
}
```

### Client (in the simulation)

```rust
use render_bridge::RenderBridgeClient;
use impeller2::types::Timestamp;
use std::time::Duration;

let mut client = RenderBridgeClient::connect(Duration::from_secs(10))?;

let frames = client.render_cameras(
    &["front_camera", "rear_camera"],
    Timestamp(1_000_000),
)?;

for frame in &frames {
    println!("{}: {} bytes", frame.camera_name, frame.data.len());
}
```

## Development

```bash
cargo check -p render-bridge
cargo test -p render-bridge
```

## License

See the repository's LICENSE file for details.
