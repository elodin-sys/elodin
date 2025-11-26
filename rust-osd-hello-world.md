````markdown
# Avatar OSD Dev Project – Debug Backend + MSP DisplayPort

This document updates the previous “HELLO AVATAR” Rust project to support **two backends**:

- **DisplayPort backend** – sends MSP DisplayPort over a serial port (for Walksnail Avatar).
- **Debug terminal backend** – draws the OSD grid in your terminal (curses-style) for fast local iteration.

You’ll be able to do:

- `cargo run -- debug` → see your OSD layout live in a terminal window.
- `cargo run -- serial /dev/ttyTHS7` → send the same layout to your Avatar over MSP.

---

## 1. Project layout

Suggested file structure:

```text
msp-osd/
  Cargo.toml
  src/
    main.rs
    osd_grid.rs
    layout.rs
    backends/
      mod.rs
      displayport.rs
      debug_terminal.rs
````

High-level design:

* `OsdGrid` – simple text grid (`rows × cols`) for OSD contents.
* `layout` – functions that populate an `OsdGrid` from some state (for now: a demo that animates over time).
* `backends::DisplayPortBackend` – converts the grid into MSP DisplayPort frames and writes them to a serial port.
* `backends::DebugTerminalBackend` – clears the terminal and prints the grid each frame.

---

## 2. Cargo.toml

Minimal `Cargo.toml`:

```toml
[package]
name = "msp-osd"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
serialport = "4.8"
crossterm = "0.27"
```

---

## 3. `src/osd_grid.rs` – in-memory text grid

```rust
// src/osd_grid.rs
pub struct OsdGrid {
    pub rows: u8,
    pub cols: u8,
    cells: Vec<char>,
}

impl OsdGrid {
    pub fn new(rows: u8, cols: u8) -> Self {
        let len = rows as usize * cols as usize;
        Self {
            rows,
            cols,
            cells: vec![' '; len],
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            *c = ' ';
        }
    }

    /// Write ASCII/UTF-8 text at (row, col), clipped to the grid.
    pub fn write_text(&mut self, row: u8, col: u8, text: &str) {
        let row = row as usize;
        let mut col = col as usize;
        let cols = self.cols as usize;

        if row >= self.rows as usize || col >= cols {
            return;
        }

        for ch in text.chars() {
            if col >= cols {
                break;
            }
            let idx = row * cols + col;
            self.cells[idx] = ch;
            col += 1;
        }
    }

    /// Return a given row as a String (for backends to render/encode).
    pub fn line_as_str(&self, row: u8) -> String {
        let row = row as usize;
        let cols = self.cols as usize;
        let start = row * cols;
        let end = start + cols;
        self.cells[start..end].iter().collect()
    }
}
```

---

## 4. `src/layout.rs` – demo layout engine

This is where you’ll eventually plug your real nav state. For now, we just animate a few numbers based on `t` (seconds since start).

```rust
// src/layout.rs
use crate::osd_grid::OsdGrid;

/// Render a simple animated demo layout into the grid.
///
/// `t` is seconds since start; used just to wiggle values a bit so you can see it update.
pub fn render_demo(t: f32, grid: &mut OsdGrid) {
    grid.clear();

    // Top status line
    grid.write_text(0, 0, &format!("DEMO OSD   t={:5.1}s", t));

    // Fake "flight data"
    let alt_m = 100.0 + 20.0 * (t / 5.0).sin();
    let gs_kt = 120.0 + 10.0 * (t / 3.0).cos();
    let hdg_deg = (t * 12.0) % 360.0;
    let vspd = 1.5 * (t / 7.0).sin();

    grid.write_text(
        2,
        0,
        &format!(
            "ALT:{:5.0}m  GS:{:5.1}kt  HDG:{:6.1}°  VS:{:+4.1}m/s",
            alt_m, gs_kt, hdg_deg, vspd
        ),
    );

    // Simple nav cue to "WP1"
    grid.write_text(4, 0, "NAV → WP1");

    // Simple horizon-ish band in the middle
    let mid_row = grid.rows / 2;
    let band = "--------------------+--------------------";
    let cols = grid.cols as usize;
    let col = (cols / 2).saturating_sub(band.len() / 2) as u8;
    grid.write_text(mid_row.saturating_sub(1), col, band);
    grid.write_text(mid_row, col, "         << LEVEL FLIGHT >>         ");
    grid.write_text(mid_row + 1, col, band);

    // Bottom status
    grid.write_text(
        grid.rows - 1,
        0,
        "SAT:12  NAV:OK  LINK:GOOD  WARN:none",
    );
}
```

---

## 5. `src/backends/displayport.rs` – MSP DisplayPort backend

This is the serial/MSP side that will talk to the Avatar.

```rust
// src/backends/displayport.rs
use crate::osd_grid::OsdGrid;
use anyhow::Result;
use serialport::SerialPort;
use std::io::Write;
use std::time::Duration;

const MSP_DISPLAYPORT: u8 = 182;

// DisplayPort subcommands (see Betaflight DisplayPort MSP spec)
const MSP_DP_CLEAR_SCREEN: u8 = 2;
const MSP_DP_WRITE_STRING: u8 = 3;
const MSP_DP_DRAW_SCREEN: u8 = 4;

/// Encode an MSP v1 packet.
fn encode_msp(cmd: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + payload.len());
    out.push(b'$');
    out.push(b'M');
    out.push(b'>'); // direction: FC -> OSD
    let size = payload.len() as u8;
    out.push(size);
    out.push(cmd);

    let mut cksum = size ^ cmd;
    for &b in payload {
        out.push(b);
        cksum ^= b;
    }
    out.push(cksum);
    out
}

fn msp_dp_clear_screen() -> Vec<u8> {
    let payload = [MSP_DP_CLEAR_SCREEN];
    encode_msp(MSP_DISPLAYPORT, &payload)
}

fn msp_dp_write_string(row: u8, col: u8, text: &str) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + text.len() + 1);
    payload.push(MSP_DP_WRITE_STRING);
    payload.push(row);
    payload.push(col);
    payload.push(0); // attribute: 0 = default font/no blink

    payload.extend_from_slice(text.as_bytes());
    payload.push(0); // null terminator

    encode_msp(MSP_DISPLAYPORT, &payload)
}

fn msp_dp_draw_screen() -> Vec<u8> {
    let payload = [MSP_DP_DRAW_SCREEN];
    encode_msp(MSP_DISPLAYPORT, &payload)
}

pub struct DisplayPortBackend {
    port: Box<dyn SerialPort>,
}

impl DisplayPortBackend {
    /// Open the given serial device at 115200 baud.
    pub fn new(device: &str) -> Result<Self> {
        let port = serialport::new(device, 115_200)
            .timeout(Duration::from_millis(50))
            .open()?;

        Ok(Self { port })
    }

    /// Send a full frame: clear, write all visible lines, then draw.
    pub fn send_frame(&mut self, grid: &OsdGrid) -> Result<()> {
        // 1. Clear screen
        self.port.write_all(&msp_dp_clear_screen())?;

        // 2. Send each non-empty row as a WRITE_STRING from col 0
        for row in 0..grid.rows {
            let line = grid.line_as_str(row);
            let trimmed = line.trim_end();
            if !trimmed.is_empty() {
                let pkt = msp_dp_write_string(row, 0, trimmed);
                self.port.write_all(&pkt)?;
            }
        }

        // 3. Present
        self.port.write_all(&msp_dp_draw_screen())?;
        self.port.flush()?;
        Ok(())
    }
}
```

> Note: eventually you’ll also parse `MSP_SET_OSD_CANVAS` from the Avatar to get the real rows/cols. For now we just pick a reasonable grid size and draw into it.

---

## 6. `src/backends/debug_terminal.rs` – curses-style terminal backend

This uses `crossterm` to clear and redraw the whole grid at the top of the terminal each frame.

```rust
// src/backends/debug_terminal.rs
use crate::osd_grid::OsdGrid;
use anyhow::Result;
use crossterm::{cursor, execute, terminal};
use std::io::{stdout, Write};

pub struct DebugTerminalBackend;

impl DebugTerminalBackend {
    pub fn new() -> Self {
        Self
    }

    /// Draw the current grid to the terminal.
    pub fn draw(&self, grid: &OsdGrid) -> Result<()> {
        let mut out = stdout();

        // Clear screen and move cursor to top-left
        execute!(
            out,
            cursor::MoveTo(0, 0),
            terminal::Clear(terminal::ClearType::All),
        )?;

        // Print each row as one line
        for row in 0..grid.rows {
            let line = grid.line_as_str(row);
            writeln!(out, "{}", line)?;
        }

        out.flush()?;
        Ok(())
    }
}
```

This is intentionally simple: no raw mode, no alternate screen. It just keeps re-painting the top of the same terminal window, like a lightweight curses dashboard.

---

## 7. `src/backends/mod.rs` – re-export backends

```rust
// src/backends/mod.rs
pub mod displayport;
pub mod debug_terminal;

pub use displayport::DisplayPortBackend;
pub use debug_terminal::DebugTerminalBackend;
```

---

## 8. `src/main.rs` – select backend & run the loop

We support two modes:

* `debug` → draw to terminal
* `serial <device>` → send MSP DisplayPort to serial (e.g. `/dev/ttyTHS7`)

```rust
// src/main.rs
mod backends;
mod layout;
mod osd_grid;

use anyhow::Result;
use backends::{DebugTerminalBackend, DisplayPortBackend};
use osd_grid::OsdGrid;
use std::env;
use std::thread;
use std::time::{Duration, Instant};

enum Mode {
    Debug,
    Serial(String),
}

fn parse_args() -> Mode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("serial") => {
            let dev = args
                .next()
                .unwrap_or_else(|| "/dev/ttyTHS7".to_string());
            Mode::Serial(dev)
        }
        _ => {
            // Default to debug mode if no or unknown args
            Mode::Debug
        }
    }
}

fn main() -> Result<()> {
    let mode = parse_args();

    // You can tune this; Avatar canvas will tell you the real size via MSP.
    let rows: u8 = 18;
    let cols: u8 = 50;
    let mut grid = OsdGrid::new(rows, cols);

    match mode {
        Mode::Debug => run_debug(&mut grid),
        Mode::Serial(dev) => run_serial(&mut grid, &dev),
    }
}

fn run_debug(grid: &mut OsdGrid) -> Result<()> {
    let backend = DebugTerminalBackend::new();
    let start = Instant::now();
    let frame_period = Duration::from_millis(200); // 5 Hz

    loop {
        let t = start.elapsed().as_secs_f32();
        layout::render_demo(t, grid);
        backend.draw(grid)?;
        thread::sleep(frame_period);
    }
}

fn run_serial(grid: &mut OsdGrid, device: &str) -> Result<()> {
    let mut backend = DisplayPortBackend::new(device)?;
    let start = Instant::now();
    let frame_period = Duration::from_millis(200); // 5 Hz

    loop {
        let t = start.elapsed().as_secs_f32();
        layout::render_demo(t, grid);
        backend.send_frame(grid)?;
        thread::sleep(frame_period);
    }
}
```

---

## 9. Using it

### 9.1 Debug backend (local curses-style view)

From the project root:

```bash
cargo run -- debug
```

You should see something like a little HUD animating in your terminal, refreshing at ~5 Hz.

Edit `layout.rs`, save, re-run `cargo run -- debug` to iterate on layout quickly.
For an even tighter loop:

```bash
cargo install cargo-watch
cargo watch -x 'run -- debug'
```

### 9.2 DisplayPort backend (Avatar / other MSP DisplayPort target)

Once you’re happy with the look in the debug backend:

```bash
cargo run -- serial /dev/ttyTHS7
```

…with UART7 on the Orin wired to Walksnail’s MSP OSD UART (115200 8N1), and the VTX configured for MSP OSD input.

Your HUD should now show up in the goggles in roughly the same positions you saw in the terminal grid.

---

## 10. Systemd reminder

If you already have a NixOS/systemd unit from before, you can now point it at the new binary in **serial** mode, e.g.:

```nix
ExecStart = "${msp-osd}/bin/msp-osd serial /dev/ttyTHS7";
```

For development, you just run `cargo run -- debug` in a shell and enjoy the terminal OSD while you tweak the layout.

---

This gives you a clean separation:

* **Layout**: just about text and positions (and soon: your real nav state).
* **Backend**: either terminal for dev, or MSP DisplayPort for the jet.

You can now extend `layout.rs` to use your actual nav solution (altitude, speed, waypoints, etc.) without touching the backends at all.

```
::contentReference[oaicite:0]{index=0}
```
