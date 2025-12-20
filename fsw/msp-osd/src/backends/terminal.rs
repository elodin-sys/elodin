use crate::backends::Backend;
use crate::osd_grid::OsdGrid;
use anyhow::Result;
use async_trait::async_trait;
use crossterm::{cursor, execute, terminal};
use std::io::{stdout, Write};

pub struct DebugTerminalBackend {
    frame_count: usize,
}

impl DebugTerminalBackend {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

#[async_trait]
impl Backend for DebugTerminalBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    async fn render(&mut self, grid: &OsdGrid) -> Result<()> {
        let mut out = stdout();

        // Clear screen and move to top
        execute!(
            out,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        // Draw top border
        writeln!(out, "┌{}┐", "─".repeat(grid.cols as usize))?;

        // Print each row with side borders
        for row in 0..grid.rows {
            let line = grid.line_as_str(row);
            writeln!(out, "│{}│", line)?;
        }

        // Bottom border
        writeln!(out, "└{}┘", "─".repeat(grid.cols as usize))?;

        out.flush()?;
        self.frame_count += 1;
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<()> {
        let mut out = stdout();
        execute!(
            out,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        writeln!(out, "MSP OSD stopped.")?;
        out.flush()?;
        Ok(())
    }
}

impl DebugTerminalBackend {
    /// Render with status information
    pub async fn render_with_status(
        &mut self,
        grid: &OsdGrid,
        state: &crate::telemetry::TelemetryState,
    ) -> Result<()> {
        let mut out = stdout();

        // Clear screen and move to top
        execute!(
            out,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        // Draw top border
        writeln!(out, "┌{}┐", "─".repeat(grid.cols as usize))?;

        // Print each row with side borders
        for row in 0..grid.rows {
            let line = grid.line_as_str(row);
            writeln!(out, "│{}│", line)?;
        }

        // Bottom border
        writeln!(out, "└{}┘", "─".repeat(grid.cols as usize))?;

        // Status lines
        let db_status = if state.db_connected {
            "✓ Connected"
        } else {
            "✗ Disconnected"
        };

        let updates_text = if state.update_count > 0 {
            format!("{} updates", state.update_count)
        } else {
            "No data".to_string()
        };

        let staleness = state.last_update.elapsed().as_secs();
        let data_status = if staleness < 2 {
            "Fresh"
        } else if staleness < 5 {
            "Stale"
        } else {
            "Very stale"
        };

        writeln!(out)?;
        writeln!(
            out,
            "MSP OSD Debug Mode | Frame {} | {}x{} grid",
            self.frame_count, grid.cols, grid.rows
        )?;
        writeln!(
            out,
            "Database: {} | Telemetry: {} ({}) | Press Ctrl-C to exit",
            db_status, updates_text, data_status
        )?;

        out.flush()?;
        self.frame_count += 1;
        Ok(())
    }
}

impl Drop for DebugTerminalBackend {
    fn drop(&mut self) {
        // Best effort cleanup
        let _ = futures_lite::future::block_on(self.cleanup());
    }
}
