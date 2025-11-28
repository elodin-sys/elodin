pub mod displayport;
pub mod terminal;

pub use displayport::DisplayPortBackend;
pub use terminal::DebugTerminalBackend;

use crate::osd_grid::OsdGrid;
use anyhow::Result;
use async_trait::async_trait;

/// Backend trait for rendering OSD output
#[async_trait]
pub trait Backend: Send + Sync {
    /// Render the current grid state
    async fn render(&mut self, grid: &OsdGrid) -> Result<()>;

    /// Optional: Handle initialization
    #[allow(dead_code)]
    async fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Optional: Handle cleanup
    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Downcast to Any for type checking
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
