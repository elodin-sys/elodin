//! Env-var-gated screenshot harness shared by every binary that renders a
//! terrain. Lets CI / scripts validate each upgrade phase, every example, and
//! the top-level `world_mesh` app from one shell loop without involving macOS
//! Screen Recording permissions.
//!
//! Activated when `WORLD_MESH_SCREENSHOT=path/to/out.png` is set:
//!
//! - the system spawns a `Screenshot::primary_window()` after
//!   `WORLD_MESH_SCREENSHOT_DELAY` seconds (default 8.0)
//! - the `save_to_disk` observer writes the PNG asynchronously
//! - we poll for the file on disk before sending `AppExit::Success`,
//!   gated on `WORLD_MESH_SCREENSHOT_EXIT=1`
//!
//! Bevy's `save_screenshot_to_disk` is asynchronous: the PNG appears a few
//! frames after the request, so we _wait for the file to exist_ before
//! exiting — otherwise the render thread tears down mid-readback and we
//! silently lose the screenshot.

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{save_to_disk, Screenshot};
use std::path::PathBuf;

pub struct EnvScreenshotPlugin;

impl Plugin for EnvScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, capture_screenshot);
    }
}

#[derive(Default, PartialEq, Eq)]
enum ScreenshotState {
    #[default]
    Pending,
    Queued,
    Done,
}

fn capture_screenshot(
    mut commands: Commands,
    time: Res<Time>,
    mut state: Local<ScreenshotState>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(out) = std::env::var("WORLD_MESH_SCREENSHOT") else {
        return;
    };
    let path = PathBuf::from(out);

    match *state {
        ScreenshotState::Pending => {
            let delay: f32 = std::env::var("WORLD_MESH_SCREENSHOT_DELAY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8.0);
            if time.elapsed_secs() < delay {
                return;
            }
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
            eprintln!("screenshot queued for {}", path.display());
            *state = ScreenshotState::Queued;
        }
        ScreenshotState::Queued => {
            let nonempty = std::fs::metadata(&path)
                .map(|m| m.len() > 0)
                .unwrap_or(false);
            if nonempty {
                eprintln!("screenshot written to {}", path.display());
                *state = ScreenshotState::Done;
                if std::env::var("WORLD_MESH_SCREENSHOT_EXIT").is_ok() {
                    exit.write(AppExit::Success);
                }
            }
        }
        ScreenshotState::Done => {}
    }
}
