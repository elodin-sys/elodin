//! Env-var-gated screenshot harness for the editor, mirroring
//! `bevy_world_mesh`'s `EnvScreenshotPlugin`. Lets scripts capture the full
//! editor window (3D viewports + egui UI) using Bevy's native `Screenshot`
//! API, without macOS Screen Recording permissions.
//!
//! Activated when `ELODIN_SCREENSHOT=path/to/out.png` is set:
//!
//! - the system spawns a `Screenshot::primary_window()` after
//!   `ELODIN_SCREENSHOT_DELAY` seconds (default 8.0)
//! - the `save_to_disk` observer writes the PNG asynchronously
//! - we poll for the file on disk before sending `AppExit::Success`,
//!   gated on `ELODIN_SCREENSHOT_EXIT=1`
//!
//! Bevy's screenshot readback is asynchronous: the PNG appears a few frames
//! after the request, so we wait for the file to exist before exiting —
//! otherwise the render thread tears down mid-readback and the capture is
//! silently lost.

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use std::path::PathBuf;

pub struct EnvScreenshotPlugin;

impl Plugin for EnvScreenshotPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("ELODIN_SCREENSHOT").is_ok() {
            app.add_systems(Update, capture_screenshot);
        }
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
    let Ok(out) = std::env::var("ELODIN_SCREENSHOT") else {
        return;
    };
    let path = PathBuf::from(out);

    match *state {
        ScreenshotState::Pending => {
            let delay: f32 = std::env::var("ELODIN_SCREENSHOT_DELAY")
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
            info!("screenshot queued for {}", path.display());
            *state = ScreenshotState::Queued;
        }
        ScreenshotState::Queued => {
            let nonempty = std::fs::metadata(&path)
                .map(|m| m.len() > 0)
                .unwrap_or(false);
            if nonempty {
                info!("screenshot written to {}", path.display());
                *state = ScreenshotState::Done;
                if std::env::var("ELODIN_SCREENSHOT_EXIT").is_ok() {
                    exit.write(AppExit::Success);
                }
            }
        }
        ScreenshotState::Done => {}
    }
}
