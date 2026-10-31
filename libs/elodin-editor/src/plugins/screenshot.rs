//! Env-var-gated screenshot harness for the editor, mirroring
//! `bevy_world_mesh`'s `EnvScreenshotPlugin`. Lets scripts capture the full
//! editor window (3D viewports + egui UI) using Bevy's native `Screenshot`
//! API, without macOS Screen Recording permissions.
//!
//! Activated when `ELODIN_SCREENSHOT=path/to/out.png` is set:
//!
//! - the system spawns a `Screenshot::primary_window()` after
//!   `ELODIN_SCREENSHOT_DELAY` seconds (default 8.0)
//! - `ELODIN_SCREENSHOT_TIME` (seconds after the recording's first sample)
//!   holds the playhead there, paused, until the capture
//! - `ELODIN_SCREENSHOT_TAB` (a named split's title) keeps that tab selected
//! - the `save_to_disk` observer writes the PNG asynchronously
//! - we poll for the file on disk before sending `AppExit::Success`,
//!   gated on `ELODIN_SCREENSHOT_EXIT=1`
//!
//! Bevy's screenshot readback is asynchronous: the PNG appears a few frames
//! after the request, so we wait for the file to exist before exiting —
//! otherwise the render thread tears down mid-readback and the capture is
//! silently lost.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy::window::PrimaryWindow;
use impeller::types::Timestamp;
use impeller_wkt::{CurrentTimestamp, EarliestTimestamp, LastUpdated};
use std::path::PathBuf;

use crate::ui::Paused;
use crate::ui::tiles::{TileState, WindowState};
use crate::ui::timeline::{AutoFollowLatestState, LatestFollow};

pub struct EnvScreenshotPlugin;

impl Plugin for EnvScreenshotPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("ELODIN_SCREENSHOT").is_ok() {
            app.add_systems(Update, (stage_screenshot, capture_screenshot).chain());
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

#[derive(SystemParam)]
struct Playback<'w> {
    current: Option<ResMut<'w, CurrentTimestamp>>,
    earliest: Option<Res<'w, EarliestTimestamp>>,
    latest: Option<Res<'w, LastUpdated>>,
    paused: Option<ResMut<'w, Paused>>,
    latest_follow: Option<ResMut<'w, LatestFollow>>,
    auto_follow: Option<ResMut<'w, AutoFollowLatestState>>,
}

/// Holds the playhead at `ELODIN_SCREENSHOT_TIME` and the tab named by
/// `ELODIN_SCREENSHOT_TAB` every frame before the capture, so a late
/// schematic load or follow-latest cannot move them.
fn stage_screenshot(
    mut playback: Playback,
    mut windows: Query<&mut WindowState, With<PrimaryWindow>>,
) {
    let time = std::env::var("ELODIN_SCREENSHOT_TIME")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    if let (Some(secs), Some(current), Some(earliest), Some(latest)) = (
        time,
        playback.current.as_mut(),
        playback.earliest.as_ref(),
        playback.latest.as_ref(),
    ) && let Some(target) = seek_target(secs, earliest.0, latest.0)
    {
        current.0 = target;
        if let Some(paused) = playback.paused.as_mut() {
            paused.0 = true;
        }
        if let Some(follow) = playback.latest_follow.as_mut() {
            follow.0 = false;
        }
        if let Some(auto_follow) = playback.auto_follow.as_mut() {
            auto_follow.cancel();
        }
    }
    if let Ok(tab) = std::env::var("ELODIN_SCREENSHOT_TAB") {
        for mut window in &mut windows {
            select_tab(&mut window.tile_state, &tab);
        }
    }
}

/// `secs` after the recording's first sample, kept within the recording;
/// `None` until the recording's bounds are known.
fn seek_target(secs: f64, earliest: Timestamp, latest: Timestamp) -> Option<Timestamp> {
    if earliest.0 == i64::MAX || latest.0 < earliest.0 {
        return None;
    }
    let target = earliest.0.saturating_add((secs * 1e6) as i64);
    Some(Timestamp(target.clamp(earliest.0, latest.0)))
}

/// Selects the tab titled `name`; false when no tab has that title.
fn select_tab(tile_state: &mut TileState, name: &str) -> bool {
    let Some(id) = tile_state
        .container_titles
        .iter()
        .find_map(|(id, title)| (title == name).then_some(*id))
    else {
        return false;
    };
    tile_state.tree.make_active(|tile, _| tile == id);
    true
}

fn capture_screenshot(
    mut commands: Commands,
    time: Res<Time>,
    mut state: Local<ScreenshotState>,
    mut exit: MessageWriter<AppExit>,
    windows: Query<&WindowState, With<PrimaryWindow>>,
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
            if let Ok(tab) = std::env::var("ELODIN_SCREENSHOT_TAB")
                && !windows
                    .iter()
                    .any(|w| w.tile_state.container_titles.values().any(|t| *t == tab))
            {
                warn!("ELODIN_SCREENSHOT_TAB: no tab titled {tab:?}");
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

#[cfg(test)]
mod tests {
    use super::{seek_target, select_tab};
    use crate::ui::tiles::TileState;
    use egui_tiles::{Container, Linear, LinearDir, Tile};
    use impeller::types::Timestamp;

    #[test]
    fn the_seek_waits_for_the_recording_and_stays_inside_it() {
        assert_eq!(seek_target(1.0, Timestamp(i64::MAX), Timestamp(0)), None);
        let (earliest, latest) = (Timestamp(1_000_000), Timestamp(11_000_000));
        assert_eq!(
            seek_target(2.5, earliest, latest),
            Some(Timestamp(3_500_000))
        );
        assert_eq!(seek_target(60.0, earliest, latest), Some(latest));
        assert_eq!(seek_target(-1.0, earliest, latest), Some(earliest));
    }

    #[test]
    fn a_tab_is_selected_by_its_title() {
        let mut tile_state = TileState::new(bevy_egui::egui::Id::new("screenshot_tabs"));
        let mut tab = |title: &str| {
            let linear = Tile::Container(Container::Linear(Linear::new(
                LinearDir::Horizontal,
                vec![],
            )));
            let id = tile_state.insert_tile(linear, None, false).expect("tab");
            tile_state.container_titles.insert(id, title.to_string());
            id
        };
        let (_first, second) = (tab("First"), tab("Second"));

        assert!(select_tab(&mut tile_state, "Second"));
        assert!(tile_state.tree.active_tiles().contains(&second));
        assert!(!select_tab(&mut tile_state, "Missing"));
    }
}
