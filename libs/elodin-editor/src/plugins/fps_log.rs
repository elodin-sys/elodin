//! Env-var-gated FPS logger for headless perf measurement, mirroring
//! `screenshot.rs`. When `ELODIN_FPS_LOG=path/to/out.csv` is set, appends one
//! `elapsed_s,fps,frame_time_ms` line per second from the smoothed Bevy frame
//! diagnostics, so scripted runs can compare scenarios without reading the
//! status bar from screenshots.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use std::io::Write;

pub struct EnvFpsLogPlugin;

impl Plugin for EnvFpsLogPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("ELODIN_FPS_LOG").is_ok() {
            app.add_systems(Update, log_fps);
        }
    }
}

fn log_fps(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut file: Local<Option<std::fs::File>>,
    mut last_logged_s: Local<f64>,
) {
    let elapsed = time.elapsed_secs_f64();
    if elapsed - *last_logged_s < 1.0 {
        return;
    }
    *last_logged_s = elapsed;

    let file = file.get_or_insert_with(|| {
        let path = std::env::var("ELODIN_FPS_LOG").expect("checked in plugin build");
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut f = std::fs::File::create(&path).expect("create ELODIN_FPS_LOG file");
        let _ = writeln!(f, "elapsed_s,fps,frame_time_ms");
        f
    });

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed());
    let frame_time = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed());
    if let (Some(fps), Some(frame_time)) = (fps, frame_time) {
        let _ = writeln!(file, "{elapsed:.1},{fps:.2},{frame_time:.3}");
    }
}
