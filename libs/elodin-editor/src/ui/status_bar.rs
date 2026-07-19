use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::{
        query::With,
        system::{Query, Res, SystemParam, SystemState},
        world::World,
    },
    prelude::Entity,
    window::PrimaryWindow,
};
use std::time::{Duration, Instant};
use bevy_ai_skybox::prelude::{SkyboxCacheHealth, SkyboxGenerationUi};
use impeller2_bevy::{ConnectionStatus, ThreadConnectionStatus};
use impeller2_wkt::SimulationTimeStep;

use crate::ui::{
    colors::get_scheme,
    input_owner::{PointerOwnerPriority, UiBlocker},
    register_window_input_blocker,
};

use super::RootWidgetSystem;

#[derive(SystemParam)]
pub struct StatusBar<'w, 's> {
    tick_time: Res<'w, SimulationTimeStep>,
    diagnostics: Res<'w, DiagnosticsStore>,
    connection_status: Res<'w, ThreadConnectionStatus>,
    primary_window: Query<'w, 's, Entity, With<PrimaryWindow>>,
    skybox_ui: Res<'w, SkyboxGenerationUi>,
    skybox_cache: Res<'w, SkyboxCacheHealth>,
}

impl RootWidgetSystem for StatusBar<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world).expect("system params invalid");
        let Ok(target_window) = state_mut.primary_window.single() else {
            return;
        };

        let tick_time = state_mut.tick_time;
        let diagnostics = state_mut.diagnostics;
        let skybox_ui = &state_mut.skybox_ui;
        let skybox_cache = &state_mut.skybox_cache;

        #[allow(deprecated, reason = "bevy_egui exposes a Context, not a root Ui")]
        let panel = egui::Panel::bottom("status_bar")
            .frame(egui::Frame {
                fill: get_scheme().bg_primary,
                inner_margin: egui::Margin::symmetric(16, 4),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let style = ui.style_mut();
                    style.spacing.item_spacing = [20.0, 8.0].into();

                    // Status

                    ui.add(editor_status_label(state_mut.connection_status.status()));

                    // Editor FPS

                    let render_fps_str = diagnostics
                        .get(&FrameTimeDiagnosticsPlugin::FPS)
                        .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                        .map_or(" N/A".to_string(), |value| format!("{value:>6.1}"));

                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("FPS {render_fps_str}"))
                            .text_style(egui::TextStyle::Small)
                            .color(get_scheme().text_secondary),
                    ));

                    // Simulator TPS

                    let sim_fps = if tick_time.0 > 0.0 {
                        format!("{:>6.1}", 1.0 / tick_time.0)
                    } else {
                        String::from("N/A")
                    };

                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("TPS {sim_fps}"))
                            .text_style(egui::TextStyle::Small)
                            .color(get_scheme().text_secondary),
                    ));

                    let ram_str = process_resident_memory_gb()
                        .map(|gb| format!("{gb:.1}"))
                        .unwrap_or_else(|| "N/A".to_string());
                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("RAM Usage: {ram_str} GB"))
                            .text_style(egui::TextStyle::Small)
                            .color(get_scheme().text_secondary),
                    ));

                    super::skybox_status::draw_skybox_status_bar(ui, skybox_ui, skybox_cache);
                });
            });

        register_window_input_blocker(
            world,
            target_window,
            panel.response.rect,
            UiBlocker::OtherPanel,
            PointerOwnerPriority::Panel,
        );
    }
}

/// Process resident set size in GiB. Cached briefly — the status bar paints every frame.
///
/// Intentionally does **not** use Bevy's `SystemInformationDiagnosticsPlugin`:
/// on macOS that plugin's `sysinfo` build enables `apple-app-store`, which
/// cannot observe the current process and always reports 0 GiB.
fn process_resident_memory_gb() -> Option<f64> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(Instant, f64)>> = Mutex::new(None);
    const TTL: Duration = Duration::from_millis(500);

    let mut guard = CACHE.lock().ok()?;
    if let Some((at, gb)) = *guard
        && at.elapsed() < TTL
    {
        return Some(gb);
    }
    let gb = process_resident_memory_bytes()? as f64 / (1024.0 * 1024.0 * 1024.0);
    *guard = Some((Instant::now(), gb));
    Some(gb)
}

#[cfg(target_os = "linux")]
fn process_resident_memory_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb.saturating_mul(1024));
    }
    None
}

#[cfg(target_os = "macos")]
fn process_resident_memory_bytes() -> Option<u64> {
    // MACH_TASK_BASIC_INFO — current resident size (not ru_maxrss peak).
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: i32 = 20;
    const MACH_TASK_BASIC_INFO_COUNT: u32 =
        (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;

    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_outCnt: *mut u32,
        ) -> i32;
    }

    let mut info = MachTaskBasicInfo::default();
    let mut count = MACH_TASK_BASIC_INFO_COUNT;
    let kr = unsafe { task_info(mach_task_self(), MACH_TASK_BASIC_INFO, &mut info, &mut count) };
    if kr == 0 {
        Some(info.resident_size)
    } else {
        None
    }
}

#[cfg(target_os = "windows")]
fn process_resident_memory_bytes() -> Option<u64> {
    use std::mem::{size_of, zeroed};
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            ppsmemCounters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    unsafe {
        let mut counters: ProcessMemoryCounters = zeroed();
        counters.cb = size_of::<ProcessMemoryCounters>() as u32;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
            Some(counters.working_set_size as u64)
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn process_resident_memory_bytes() -> Option<u64> {
    None
}

fn editor_status_label_ui(ui: &mut egui::Ui, status: ConnectionStatus) -> egui::Response {
    let style = ui.style_mut();
    let font_id = egui::TextStyle::Small.resolve(style);

    let text_color = get_scheme().text_secondary;

    let (status_label, status_color) = match status {
        ConnectionStatus::NoConnection => ("DISCONNECTED", get_scheme().error),
        ConnectionStatus::Success => ("CONNECTED", get_scheme().success),
        ConnectionStatus::Connecting => ("CONNECTING", get_scheme().blue),
        ConnectionStatus::Error => ("CONNECTION ERROR", get_scheme().error),
    };

    // Set widget size and allocate space

    let galley = ui
        .painter()
        .layout_no_wrap(status_label.to_string(), font_id.clone(), text_color);
    let circle_diameter = galley.size().y / 2.0;
    let spacing = circle_diameter * 1.5;

    let desired_size = egui::vec2(circle_diameter + spacing + galley.size().x, galley.size().y);

    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    // Paint the UI
    if ui.is_rect_visible(rect) {
        // Background
        let circle_radius = circle_diameter / 2.0;
        ui.painter().circle_filled(
            egui::pos2(rect.left_center().x + circle_radius, rect.left_center().y),
            circle_radius,
            status_color,
        );

        // Label
        ui.painter().text(
            egui::pos2(
                rect.left_center().x + circle_diameter + spacing,
                rect.left_center().y,
            ),
            egui::Align2::LEFT_CENTER,
            status_label,
            font_id,
            text_color,
        );
    }

    response
}

pub fn editor_status_label(status: ConnectionStatus) -> impl egui::Widget {
    move |ui: &mut egui::Ui| editor_status_label_ui(ui, status)
}
