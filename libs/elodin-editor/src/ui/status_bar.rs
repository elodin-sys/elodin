use bevy::{
    diagnostic::{
        DiagnosticsStore, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
    },
    ecs::{
        query::With,
        system::{Query, Res, SystemParam, SystemState},
        world::World,
    },
    prelude::Entity,
    window::PrimaryWindow,
};
use bevy_ai_skybox::prelude::{SkyboxCacheHealth, SkyboxGenerationUi};
use impeller2_bevy::{ConnectionStatus, ThreadConnectionStatus};
use impeller2_wkt::SimulationTimeStep;
use std::time::{Duration, Instant};

use crate::ui::{
    input_owner::{PointerOwnerPriority, UiBlocker},
    register_window_input_blocker,
};
use crate::{
    plugins::hw_stats::HardwareStats,
    ui::{
        colors::{PUMPKIN_DEFAULT, get_scheme},
        plot::PlotGpuBufferPool,
    },
};

use super::RootWidgetSystem;
use crate::ui::widgets::SystemStateExt;
use impeller2_wkt::DbConfig;

#[derive(SystemParam)]
pub struct StatusBar<'w, 's> {
    tick_time: Res<'w, SimulationTimeStep>,
    diagnostics: Res<'w, DiagnosticsStore>,
    connection_status: Res<'w, ThreadConnectionStatus>,
    primary_window: Query<'w, 's, Entity, With<PrimaryWindow>>,
    skybox_ui: Res<'w, SkyboxGenerationUi>,
    skybox_cache: Res<'w, SkyboxCacheHealth>,
    hardware_stats: Res<'w, HardwareStats>,
    plot_gpu_pool: Res<'w, PlotGpuBufferPool>,
    db_config: Res<'w, DbConfig>,
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
        let state_mut = state.params_mut(world);
        let Ok(target_window) = state_mut.primary_window.single() else {
            return;
        };

        let tick_time = state_mut.tick_time;
        let diagnostics = state_mut.diagnostics;
        let skybox_ui = &state_mut.skybox_ui;
        let skybox_cache = &state_mut.skybox_cache;
        let hardware_stats = &state_mut.hardware_stats;
        let plot_gpu_pool = &state_mut.plot_gpu_pool;
        let build_error = state_mut
            .db_config
            .metadata
            .get("ui.build_error")
            .filter(|s| !s.is_empty())
            .cloned();

        let panel = super::utils::show_panel(
            egui::Panel::bottom("status_bar").frame(egui::Frame {
                fill: get_scheme().bg_primary,
                inner_margin: egui::Margin::symmetric(16, 4),
                ..Default::default()
            }),
            ctx,
            |ui| {
                ui.horizontal(|ui| {
                    let style = ui.style_mut();
                    style.spacing.item_spacing = [20.0, 8.0].into();

                    // Status

                    ui.add(editor_status_label(state_mut.connection_status.status()));

                    if let Some(err) = &build_error {
                        ui.add(egui::Label::new(
                            egui::RichText::new(format!("Schematic build error: {err}"))
                                .text_style(egui::TextStyle::Small)
                                .color(get_scheme().error),
                        ));
                    }

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
                    let system_memory_percent = diagnostics
                        .get(&SystemInformationDiagnosticsPlugin::SYSTEM_MEM_USAGE)
                        .and_then(|diagnostic| diagnostic.smoothed());
                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("RAM Usage: {ram_str} GB"))
                            .text_style(egui::TextStyle::Small)
                            .color(pressure_color(system_memory_percent)),
                    ))
                    .on_hover_text(format!(
                        "Editor resident memory: {ram_str} GiB\nSystem memory used: {}",
                        format_percent(system_memory_percent)
                    ));

                    let process_cpu = diagnostics
                        .get(&SystemInformationDiagnosticsPlugin::PROCESS_CPU_USAGE)
                        .and_then(|diagnostic| diagnostic.smoothed());
                    let system_cpu = diagnostics
                        .get(&SystemInformationDiagnosticsPlugin::SYSTEM_CPU_USAGE)
                        .and_then(|diagnostic| diagnostic.smoothed());
                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("CPU {}", format_percent(process_cpu)))
                            .text_style(egui::TextStyle::Small)
                            .color(get_scheme().text_secondary),
                    ))
                    .on_hover_text(format!(
                        "Editor CPU: {}\nSystem CPU: {}",
                        format_percent(process_cpu),
                        format_percent(system_cpu)
                    ));

                    let pool = plot_gpu_pool.snapshot();
                    let plot_label = match pool.shard_occupancy_percent() {
                        Some(percent) => {
                            format!("{} · {percent:.0}% shards", format_plot_bytes(pool.resident_bytes()))
                        }
                        None => format_plot_bytes(pool.resident_bytes()),
                    };
                    let (gpu_label, gpu_pressure) =
                        if let Some(memory) = hardware_stats.device_memory {
                            let pressure = if memory.total_bytes > 0 {
                                Some(memory.used_bytes as f64 / memory.total_bytes as f64 * 100.0)
                            } else {
                                None
                            };
                            (
                                format!(
                                    "GPU {:.1}/{:.1} GB · {} · {plot_label}",
                                    bytes_to_gib(memory.used_bytes),
                                    bytes_to_gib(memory.total_bytes),
                                    format_percent(
                                        hardware_stats.gpu_utilization_percent.map(f64::from)
                                    )
                                ),
                                pressure,
                            )
                        } else {
                            (
                                format!(
                                    "GPU app {:.1} GB · {plot_label}",
                                    bytes_to_gib(hardware_stats.app_gpu_bytes()),
                                ),
                                None,
                            )
                        };
                    ui.add(egui::Label::new(
                        egui::RichText::new(gpu_label)
                            .text_style(egui::TextStyle::Small)
                            .color(pressure_color(gpu_pressure)),
                    ))
                    .on_hover_text(format!(
                        "App GPU allocations\nBuffers: {:.2} GiB ({} objects)\nTextures: {:.2} GiB ({} objects)\n\nPlot buffers\nResident: {:.2} GiB ({} value, {} index)\nPooled: {:.2} GiB\nShard occupancy: {} / {} ({})\nReady: {} value, {} index\nQuarantined: {} value, {} index\nCumulative: {} value alloc / {} reuse / {} destroyed, {} index alloc / {} reuse / {} destroyed",
                        bytes_to_gib(hardware_stats.app_buffer_bytes),
                        format_object_count(hardware_stats.app_buffer_count),
                        bytes_to_gib(hardware_stats.app_texture_bytes),
                        format_object_count(hardware_stats.app_texture_count),
                        bytes_to_gib(pool.resident_bytes()),
                        pool.value_live,
                        pool.index_live,
                        bytes_to_gib(pool.pooled_bytes()),
                        pool.value_shards_used,
                        pool.value_shards_capacity,
                        format_percent(pool.shard_occupancy_percent()),
                        pool.value_ready,
                        pool.index_ready,
                        pool.value_quarantined,
                        pool.index_quarantined,
                        pool.value_allocations,
                        pool.value_reuses,
                        pool.value_destroyed,
                        pool.index_allocations,
                        pool.index_reuses,
                        pool.index_destroyed,
                    ));

                    super::skybox_status::draw_skybox_status_bar(ui, skybox_ui, skybox_cache);
                });
            },
        );

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
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info,
            &mut count,
        )
    };
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

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn format_plot_bytes(bytes: u64) -> String {
    if bytes < 1024 * 1024 * 1024 {
        format!("PLOT {:.0} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("PLOT {:.1} GB", bytes_to_gib(bytes))
    }
}

fn format_percent(value: Option<f64>) -> String {
    value
        .map(|percent| format!("{percent:.0}%"))
        .unwrap_or_else(|| "N/A".to_string())
}

fn format_object_count(value: Option<u64>) -> String {
    value
        .map(|count| count.to_string())
        .unwrap_or_else(|| "N/A".to_string())
}

fn pressure_color(percent: Option<f64>) -> egui::Color32 {
    match percent {
        Some(percent) if percent >= 90.0 => get_scheme().error,
        Some(percent) if percent >= 80.0 => PUMPKIN_DEFAULT,
        _ => get_scheme().text_secondary,
    }
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
