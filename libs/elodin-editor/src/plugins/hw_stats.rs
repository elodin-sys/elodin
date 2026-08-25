use std::{
    collections::BTreeMap,
    fmt::Write as _,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{
    app::{App, AppExit, Plugin, Update},
    ecs::schedule::IntoScheduleConfigs,
    render::{
        error_handler::{ErrorType, RenderError, RenderErrorHandler, RenderErrorPolicy},
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    time::common_conditions::on_timer,
};
use nvml_wrapper::Nvml;

use crate::ui::plot::{PlotGpuBufferPool, data::PlotGpuPoolSnapshot, gpu::evict_all_plot_gpu};

const NVIDIA_VENDOR_ID: u32 = 0x10de;
const GPU_MEMORY_PRESSURE_TRIM_PERCENT: u64 = 90;
const OOM_RECOVERY_COOLDOWN: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceMemoryStats {
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[derive(bevy::prelude::Resource, Clone, Copy, Debug, Default)]
pub struct HardwareStats {
    pub app_buffer_bytes: u64,
    pub app_texture_bytes: u64,
    pub app_buffer_count: Option<u64>,
    pub app_texture_count: Option<u64>,
    pub device_memory: Option<DeviceMemoryStats>,
    pub gpu_utilization_percent: Option<u32>,
}

impl HardwareStats {
    pub fn app_gpu_bytes(&self) -> u64 {
        self.app_buffer_bytes.saturating_add(self.app_texture_bytes)
    }
}

#[derive(Default)]
enum NvmlState {
    #[default]
    Uninitialized,
    Ready(Nvml),
    Unavailable,
}

#[derive(bevy::prelude::Resource, Default)]
struct NvmlSampler {
    state: NvmlState,
}

#[derive(bevy::prelude::Resource, Default)]
struct OomRecoveryState {
    last_oom: Option<Instant>,
}

fn allow_oom_recovery(state: &mut OomRecoveryState, now: Instant) -> bool {
    let repeated = state
        .last_oom
        .is_some_and(|last| now.duration_since(last) < OOM_RECOVERY_COOLDOWN);
    state.last_oom = Some(now);
    !repeated
}

pub struct HardwareStatsPlugin;

impl Plugin for HardwareStatsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HardwareStats>()
            .init_resource::<NvmlSampler>()
            .init_resource::<OomRecoveryState>()
            .insert_resource(RenderErrorHandler(log_render_error))
            .add_systems(
                Update,
                sample_hardware_stats.run_if(on_timer(Duration::from_millis(500))),
            );
    }
}

fn sample_hardware_stats(
    render_device: Option<bevy::prelude::Res<RenderDevice>>,
    adapter: Option<bevy::prelude::Res<RenderAdapterInfo>>,
    mut sampler: bevy::prelude::ResMut<NvmlSampler>,
    mut stats: bevy::prelude::ResMut<HardwareStats>,
    mut plot_pool: Option<bevy::prelude::ResMut<PlotGpuBufferPool>>,
) {
    if let Some(render_device) = render_device {
        let counters = render_device.wgpu_device().get_internal_counters();
        stats.app_buffer_bytes = counter_u64(counters.hal.buffer_memory.read());
        stats.app_texture_bytes = counter_u64(counters.hal.texture_memory.read());
        stats.app_buffer_count =
            supported_counter(counters.hal.buffers.read(), stats.app_buffer_bytes);
        stats.app_texture_count =
            supported_counter(counters.hal.textures.read(), stats.app_texture_bytes);
    }

    stats.device_memory = None;
    stats.gpu_utilization_percent = None;

    let Some(adapter) = adapter else {
        return;
    };
    let Some(nvml) = sampler.nvml(adapter.vendor) else {
        return;
    };
    let device = nvml
        .device_by_pci_bus_id(adapter.device_pci_bus_id.as_str())
        .or_else(|_| nvml.device_by_index(0));
    let Ok(device) = device else {
        return;
    };

    stats.device_memory = device.memory_info().ok().map(|memory| DeviceMemoryStats {
        used_bytes: memory.used,
        total_bytes: memory.total,
    });
    stats.gpu_utilization_percent = device.utilization_rates().ok().map(|rates| rates.gpu);

    if let (Some(memory), Some(pool)) = (stats.device_memory, plot_pool.as_deref_mut())
        && memory.total_bytes > 0
        && memory.used_bytes.saturating_mul(100)
            >= memory
                .total_bytes
                .saturating_mul(GPU_MEMORY_PRESSURE_TRIM_PERCENT)
    {
        let trim = pool.trim_ready();
        if trim.values > 0 || trim.indices > 0 {
            tracing::info!(
                value_buffers = trim.values,
                index_buffers = trim.indices,
                used_bytes = memory.used_bytes,
                total_bytes = memory.total_bytes,
                "Trimmed ready plot GPU buffers under memory pressure"
            );
        }
    }
}

impl NvmlSampler {
    fn nvml(&mut self, vendor_id: u32) -> Option<&Nvml> {
        if matches!(self.state, NvmlState::Uninitialized) {
            self.state = if vendor_id == NVIDIA_VENDOR_ID {
                match Nvml::init() {
                    Ok(nvml) => NvmlState::Ready(nvml),
                    Err(error) => {
                        tracing::debug!(%error, "NVML GPU metrics unavailable");
                        NvmlState::Unavailable
                    }
                }
            } else {
                NvmlState::Unavailable
            };
        }

        match &self.state {
            NvmlState::Ready(nvml) => Some(nvml),
            NvmlState::Uninitialized | NvmlState::Unavailable => None,
        }
    }
}

fn counter_u64(value: isize) -> u64 {
    value.max(0) as u64
}

fn requires_renderer_recovery(error_type: ErrorType) -> bool {
    matches!(error_type, ErrorType::OutOfMemory | ErrorType::DeviceLost)
}

fn supported_counter(value: isize, attributed_bytes: u64) -> Option<u64> {
    let value = counter_u64(value);
    (value > 0 || attributed_bytes == 0).then_some(value)
}

pub(crate) struct GpuAllocatorDump {
    pub path: PathBuf,
    pub total_allocated_bytes: u64,
    pub total_reserved_bytes: u64,
    pub block_count: usize,
    pub allocation_count: usize,
    pub largest_groups: String,
}

pub(crate) fn dump_gpu_allocations(
    render_device: &RenderDevice,
    reason: &str,
    plot_snapshot: Option<PlotGpuPoolSnapshot>,
) -> Result<GpuAllocatorDump, String> {
    let report = render_device
        .wgpu_device()
        .generate_allocator_report()
        .ok_or_else(|| "GPU allocator reporting is unavailable on this backend".to_string())?;

    let mut groups: BTreeMap<(String, u64), usize> = BTreeMap::new();
    for allocation in &report.allocations {
        let label = if allocation.name.is_empty() {
            "(unlabeled)".to_string()
        } else {
            allocation.name.clone()
        };
        *groups.entry((label, allocation.size)).or_default() += 1;
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by_key(|((_, size), count)| std::cmp::Reverse(size.saturating_mul(*count as u64)));
    let mut largest_groups = String::new();
    for ((label, size), count) in groups.iter().take(25) {
        let total = size.saturating_mul(*count as u64);
        let _ = writeln!(
            largest_groups,
            "{label}: {count} × {} = {}",
            format_bytes(*size),
            format_bytes(total)
        );
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    #[cfg(unix)]
    let dump_dir = PathBuf::from("/tmp");
    #[cfg(not(unix))]
    let dump_dir = std::env::temp_dir();
    let path = dump_dir.join(format!("elodin-gpu-{reason}-{timestamp}.txt"));
    let mut full = String::new();
    let _ = writeln!(full, "reason: {reason}");
    let _ = writeln!(
        full,
        "allocated: {} ({})",
        report.total_allocated_bytes,
        format_bytes(report.total_allocated_bytes)
    );
    let _ = writeln!(
        full,
        "reserved: {} ({})",
        report.total_reserved_bytes,
        format_bytes(report.total_reserved_bytes)
    );
    let _ = writeln!(full, "blocks: {}", report.blocks.len());
    let _ = writeln!(full, "allocations: {}", report.allocations.len());
    if let Some(snapshot) = plot_snapshot {
        let _ = writeln!(full, "plot pool: {snapshot:?}");
        let _ = writeln!(
            full,
            "plot shard occupancy: {} / {} ({:.1}%)",
            snapshot.value_shards_used,
            snapshot.value_shards_capacity,
            snapshot.shard_occupancy_percent().unwrap_or(0.0)
        );
    }
    let _ = writeln!(full, "\naggregated by label and size:\n{largest_groups}");
    let _ = writeln!(full, "memory blocks:");
    for (index, block) in report.blocks.iter().enumerate() {
        let _ = writeln!(
            full,
            "  block {index}: size={} allocations={:?}",
            block.size, block.allocations
        );
    }
    let _ = writeln!(full, "\nall allocations:");
    for (index, allocation) in report.allocations.iter().enumerate() {
        let label = if allocation.name.is_empty() {
            "(unlabeled)"
        } else {
            &allocation.name
        };
        let _ = writeln!(
            full,
            "  {index}: size={} offset={} label={label:?}",
            allocation.size, allocation.offset
        );
    }
    std::fs::write(&path, full)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;

    Ok(GpuAllocatorDump {
        path,
        total_allocated_bytes: report.total_allocated_bytes,
        total_reserved_bytes: report.total_reserved_bytes,
        block_count: report.blocks.len(),
        allocation_count: report.allocations.len(),
        largest_groups,
    })
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} B")
    }
}

fn log_render_error(
    error: &RenderError,
    main_world: &mut bevy::prelude::World,
    render_world: &mut bevy::prelude::World,
) -> RenderErrorPolicy {
    let recoverable = requires_renderer_recovery(error.ty);
    let mut dump_path = None;
    let plot_snapshot = main_world
        .get_resource::<PlotGpuBufferPool>()
        .map(|pool| pool.snapshot());
    if recoverable && let Some(render_device) = main_world.get_resource::<RenderDevice>() {
        let reason = match error.ty {
            ErrorType::DeviceLost => "device-lost",
            _ => "oom",
        };
        match dump_gpu_allocations(render_device, reason, plot_snapshot) {
            Ok(dump) => {
                tracing::error!(
                    path = %dump.path.display(),
                    total_allocated_bytes = dump.total_allocated_bytes,
                    total_reserved_bytes = dump.total_reserved_bytes,
                    block_count = dump.block_count,
                    allocation_count = dump.allocation_count,
                    largest_groups = %dump.largest_groups,
                    "GPU allocator report"
                );
                dump_path = Some(dump.path);
            }
            Err(error) => tracing::error!(%error, "GPU allocator report unavailable"),
        }
    }
    let dump_path_label = dump_path
        .as_deref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    if let Some(pool) = main_world.get_resource::<PlotGpuBufferPool>() {
        tracing::error!(
            error_type = ?error.ty,
            plot_gpu_pool = ?pool.snapshot(),
            "Render failure GPU state"
        );
    }
    if let Some(stats) = main_world.get_resource::<HardwareStats>() {
        tracing::error!(
            error_type = ?error.ty,
            app_buffer_bytes = stats.app_buffer_bytes,
            app_texture_bytes = stats.app_texture_bytes,
            app_buffer_count = ?stats.app_buffer_count,
            app_texture_count = ?stats.app_texture_count,
            device_memory = ?stats.device_memory,
            gpu_utilization_percent = ?stats.gpu_utilization_percent,
            "Render failure hardware state"
        );
    }

    if recoverable {
        let now = Instant::now();
        let repeated = main_world
            .get_resource_mut::<OomRecoveryState>()
            .is_some_and(|mut state| !allow_oom_recovery(&mut state, now));
        if !repeated {
            let trim = if main_world.contains_resource::<PlotGpuBufferPool>() {
                evict_all_plot_gpu(main_world, render_world)
            } else {
                Default::default()
            };
            tracing::error!(
                value_buffers = trim.values,
                index_buffers = trim.indices,
                error_type = ?error.ty,
                "Evicted plot GPU caches and recreating renderer; GPU allocation report: {}",
                dump_path_label
            );
            return RenderErrorPolicy::Recover(crate::editor_wgpu_settings().into());
        }
        tracing::error!(error_type = ?error.ty, "Repeated GPU failure during recovery window");
    }

    tracing::error!(
        "Quitting the application due to {:?} RenderError; GPU allocation report: {}",
        error.ty,
        dump_path_label
    );
    main_world.write_message(AppExit::error());
    RenderErrorPolicy::StopRendering
}

#[cfg(test)]
mod tests {
    use super::{
        OomRecoveryState, allow_oom_recovery, counter_u64, requires_renderer_recovery,
        supported_counter,
    };
    use bevy::render::error_handler::ErrorType;
    use std::time::{Duration, Instant};

    #[test]
    fn negative_backend_counters_are_clamped() {
        assert_eq!(counter_u64(-1), 0);
        assert_eq!(counter_u64(42), 42);
        assert_eq!(supported_counter(0, 1024), None);
        assert_eq!(supported_counter(3, 1024), Some(3));
    }

    #[test]
    fn oom_and_device_loss_require_renderer_recovery() {
        assert!(requires_renderer_recovery(ErrorType::OutOfMemory));
        assert!(requires_renderer_recovery(ErrorType::DeviceLost));
        assert!(!requires_renderer_recovery(ErrorType::Validation));
    }

    #[test]
    fn oom_recovery_allows_one_attempt_per_window() {
        let start = Instant::now();
        let mut state = OomRecoveryState::default();
        assert!(allow_oom_recovery(&mut state, start));
        assert!(!allow_oom_recovery(
            &mut state,
            start + Duration::from_secs(1)
        ));
        assert!(allow_oom_recovery(
            &mut state,
            start + Duration::from_secs(12)
        ));
    }
}
