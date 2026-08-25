use std::time::Duration;

use bevy::{
    app::{App, AppExit, Plugin, Update},
    ecs::schedule::IntoScheduleConfigs,
    render::{
        error_handler::{RenderError, RenderErrorHandler, RenderErrorPolicy},
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    time::common_conditions::on_timer,
};
use nvml_wrapper::Nvml;

use crate::ui::plot::PlotGpuBufferPool;

const NVIDIA_VENDOR_ID: u32 = 0x10de;

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceMemoryStats {
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[derive(bevy::prelude::Resource, Clone, Copy, Debug, Default)]
pub struct HardwareStats {
    pub app_buffer_bytes: u64,
    pub app_texture_bytes: u64,
    pub app_memory_allocations: u64,
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

pub struct HardwareStatsPlugin;

impl Plugin for HardwareStatsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HardwareStats>()
            .init_resource::<NvmlSampler>()
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
) {
    if let Some(render_device) = render_device {
        let counters = render_device.wgpu_device().get_internal_counters();
        stats.app_buffer_bytes = counter_u64(counters.hal.buffer_memory.read());
        stats.app_texture_bytes = counter_u64(counters.hal.texture_memory.read());
        stats.app_memory_allocations = counter_u64(counters.hal.memory_allocations.read());
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

fn log_render_error(
    error: &RenderError,
    main_world: &mut bevy::prelude::World,
    _render_world: &mut bevy::prelude::World,
) -> RenderErrorPolicy {
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
            app_memory_allocations = stats.app_memory_allocations,
            device_memory = ?stats.device_memory,
            gpu_utilization_percent = ?stats.gpu_utilization_percent,
            "Render failure hardware state"
        );
    }
    tracing::error!("Quitting the application due to {:?} RenderError", error.ty);
    main_world.write_message(AppExit::error());
    RenderErrorPolicy::StopRendering
}

#[cfg(test)]
mod tests {
    use super::counter_u64;

    #[test]
    fn negative_backend_counters_are_clamped() {
        assert_eq!(counter_u64(-1), 0);
        assert_eq!(counter_u64(42), 42);
    }
}
