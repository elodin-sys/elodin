use bevy::{
    app::Plugin,
    prelude::{App, Local, Res},
    render::{Render, RenderApp, renderer::RenderAdapterInfo},
};

pub struct GpuInfoPlugin;

impl Plugin for GpuInfoPlugin {
    fn build(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).add_systems(Render, log_gpu_info);
    }
}

fn log_gpu_info(adapter: Res<RenderAdapterInfo>, mut logged: Local<bool>) {
    if *logged {
        return;
    }
    tracing::info!(
        name = %adapter.name,
        device_type = ?adapter.device_type,
        backend = ?adapter.backend,
        driver = %adapter.driver,
        driver_info = %adapter.driver_info,
        vendor_id = %format_args!("{:#06x}", adapter.vendor),
        device_id = %format_args!("{:#06x}", adapter.device),
        pci_bus = %adapter.device_pci_bus_id,
        "Graphics adapter initialized"
    );
    *logged = true;
}
