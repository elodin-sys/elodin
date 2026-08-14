use std::{env, ffi::OsStr, sync::Once};

use bevy::{
    app::Plugin,
    prelude::{App, Local, Res},
    render::{Render, RenderApp, renderer::RenderAdapterInfo},
};

const GPU_FAILURE_HELP: &str = "\
elodin: unable to initialize graphics because no GPU was found.
elodin: select a graphics path before entering the development shell:
  ELODIN_GPU=nvidia nix develop
  ELODIN_GPU=mesa nix develop
  ELODIN_GPU=nvk nix develop";

pub struct GpuInfoPlugin;

impl Plugin for GpuInfoPlugin {
    fn build(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).add_systems(Render, log_gpu_info);
    }
}

pub fn install_gpu_panic_handler() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if panic_message(info).is_some_and(is_gpu_init_panic) {
                eprintln!("{GPU_FAILURE_HELP}");
            }
            previous(info);
        }));
    });

    if should_force_gpu_panic(env::var_os("ELODIN_GPU_PANIC").as_deref()) {
        panic!("Unable to find a GPU! Make sure you have installed required drivers!");
    }
}

fn panic_message<'a>(info: &'a std::panic::PanicHookInfo<'_>) -> Option<&'a str> {
    info.payload()
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
}

fn is_gpu_init_panic(message: &str) -> bool {
    message.contains("Unable to find a GPU")
}

fn should_force_gpu_panic(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("true"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_gpu_initialization_panic() {
        assert!(is_gpu_init_panic(
            "Unable to find a GPU! Make sure you have installed required drivers!"
        ));
        assert!(!is_gpu_init_panic("another panic"));
    }

    #[test]
    fn force_panic_requires_true() {
        assert!(should_force_gpu_panic(Some(OsStr::new("true"))));
        assert!(!should_force_gpu_panic(Some(OsStr::new("1"))));
        assert!(!should_force_gpu_panic(Some(OsStr::new("false"))));
        assert!(!should_force_gpu_panic(None));
    }
}
