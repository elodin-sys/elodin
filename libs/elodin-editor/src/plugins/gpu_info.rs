use std::{env, ffi::OsStr, sync::Once};

use bevy::{
    app::Plugin,
    prelude::{App, Local, Res},
    render::{Render, RenderApp, renderer::RenderAdapterInfo},
};

fn gpu_failure_help() -> &'static str {
    #[cfg(target_arch = "wasm32")]
    {
        "\
elodin: unable to initialize graphics because no GPU was found.
elodin: this browser blocked WebGPU and WebGL.
elodin: allow graphics and hardware acceleration for this site."
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        "\
elodin: unable to initialize graphics because no GPU was found.
elodin: select a graphics path before entering the development shell:
  ELODIN_GPU=nvidia nix develop
  ELODIN_GPU=mesa nix develop
  ELODIN_GPU=nvk nix develop"
    }
}

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
                #[cfg(target_arch = "wasm32")]
                web_sys::console::error_1(&gpu_failure_help().into());
                #[cfg(not(target_arch = "wasm32"))]
                eprintln!("{}", gpu_failure_help());
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

pub(crate) struct GraphicsBackendStatus {
    pub label: &'static str,
    pub accelerated: bool,
}

pub(crate) fn graphics_backend_status(
    backend: wgpu::Backend,
    device_type: wgpu::DeviceType,
    wasm: bool,
) -> GraphicsBackendStatus {
    let software = device_type == wgpu::DeviceType::Cpu;
    let label = match backend {
        wgpu::Backend::Noop => "No GPU",
        wgpu::Backend::BrowserWebGpu => {
            if software {
                "WebGPU software"
            } else {
                "WebGPU"
            }
        }
        wgpu::Backend::Gl => match (wasm, software) {
            (true, false) => "WebGL",
            (true, true) => "WebGL software",
            (false, false) => "OpenGL",
            (false, true) => "OpenGL software",
        },
        wgpu::Backend::Vulkan => {
            if software {
                "Vulkan software"
            } else {
                "Vulkan"
            }
        }
        wgpu::Backend::Metal => {
            if software {
                "Metal software"
            } else {
                "Metal"
            }
        }
        wgpu::Backend::Dx12 => {
            if software {
                "D3D12 software"
            } else {
                "D3D12"
            }
        }
    };
    let accelerated = !software
        && backend != wgpu::Backend::Noop
        && matches!(
            device_type,
            wgpu::DeviceType::IntegratedGpu
                | wgpu::DeviceType::DiscreteGpu
                | wgpu::DeviceType::VirtualGpu
        );
    GraphicsBackendStatus { label, accelerated }
}

pub(crate) fn graphics_backend_hover(
    accelerated: bool,
    name: &str,
    driver: &str,
    driver_info: &str,
) -> String {
    let mut lines = vec![
        if accelerated {
            "GPU accelerated"
        } else {
            "Not GPU accelerated"
        },
        name,
    ];
    if !driver.is_empty() {
        lines.push(driver);
    }
    if !driver_info.is_empty() && driver_info != driver {
        lines.push(driver_info);
    }
    lines.join("\n")
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
    fn gpu_failure_help_matches_this_target() {
        let help = gpu_failure_help();
        #[cfg(target_arch = "wasm32")]
        assert!(help.contains("hardware acceleration"));
        #[cfg(not(target_arch = "wasm32"))]
        assert!(help.contains("ELODIN_GPU"));
    }

    #[test]
    fn wasm_webgpu_on_a_real_gpu_is_accelerated() {
        let status = graphics_backend_status(
            wgpu::Backend::BrowserWebGpu,
            wgpu::DeviceType::IntegratedGpu,
            true,
        );
        assert_eq!(status.label, "WebGPU");
        assert!(status.accelerated);
    }

    #[test]
    fn wasm_gl_is_webgl() {
        let status =
            graphics_backend_status(wgpu::Backend::Gl, wgpu::DeviceType::DiscreteGpu, true);
        assert_eq!(status.label, "WebGL");
        assert!(status.accelerated);
    }

    #[test]
    fn native_backends_name_the_api() {
        assert_eq!(
            graphics_backend_status(wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu, false)
                .label,
            "Vulkan"
        );
        assert_eq!(
            graphics_backend_status(wgpu::Backend::Metal, wgpu::DeviceType::IntegratedGpu, false)
                .label,
            "Metal"
        );
        assert_eq!(
            graphics_backend_status(wgpu::Backend::Dx12, wgpu::DeviceType::DiscreteGpu, false)
                .label,
            "D3D12"
        );
        assert_eq!(
            graphics_backend_status(wgpu::Backend::Gl, wgpu::DeviceType::IntegratedGpu, false)
                .label,
            "OpenGL"
        );
        assert!(
            graphics_backend_status(wgpu::Backend::Vulkan, wgpu::DeviceType::VirtualGpu, false)
                .accelerated
        );
    }

    #[test]
    fn cpu_and_noop_are_not_gpu_accelerated() {
        let software = graphics_backend_status(wgpu::Backend::Vulkan, wgpu::DeviceType::Cpu, false);
        assert_eq!(software.label, "Vulkan software");
        assert!(!software.accelerated);

        let webgl_software =
            graphics_backend_status(wgpu::Backend::Gl, wgpu::DeviceType::Cpu, true);
        assert_eq!(webgl_software.label, "WebGL software");
        assert!(!webgl_software.accelerated);

        let noop = graphics_backend_status(wgpu::Backend::Noop, wgpu::DeviceType::Other, false);
        assert_eq!(noop.label, "No GPU");
        assert!(!noop.accelerated);

        let unknown =
            graphics_backend_status(wgpu::Backend::Vulkan, wgpu::DeviceType::Other, false);
        assert_eq!(unknown.label, "Vulkan");
        assert!(!unknown.accelerated);
    }

    #[test]
    fn hover_says_whether_the_adapter_is_accelerated() {
        let accelerated = graphics_backend_hover(true, "Arc Graphics", "", "WebGL 2.0");
        assert!(accelerated.starts_with("GPU accelerated"));
        assert!(accelerated.contains("Arc Graphics"));
        assert!(accelerated.contains("WebGL 2.0"));

        let software = graphics_backend_hover(false, "llvmpipe", "llvmpipe", "llvmpipe");
        assert!(software.starts_with("Not GPU accelerated"));
        assert!(software.contains("llvmpipe"));
        assert_eq!(software.matches("llvmpipe").count(), 2);
    }

    #[test]
    fn force_panic_requires_true() {
        assert!(should_force_gpu_panic(Some(OsStr::new("true"))));
        assert!(!should_force_gpu_panic(Some(OsStr::new("1"))));
        assert!(!should_force_gpu_panic(Some(OsStr::new("false"))));
        assert!(!should_force_gpu_panic(None));
    }
}
