use std::{env, ffi::OsStr, sync::Once};

use bevy::{
    app::Plugin,
    prelude::{App, Commands, First, Res, Resource},
    render::renderer::RenderAdapterInfo,
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
        // RenderPlugin installs the adapter before the first update. Publish
        // GpuInfo before shader assets are extracted on that first frame.
        app.add_systems(First, copy_adapter_info);
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

#[derive(Resource)]
pub(crate) struct GpuInfo(pub(crate) RenderAdapterInfo);

impl GpuInfo {
    pub(crate) fn label(&self) -> &'static str {
        self.label_for(cfg!(target_arch = "wasm32"))
    }

    fn label_for(&self, wasm: bool) -> &'static str {
        let software = self.0.device_type == wgpu::DeviceType::Cpu;
        match self.0.backend {
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
        }
    }

    pub(crate) fn accelerated(&self) -> bool {
        self.0.device_type != wgpu::DeviceType::Cpu
            && self.0.backend != wgpu::Backend::Noop
            && matches!(
                self.0.device_type,
                wgpu::DeviceType::IntegratedGpu
                    | wgpu::DeviceType::DiscreteGpu
                    | wgpu::DeviceType::VirtualGpu
            )
    }

    pub(crate) fn hover(&self) -> String {
        let mut lines = vec![
            if self.accelerated() {
                "GPU accelerated"
            } else {
                "Not GPU accelerated"
            },
            self.0.name.as_str(),
        ];
        if !self.0.driver.is_empty() {
            lines.push(self.0.driver.as_str());
        }
        if !self.0.driver_info.is_empty() && self.0.driver_info != self.0.driver {
            lines.push(self.0.driver_info.as_str());
        }
        lines.join("\n")
    }
}

fn copy_adapter_info(
    adapter: Res<RenderAdapterInfo>,
    existing: Option<Res<GpuInfo>>,
    mut commands: Commands,
) {
    if existing.is_some() {
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
    commands.insert_resource(GpuInfo(adapter.clone()));
}

#[cfg(test)]
pub(crate) mod tests {
    use bevy::render::renderer::WgpuWrapper;

    use super::*;

    fn graphics_backend(
        api: wgpu::Backend,
        device_type: wgpu::DeviceType,
        name: &str,
        driver: &str,
        driver_info: &str,
    ) -> GpuInfo {
        GpuInfo(RenderAdapterInfo(WgpuWrapper::new(wgpu::AdapterInfo {
            name: name.to_owned(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: driver.to_owned(),
            driver_info: driver_info.to_owned(),
            backend: api,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: false,
        })))
    }

    pub(crate) fn backend(api: wgpu::Backend, device_type: wgpu::DeviceType) -> GpuInfo {
        graphics_backend(api, device_type, "", "", "")
    }

    #[test]
    fn adapter_info_is_available_before_the_first_update() {
        use bevy::prelude::Update;

        let mut app = App::new();
        app.insert_resource(backend(wgpu::Backend::BrowserWebGpu, wgpu::DeviceType::Other).0)
            .add_plugins(GpuInfoPlugin)
            .add_systems(Update, |gpu: Res<GpuInfo>| {
                assert_eq!(gpu.0.backend, wgpu::Backend::BrowserWebGpu);
            });
        app.update();
    }

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
        let status = backend(
            wgpu::Backend::BrowserWebGpu,
            wgpu::DeviceType::IntegratedGpu,
        );
        assert_eq!(status.label_for(true), "WebGPU");
        assert!(status.accelerated());
    }

    #[test]
    fn wasm_gl_is_webgl() {
        let status = backend(wgpu::Backend::Gl, wgpu::DeviceType::DiscreteGpu);
        assert_eq!(status.label_for(true), "WebGL");
        assert!(status.accelerated());
    }

    #[test]
    fn native_backends_name_the_api() {
        assert_eq!(
            backend(wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu).label_for(false),
            "Vulkan"
        );
        assert_eq!(
            backend(wgpu::Backend::Metal, wgpu::DeviceType::IntegratedGpu).label_for(false),
            "Metal"
        );
        assert_eq!(
            backend(wgpu::Backend::Dx12, wgpu::DeviceType::DiscreteGpu).label_for(false),
            "D3D12"
        );
        assert_eq!(
            backend(wgpu::Backend::Gl, wgpu::DeviceType::IntegratedGpu).label_for(false),
            "OpenGL"
        );
        assert!(backend(wgpu::Backend::Vulkan, wgpu::DeviceType::VirtualGpu).accelerated());
    }

    #[test]
    fn label_uses_this_targets_api_name() {
        let status = backend(wgpu::Backend::Gl, wgpu::DeviceType::DiscreteGpu);
        #[cfg(target_arch = "wasm32")]
        assert_eq!(status.label(), "WebGL");
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(status.label(), "OpenGL");
    }

    #[test]
    fn cpu_and_noop_are_not_gpu_accelerated() {
        let software = backend(wgpu::Backend::Vulkan, wgpu::DeviceType::Cpu);
        assert_eq!(software.label_for(false), "Vulkan software");
        assert!(!software.accelerated());

        let webgl_software = backend(wgpu::Backend::Gl, wgpu::DeviceType::Cpu);
        assert_eq!(webgl_software.label_for(true), "WebGL software");
        assert!(!webgl_software.accelerated());

        let noop = backend(wgpu::Backend::Noop, wgpu::DeviceType::Other);
        assert_eq!(noop.label_for(false), "No GPU");
        assert!(!noop.accelerated());

        let unknown = backend(wgpu::Backend::Vulkan, wgpu::DeviceType::Other);
        assert_eq!(unknown.label_for(false), "Vulkan");
        assert!(!unknown.accelerated());
    }

    #[test]
    fn hover_says_whether_the_adapter_is_accelerated() {
        let accelerated = graphics_backend(
            wgpu::Backend::Gl,
            wgpu::DeviceType::IntegratedGpu,
            "Arc Graphics",
            "",
            "WebGL 2.0",
        )
        .hover();
        assert!(accelerated.starts_with("GPU accelerated"));
        assert!(accelerated.contains("Arc Graphics"));
        assert!(accelerated.contains("WebGL 2.0"));

        let software = graphics_backend(
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::Cpu,
            "llvmpipe",
            "llvmpipe",
            "llvmpipe",
        )
        .hover();
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
