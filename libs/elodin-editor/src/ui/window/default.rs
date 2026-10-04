use bevy::window::{CompositeAlphaMode, PresentMode, Window, WindowTheme};

#[cfg(feature = "tracy")]
pub const fn default_present_mode() -> PresentMode {
    PresentMode::AutoNoVsync
}

#[cfg(not(feature = "tracy"))]
pub const fn default_present_mode() -> PresentMode {
    PresentMode::Fifo
}

pub const fn default_window_theme() -> Option<WindowTheme> {
    Some(WindowTheme::Dark)
}

pub fn default_composite_alpha_mode() -> CompositeAlphaMode {
    if cfg!(target_os = "macos") {
        CompositeAlphaMode::PostMultiplied
    } else {
        CompositeAlphaMode::Opaque
    }
}

/// `ELODIN_PRESENT_MODE=novsync|vsync|fifo` overrides the compiled-in default;
/// used to diagnose presentation-pacing FPS caps (e.g. ProMotion settling low).
pub fn present_mode_from_env() -> PresentMode {
    match std::env::var("ELODIN_PRESENT_MODE").as_deref() {
        Ok("novsync") => PresentMode::AutoNoVsync,
        Ok("vsync") => PresentMode::AutoVsync,
        Ok("fifo") => PresentMode::Fifo,
        _ => default_present_mode(),
    }
}

pub fn base_window() -> Window {
    Window {
        present_mode: present_mode_from_env(),
        window_theme: default_window_theme(),
        composite_alpha_mode: default_composite_alpha_mode(),
        ..Default::default()
    }
}

pub fn window_theme_for_mode(mode: Option<&str>) -> Option<WindowTheme> {
    match mode.map(|m| m.to_ascii_lowercase()) {
        Some(mode) if mode == "light" => Some(WindowTheme::Light),
        _ => default_window_theme(),
    }
}
