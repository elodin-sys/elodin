use bevy::window::{CompositeAlphaMode, PresentMode, Window, WindowTheme};

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

pub fn base_window() -> Window {
    Window {
        present_mode: default_present_mode(),
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
