use bevy::prelude::*;
use bevy::window::{
    CompositeAlphaMode,
    PresentMode,
    Window,
    WindowLevel,
    WindowResolution,
    WindowTheme,
    WindowPosition,
};

#[derive(Component)]
pub struct MotorPanelWindow;

#[derive(Component)]
pub struct RatePanelWindow;

fn secondary_window(title: &str, position: WindowPosition) -> Window {
    let composite_alpha_mode = if cfg!(target_os = "macos") {
        CompositeAlphaMode::PostMultiplied
    } else {
        CompositeAlphaMode::Opaque
    };

    Window {
        title: title.into(),
        resolution: WindowResolution::new(960.0, 720.0),
        present_mode: PresentMode::AutoVsync,
        window_theme: Some(WindowTheme::Dark),
        decorations: cfg!(target_os = "macos"),
        visible: true,
        resizable: true,
        composite_alpha_mode,
        prevent_default_event_handling: true,
        position,
        window_level: WindowLevel::Normal,
        ..default()
    }
}

pub fn spawn_secondary_windows(mut commands: Commands) {
    // Position the windows so they do not overlap the primary window entirely.
    let motor_position = WindowPosition::At(IVec2::new(60, 60));
    let rate_position = WindowPosition::At(IVec2::new(80 + 960, 80));

    commands.spawn((
        secondary_window("Motor Panel", motor_position),
        MotorPanelWindow,
    ));

    commands.spawn((
        secondary_window("Rate Control Panel", rate_position),
        RatePanelWindow,
    ));
}
