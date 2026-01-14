use std::{collections::HashMap, time::Duration};

#[cfg(not(target_os = "macos"))]
use bevy::log::error;
use bevy::{
    app::AppExit,
    log::{info, warn},
    prelude::*,
    window::{PrimaryWindow, WindowCloseRequested},
    winit::WINIT_WINDOWS,
};
use bevy_defer::{AccessError, AsyncCommandsExtension, AsyncWorld};
use impeller2_wkt::WindowRect;
#[cfg(target_os = "macos")]
use winit::dpi::LogicalPosition;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    monitor::MonitorHandle,
    window::Window as WinitWindow,
};

use crate::ui::tiles::{WindowId, WindowRelayout, WindowState};

pub fn handle_window_close(
    mut events: EventReader<WindowCloseRequested>,
    primary: Query<&WindowId, With<PrimaryWindow>>,
    mut exit: EventWriter<AppExit>,
) {
    for evt in events.read() {
        let entity = evt.window;
        if primary
            .get(entity)
            .map(|window_id| window_id.is_primary())
            .unwrap_or(false)
        {
            exit.write(AppExit::Success);
        }
    }
}

pub async fn wait_for_winit_window(
    window_id: Entity,
    timeout: Duration,
) -> Result<bool, AccessError> {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        let window_ready = AsyncWorld.run(|_world| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                winit_windows.get_window(window_id).is_some()
            })
        });
        if window_ready {
            return Ok(true);
        }
        AsyncWorld.yield_now().await;
    }
    Ok(false)
}

#[cfg(target_os = "macos")]
pub async fn apply_physical_screen_rect(
    window_entity: Entity,
    screen_index: usize,
    rect: WindowRect,
) -> bevy_defer::AccessResult {
    if !wait_for_winit_window(window_entity, Duration::from_millis(2000)).await? {
        warn!(%window_entity, "apply_physical_screen_rect: winit window not ready");
        return Ok(());
    }

    let target = AsyncWorld.run(
        |_world| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
            let any_window = winit_windows.windows.values().next()?;
            let screens = collect_sorted_screens(any_window);
            log_screens("macos.load.screens", &screens);
            let screen = screens.get(screen_index)?;
            let pos = screen_physical_position(screen);
            let size = screen_physical_size(screen);
            if size.width <= 0.0 || size.height <= 0.0 {
                return None;
            }
            let req_x = pos.x + (rect.x as f64 / 100.0) * size.width;
            let req_y = pos.y + (rect.y as f64 / 100.0) * size.height;
            let req_w = ((rect.width as f64 / 100.0) * size.width).round().max(1.0);
            let req_h = ((rect.height as f64 / 100.0) * size.height)
                .round()
                .max(1.0);

            let max_x = pos.x + size.width - req_w;
            let max_y = pos.y + size.height - req_h;
            let clamped_x = req_x.clamp(pos.x, max_x.max(pos.x));
            let clamped_y = req_y.clamp(pos.y, max_y.max(pos.y));
            info!(
                target_screen = screen_index,
                screen_pos = ?pos,
                screen_size = ?size,
                req_pos = ?LogicalPosition::new(req_x, req_y),
                req_size = ?LogicalSize::new(req_w, req_h),
                clamped_pos = ?LogicalPosition::new(clamped_x, clamped_y),
                "mac apply_physical_screen_rect request"
            );

            Some((
                LogicalPosition::new(clamped_x, clamped_y),
                LogicalSize::new(req_w, req_h),
            ))
            })
        },
    );

    let Some((pos, size)) = target else {
        warn!(
            ?screen_index,
            "apply_physical_screen_rect: no screen found for index"
        );
        return Ok(());
    };

    AsyncWorld.run(|_world| {
        WINIT_WINDOWS.with_borrow(|winit_windows| {
            let Some(window) = winit_windows.get_window(window_entity) else {
                warn!(%window_entity, "apply_physical_screen_rect: winit window not found");
                return;
            };

            if let Some(screen) = window.current_monitor() {
                info!(
                    current_screen = ?screen.name(),
                    current_pos = ?screen.position(),
                    current_size = ?screen.size(),
                    "apply_physical_screen_rect: current screen info"
                );
            }

            // Safety: macOS expects physical coords for resize/move.
            let (physical_size, physical_pos) = {
                let scale_factor = window.scale_factor().max(0.0001);
                let size = PhysicalSize::new(
                    (size.width * scale_factor).round().max(1.0),
                    (size.height * scale_factor).round().max(1.0),
                );
                let pos = winit::dpi::PhysicalPosition::new(
                    (pos.x * scale_factor).round(),
                    (pos.y * scale_factor).round(),
                );
                (size, pos)
            };

            let _ = window.request_inner_size(physical_size);
            window.set_outer_position(physical_pos);
        })
    });
    info!(
        window = %window_entity,
        target_screen = screen_index,
        rect = ?rect,
        "mac apply_physical_screen_rect applied"
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
async fn apply_window_screen(entity: Entity, screen: usize) -> Result<(), bevy_defer::AccessError> {
    info!(
        window = %entity,
        %screen,
        "apply_window_screen start"
    );
    if !wait_for_winit_window(entity, Duration::from_millis(2000)).await? {
        error!(%entity, "Unable to apply window to screen: winit window not found.");
        return Ok(());
    }
    let window_states = AsyncWorld.query::<&WindowState>();
    let mut state = window_states
        .entity(entity)
        .get_mut(|state| state.clone())?;
    let target_monitor_maybe = AsyncWorld.run(|_world| {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        let Some(window) = winit_windows.get_window(entity) else {
            error!(%entity, "No winit window in apply window screen");
            return None;
        };

        let screens = collect_sorted_screens(window);

        if window_on_target_screen(&mut state, window, &screens) {
            if crate::ui::platform::LINUX_MULTI_WINDOW {
                exit_fullscreen(window);
                force_windowed(window);
            } else if window.fullscreen().is_some() {
                exit_fullscreen(window);
            }
            return None;
        }

        if let Some(target_monitor) = screens.get(screen).cloned() {
            assign_window_to_screen(window, target_monitor.clone());
            Some(target_monitor)
        } else {
            warn!(
                screen,
                path = ?state.descriptor.path,
                "screen out of range; skipping screen assignment"
            );
            state.descriptor.screen = None;
            warn!(%entity, "screen out of range");
            None
        }
    })
    });
    if let Some(target_monitor) = target_monitor_maybe {
        let success =
            wait_for_window_to_change_screens(entity, screen, Duration::from_millis(1000)).await?;
        AsyncWorld.run(|_world| {
        WINIT_WINDOWS.with_borrow(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                error!(%entity, "No winit window in apply screen");
                return;
            };
            let screens = collect_sorted_screens(window);
            let detected_screen = detect_window_screen(window, &screens);
            let on_target = detected_screen == Some(screen);
            if success || on_target {
                if crate::ui::platform::LINUX_MULTI_WINDOW {
                    exit_fullscreen(window);
                    force_windowed(window);
                } else if window.fullscreen().is_some() {
                    exit_fullscreen(window);
                }
            } else {
                recenter_window_on_screen(window, &target_monitor);
            }
        })
        });
    }
    Ok(())
}

pub fn handle_window_relayout_events(
    mut relayout_events: EventReader<WindowRelayout>,
    mut commands: Commands,
    mut per_window: Local<HashMap<Entity, Vec<WindowRelayout>>>,
) {
    if relayout_events.is_empty() {
        return;
    }
    per_window.clear();
    for relayout_event in relayout_events.read() {
        match relayout_event {
            e @ WindowRelayout::Screen { window, screen: _ } => {
                per_window.entry(*window).or_default().push(e.clone());
            }
            e @ WindowRelayout::Rect { window, rect: _ } => {
                per_window.entry(*window).or_default().push(e.clone());
            }
            WindowRelayout::UpdateDescriptors => {}
        }
    }

    for (_id, relayout_events) in per_window.drain() {
        commands.spawn_task(move || async {
            for relayout_event in relayout_events {
                match relayout_event {
                    WindowRelayout::Screen { window, screen } => {
                        info!(
                            target_screen = screen,
                            "Attempting window screen assignment"
                        );
                        #[cfg(target_os = "macos")]
                        {
                            let window_states = AsyncWorld.query::<&WindowState>();
                            if let Ok(Some(rect)) = window_states
                                .entity(window)
                                .get(|state| state.descriptor.screen_rect)
                            {
                                apply_physical_screen_rect(window, screen, rect).await.ok();
                                continue;
                            }
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            apply_window_screen(window, screen).await?;
                        }
                    }
                    WindowRelayout::Rect { window, rect } => {
                        #[cfg(target_os = "macos")]
                        {
                            if let Some(screen) = AsyncWorld
                                .query::<&WindowState>()
                                .entity(window)
                                .get(|state| state.descriptor.screen)
                                .ok()
                                .flatten()
                            {
                                apply_physical_screen_rect(window, screen, rect).await.ok();
                                continue;
                            }
                            apply_physical_screen_rect(window, 0, rect).await.ok();
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            apply_window_rect(rect, window, Duration::from_millis(1000)).await?;
                        }
                    }
                    WindowRelayout::UpdateDescriptors => {
                        unreachable!();
                    }
                }
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "macos"))]
async fn apply_window_rect(
    rect: WindowRect,
    entity: Entity,
    timeout: Duration,
) -> Result<(), AccessError> {
    info!(
        window = %entity,
        ?rect,
        "apply_window_rect start"
    );

    let window_states = AsyncWorld.query::<&WindowState>();
    let state = window_states.entity(entity).get_mut(|state| state.clone())?;

    let is_full_rect = crate::ui::platform::LINUX_MULTI_WINDOW
        && rect.x == 0
        && rect.y == 0
        && rect.width == 100
        && rect.height == 100;

    let start = std::time::Instant::now();
    let mut wait = true;
    while wait && start.elapsed() < timeout {
        AsyncWorld.yield_now().await;
        wait = AsyncWorld.run(|_world| {
        WINIT_WINDOWS.with_borrow(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                error!(%entity, "No winit window in apply rect");
                return true;
            };

            if rect.width == 0 && rect.height == 0 {
                linux_request_minimize(window);
                info!("Applied minimize rect");
                return false;
            }

            let screen_handle = if let Some(idx) = state.descriptor.screen {
                let screens = collect_sorted_screens(window);
                screens
                    .get(idx)
                    .cloned()
                    .or_else(|| window.current_monitor())
            } else {
                window.current_monitor()
            };

            let Some(screen_handle) = screen_handle else {
                warn!("No screen handle available; retrying");
                return true;
            };

            let scale_factor = window.scale_factor().max(0.0001);
            let scale_with_monitor = screen_handle.scale_factor().max(0.0001);
            let LogicalSize { width, height } = screen_physical_size(&screen_handle);
            let screen_width = width.max(1.0);
            let screen_height = height.max(1.0);
            let req_x = (rect.x as f64 / 100.0) * screen_width;
            let req_y = (rect.y as f64 / 100.0) * screen_height;
            let req_w = (rect.width as f64 / 100.0) * screen_width;
            let req_h = (rect.height as f64 / 100.0) * screen_height;

            let pos_x = req_x + screen_handle.position().x as f64;
            let pos_y = req_y + screen_handle.position().y as f64;
            let size = PhysicalSize::new(
                (req_w * scale_factor).round() as u32,
                (req_h * scale_factor).round() as u32,
            );
            let pos = winit::dpi::PhysicalPosition::new(
                (pos_x * scale_with_monitor).round(),
                (pos_y * scale_with_monitor).round(),
            );
            window.set_outer_position(pos);
            let _ = window.request_inner_size(size);
            if is_full_rect {
                info!("Applied full-screen rect via positioning");
            }
            false
        })
        });
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
async fn wait_for_window_to_change_screens(
    entity: Entity,
    target_screen: usize,
    timeout: Duration,
) -> Result<bool, AccessError> {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout {
            info!(
                window = %entity,
                timeout_ms = timeout.as_millis(),
                "Timed out waiting for window to change screens"
            );
            return Ok(false);
        }

        AsyncWorld.yield_now().await;

        let on_target = AsyncWorld.run(|_world| {
        WINIT_WINDOWS.with_borrow(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                warn!(%entity, "No winit window when waiting for screen change");
                return false;
            };
            let screens = collect_sorted_screens(window);
            let screen = detect_window_screen(window, &screens);
            if screen == Some(target_screen) {
                info!(window = %entity, %target_screen, "Window now on target screen");
                true
            } else {
                false
            }
        })
        });
        if on_target {
            return Ok(true);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn assign_window_to_screen(window: &WinitWindow, target_monitor: MonitorHandle) {
    info!(
        target = ?target_monitor.name(),
        pos = ?target_monitor.position(),
        size = ?target_monitor.size(),
        "Assign window to screen"
    );
    if crate::ui::platform::LINUX_MULTI_WINDOW {
        exit_fullscreen(window);
        force_windowed(window);
        linux_clear_minimized(window);
    } else if window.fullscreen().is_some() {
        exit_fullscreen(window);
    }
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(
        target_monitor,
    ))));
}

#[cfg(not(target_os = "macos"))]
fn recenter_window_on_screen(window: &WinitWindow, target_monitor: &MonitorHandle) {
    let physical_size = window.inner_size();
    let LogicalSize { width, height } = screen_physical_size(target_monitor);
    let screen_size = PhysicalSize::new(width as u32, height as u32);
    let mut x =
        target_monitor.position().x + (screen_size.width as i32 - physical_size.width as i32) / 2;
    let mut y =
        target_monitor.position().y + (screen_size.height as i32 - physical_size.height as i32) / 2;
    x = x.max(target_monitor.position().x);
    y = y.max(target_monitor.position().y);
    let target_pos = winit::dpi::PhysicalPosition::new(x, y);
    window.set_outer_position(target_pos);
    info!(
        ?target_pos,
        target_screen = ?target_monitor.name(),
        "recenter window on screen"
    );
}

pub fn collect_sorted_screens(window: &WinitWindow) -> Vec<MonitorHandle> {
    let mut monitors: Vec<MonitorHandle> = window.available_monitors().collect();
    monitors.sort_by_key(|m| {
        let pos = m.position();
        (pos.x, pos.y)
    });
    monitors
}

fn screens_match(a: &MonitorHandle, b: &MonitorHandle) -> bool {
    a.name() == b.name() && a.position() == b.position() && a.size() == b.size()
}

#[cfg(target_os = "macos")]
fn log_screens(label: &str, screens: &[MonitorHandle]) {
    let named: Vec<_> = screens
        .iter()
        .map(|s| {
            (
                s.name().unwrap_or_else(|| "Unknown".to_string()),
                s.position(),
                s.size(),
            )
        })
        .collect();
    info!(?named, label);
}

#[cfg(target_os = "macos")]
fn screen_physical_position(screen: &MonitorHandle) -> LogicalPosition<f64> {
    let pos = screen.position();
    LogicalPosition::new(pos.x as f64, pos.y as f64)
}

#[cfg(target_os = "macos")]
fn screen_physical_size(screen: &MonitorHandle) -> LogicalSize<f64> {
    let size = screen.size();
    let scale = screen.scale_factor().max(0.0001);
    LogicalSize::new(size.width as f64 / scale, size.height as f64 / scale)
}

#[cfg(not(target_os = "macos"))]
fn screen_physical_size(screen: &MonitorHandle) -> LogicalSize<f64> {
    let size = screen.size();
    LogicalSize::new(size.width as f64, size.height as f64)
}

#[cfg(not(target_os = "macos"))]
fn window_on_target_screen(
    state: &mut WindowState,
    window: &WinitWindow,
    screens: &[MonitorHandle],
) -> bool {
    let Some(target_screen) = state.descriptor.screen else {
        return false;
    };
    if let Some(current) = window.current_monitor()
        && screens_match(&current, &screens[target_screen])
    {
        info!(
            current_screen = target_screen,
            "window already on target screen"
        );
        return true;
    }
    false
}

#[cfg(not(target_os = "macos"))]
fn exit_fullscreen(window: &WinitWindow) {
    window.set_fullscreen(None);
}

#[cfg(not(target_os = "macos"))]
fn force_windowed(window: &WinitWindow) {
    window.set_decorations(true);
    window.set_resizable(true);
}

#[cfg(not(target_os = "macos"))]
fn linux_clear_minimized(_window: &WinitWindow) {
    _window.set_minimized(false);
    info!("Cleared minimized state on X11 window");
}

#[cfg(not(target_os = "macos"))]
fn linux_request_minimize(_window: &WinitWindow) {
    #[cfg(target_os = "linux")]
    {
        _window.set_minimized(true);
    }
}

pub fn detect_window_screen(window: &WinitWindow, screens: &[MonitorHandle]) -> Option<usize> {
    if let Some(current) = window.current_monitor() {
        for (idx, screen) in screens.iter().enumerate() {
            if screens_match(&current, screen) {
                return Some(idx);
            }
        }
    }
    None
}
