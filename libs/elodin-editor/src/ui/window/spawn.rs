use std::collections::HashMap;

use bevy::{
    camera::{Camera2d, RenderTarget},
    log::info,
    prelude::*,
    ui::UiTargetCamera,
    window::{EnabledButtons, Window, WindowPosition, WindowRef, WindowResolution},
    winit::WINIT_WINDOWS,
    ecs::system::NonSendMarker,
};
use bevy_egui::EguiContextSettings;
use egui_tiles::Tile;

use crate::ui::{
    UI_ORDER_BASE, base_window, create_egui_context,
    tiles::{PaneRenderTargets, WindowId, WindowRelayout, WindowState},
    window_theme_for_mode,
};

use super::placement::collect_sorted_screens;

pub fn sync_windows(
    mut commands: Commands,
    mut windows_state: Query<(Entity, &WindowId, &mut WindowState, Option<&mut Window>)>,
    mut cameras: Query<&mut Camera>,
    children: Query<&Children>,
    mut existing_map: Local<HashMap<WindowId, Entity>>,
    _non_send_marker: NonSendMarker,
) {
    let screens_any = WINIT_WINDOWS.with_borrow(|winit_windows| {
        winit_windows
            .windows
            .values()
            .next()
            .map(|w| collect_sorted_screens(w))
    });
    if screens_any.is_none() {
        warn!("No screen info available; windows will use default sizing/position");
    }

    for (entity, marker, mut state, window_maybe) in &mut windows_state {
        let PaneRenderTargets {
            cameras: camera_targets,
            ui_nodes,
        } = state.tile_state.collect_render_targets();
        state.graph_entities = camera_targets;
        for ui_node in ui_nodes {
            assign_ui_target_camera(&mut commands, &children, ui_node, entity);
        }

        if let Some(mut window) = window_maybe {
            window.window_theme = window_theme_for_mode(state.descriptor.mode.as_deref());
            existing_map.insert(*marker, entity);
            for (index, &graph) in state.graph_entities.iter().enumerate() {
                if let Ok(mut camera) = cameras.get_mut(graph) {
                    retarget_camera(&mut camera, entity);
                    camera.is_active = true;
                    let base_order = window_graph_order_base(*marker);
                    camera.order = base_order + index as isize;
                }
            }
            continue;
        }

        for &graph in &state.graph_entities {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                camera.is_active = false;
            }
        }

        let title = compute_window_title(&state);

        let (resolution, position, _pre_applied_screen) = if let Some(rect) =
            state.descriptor.screen_rect
            && let Some(screen_idx) = state.descriptor.screen
            && let Some(screens) = screens_any.as_ref()
            && let Some(screen) = screens.get(screen_idx)
        {
            let screen_pos = screen.position();
            let screen_size = screen.size();
            let width_px = ((rect.width as f64 / 100.0) * screen_size.width as f64)
                .round()
                .max(1.0);
            let height_px = ((rect.height as f64 / 100.0) * screen_size.height as f64)
                .round()
                .max(1.0);
            let x =
                screen_pos.x + ((rect.x as f64 / 100.0) * screen_size.width as f64).round() as i32;
            let y =
                screen_pos.y + ((rect.y as f64 / 100.0) * screen_size.height as f64).round() as i32;
            (
                WindowResolution::new(width_px as u32, height_px as u32),
                Some(WindowPosition::At(IVec2::new(x, y))),
                Some(screen_idx),
            )
        } else {
            (
                WindowResolution::new(640, 480),
                None,
                state.descriptor.screen,
            )
        };

        let egui_context = create_egui_context();

        let window_component = Window {
            title,
            resolution,
            position: position.unwrap_or(WindowPosition::Automatic),
            enabled_buttons: EnabledButtons {
                close: true,
                minimize: true,
                maximize: true,
            },
            window_theme: window_theme_for_mode(state.descriptor.mode.as_deref()),
            ..base_window()
        };

        let window_entity = commands
            .entity(entity)
            .insert((
                window_component,
                *marker,
                egui_context,
                EguiContextSettings::default(),
                Camera2d,
            ))
            .id();

        let camera = Camera {
            order: UI_ORDER_BASE,
            target: RenderTarget::Window(WindowRef::Entity(window_entity)),
            ..Default::default()
        };

        commands.entity(entity).insert(camera);

        if let Some(screen) = state.descriptor.screen.as_ref() {
            commands.send_event(WindowRelayout::Screen {
                window: window_entity,
                screen: *screen,
            });
        }

        if let Some(rect) = state.descriptor.screen_rect.as_ref() {
            commands.send_event(WindowRelayout::Rect {
                window: window_entity,
                rect: *rect,
            });
        }
        #[cfg(target_os = "macos")]
        {
            if let (Some(screen_idx), Some(rect)) =
                (state.descriptor.screen, state.descriptor.screen_rect)
            {
                info!(
                    window = %window_entity,
                    target_screen = screen_idx,
                    rect = ?rect,
                    "mac spawn: apply physical rect"
                );
                use bevy_defer::AsyncCommandsExtension;
                commands.spawn_task(move || async move {
                    super::placement::apply_physical_screen_rect(window_entity, screen_idx, rect)
                        .await
                        .ok();
                    Ok(())
                });
            }
        }
        existing_map.insert(*marker, window_entity);
        info!(
            "Created window entity {window_entity} with window id {:?}",
            marker
        );
        for (index, &graph) in state.graph_entities.iter().enumerate() {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                retarget_camera(&mut camera, window_entity);
                camera.is_active = true;
                let base_order = window_graph_order_base(*marker);
                camera.order = base_order + index as isize;
            }
        }
    }
}

fn retarget_camera(camera: &mut Camera, window_entity: Entity) {
    let matches_target = matches!(
        camera.target,
        RenderTarget::Window(WindowRef::Entity(entity)) if entity == window_entity
    );
    if !matches_target {
        // Force bevy's camera_system to recompute target_info after window retargeting.
        camera.target = RenderTarget::Window(WindowRef::Entity(window_entity));
        camera.computed = Default::default();
    }
}

fn assign_ui_target_camera(
    commands: &mut Commands,
    children: &Query<&Children>,
    root: Entity,
    target: Entity,
) {
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        commands.entity(entity).insert(UiTargetCamera(target));
        if let Ok(children) = children.get(entity) {
            stack.extend(children.iter());
        }
    }
}

pub fn window_graph_order_base(id: WindowId) -> isize {
    const SECONDARY_GRAPH_ORDER_BASE: isize = 1000;
    SECONDARY_GRAPH_ORDER_BASE + id.0 as isize * SECONDARY_GRAPH_ORDER_STRIDE
}

const SECONDARY_GRAPH_ORDER_STRIDE: isize = 50;

fn find_named_container_title(
    tree: &egui_tiles::Tree<crate::ui::tiles::Pane>,
    titles: &HashMap<egui_tiles::TileId, String>,
    tile_id: egui_tiles::TileId,
) -> Option<String> {
    if let Some(title) = titles
        .get(&tile_id)
        .and_then(|value| normalize_title(value))
    {
        return Some(title);
    }

    let tile = tree.tiles.get(tile_id)?;
    if let Tile::Container(container) = tile {
        match container {
            egui_tiles::Container::Tabs(tabs) => {
                for child in &tabs.children {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
            egui_tiles::Container::Linear(linear) => {
                for child in &linear.children {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
            egui_tiles::Container::Grid(grid) => {
                for child in grid.children() {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
        }
    }

    None
}

fn normalize_title(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn friendly_title_from_stem(stem: &str) -> Option<String> {
    let words: Vec<String> = stem
        .split(|c: char| !c.is_alphanumeric())
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            let mut chars = segment.chars();
            let mut word = String::new();
            if let Some(first) = chars.next() {
                word.extend(first.to_uppercase());
            }
            for ch in chars {
                word.extend(ch.to_lowercase());
            }
            word
        })
        .filter(|word| !word.is_empty())
        .collect();

    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

fn window_container_title(state: &WindowState) -> Option<String> {
    let root = state.tile_state.tree.root()?;
    find_named_container_title(
        &state.tile_state.tree,
        &state.tile_state.container_titles,
        root,
    )
}

pub fn compute_window_title(state: &WindowState) -> String {
    state
        .descriptor
        .title
        .clone()
        .or_else(|| window_container_title(state))
        .or_else(|| {
            state
                .descriptor
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
                .and_then(|s| friendly_title_from_stem(&s.to_string_lossy()))
        })
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "Panel".to_string())
}
