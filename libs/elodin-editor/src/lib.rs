#![recursion_limit = "256"]

use std::{collections::HashMap, ops::Range, sync::Arc, time::Duration};

use crate::plugins::editor_cam_touch;
use bevy::{
    DefaultPlugins,
    asset::{UnapprovedPathMode, embedded_asset},
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    log::LogPlugin,
    math::{DQuat, DVec3},
    pbr::{
        DirectionalLightShadowMap,
        wireframe::{WireframeConfig, WireframePlugin},
    },
    prelude::*,
    window::{PresentMode, WindowResolution, WindowTheme},
    winit::WinitSettings,
};
use bevy_egui::{EguiContextSettings, EguiPlugin};
use big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};
use convert_case::{Case, Casing};
use impeller2::types::{ComponentId, OwnedPacket};
use impeller2::types::{Msg, Timestamp};
use impeller2_bevy::{
    ComponentMetadataRegistry, ComponentPathRegistry, ComponentSchemaRegistry, ComponentValueMap,
    CurrentStreamId, EntityMap, PacketHandlerInput, PacketHandlers, PacketTx,
};
use impeller2_wkt::{CurrentTimestamp, NewConnection, SetStreamState, WorldPos};
use impeller2_wkt::{EarliestTimestamp, LastUpdated};
use nox::Tensor;
use object_3d::create_object_3d_entity;
use plugins::navigation_gizmo::{NavigationGizmoPlugin, RenderLayerAlloc};
use ui::{
    SelectedObject,
    colors::{ColorExt, get_scheme},
    tiles::{self, TileState},
    utils::FriendlyEpoch,
    widgets::{
        inspector::viewport::set_viewport_pos,
        plot::{CollectedGraphData, gpu::LineHandle},
    },
};

pub mod object_3d;
mod offset_parse;
mod plugins;
pub mod ui;

#[cfg(not(target_family = "wasm"))]
pub mod run;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct EmbeddedAssetPlugin;

impl Plugin for EmbeddedAssetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "assets/logo.png");
        embedded_asset!(app, "assets/icons/play.png");
        embedded_asset!(app, "assets/icons/pause.png");
        embedded_asset!(app, "assets/icons/scrub.png");
        embedded_asset!(app, "assets/icons/jump_to_end.png");
        embedded_asset!(app, "assets/icons/jump_to_start.png");
        embedded_asset!(app, "assets/icons/frame_forward.png");
        embedded_asset!(app, "assets/icons/frame_back.png");
        embedded_asset!(app, "assets/icons/search.png");
        embedded_asset!(app, "assets/icons/add.png");
        embedded_asset!(app, "assets/icons/subtract.png");
        embedded_asset!(app, "assets/icons/close.png");
        embedded_asset!(app, "assets/icons/chart.png");
        embedded_asset!(app, "assets/icons/left-side-bar.png");
        embedded_asset!(app, "assets/icons/right-side-bar.png");
        embedded_asset!(app, "assets/icons/fullscreen.png");
        embedded_asset!(app, "assets/icons/exit-fullscreen.png");
        embedded_asset!(app, "assets/icons/setting.png");
        embedded_asset!(app, "assets/icons/lightning.png");
        embedded_asset!(app, "assets/icons/link.png");
        embedded_asset!(app, "assets/icons/loop.png");
        embedded_asset!(app, "assets/icons/tile_3d_viewer.png");
        embedded_asset!(app, "assets/icons/tile_graph.png");
        embedded_asset!(app, "assets/icons/ip-addr.png");
        embedded_asset!(app, "assets/icons/folder.png");
        embedded_asset!(app, "assets/logo-full.png");
        embedded_asset!(app, "assets/icons/chevron_right.png");
        embedded_asset!(app, "assets/icons/vertical-chevrons.png");
    }
}

#[derive(Default)]
pub struct EditorPlugin {
    window_resolution: WindowResolution,
}

impl EditorPlugin {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            window_resolution: WindowResolution::new(width, height),
        }
    }
}

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        let composite_alpha_mode = if cfg!(target_os = "macos") {
            bevy::window::CompositeAlphaMode::PostMultiplied
        } else {
            bevy::window::CompositeAlphaMode::Opaque
        };
        let winit_settings = if cfg!(target_os = "macos") {
            WinitSettings {
                focused_mode: bevy::winit::UpdateMode::Reactive {
                    wait: Duration::from_millis(16),
                    react_to_device_events: true,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
                unfocused_mode: bevy::winit::UpdateMode::Reactive {
                    wait: Duration::from_millis(32),
                    react_to_device_events: false,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
            }
        } else {
            WinitSettings::game()
        };
        app
            //.insert_resource(AssetMetaCheck::Never)
            .add_plugins(plugins::WebAssetPlugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            window_theme: Some(WindowTheme::Dark),
                            title: "Elodin".into(),
                            present_mode: PresentMode::AutoVsync,
                            canvas: Some("#editor".to_string()),
                            resolution: self.window_resolution.clone(),
                            resize_constraints: WindowResizeConstraints {
                                min_width: 400.,
                                min_height: 400.,
                                ..Default::default()
                            },
                            composite_alpha_mode,
                            prevent_default_event_handling: true,
                            decorations: cfg!(target_os = "macos"),
                            visible: cfg!(target_os = "linux"),
                            ..default()
                        }),
                        ..default()
                    })
                    .set(AssetPlugin {
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        ..default()
                    })
                    .disable::<TransformPlugin>()
                    .disable::<DiagnosticsPlugin>()
                    .disable::<LogPlugin>()
                    .build(),
            )
            .insert_resource(winit_settings)
            .init_resource::<tiles::ViewportContainsPointer>()
            .add_plugins(bevy_framepace::FramepacePlugin)
            //.add_plugins(DefaultPickingPlugins)
            .add_plugins(big_space::FloatingOriginPlugin::<i128>::new(16_000., 100.))
            .add_plugins(bevy_editor_cam::DefaultEditorCamPlugins)
            .add_plugins(EmbeddedAssetPlugin)
            .add_plugins(EguiPlugin {
                enable_multipass_for_primary_context: false,
            })
            .add_plugins(bevy_infinite_grid::InfiniteGridPlugin)
            .add_plugins(NavigationGizmoPlugin)
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            //.add_plugins(crate::plugins::gizmos::GizmoPlugin)
            .add_plugins(ui::UiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_plugins(WireframePlugin::default())
            .add_plugins(editor_cam_touch::EditorCamTouchPlugin)
            .add_plugins(crate::ui::widgets::PlotPlugin)
            .add_plugins(crate::plugins::LogicalKeyPlugin)
            .add_systems(Startup, setup_floating_origin)
            .add_systems(Startup, setup_window_icon)
            //.add_systems(Startup, spawn_clear_bg)
            .add_systems(Startup, setup_clear_state)
            .add_systems(Update, setup_egui_context)
            //.add_systems(Update, make_entities_selectable)
            .add_systems(PreUpdate, setup_cell)
            .add_systems(PreUpdate, sync_res::<CurrentTimestamp>)
            .add_systems(PreUpdate, sync_res::<impeller2_wkt::SimulationTimeStep>)
            .add_systems(PreUpdate, sync_pos)
            .add_systems(PreUpdate, sync_object_3d)
            .add_systems(Update, sync_paused)
            .add_systems(PreUpdate, set_selected_range)
            .add_systems(PreUpdate, set_floating_origin.after(sync_pos))
            .add_systems(PreUpdate, update_eql_context)
            .add_systems(PreUpdate, set_eql_context_range.after(update_eql_context))
            .add_systems(Startup, spawn_ui_cam)
            .add_systems(PostUpdate, ui::video_stream::set_visibility)
            .add_systems(PostUpdate, set_clear_color)
            .add_systems(Update, set_viewport_pos)
            //.add_systems(Update, clamp_current_time)
            .insert_resource(WireframeConfig {
                global: false,
                default_color: Color::WHITE,
            })
            .insert_resource(ClearColor(get_scheme().bg_secondary.into_bevy()))
            .insert_resource(TimeRangeBehavior::default())
            .insert_resource(SelectedTimeRange(Timestamp(i64::MIN)..Timestamp(i64::MAX)))
            .init_resource::<EqlContext>()
            .init_resource::<SyncedObject3d>()
            .add_plugins(object_3d::Object3DPlugin);
        if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
            app.add_systems(Update, handle_drag_resize);
        }

        #[cfg(not(target_family = "wasm"))]
        app.add_plugins(crate::ui::startup_window::StartupPlugin);

        #[cfg(target_os = "macos")]
        app.add_systems(Update, setup_titlebar);

        // For adding features incompatible with wasm:
        embedded_asset!(app, "./assets/diffuse.ktx2");
        embedded_asset!(app, "./assets/specular.ktx2");
        if cfg!(not(target_arch = "wasm32")) {
            app.insert_resource(DirectionalLightShadowMap { size: 8192 });
        }
    }
}

fn setup_egui_context(mut contexts: Query<&mut EguiContextSettings>) {
    for mut context in &mut contexts {
        context.capture_pointer_input = false;
    }
}

#[derive(Component, Clone)]
pub struct MainCamera;

#[derive(Component)]
pub struct GridHandle {
    pub grid: Entity,
}

fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::<i128>::default(),
        Transform::IDENTITY,
    ));
}

fn spawn_ui_cam(mut commands: Commands) {
    commands.spawn((Camera2d, IsDefaultUiCamera));
}

fn set_clear_color(mut clear_color: ResMut<ClearColor>) {
    clear_color.0 = get_scheme().bg_secondary.into_bevy();
}

// NOTE(sphw): enabling this causes weird flickering issues when spawning too many 2d cameras
// This issue (https://github.com/bevyengine/bevy/issues/18897) looks to be the same thing
//
// fn spawn_clear_bg(mut commands: Commands) {
//     commands.spawn((Camera2d, IsDefaultUiCamera));
//     let bg_color = Color::Srgba(Srgba::hex("#0C0C0C").unwrap());
//     // root node
//     commands
//         .spawn(Node {
//             width: Val::Percent(100.0),
//             height: Val::Percent(100.0),
//             justify_content: JustifyContent::Stretch,
//             flex_direction: FlexDirection::Column,
//             ..default()
//         })
//         .with_children(|parent| {
//             parent
//                 .spawn(Node {
//                     height: Val::Px(56.0),
//                     ..default()
//                 })
//                 .insert(BackgroundColor(if cfg!(target_os = "macos") {
//                     Color::NONE
//                 } else {
//                     bg_color
//                 }));

//             parent
//                 .spawn(Node {
//                     height: Val::Percent(100.0),
//                     ..default()
//                 })
//                 .insert(BackgroundColor(bg_color));
//         });
// }

#[allow(clippy::type_complexity)]
fn set_floating_origin(
    query: Query<(&Transform, &GridCell<i128>), (With<MainCamera>, Without<FloatingOrigin>)>,
    mut floating_origin: Query<(&mut Transform, &mut GridCell<i128>), With<FloatingOrigin>>,
) {
    let Some((transform, grid_cell)) = query.iter().next() else {
        return;
    };
    for (mut origin, mut cell) in floating_origin.iter_mut() {
        *origin = *transform;
        *cell = *grid_cell;
    }
}

#[cfg(target_os = "macos")]
#[derive(Component)]
struct SetupTitlebar;

#[cfg(target_os = "macos")]
fn setup_titlebar(
    windows: Query<Entity, Without<SetupTitlebar>>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut commands: Commands,
) {
    use cocoa::{
        appkit::{
            NSColor, NSToolbar, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility,
            NSWindowToolbarStyle,
        },
        base::{BOOL, id, nil},
    };
    use objc::{
        class,
        declare::ClassDecl,
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    for id in &windows {
        let Some(window) = winit_windows.get_window(id) else {
            continue;
        };
        window.set_blur(true);
        use raw_window_handle::HasRawWindowHandle;
        let handle = window.raw_window_handle();
        let raw_window_handle::RawWindowHandle::AppKit(handle) = handle else {
            error!("non AppKit window on macOS");
            continue;
        };
        let window = handle.ns_window;
        let window: cocoa::base::id = unsafe { std::mem::transmute(window) };
        if window.is_null() {
            continue;
        }
        unsafe {
            // define an objective class for NSToolbar's delegate,
            // because NSToolbar will complain if we do not
            let toolbar_delegate_class = {
                let superclass = class!(NSObject);
                let Some(mut decl) = ClassDecl::new(&format!("ToolbarDel{}", id), superclass)
                else {
                    continue;
                };
                extern "C" fn toolbar(_: &Object, _: Sel, _: id, _: id, _: BOOL) -> *const Object {
                    nil
                }
                extern "C" fn allowed(_: &Object, _: Sel, _: id) -> id {
                    nil
                }
                decl.add_method(
                    objc::sel!(toolbar:itemForItemIdentifier:willBeInsertedIntoToolbar:),
                    toolbar as extern "C" fn(&Object, Sel, id, id, bool) -> *const Object,
                );
                decl.add_method(
                    objc::sel!(toolbarAllowedItemIdentifiers:),
                    allowed as extern "C" fn(&Object, Sel, id) -> id,
                );
                decl.add_method(
                    objc::sel!(toolbarDefaultItemIdentifiers:),
                    allowed as extern "C" fn(&Object, Sel, id) -> id,
                );
                decl.register()
            };
            let del: id = msg_send![toolbar_delegate_class, alloc];
            let del = msg_send![del, init];
            let toolbar = NSToolbar::alloc(nil);
            let toolbar = toolbar.init_();
            toolbar.setDelegate_(del);
            if toolbar.is_null() {
                continue;
            }
            window.setTitlebarAppearsTransparent_(true);
            let color = NSColor::clearColor(nil);
            window.setBackgroundColor_(NSColor::colorWithRed_green_blue_alpha_(
                color,
                (0x0C / 0xFF) as f64,
                (0x0C / 0xFF) as f64,
                (0x0C / 0xFF) as f64,
                0.7,
            ));
            window.setStyleMask_(
                NSWindowStyleMask::NSFullSizeContentViewWindowMask
                    | NSWindowStyleMask::NSResizableWindowMask
                    | NSWindowStyleMask::NSTitledWindowMask
                    | NSWindowStyleMask::NSClosableWindowMask
                    | NSWindowStyleMask::NSMiniaturizableWindowMask
                    | NSWindowStyleMask::NSUnifiedTitleAndToolbarWindowMask,
            );
            window.setToolbarStyle_(NSWindowToolbarStyle::NSWindowToolbarStyleUnified);
            window.setTitleVisibility_(NSWindowTitleVisibility::NSWindowTitleHidden);
            window.setToolbar_(toolbar);
            commands.entity(id).insert(SetupTitlebar);
        }
    }
}

fn handle_drag_resize(
    windows: Query<(Entity, &Window, &bevy::window::PrimaryWindow)>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut just_set_cursor: Local<bool>,
) {
    for (id, window, _) in &windows {
        let Some(cursor_pos) = window.physical_cursor_position() else {
            continue;
        };
        let size = window.physical_size().as_vec2();
        let window = winit_windows.get_window(id).unwrap();
        const RESIZE_ZONE: f32 = 5.0;
        let resize_west = cursor_pos.x < RESIZE_ZONE;
        let resize_east = cursor_pos.x > size.x - RESIZE_ZONE;
        let resize_north = cursor_pos.y < RESIZE_ZONE;
        let resize_south = cursor_pos.y > size.y - RESIZE_ZONE;
        if cursor_pos.y < 45.0
            && !resize_north
            && !resize_east
            && !resize_west
            && mouse_buttons.pressed(MouseButton::Left)
        {
            let _ = window.drag_window();
        }
        let resize_dir = match (resize_west, resize_east, resize_north, resize_south) {
            (true, _, true, _) => Some(winit::window::ResizeDirection::NorthWest),
            (_, true, true, _) => Some(winit::window::ResizeDirection::NorthEast),

            (true, _, _, true) => Some(winit::window::ResizeDirection::SouthWest),
            (_, true, _, true) => Some(winit::window::ResizeDirection::SouthEast),
            (true, _, _, _) => Some(winit::window::ResizeDirection::West),
            (_, true, _, _) => Some(winit::window::ResizeDirection::East),
            (_, _, true, _) => Some(winit::window::ResizeDirection::North),
            (_, _, _, true) => Some(winit::window::ResizeDirection::South),
            _ => None,
        };
        if let Some(resize_dir) = resize_dir {
            if mouse_buttons.pressed(MouseButton::Left) {
                let _ = window.drag_resize_window(resize_dir);
            }
            window.set_cursor(winit::window::CursorIcon::from(resize_dir));
            *just_set_cursor = true
        } else if *just_set_cursor {
            *just_set_cursor = false;
            window.set_cursor(winit::window::CursorIcon::Default);
        }
    }
}

fn setup_window_icon(
    _windows: Query<(Entity, &bevy::window::PrimaryWindow)>,
    // this is load bearing, because it ensures that there is at
    // least one window spawned
    _winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    #[cfg(target_os = "macos")]
    set_icon_mac();

    if !_windows.is_empty() {
        #[cfg(target_os = "windows")]
        set_icon_windows();
    }
}

#[cfg(target_os = "windows")]
fn set_icon_windows() {
    use winapi::um::winuser;
    let png_bytes = include_bytes!("../assets/win-512x512@2x.png");
    let unscaled_image =
        image::ImageReader::with_format(std::io::Cursor::new(png_bytes), image::ImageFormat::Png)
            .decode()
            .unwrap();
    let window_handle = unsafe { winuser::GetActiveWindow() };
    if window_handle.is_null() {
        return;
    }

    unsafe {
        let margins = winapi::um::uxtheme::MARGINS {
            cxLeftWidth: 0,
            cxRightWidth: 0,
            cyTopHeight: -40,
            cyBottomHeight: 40,
        };
        winapi::um::dwmapi::DwmExtendFrameIntoClientArea(window_handle, &margins);
        let mut rect: winapi::shared::windef::RECT = std::mem::zeroed();
        winapi::um::winuser::GetWindowRect(window_handle, &mut rect);
    }

    fn resize_image(image: &image::DynamicImage, size: usize) -> Option<Vec<u8>> {
        let image_scaled =
            image::imageops::resize(image, size as _, size as _, image::imageops::Lanczos3);
        let mut image_scaled_bytes: Vec<u8> = Vec::new();
        if image_scaled
            .write_to(
                &mut std::io::Cursor::new(&mut image_scaled_bytes),
                image::ImageFormat::Png,
            )
            .is_ok()
        {
            Some(image_scaled_bytes)
        } else {
            None
        }
    }

    let icon_size_big = unsafe { winuser::GetSystemMetrics(winuser::SM_CXICON) };
    let Some(big_image_bytes) = resize_image(&unscaled_image, icon_size_big as _) else {
        return;
    };

    unsafe {
        let icon = winuser::CreateIconFromResourceEx(
            big_image_bytes.as_ptr() as *mut _,
            big_image_bytes.len() as u32,
            1,          // Means this is an icon, not a cursor.
            0x00030000, // Version number of the HICON
            icon_size_big,
            icon_size_big,
            winuser::LR_DEFAULTCOLOR,
        );
        winuser::SendMessageW(
            window_handle,
            winuser::WM_SETICON,
            winuser::ICON_BIG as usize,
            icon as isize,
        );
    }

    let icon_size_small = unsafe { winuser::GetSystemMetrics(winuser::SM_CXSMICON) };
    let Some(small_image_bytes) = resize_image(&unscaled_image, icon_size_small as _) else {
        return;
    };

    unsafe {
        let icon = winuser::CreateIconFromResourceEx(
            small_image_bytes.as_ptr() as *mut _,
            small_image_bytes.len() as u32,
            1,          // Means this is an icon, not a cursor.
            0x00030000, // Version number of the HICON
            icon_size_small,
            icon_size_small,
            winuser::LR_DEFAULTCOLOR,
        );
        winuser::SendMessageW(
            window_handle,
            winuser::WM_SETICON,
            winuser::ICON_SMALL as usize,
            icon as isize,
        );
    }
}

/// source: https://github.com/emilk/egui/blob/15370bbea0b468cf719a75cc6d1e39eb00c420d8/crates/eframe/src/native/app_icon.rs#L199C1-L268C2
#[cfg(target_os = "macos")]
fn set_icon_mac() {
    use cocoa::{
        appkit::{NSApp, NSApplication, NSImage},
        base::nil,
        foundation::NSData,
    };

    let png_bytes = include_bytes!("../assets/512x512@2x.png");

    // SAFETY: Accessing raw data from icon in a read-only manner. Icon data is static!
    unsafe {
        let app = NSApp();
        if app.is_null() {
            panic!("NSApp was null when setting app icon")
        }

        let data = NSData::dataWithBytes_length_(
            nil,
            png_bytes.as_ptr().cast::<std::ffi::c_void>(),
            png_bytes.len() as u64,
        );

        let app_icon = NSImage::initWithData_(NSImage::alloc(nil), data);

        app.setApplicationIconImage_(app_icon);
    }
}

#[derive(Component)]
pub struct EntityConfigured;

pub fn setup_cell(
    query: Query<Entity, (With<WorldPos>, Without<GridCell<i128>>)>,
    mut cmds: Commands,
) {
    for e in query.iter() {
        cmds.entity(e)
            .insert((Transform::default(), GridCell::<i128>::default()));
    }
}

pub trait WorldPosExt {
    fn bevy_pos(&self) -> DVec3;
    fn bevy_att(&self) -> DQuat;
}

impl WorldPosExt for WorldPos {
    fn bevy_pos(&self) -> DVec3 {
        let [x, y, z] = self.pos.parts().map(Tensor::into_buf);
        DVec3::new(x, z, -y)
    }

    fn bevy_att(&self) -> DQuat {
        let [i, j, k, w] = self.att.parts().map(Tensor::into_buf);
        let x = i;
        let y = k;
        let z = -j;
        DQuat::from_xyzw(x, y, z, w)
    }
}

pub fn sync_paused(
    paused: Res<ui::Paused>,
    event: ResMut<PacketTx>,
    stream_id: Res<CurrentStreamId>,
) {
    if paused.is_changed() {
        event.send_msg(SetStreamState {
            id: stream_id.0,
            playing: Some(!paused.0),
            timestamp: None,
            time_step: None,
            frequency: None,
        })
    }
}

pub fn sync_pos(
    mut query: Query<(&mut Transform, &mut GridCell<i128>, &WorldPos)>,
    floating_origin: Res<FloatingOriginSettings>,
) {
    query
        .iter_mut()
        .for_each(|(mut transform, mut grid_cell, world_pos)| {
            // Converts from Z-up to Y-up
            let pos = world_pos.bevy_pos();
            let att = world_pos.bevy_att();
            let (new_grid_cell, translation) = floating_origin.translation_to_grid(pos);
            *grid_cell = new_grid_cell;
            *transform = bevy::prelude::Transform {
                translation,
                rotation: att.as_quat(),
                ..Default::default()
            }
        });
}

pub trait BevyExt {
    type Bevy;
    fn into_bevy(self) -> Self::Bevy;
}
impl BevyExt for impeller2_wkt::Mesh {
    type Bevy = Mesh;

    fn into_bevy(self) -> Self::Bevy {
        match self {
            impeller2_wkt::Mesh::Sphere { radius } => {
                bevy::math::primitives::Sphere { radius }.into()
            }
            impeller2_wkt::Mesh::Box { x, y, z } => {
                bevy::math::primitives::Cuboid::new(x, y, z).into()
            }
            impeller2_wkt::Mesh::Cylinder { radius, height } => {
                bevy::math::primitives::Cylinder::new(radius, height).into()
            }
        }
    }
}

impl BevyExt for impeller2_wkt::Material {
    type Bevy = StandardMaterial;

    fn into_bevy(self) -> Self::Bevy {
        bevy::prelude::StandardMaterial {
            base_color: Color::srgb(self.base_color.r, self.base_color.g, self.base_color.b),
            ..Default::default()
        }
    }
}

#[derive(Default, Resource)]
struct SyncedObject3d(HashMap<Entity, Entity>);

#[allow(clippy::too_many_arguments)]
fn sync_object_3d(
    query: Query<(Entity, &ComponentId), With<impeller2_wkt::WorldPos>>,
    meshes: Query<&impeller2_wkt::Mesh>,
    materials: Query<&impeller2_wkt::Material>,
    glbs: Query<&impeller2_wkt::Glb>,
    mut synced_object_3d: ResMut<SyncedObject3d>,
    entity_map: ResMut<EntityMap>,
    path_reg: Res<ComponentPathRegistry>,
    ctx: Res<EqlContext>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    assets: Res<AssetServer>,
) {
    for (entity, id) in &query {
        if synced_object_3d.0.contains_key(&entity) {
            continue;
        }
        let Some(path) = dbg!(path_reg.get(id)) else {
            continue;
        };
        let parent = path.path.first().unwrap();

        let glb = entity_map
            .get(&ComponentId::new(&format!(
                "{}.asset_handle_glb",
                parent.name
            )))
            .and_then(|e| glbs.get(*e).ok());
        let mesh = entity_map
            .get(&ComponentId::new(&format!(
                "{}.asset_handle_mesh",
                parent.name
            )))
            .and_then(|e| meshes.get(*e).ok());
        let material = entity_map
            .get(&ComponentId::new(&format!(
                "{}.asset_handle_material",
                parent.name
            )))
            .and_then(|e| materials.get(*e).ok());

        let mesh_source = match dbg!((glb, mesh, material)) {
            (Some(glb), _, _) => impeller2_wkt::Object3D::Glb(glb.0.clone()),
            (_, Some(mesh), Some(mat)) => impeller2_wkt::Object3D::Mesh {
                mesh: mesh.clone(),
                material: mat.clone(),
            },
            _ => continue,
        };

        let eql = format!("{}.world_pos", parent.name.to_case(Case::Snake));
        let Ok(expr) = dbg!(ctx.0.parse_str(&eql)) else {
            continue;
        };

        let object_entity = create_object_3d_entity(
            &mut commands,
            eql,
            expr,
            Some(mesh_source),
            &mut material_assets,
            &mut mesh_assets,
            &assets,
        );
        synced_object_3d.0.insert(entity, object_entity);
    }
}

fn sync_res<R: Component + Resource + Clone>(q: Query<&R>, mut res: ResMut<R>) {
    for t in &q {
        *res = t.clone();
    }
}

pub fn setup_clear_state(mut packet_handlers: ResMut<PacketHandlers>, mut commands: Commands) {
    let sys = commands.register_system(clear_state_new_connection);
    packet_handlers.0.push(sys);
}

#[allow(clippy::too_many_arguments)]
fn clear_state_new_connection(
    PacketHandlerInput { packet, .. }: PacketHandlerInput,
    mut entity_map: ResMut<EntityMap>,
    mut ui_state: ResMut<TileState>,
    mut selected_object: ResMut<SelectedObject>,
    mut render_layer_alloc: ResMut<RenderLayerAlloc>,
    mut value_map: Query<&mut ComponentValueMap>,
    mut graph_data: ResMut<CollectedGraphData>,
    lines: Query<Entity, With<LineHandle>>,
    mut synced_glbs: ResMut<SyncedObject3d>,
    mut eql_context: ResMut<EqlContext>,
    mut commands: Commands,
) {
    match packet {
        OwnedPacket::Msg(m) if m.id == NewConnection::ID => {}
        _ => return,
    }
    eql_context.0.component_parts.clear();
    entity_map.0.retain(|_, entity| {
        if let Ok(mut entity_commands) = commands.get_entity(*entity) {
            entity_commands.despawn();
        }
        false
    });
    value_map.iter_mut().for_each(|mut map| {
        map.0.clear();
    });
    for line in lines.iter() {
        if let Ok(mut entity_commands) = commands.get_entity(line) {
            entity_commands.despawn();
        }
    }
    synced_glbs.0.clear();
    ui_state.clear(&mut commands, &mut selected_object);
    *graph_data = CollectedGraphData::default();
    render_layer_alloc.free_all();
}

#[derive(Resource, Clone)]
pub struct SelectedTimeRange(Range<Timestamp>);
impl Default for SelectedTimeRange {
    fn default() -> Self {
        Self(Timestamp(i64::MIN)..Timestamp(i64::MAX))
    }
}

pub fn set_selected_range(
    mut selected_range: ResMut<SelectedTimeRange>,
    earliest: Res<EarliestTimestamp>,
    latest: Res<LastUpdated>,
    behavior: Res<TimeRangeBehavior>,
) {
    selected_range.0 = behavior.calculate_selected_range(earliest.0, latest.0);
}

#[derive(Resource, PartialEq, Eq, Clone, Copy)]
pub struct TimeRangeBehavior {
    start: Offset,
    end: Offset,
}

#[derive(Resource, PartialEq, Eq, Clone, Copy, Debug)]
enum Offset {
    Earliest(Duration),
    Latest(Duration),
    Fixed(Timestamp),
}

impl std::fmt::Display for Offset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Offset::Earliest(duration) => {
                let d = hifitime::Duration::from(*duration)
                    .to_string()
                    .to_uppercase();
                write!(f, "+{d}")
            }
            Offset::Latest(duration) => {
                let d = hifitime::Duration::from(*duration)
                    .to_string()
                    .to_uppercase();
                write!(f, "-{d}")
            }
            Offset::Fixed(timestamp) => {
                let timestamp = FriendlyEpoch(hifitime::Epoch::from(*timestamp));
                write!(f, "{timestamp}")
            }
        }
    }
}

impl std::fmt::Display for TimeRangeBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start, self.end) {
            (Offset::Earliest(start), Offset::Latest(end)) if end.is_zero() && start.is_zero() => {
                write!(f, "FULL RANGE")
            }
            (Offset::Latest(start), Offset::Latest(end)) if end.is_zero() => {
                let start = hifitime::Duration::from(start).to_string().to_uppercase();
                write!(f, "LAST {start}")
            }
            (start, end) => {
                write!(f, "{start} ↔ {end}")
            }
        }
    }
}

impl Default for TimeRangeBehavior {
    fn default() -> Self {
        Self::FULL
    }
}

impl TimeRangeBehavior {
    const FULL: Self = TimeRangeBehavior {
        start: Offset::Earliest(Duration::ZERO),
        end: Offset::Latest(Duration::ZERO),
    };
    const LAST_30S: Self = Self::last(Duration::from_secs(30));
    const LAST_1M: Self = Self::last(Duration::from_secs(60));
    const LAST_5M: Self = Self::last(Duration::from_secs(60 * 5));
    const LAST_15M: Self = Self::last(Duration::from_secs(60 * 15));
    const LAST_30M: Self = Self::last(Duration::from_secs(60 * 30));
    const LAST_1H: Self = Self::last(Duration::from_secs(60 * 60));
    const LAST_6H: Self = Self::last(Duration::from_secs(60 * 60 * 6));
    const LAST_12H: Self = Self::last(Duration::from_secs(60 * 60 * 12));
    const LAST_24H: Self = Self::last(Duration::from_secs(60 * 60 * 24));

    pub const fn last(duration: Duration) -> Self {
        TimeRangeBehavior {
            start: Offset::Latest(duration),
            end: Offset::Latest(Duration::ZERO),
        }
    }

    fn calculate_selected_range(&self, earliest: Timestamp, latest: Timestamp) -> Range<Timestamp> {
        let start = match self.start {
            Offset::Earliest(duration) => earliest + duration,
            Offset::Latest(duration) => latest - duration,
            Offset::Fixed(timestamp) => timestamp,
        };
        let end = match self.end {
            Offset::Earliest(duration) => earliest + duration,
            Offset::Latest(duration) => latest - duration,
            Offset::Fixed(timestamp) => timestamp,
        };

        clamp_range(earliest..latest, start..end)
    }

    pub fn is_subset(&self, earliest: Timestamp, latest: Timestamp) -> bool {
        let start = match self.start {
            Offset::Earliest(duration) => earliest + duration,
            Offset::Latest(duration) => latest - duration,
            Offset::Fixed(timestamp) => timestamp,
        };
        let end = match self.end {
            Offset::Earliest(duration) => earliest + duration,
            Offset::Latest(duration) => latest - duration,
            Offset::Fixed(timestamp) => timestamp,
        };
        let full_range = earliest..=latest;
        full_range.contains(&start) && full_range.contains(&end)
    }
}

fn clamp_range(total_range: Range<Timestamp>, b: Range<Timestamp>) -> Range<Timestamp> {
    let start = total_range.start.max(b.start);
    let end = total_range.end.min(b.end);
    start..end
}

pub fn clamp_current_time(
    range: Res<SelectedTimeRange>,
    current_timestamp: ResMut<CurrentTimestamp>,
    packet_tx: Res<PacketTx>,
    current_stream_id: Res<CurrentStreamId>,
) {
    if range.0.start > range.0.end {
        return;
    }
    let new_timestamp = current_timestamp.0.clamp(range.0.start, range.0.end);
    if new_timestamp != current_timestamp.0 {
        packet_tx.send_msg(SetStreamState::rewind(**current_stream_id, new_timestamp))
    }
}

#[derive(Resource)]
pub struct EqlContext(pub eql::Context);

impl Default for EqlContext {
    fn default() -> Self {
        Self(eql::Context::new(
            HashMap::new(),
            Timestamp(i64::MIN),
            Timestamp(i64::MAX),
        ))
    }
}

pub fn update_eql_context(
    component_metadata_registry: Res<ComponentMetadataRegistry>,
    component_schema_registry: Res<ComponentSchemaRegistry>,
    path_reg: Res<ComponentPathRegistry>,
    mut eql_context: ResMut<EqlContext>,
) {
    eql_context.0 = eql::Context::from_leaves(
        path_reg.0.iter().filter_map(|(id, path)| {
            let schema = component_schema_registry.0.get(id)?;
            let metadata = component_metadata_registry.0.get(id)?;
            let mut component = eql::Component::new(metadata.name.clone(), path.id, schema.clone());
            if !metadata.element_names().is_empty() {
                component.element_names = metadata
                    .element_names()
                    .split(",")
                    .map(str::to_string)
                    .collect();
            }
            Some(Arc::new(component))
        }),
        Timestamp(i64::MIN),
        Timestamp(i64::MAX),
    );
}

pub fn set_eql_context_range(time_range: Res<SelectedTimeRange>, mut eql: ResMut<EqlContext>) {
    eql.0.earliest_timestamp = time_range.0.start;
    eql.0.last_timestamp = time_range.0.end;
}

pub fn dirs() -> directories::ProjectDirs {
    directories::ProjectDirs::from("systems", "elodin", "editor").unwrap()
}
