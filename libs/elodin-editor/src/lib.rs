use std::time::Duration;

use bevy::{
    asset::embedded_asset,
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    log::LogPlugin,
    pbr::{
        wireframe::{WireframeConfig, WireframePlugin},
        DirectionalLightShadowMap,
    },
    prelude::*,
    render::{
        camera::{Exposure, PhysicalCameraParameters},
        view::RenderLayers,
    },
    window::{PresentMode, WindowResolution, WindowTheme},
    winit::WinitSettings,
    DefaultPlugins,
};
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::prelude::OrbitConstraint;
use bevy_egui::EguiPlugin;
use bevy_mod_picking::prelude::*;
use big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};
use conduit::{
    well_known::{Viewport, WorldPos},
    ControlMsg, EntityId,
};
use plugins::navigation_gizmo::{spawn_gizmo, NavigationGizmoPlugin, RenderLayerAlloc};
use ui::{tiles, EntityPair, HoveredEntity};

use crate::plugins::editor_cam_touch;

pub mod chunks;
mod plugins;
pub mod ui;

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
        app
            //.insert_resource(AssetMetaCheck::Never)
            .add_plugins(plugins::WebAssetPlugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            window_theme: Some(WindowTheme::Dark),
                            title: "Elodin".into(),
                            present_mode: PresentMode::AutoNoVsync,
                            canvas: Some("#editor".to_string()),
                            resolution: self.window_resolution.clone(),
                            resize_constraints: WindowResizeConstraints {
                                min_width: 400.,
                                min_height: 400.,
                                ..Default::default()
                            },
                            composite_alpha_mode,
                            prevent_default_event_handling: true,
                            ..default()
                        }),
                        ..default()
                    })
                    .disable::<TransformPlugin>()
                    .disable::<DiagnosticsPlugin>()
                    .disable::<LogPlugin>()
                    .build(),
            )
            .insert_resource(WinitSettings {
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
            })
            .init_resource::<tiles::ViewportContainsPointer>()
            .add_plugins(bevy_framepace::FramepacePlugin)
            .add_plugins(DefaultPickingPlugins)
            .add_plugins(big_space::FloatingOriginPlugin::<i128>::new(16_000., 100.))
            .add_plugins(bevy_editor_cam::DefaultEditorCamPlugins)
            .add_plugins(EmbeddedAssetPlugin)
            .add_plugins(EguiPlugin)
            .add_plugins(bevy_infinite_grid::InfiniteGridPlugin)
            .add_plugins(NavigationGizmoPlugin)
            .add_plugins(crate::plugins::gizmos::GizmoPlugin)
            .add_plugins(ui::UiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin)
            .add_plugins(WireframePlugin)
            .add_plugins(editor_cam_touch::EditorCamTouchPlugin)
            .add_plugins(crate::ui::widgets::PlotPlugin)
            .add_systems(Startup, setup_floating_origin)
            .add_systems(Startup, setup_window_icon)
            .add_systems(Startup, spawn_clear_bg)
            .add_systems(Update, make_entities_selectable)
            .add_systems(PreUpdate, sync_pos)
            .add_systems(Update, sync_paused)
            .add_systems(PreUpdate, set_floating_origin.after(sync_pos))
            .insert_resource(WireframeConfig {
                global: false,
                default_color: Color::WHITE,
            })
            .insert_resource(ClearColor(Color::NONE));

        #[cfg(target_os = "macos")]
        app.add_systems(Startup, setup_titlebar);

        app.insert_resource(Msaa::default());

        // For adding features incompatible with wasm:
        embedded_asset!(app, "./assets/diffuse.ktx2");
        embedded_asset!(app, "./assets/specular.ktx2");
        if cfg!(not(target_arch = "wasm32")) {
            app.insert_resource(DirectionalLightShadowMap { size: 8192 });
        }
    }
}

#[derive(Component)]
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

fn spawn_clear_bg(mut commands: Commands) {
    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    let bg_color = Color::Srgba(Srgba::hex("#0C0C0C").unwrap());
    // root node
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Stretch,
                flex_direction: FlexDirection::Column,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent.spawn(NodeBundle {
                style: Style {
                    height: Val::Px(56.0),
                    ..default()
                },
                background_color: if cfg!(target_os = "macos") {
                    Color::NONE.into()
                } else {
                    bg_color.into()
                },
                ..default()
            });

            parent.spawn(NodeBundle {
                style: Style {
                    height: Val::Percent(100.0),
                    ..default()
                },
                background_color: bg_color.into(),
                ..default()
            });
        });
}

fn spawn_main_camera(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
    viewport: &Viewport,
) -> (Entity, Option<Entity>, Option<Entity>) {
    let mut main_camera_layers = RenderLayers::default();
    let mut grid_layers = RenderLayers::none();
    if let Some(grid_layer) = render_layer_alloc.alloc() {
        main_camera_layers = main_camera_layers.with(grid_layer);
        grid_layers = grid_layers.with(grid_layer);
    }
    let grid_visibility = if viewport.show_grid {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    let grid_id = commands
        .spawn((
            bevy_infinite_grid::InfiniteGridBundle {
                settings: bevy_infinite_grid::InfiniteGridSettings {
                    minor_line_color: Color::srgba(1.0, 1.0, 1.0, 0.02),
                    major_line_color: Color::srgba(1.0, 1.0, 1.0, 0.05),
                    z_axis_color: Color::Srgba(Srgba::hex("#264FFF").unwrap()),
                    x_axis_color: Color::Srgba(Srgba::hex("#EE3A43").unwrap()),
                    fadeout_distance: 50_000.0,
                    scale: 0.1,
                    ..Default::default()
                },
                visibility: grid_visibility,
                ..Default::default()
            },
            grid_layers,
        ))
        .id();
    let mut camera = commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(5.0, 5.0, 10.0))
                .looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                hdr: false,
                clear_color: ClearColorConfig::None,
                order: 1,
                ..Default::default()
            },
            tonemapping: Tonemapping::TonyMcMapface,
            exposure: Exposure::from_physical_camera(PhysicalCameraParameters {
                aperture_f_stops: 2.8,
                shutter_speed_s: 1.0 / 200.0,
                sensitivity_iso: 400.0,
                // full frame sensor height
                sensor_height: 24.0 / 1000.0,
            }),

            ..default()
        },
        main_camera_layers,
        MainCamera,
        GridCell::<i128>::default(),
        EditorCam {
            orbit_constraint: OrbitConstraint::Fixed {
                up: Vec3::Y,
                can_pass_tdc: false,
            },
            last_anchor_depth: 2.0,
            ..Default::default()
        },
        conduit::bevy::Persistent,
        GridHandle { grid: grid_id },
    ));

    camera.insert(BloomSettings { ..default() });
    camera.insert(EnvironmentMapLight {
        diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
        specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
        intensity: 2000.0,
    });

    let camera = camera.id();

    let (nav_gizmo, nav_gizmo_camera) =
        spawn_gizmo(camera, commands, meshes, materials, render_layer_alloc);
    (camera, nav_gizmo, nav_gizmo_camera)
}

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
fn setup_titlebar(
    windows: Query<(Entity, &bevy::window::PrimaryWindow)>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    use cocoa::{
        appkit::{
            NSColor, NSToolbar, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility,
            NSWindowToolbarStyle,
        },
        base::{id, nil, BOOL},
    };
    use objc::{
        class,
        declare::ClassDecl,
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    for (id, _) in &windows {
        let window = winit_windows.get_window(id).unwrap();
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
            panic!("null window");
        }
        unsafe {
            // define an objective class for NSToolbar's delegate,
            // because NSToolbar will complain if we do not
            let toolbar_delegate_class = {
                let superclass = class!(NSObject);
                let mut decl = ClassDecl::new("ToolbarDel", superclass).unwrap();
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
                panic!("null toolbar");
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

fn make_entities_selectable(
    mut commands: Commands,
    entities: Query<(&EntityId, Entity), Without<EntityConfigured>>,
) {
    for (entity_id, entity) in entities.iter() {
        let entity_id = entity_id.to_owned();
        commands.entity(entity).insert((
            EntityConfigured,
            On::<Pointer<Out>>::run(
                move |_event: Listener<Pointer<Out>>, mut hovered_entity: ResMut<HoveredEntity>| {
                    hovered_entity.0 = None
                },
            ),
            On::<Pointer<Over>>::run(
                move |_event: Listener<Pointer<Over>>,
                      mut hovered_entity: ResMut<HoveredEntity>,
                      viewport_contains_pointer: Res<tiles::ViewportContainsPointer>| {
                    hovered_entity.0 = if viewport_contains_pointer.0 {
                        Some(EntityPair {
                            bevy: entity,
                            conduit: entity_id,
                        })
                    }
                    else {
                        None
                    };
                },
            ),
        ));
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

pub fn sync_paused(paused: Res<ui::Paused>, mut event: EventWriter<ControlMsg>) {
    if paused.is_changed() {
        event.send(ControlMsg::SetPlaying(!paused.0));
    }
}
