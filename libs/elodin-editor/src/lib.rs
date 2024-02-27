use std::time::Duration;

use bevy::{
    asset::embedded_asset,
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    log::LogPlugin,
    pbr::{
        DirectionalLightShadowMap, ScreenSpaceAmbientOcclusionQualityLevel,
        ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    render::view::RenderLayers,
    window::{PresentMode, WindowTheme},
    winit::WinitSettings,
    DefaultPlugins,
};
use bevy_egui::EguiPlugin;
use bevy_infinite_grid::{
    GridShadowCamera, InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings,
};
use bevy_mod_picking::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_polyline::PolylinePlugin;
use conduit::{well_known::WorldPos, ControlMsg};
use plugins::navigation_gizmo::NavigationGizmoPlugin;
use traces::TracesPlugin;

mod plugins;
pub(crate) mod traces;
mod ui;

struct EmbeddedAssetPlugin;

impl Plugin for EmbeddedAssetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "assets/icons/icon_play.png");
        embedded_asset!(app, "assets/icons/icon_pause.png");
        embedded_asset!(app, "assets/icons/icon_scrub.png");
        embedded_asset!(app, "assets/icons/icon_jump_to_end.png");
        embedded_asset!(app, "assets/icons/icon_jump_to_start.png");
        embedded_asset!(app, "assets/icons/icon_frame_forward.png");
        embedded_asset!(app, "assets/icons/icon_frame_back.png");
        embedded_asset!(app, "assets/textures/cube_side_top.png");
        embedded_asset!(app, "assets/textures/cube_side_bottom.png");
        embedded_asset!(app, "assets/textures/cube_side_front.png");
        embedded_asset!(app, "assets/textures/cube_side_back.png");
        embedded_asset!(app, "assets/textures/cube_side_right.png");
        embedded_asset!(app, "assets/textures/cube_side_left.png");
        embedded_asset!(app, "assets/textures/cube_side_corner.png");
        embedded_asset!(app, "assets/textures/cube_side_edge.png");
    }
}

const NAVIGATION_GIZMO_LAYER: RenderLayers = RenderLayers::layer(1);

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        window_theme: Some(WindowTheme::Dark),
                        title: "Elodin Editor".into(),
                        present_mode: PresentMode::AutoVsync,
                        fit_canvas_to_parent: true,
                        canvas: Some("#editor".to_string()),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<DiagnosticsPlugin>()
                .disable::<LogPlugin>()
                .build(),
        )
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::ReactiveLowPower {
                wait: Duration::from_secs(1),
            },
            ..Default::default()
        })
        .init_resource::<ui::Paused>()
        .init_resource::<ui::ShowStats>()
        .add_plugins(
            DefaultPickingPlugins
                .build()
                .disable::<DebugPickingPlugin>()
                .disable::<DefaultHighlightingPlugin>(),
        )
        .add_plugins(EmbeddedAssetPlugin)
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PolylinePlugin)
        .add_plugins(TracesPlugin)
        .add_plugins(NavigationGizmoPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_systems(Startup, setup_main_camera)
        .add_systems(Startup, setup_grid)
        .add_systems(Startup, setup_window_icon)
        .add_systems(Update, (ui::shortcuts, ui::render))
        .add_systems(Update, make_pickable)
        .add_systems(Update, sync_pos)
        .add_systems(Update, sync_paused)
        .add_systems(PostUpdate, ui::set_camera_viewport.after(ui::render))
        .insert_resource(ClearColor(Color::hex("#0D0D0D").unwrap()))
        .insert_resource(Msaa::Off);

        #[cfg(target_os = "macos")]
        app.add_systems(Startup, setup_titlebar);

        // For adding features incompatible with wasm:
        if cfg!(not(target_arch = "wasm32")) {
            embedded_asset!(app, "./assets/diffuse.ktx2");
            embedded_asset!(app, "./assets/specular.ktx2");
            app.insert_resource(DirectionalLightShadowMap { size: 8192 });
        }
    }
}

fn setup_grid(mut commands: Commands) {
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            minor_line_color: Color::rgba(1.0, 1.0, 1.0, 0.05),
            major_line_color: Color::rgba(1.0, 1.0, 1.0, 0.05),
            z_axis_color: Color::hex("#264FFF").unwrap(),
            x_axis_color: Color::hex("#EE3A43").unwrap(),
            shadow_color: None,
            ..Default::default()
        },
        ..default()
    });
}

#[derive(Component)]
pub struct MainCamera;

fn setup_main_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    // return the id so it can be fetched below
    let mut camera = commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(5.0, 5.0, 10.0)),
            camera: Camera {
                hdr: true,
                ..Default::default()
            },
            tonemapping: Tonemapping::TonyMcMapface,
            ..default()
        },
        // NOTE: Layers should be specified for all cameras otherwise `bevy_mod_picking` will use all layers
        RenderLayers::default(),
        MainCamera,
    ));

    camera
        .insert(BloomSettings { ..default() })
        .insert(PanOrbitCamera::default())
        .insert(GridShadowCamera);

    // For adding features incompatible with wasm:
    if cfg!(not(target_arch = "wasm32")) {
        camera.insert(EnvironmentMapLight {
            diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
            specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
        });
        // .insert(ScreenSpaceAmbientOcclusionBundle {
        //     settings: ScreenSpaceAmbientOcclusionSettings {
        //         quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
        //     },
        //     ..Default::default()
        // });
        // NOTE: Crashes custom camera viewport
        // .insert(TemporalAntiAliasBundle::default());

        commands.spawn(ScreenSpaceAmbientOcclusionSettings {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
        });
    }
}

#[cfg(target_os = "macos")]
fn setup_titlebar(
    windows: Query<(Entity, &bevy::window::PrimaryWindow)>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    use cocoa::{
        appkit::{
            NSToolbar, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility, NSWindowToolbarStyle,
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

fn setup_window_icon() {
    #[cfg(target_os = "macos")]
    set_icon_mac()
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

#[allow(clippy::type_complexity)]
fn make_pickable(
    mut commands: Commands,
    meshes: Query<Entity, (With<Handle<Mesh>>, Without<Pickable>, Changed<Handle<Mesh>>)>,
) {
    for entity in meshes.iter() {
        commands.entity(entity).insert((PickableBundle::default(),));
    }
}

pub fn sync_pos(mut query: Query<(&mut Transform, &WorldPos)>) {
    query.iter_mut().for_each(|(mut transform, pos)| {
        let WorldPos { pos, att } = pos;
        *transform = bevy::prelude::Transform {
            translation: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
            rotation: Quat::from_xyzw(att.i as f32, att.j as f32, att.k as f32, att.w as f32),
            ..Default::default()
        }
    });
}

pub fn sync_paused(paused: Res<ui::Paused>, mut event: EventWriter<ControlMsg>) {
    if paused.is_changed() {
        event.send(ControlMsg::SetPlaying(!paused.0));
    }
}
