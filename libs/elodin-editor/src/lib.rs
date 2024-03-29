use std::{collections::BTreeMap, time::Duration};

use bevy::{
    asset::{embedded_asset, AssetMetaCheck},
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    log::LogPlugin,
    math::DVec3,
    pbr::{
        DirectionalLightShadowMap, ScreenSpaceAmbientOcclusionQualityLevel,
        ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    render::{
        camera::{Exposure, PhysicalCameraParameters},
        view::RenderLayers,
    },
    window::{PresentMode, WindowTheme},
    winit::WinitSettings,
    DefaultPlugins,
};
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::prelude::OrbitConstraint;
use bevy_egui::EguiPlugin;
use bevy_mod_picking::prelude::*;
use bevy_polyline::PolylinePlugin;
use bevy_tweening::TweeningPlugin;
use big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};
use conduit::{
    bevy::{ComponentValueMap, Tick},
    well_known::WorldPos,
    ComponentId, ControlMsg, EntityId,
};
use plugins::navigation_gizmo::{spawn_gizmo, NavigationGizmoPlugin, RenderLayerAlloc};
use traces::TracesPlugin;
use ui::{utils, EntityPair, HoveredEntity};

mod plugins;
pub(crate) mod traces;
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
        embedded_asset!(app, "assets/icons/close.png");
        embedded_asset!(app, "assets/icons/chart.png");
    }
}

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AssetMetaCheck::Never)
            .add_plugins(plugins::WebAssetPlugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            window_theme: Some(WindowTheme::Dark),
                            title: "Elodin Editor".into(),
                            present_mode: PresentMode::AutoVsync,
                            canvas: Some("#editor".to_string()),
                            resize_constraints: WindowResizeConstraints {
                                min_width: 400.,
                                min_height: 400.,
                                ..Default::default()
                            },
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
                // On MacOS we use a special winit fork that requests a redraw every screen refresh,
                // I believe other platforms have similar behavior, but I have not fully tested them yet
                #[cfg(target_os = "macos")]
                focused_mode: bevy::winit::UpdateMode::Reactive {
                    wait: Duration::MAX,
                },
                #[cfg(not(target_os = "macos"))]
                focused_mode: bevy::winit::UpdateMode::Continuous,
                unfocused_mode: bevy::winit::UpdateMode::ReactiveLowPower {
                    wait: Duration::from_millis(16),
                },
            })
            .init_resource::<CollectedEntityData>()
            .add_plugins(DefaultPickingPlugins)
            .add_plugins(big_space::FloatingOriginPlugin::<i128>::default())
            .add_plugins(bevy_editor_cam::DefaultEditorCamPlugins)
            .add_plugins(EmbeddedAssetPlugin)
            .add_plugins(EguiPlugin)
            //.add_plugins(InfiniteGridPlugin)
            .add_plugins(PolylinePlugin)
            .add_plugins(TracesPlugin)
            .add_plugins(NavigationGizmoPlugin)
            .add_plugins(crate::plugins::gizmos::GizmoPlugin)
            .add_plugins(ui::UiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin)
            .add_plugins(TweeningPlugin)
            .add_systems(Startup, setup_floating_origin)
            //.add_systems(Startup, setup_grid)
            .add_systems(Startup, setup_window_icon)
            .add_systems(PreUpdate, collect_entity_data)
            .add_systems(Update, make_entities_selectable)
            .add_systems(Update, sync_pos)
            .add_systems(Update, sync_paused)
            .add_systems(Update, set_floating_origin)
            .insert_resource(ClearColor(Color::hex("#0D0D0D").unwrap()));

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

#[derive(Resource, Default, Debug)]
pub struct CollectedEntityData {
    pub ticks: Vec<u64>,
    pub data: BTreeMap<EntityId, BTreeMap<ComponentId, Vec<Vec<f64>>>>,
}

pub fn collect_entity_data(
    mut collected_entity_data: ResMut<CollectedEntityData>,
    tick: Res<Tick>,
    current_entity_data: Query<(&EntityId, &ComponentValueMap)>,
) {
    let last_tick = collected_entity_data.ticks.last();

    if last_tick.is_none() || last_tick.is_some_and(|t| tick.0 > *t) {
        collected_entity_data.ticks.push(tick.0);

        for (entity_id, component_value_map) in current_entity_data.iter() {
            let mut entity_components = BTreeMap::new();

            if let Some(entity) = collected_entity_data.data.get(entity_id) {
                for (component_id, component_value) in &component_value_map.0 {
                    if let Some(entity_component) = entity.get(component_id) {
                        let mut collected_component_values = entity_component.clone();
                        collected_component_values
                            .push(utils::component_value_to_vec(component_value));

                        entity_components.insert(*component_id, collected_component_values);
                    } else {
                        entity_components.insert(
                            *component_id,
                            vec![utils::component_value_to_vec(component_value)],
                        );
                    }
                }
            } else {
                for (component_id, component_value) in &component_value_map.0 {
                    entity_components.insert(
                        *component_id,
                        vec![utils::component_value_to_vec(component_value)],
                    );
                }
            }

            collected_entity_data
                .data
                .insert(*entity_id, entity_components);
        }
    }
}

#[derive(Component)]
pub struct MainCamera;

fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::<i128>::default(),
        Transform::IDENTITY,
    ));
}

fn spawn_main_camera(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
) -> Entity {
    // For adding features incompatible with wasm:
    if cfg!(not(target_arch = "wasm32")) {
        // .insert(ScreenSpaceAmbientOcclusionBundle {
        //     settings: ScreenSpaceAmbientOcclusionSettings {
        //         quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
        //     },
        //     ..Default::default()
        // });
        // NOTE: Crashes custom camera viewport
        // camera.insert(TemporalAntiAliasBundle::default());

        commands.spawn(ScreenSpaceAmbientOcclusionSettings {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
        });
    }
    let mut camera = commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(5.0, 5.0, 10.0))
                .looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                hdr: true,
                ..Default::default()
            },
            tonemapping: Tonemapping::TonyMcMapface,
            exposure: Exposure::from_physical_camera(PhysicalCameraParameters {
                aperture_f_stops: 2.8,
                shutter_speed_s: 1.0 / 200.0,
                sensitivity_iso: 400.0,
            }),
            ..default()
        },
        // NOTE: Layers should be specified for all cameras otherwise `bevy_mod_picking` will use all layers
        RenderLayers::default(),
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
    ));

    camera.insert(BloomSettings { ..default() });

    camera.insert(EnvironmentMapLight {
        diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
        specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
        intensity: 2000.0,
    });

    let camera = camera.id();

    spawn_gizmo(camera, commands, meshes, materials, render_layer_alloc);
    camera
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
                      mut hovered_entity: ResMut<HoveredEntity>| {
                    hovered_entity.0 = Some(EntityPair {
                        bevy: entity,
                        conduit: entity_id,
                    })
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
        .for_each(|(mut transform, mut grid_cell, pos)| {
            let WorldPos { pos, att } = pos;
            let (new_grid_cell, translation) =
                floating_origin.translation_to_grid(DVec3::from_slice(pos.as_slice()));
            *grid_cell = new_grid_cell;
            *transform = bevy::prelude::Transform {
                translation,
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
