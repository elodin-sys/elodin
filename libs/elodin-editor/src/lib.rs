use bevy::{
    asset::embedded_asset,
    core_pipeline::{
        bloom::BloomSettings,
        experimental::taa::{TemporalAntiAliasBundle, TemporalAntiAliasPlugin},
        tonemapping::Tonemapping,
    },
    diagnostic::DiagnosticsPlugin,
    log::LogPlugin,
    pbr::{
        DirectionalLightShadowMap, ScreenSpaceAmbientOcclusionBundle,
        ScreenSpaceAmbientOcclusionQualityLevel, ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    window::{PresentMode, WindowTheme},
    DefaultPlugins,
};
use bevy_atmosphere::prelude::*;
use bevy_egui::EguiPlugin;
use bevy_infinite_grid::{
    GridShadowCamera, InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings,
};
use bevy_mod_picking::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_polyline::PolylinePlugin;
use elodin_conduit::well_known::{SimState, WorldPos};
use traces::TracesPlugin;

pub(crate) mod traces;
mod ui;

#[cfg(feature = "core")]
pub fn editor<'a, T>(func: impl elodin_core::runner::IntoSimRunner<'a, T> + Send + Sync + 'static) {
    use elodin_conduit::{
        bevy::{Msg, SubscribeEvent},
        bevy_sync::{SendPlbPlugin, SyncPlugin, DEFAULT_SUB_FILTERS},
        cid_mask,
    };

    let (server_tx, server_rx) = flume::unbounded();
    let (client_tx, client_rx) = flume::unbounded();
    let _ = std::thread::spawn(move || {
        let runner = func.into_runner();
        let mut app = runner
            .run_mode(elodin_core::runner::RunMode::RealTime)
            .build_with_plugins((SyncPlugin::new(server_rx), SendPlbPlugin));

        app.run()
    });
    server_tx
        .send(Msg::Subscribe(SubscribeEvent {
            tx: client_tx.clone(),
            filters: DEFAULT_SUB_FILTERS.to_vec(),
        }))
        .unwrap();

    client_tx
        .send(Msg::Subscribe(SubscribeEvent {
            tx: server_tx.clone(),
            filters: vec![cid_mask!(32;sim_state)],
        }))
        .unwrap();
    let mut app = App::new();
    app.add_plugins((EditorPlugin, SyncPlugin::new(client_rx)));
    app.run()
}

struct EmbeddedAssetPlugin;

impl Plugin for EmbeddedAssetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "assets/icons/icon_play.png");
        embedded_asset!(app, "assets/icons/icon_pause.png");
        embedded_asset!(app, "assets/icons/icon_scrub.png");
        embedded_asset!(app, "assets/icons/icon_skip_next.png");
        embedded_asset!(app, "assets/icons/icon_skip_prev.png");
    }
}

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        window_theme: Some(WindowTheme::Dark),
                        title: "Elodin Editor".into(),
                        present_mode: PresentMode::AutoNoVsync,
                        fit_canvas_to_parent: true,
                        canvas: Some("#editor".to_string()),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<LogPlugin>()
                .disable::<DiagnosticsPlugin>()
                .build(),
        )
        .insert_resource(SimState::default())
        .init_resource::<ui::UiState>()
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
        .add_systems(Startup, setup)
        .add_systems(Update, ui::render)
        .add_systems(Update, make_pickable)
        .add_systems(Update, sync_pos)
        .add_systems(Startup, setup_window_icon)
        .insert_resource(AmbientLight {
            color: Color::hex("#FFF").unwrap(),
            brightness: 1.0,
        })
        //.insert_resource(Editables::default())
        .insert_resource(ClearColor(Color::hex("#16161A").unwrap()))
        .insert_resource(Msaa::Off);

        // For adding features incompatible with wasm:
        if cfg!(not(target_arch = "wasm32")) {
            app.add_plugins(TemporalAntiAliasPlugin)
                .add_plugins(AtmospherePlugin)
                .insert_resource(AtmosphereModel::new(Gradient {
                    sky: Color::hex("1B2642").unwrap(),
                    horizon: Color::hex("00081E").unwrap(),
                    ground: Color::hex("#00081E").unwrap(),
                }))
                .insert_resource(DirectionalLightShadowMap { size: 8192 });
        }
    }
}

fn setup(mut commands: Commands, _asset_server: Res<AssetServer>) {
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            minor_line_color: Color::hex("#00081E").unwrap(),
            major_line_color: Color::hex("#00081E").unwrap(),
            x_axis_color: Color::hex("F46E22").unwrap(),
            shadow_color: None,
            ..Default::default()
        },
        ..default()
    });

    // return the id so it can be fetched below
    let mut camera = commands.spawn(Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(5.0, 5.0, 10.0)),
        camera: Camera {
            hdr: true,
            ..Default::default()
        },
        tonemapping: Tonemapping::TonyMcMapface,
        ..default()
    });

    camera
        .insert(BloomSettings { ..default() })
        .insert(PanOrbitCamera::default())
        .insert(GridShadowCamera);

    // For adding features incompatible with wasm:
    if cfg!(not(target_arch = "wasm32")) {
        camera
            .insert(AtmosphereCamera::default())
            // .insert(EnvironmentMapLight {
            //     diffuse_map: asset_server.load("diffuse.ktx2"),
            //     specular_map: asset_server.load("specular.ktx2"),
            // })
            .insert(ScreenSpaceAmbientOcclusionBundle {
                settings: ScreenSpaceAmbientOcclusionSettings {
                    quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
                },
                ..Default::default()
            })
            .insert(TemporalAntiAliasBundle::default());

        commands.spawn(ScreenSpaceAmbientOcclusionSettings {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
        });
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
