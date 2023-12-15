use self::ui::*;
use bevy::{
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
use paracosm::{plugin::sync_pos, SimState};

//pub(crate) mod traces;
mod ui;

// pub fn editor<'a, T>(func: impl IntoSimRunner<'a, T> + Send + Sync + 'static) {
//     let (tx, rx) = channel_pair();
//     std::thread::spawn(move || {
//         let runner = func.into_runner();
//         let mut app = runner
//             .run_mode(paracosm::runner::RunMode::RealTime)
//             .build(tx);
//         app.run()
//     });
//     let mut app = App::new();
//     app.insert_non_send_resource(rx);
//     app.add_plugins(EditorPlugin::<ClientChannel>::new());
//     app.run()
// }

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        window_theme: Some(WindowTheme::Dark),
                        title: "Paracosm Editor".into(),
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
        .add_plugins(
            DefaultPickingPlugins
                .build()
                .disable::<DebugPickingPlugin>()
                .disable::<DefaultHighlightingPlugin>(),
        )
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(InfiniteGridPlugin)
        // .add_plugins(PolylinePlugin)
        //.add_plugins(TracesPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (picked_system,))
        .add_systems(Update, make_pickable)
        .add_systems(Update, timeline_system)
        .add_systems(Update, sync_pos)
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

#[allow(clippy::type_complexity)]
fn make_pickable(
    mut commands: Commands,
    meshes: Query<Entity, (With<Handle<Mesh>>, Without<Pickable>, Changed<Handle<Mesh>>)>,
) {
    for entity in meshes.iter() {
        commands.entity(entity).insert((PickableBundle::default(),));
    }
}
