//! Editor adapter for the independently maintained bevy-cesium plugin.
use super::simulation_startup::release_simulation;
use crate::{
    MainCamera,
    sensor_camera::{SensorCamera, SensorCameraConfig, SensorCameraConfigs},
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
};
use bevy::{math::DMat4, picking::Pickable, prelude::*};
use bevy_cesium::{
    CesiumDetailViews, CesiumPlugin as StreamingPlugin, CesiumPrimitive, CesiumSelectionMode,
    CesiumSet, CesiumSource, CesiumSourceUpdate, CesiumStatus, CesiumView, CesiumViews,
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

pub struct CesiumPlugin;

#[derive(Resource, Default)]
struct RetryTerrain(bool);

fn env_positive(name: &str, default: f64, allow_zero: bool) -> f64 {
    let Ok(value) = std::env::var(name) else {
        return default;
    };
    let value: f64 = value.parse().unwrap_or_else(|_| panic!("Invalid {name}"));
    assert!(
        value.is_finite() && (value > 0.0 || allow_zero && value == 0.0),
        "Invalid {name}"
    );
    value
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureView {
    position: [f64; 3],
    direction: [f64; 3],
    up: [f64; 3],
    width: f64,
    height: f64,
    horizontal_fov: f64,
    vertical_fov: f64,
}

fn parse_capture_views(bytes: &[u8]) -> Result<Vec<CesiumView>, &'static str> {
    let input: Vec<CaptureView> =
        serde_json::from_slice(bytes).map_err(|_| "Invalid capture views JSON")?;
    if input.is_empty() {
        return Err("Capture views must not be empty");
    }
    input
        .into_iter()
        .map(|v| {
            let direction = bevy::math::DVec3::from_array(v.direction);
            let up = bevy::math::DVec3::from_array(v.up);
            if !bevy::math::DVec3::from_array(v.position).is_finite()
                || !direction.is_finite()
                || !up.is_finite()
                || !direction.cross(up).length_squared().is_finite()
                || direction.cross(up).length_squared() <= 1e-12
                || [v.width, v.height]
                    .iter()
                    .any(|x| !x.is_finite() || *x <= 0.0)
                || [v.horizontal_fov, v.vertical_fov]
                    .iter()
                    .any(|x| !x.is_finite() || *x <= 0.0 || *x >= std::f64::consts::PI)
            {
                return Err("Invalid capture camera geometry");
            }
            Ok(CesiumView {
                position: v.position,
                direction: v.direction,
                up: v.up,
                width: v.width,
                height: v.height,
                horizontal_fov: v.horizontal_fov,
                vertical_fov: v.vertical_fov,
            })
        })
        .collect()
}

/// Fixed-path captures wait for terrain rather than silently emitting coarse frames.
#[derive(Resource)]
pub(crate) struct TerrainCapture {
    started: web_time::Instant,
    timeout: std::time::Duration,
}
impl TerrainCapture {
    pub(crate) fn ready(&self, status: Option<&CesiumStatus>) -> Result<bool, &'static str> {
        if status.is_some_and(|s| s.load_failed) {
            return Err(status
                .and_then(|s| s.error)
                .unwrap_or("Terrain capture failed to load"));
        }
        if status.is_some_and(CesiumStatus::ready_for_capture) {
            return Ok(true);
        }
        if self.started.elapsed() >= self.timeout {
            return Err("Terrain capture warm-up timed out");
        }
        Ok(false)
    }
}

/// Interactive streaming needs one complete initial selection too. Afterwards,
/// normal flight can refine terrain without repeatedly stopping sensor output.
#[derive(Resource)]
pub(crate) struct TerrainStartup {
    started: web_time::Instant,
    ready: bool,
    error: Option<&'static str>,
}

impl TerrainStartup {
    pub(crate) fn waiting() -> Self {
        Self {
            started: web_time::Instant::now(),
            ready: false,
            error: None,
        }
    }

    pub(crate) fn owns_surface(&self) -> bool {
        self.error.is_none()
    }

    pub(crate) fn update(&mut self, status: &CesiumStatus) {
        if self.ready || self.error.is_some() {
            return;
        }
        if status.load_failed {
            self.error = Some(status.error.unwrap_or("Terrain capture failed to load"));
        } else {
            self.ready = status.ready_for_capture();
        }
    }

    pub(super) fn result(&self) -> Result<bool, &'static str> {
        self.error.map_or(Ok(self.ready), Err)
    }
}

pub(crate) fn capture_ready(world: &World) -> Result<bool, &'static str> {
    capture_readiness(
        world.get_resource::<TerrainStartup>(),
        world.get_resource::<TerrainCapture>(),
        world.get_resource::<CesiumStatus>(),
    )
}

pub(crate) fn capture_readiness(
    startup: Option<&TerrainStartup>,
    capture: Option<&TerrainCapture>,
    status: Option<&CesiumStatus>,
) -> Result<bool, &'static str> {
    if let Some(startup) = startup
        && let Err(error) = startup.result()
    {
        return Err(error);
    }
    if let Some(capture) = capture {
        return capture.ready(status);
    }
    Ok(startup.is_none_or(|startup| startup.ready))
}

pub(crate) fn playback_ready(world: &World) -> bool {
    world
        .get_resource::<TerrainStartup>()
        .is_none_or(|startup| startup.ready || startup.error.is_some())
}

fn camera_position_ecef(
    ecef_from_bevy: DMat4,
    grid: &FloatingOriginSettings,
    cell: &GridCell,
    transform: &Transform,
) -> bevy::math::DVec3 {
    ecef_from_bevy.transform_point3(grid.grid_position_double(cell, transform))
}

#[cfg(test)]
mod coordinate_tests {
    use super::*;
    use bevy::math::DVec3;
    use bevy_geo_frames::{GeoOrigin, Present};

    #[test]
    fn terrain_asset_waits_for_the_db_and_tracks_connection_changes() {
        let mut app = App::new();
        app.insert_resource(crate::ui::schematic::CurrentSchematic(
            impeller_wkt::Schematic {
                terrain: Some(impeller_wkt::TerrainConfig::Asset {
                    path: "db:terrain/Falcon9/tileset.json".into(),
                }),
                ..default()
            },
        ))
        .add_systems(Update, configure_terrain);
        app.update();
        assert!(!playback_ready(app.world()));
        assert!(app.world().resource::<CesiumSourceUpdate>().0.is_none());

        for (address, expected) in [
            (
                "assets.example:2240",
                "http://assets.example:2241/terrain/Falcon9/tileset.json",
            ),
            (
                "127.0.0.1:2260",
                "http://127.0.0.1:2261/terrain/Falcon9/tileset.json",
            ),
        ] {
            app.insert_resource(impeller_bevy::ConnectionAddr::from_endpoint(
                address.parse().unwrap(),
            ));
            app.update();
            let Some(CesiumSource::Prepared { url, .. }) =
                &app.world().resource::<CesiumSourceUpdate>().0
            else {
                panic!("terrain asset was not resolved through the DB endpoint");
            };
            assert_eq!(url, expected);
            assert!(
                playback_ready(app.world()),
                "prepared replay must not wait for refinement"
            );
            app.world_mut()
                .resource_mut::<TerrainStartup>()
                .update(&CesiumStatus {
                    load_failed: true,
                    error: Some("HTTP request failed"),
                    ..default()
                });
            assert!(
                playback_ready(app.world()),
                "a missing tile must not stop replay"
            );
        }

        // The assets host and prefix may change independently of the DB.
        for base in [
            "https://cdn.example/recordings/f9/assets",
            "https://cdn.example/recordings/other/assets/",
        ] {
            let endpoint = impeller_bevy::DbEndpoint::parse("db.example:2240")
                .unwrap()
                .with_assets(base.parse().unwrap());
            app.insert_resource(impeller_bevy::ConnectionAddr::from_endpoint(endpoint));
            app.update();
            let Some(CesiumSource::Prepared { url, .. }) =
                &app.world().resource::<CesiumSourceUpdate>().0
            else {
                panic!("terrain asset did not use the assets override");
            };
            assert_eq!(
                url,
                &format!(
                    "{}/terrain/Falcon9/tileset.json",
                    base.trim_end_matches('/')
                )
            );
        }
    }

    #[test]
    fn prepared_sensor_capture_waits_for_initial_terrain() {
        let mut app = App::new();
        app.init_resource::<crate::sensor_camera::HeadlessMode>()
            .insert_resource(crate::ui::schematic::CurrentSchematic(
                impeller_wkt::Schematic {
                    terrain: Some(impeller_wkt::TerrainConfig::Asset {
                        path: "terrain/Falcon9/tileset.json".into(),
                    }),
                    ..default()
                },
            ))
            .insert_resource(impeller_bevy::ConnectionAddr::from_endpoint(
                "127.0.0.1:2240".parse().unwrap(),
            ))
            .add_systems(Update, configure_terrain);
        app.update();
        assert_eq!(capture_ready(app.world()), Ok(false));
    }

    #[test]
    fn prepared_terrain_streams_by_default_but_offline_has_no_provider_credentials() {
        use super::super::terrain_playback::{TerrainMode, TerrainPlayback};
        let mut app = App::new();
        app.insert_resource(crate::ui::schematic::CurrentSchematic(
            impeller_wkt::Schematic {
                terrain: Some(impeller_wkt::TerrainConfig::Asset {
                    path: "terrain/Falcon9/tileset.json".into(),
                }),
                ..default()
            },
        ))
        .insert_resource(impeller_bevy::ConnectionAddr::from_endpoint(
            "127.0.0.1:2240".parse().unwrap(),
        ))
        .add_systems(Update, configure_terrain);
        for mode in [None, Some(TerrainMode::Offline), Some(TerrainMode::Hybrid)] {
            app.insert_resource(TerrainPlayback {
                mode,
                basemap: None,
            });
            app.update();
            let Some(CesiumSource::Prepared {
                ion_access_token, ..
            }) = &app.world().resource::<CesiumSourceUpdate>().0
            else {
                panic!("expected prepared terrain")
            };
            assert_eq!(
                ion_access_token.is_some(),
                mode != Some(TerrainMode::Offline)
            );
        }
    }

    #[test]
    fn offline_mode_rejects_unprepared_sources_before_creating_a_provider() {
        use super::super::terrain_playback::{TerrainMode, TerrainPlayback};
        let mut app = App::new();
        app.insert_resource(TerrainPlayback {
            mode: Some(TerrainMode::Offline),
            basemap: None,
        })
        .insert_resource(crate::ui::schematic::CurrentSchematic(
            impeller_wkt::Schematic {
                terrain: Some(impeller_wkt::TerrainConfig::Ion { asset_id: 2275207 }),
                ..default()
            },
        ))
        .add_systems(Update, configure_terrain);
        app.update();
        assert!(app.world().resource::<CesiumSourceUpdate>().0.is_none());
        assert!(
            app.world()
                .resource::<TerrainStartup>()
                .result()
                .unwrap_err()
                .contains("prepare")
        );
        assert!(playback_ready(app.world()));
    }

    #[test]
    fn aerial_and_photo3d_select_their_matching_sources() {
        assert!(matches!(
            terrain_source(1),
            CesiumSource::IonTerrain {
                asset_id: 1,
                imagery_asset_id: 2,
                ..
            }
        ));
        assert!(matches!(
            terrain_source(2275207),
            CesiumSource::Ion {
                asset_id: 2275207,
                ..
            }
        ));
    }

    #[test]
    fn failed_terrain_releases_playback_but_keeps_captures_failed() {
        let mut world = World::new();
        world.insert_resource(TerrainStartup::waiting());
        assert!(!playback_ready(&world));
        world
            .resource_mut::<TerrainStartup>()
            .update(&CesiumStatus {
                load_failed: true,
                error: Some("Cesium Ion unavailable"),
                ..default()
            });
        assert!(playback_ready(&world));
        assert!(!world.resource::<TerrainStartup>().owns_surface());
        assert_eq!(capture_ready(&world), Err("Cesium Ion unavailable"));
    }

    #[test]
    fn terrain_failure_releases_held_simulation() {
        let (packet_tx, _packet_rx, outgoing, _incoming) = impeller_bevy::channels::channels();
        let mut config = impeller_wkt::DbConfig::default();
        config.metadata.insert(
            impeller_wkt::EDITOR_STARTUP_GATE_METADATA_KEY.to_string(),
            "1".to_string(),
        );
        let mut app = App::new();
        app.add_message::<crate::ui::schematic::DocumentLoaded>()
            .insert_resource(TerrainStartup::waiting())
            .insert_resource(config)
            .insert_resource(packet_tx)
            .add_systems(Update, release_simulation);
        app.update();
        assert!(outgoing.try_recv().is_err());

        app.world_mut()
            .resource_mut::<TerrainStartup>()
            .update(&CesiumStatus {
                load_failed: true,
                ..default()
            });
        app.update();
        assert!(matches!(outgoing.try_recv(), Ok(Some(_))));
    }

    #[test]
    fn removing_terrain_from_the_schematic_disables_streaming() {
        let mut app = App::new();
        let terrain = Some(impeller_wkt::TerrainConfig::Ion { asset_id: 2275207 });
        app.insert_resource(crate::ui::schematic::CurrentSchematic(
            impeller_wkt::Schematic {
                terrain,
                ..default()
            },
        ))
        .add_systems(Update, configure_terrain);
        app.update();
        assert!(app.world().contains_resource::<TerrainStartup>());
        app.world_mut()
            .resource_mut::<crate::ui::schematic::CurrentSchematic>()
            .terrain = None;
        app.update();
        assert!(!app.world().contains_resource::<TerrainStartup>());
        assert!(
            app.world()
                .resource::<crate::ui::schematic::CurrentSchematic>()
                .terrain
                .is_none()
        );
        assert!(app.world().resource::<CesiumSourceUpdate>().0.is_none());
        assert!(playback_ready(app.world()));
    }

    #[test]
    fn failed_terrain_retries_only_when_requested() {
        let terrain = impeller_wkt::TerrainConfig::Ion { asset_id: 2275207 };
        let mut app = App::new();
        app.add_message::<crate::ui::schematic::DocumentLoaded>()
            .add_message::<crate::ui::schematic::DocumentReloaded>()
            .init_resource::<RetryTerrain>()
            .insert_resource(crate::ui::schematic::CurrentSchematic(
                impeller_wkt::Schematic {
                    terrain: Some(terrain.clone()),
                    ..default()
                },
            ))
            .add_systems(Update, request_terrain_retry.before(configure_terrain))
            .add_systems(Update, configure_terrain);
        app.update();
        app.world_mut().resource_mut::<TerrainStartup>().error = Some("failed");

        app.world_mut()
            .resource_mut::<crate::ui::schematic::CurrentSchematic>()
            .terrain = Some(terrain);
        app.update();
        assert_eq!(
            app.world().resource::<TerrainStartup>().error,
            Some("failed")
        );

        app.world_mut()
            .write_message(crate::ui::schematic::DocumentReloaded {
                save_path: None,
                document: crate::ui::schematic::SchematicDocumentAsset {
                    root: default(),
                    windows: Vec::new(),
                },
                changed_window_indices: Vec::new(),
            });
        app.update();
        let startup = app.world().resource::<TerrainStartup>();
        assert_eq!(startup.error, None);
        assert!(!startup.ready);
    }

    #[test]
    fn interactive_screenshot_waits_for_startup_and_reports_failure() {
        let mut world = World::new();
        assert_eq!(capture_ready(&world), Ok(true));
        world.insert_resource(TerrainStartup {
            started: web_time::Instant::now(),
            ready: false,
            error: None,
        });
        // Interactive streaming has no fixed-path TerrainCapture resource.
        assert!(!world.contains_resource::<TerrainCapture>());
        assert_eq!(capture_ready(&world), Ok(false));
        world.resource_mut::<TerrainStartup>().ready = true;
        assert_eq!(capture_ready(&world), Ok(true));
        world.resource_mut::<TerrainStartup>().ready = false;
        world.insert_resource(CesiumStatus::default());
        world.insert_resource(TerrainCapture {
            started: web_time::Instant::now(),
            timeout: std::time::Duration::ZERO,
        });
        assert_eq!(
            capture_ready(&world),
            Err("Terrain capture warm-up timed out")
        );
    }

    #[test]
    fn playback_waits_only_for_the_initial_terrain() {
        let mut world = World::new();
        assert!(playback_ready(&world));
        world.insert_resource(TerrainStartup {
            started: web_time::Instant::now(),
            ready: false,
            error: None,
        });
        assert!(!playback_ready(&world));
        world.resource_mut::<TerrainStartup>().ready = true;
        assert!(playback_ready(&world));
        world.insert_resource(CesiumStatus::default());
        assert!(playback_ready(&world));
    }

    #[test]
    fn initial_terrain_wait_has_no_timeout_and_does_not_restart_during_flight() {
        let mut startup = TerrainStartup {
            started: web_time::Instant::now() - std::time::Duration::from_secs(3600),
            ready: false,
            error: None,
        };
        startup.update(&CesiumStatus::default());
        assert_eq!(startup.result(), Ok(false));
        startup.update(&CesiumStatus {
            selection_complete: true,
            visible_tiles: 1,
            stable_frames: 8,
            ..default()
        });
        assert_eq!(startup.result(), Ok(true));
        startup.update(&CesiumStatus::default());
        assert_eq!(startup.result(), Ok(true));
    }

    #[test]
    fn inactive_sensor_keeps_terrain_requirements() {
        let mut app = App::new();
        app.insert_resource(GeoContext::from(GeoOrigin::new_from_degrees(
            35.350664,
            -117.809027,
            589.274,
        )))
        .insert_resource(FloatingOriginSettings::new(2000.0, 100.0))
        .init_resource::<CesiumViews>()
        .add_systems(Update, collect_views);
        let camera = Camera {
            is_active: false,
            viewport: Some(bevy::camera::Viewport {
                physical_size: UVec2::new(640, 480),
                ..default()
            }),
            ..default()
        };
        app.world_mut().spawn((
            camera.clone(),
            Projection::Perspective(default()),
            Transform::from_xyz(0., 100., 0.),
            GridCell::default(),
            SensorCamera { config_index: 0 },
        ));
        app.world_mut().spawn((
            camera,
            Projection::Perspective(default()),
            Transform::IDENTITY,
            MainCamera,
        ));
        app.update();
        let views = &app.world().resource::<CesiumViews>().0;
        assert_eq!(views.len(), 1);
        assert_eq!((views[0].width, views[0].height), (640., 480.));
        let first = views[0].position;
        app.update();
        assert_eq!(app.world().resource::<CesiumViews>().0[0].position, first);
    }

    #[test]
    fn aligned_zoomed_sensor_shares_coverage_before_and_after_startup() {
        let config: SensorCameraConfig = serde_json::from_value(serde_json::json!({
            "entity_name": "jet", "camera_name": "jet.color",
            "width": 640, "height": 480, "fov_degrees": 90.0,
            "near": 0.1, "far": 100_000.0, "pos_offset": [1.2, 0.0, 0.1]
        }))
        .unwrap();
        let mut zoom = config.clone();
        zoom.camera_name = "jet.ir".into();
        zoom.fov_degrees = 14.4;
        zoom.height = 512;
        let configs = vec![config.clone(), zoom.clone()];
        assert!(has_wider_sensor_view(&zoom, &configs));
        assert!(!has_wider_sensor_view(&config, &configs));
        assert!(!has_wider_sensor_view(&zoom, &[zoom.clone()]));
        let mut other_pose = zoom.clone();
        other_pose.rot_offset[1] = 30.0;
        assert!(!has_wider_sensor_view(&other_pose, &configs));

        let mut app = App::new();
        app.insert_resource(GeoContext::from(GeoOrigin::new_from_degrees(
            35.350664,
            -117.809027,
            589.274,
        )))
        .insert_resource(FloatingOriginSettings::new(2000.0, 100.0))
        .insert_resource(SensorCameraConfigs(configs))
        .insert_resource(TerrainStartup {
            started: web_time::Instant::now(),
            ready: false,
            error: None,
        })
        .init_resource::<CesiumViews>()
        .add_systems(Update, collect_views);
        for config_index in 0..2 {
            app.world_mut().spawn((
                Camera {
                    is_active: false,
                    ..default()
                },
                Projection::Perspective(default()),
                Transform::from_xyz(0., 100., 0.),
                GridCell::default(),
                SensorCamera { config_index },
            ));
        }
        app.update();
        assert_eq!(app.world().resource::<CesiumViews>().0.len(), 1);
        app.world_mut().resource_mut::<TerrainStartup>().ready = true;
        app.update();
        assert_eq!(app.world().resource::<CesiumViews>().0.len(), 1);
        app.world_mut().resource_mut::<SensorCameraConfigs>().0[1].rot_offset[1] = 30.0;
        app.update();
        assert_eq!(app.world().resource::<CesiumViews>().0.len(), 2);
    }

    #[test]
    fn fixed_capture_rejects_empty_or_invalid_paths_and_fails_closed() {
        assert!(parse_capture_views(b"[]").is_err());
        let valid = br#"[{"position":[6378200,5,-5],"direction":[-1,0,0],"up":[0,0,1],"width":1280,"height":720,"horizontal_fov":1.4,"vertical_fov":0.9}]"#;
        assert_eq!(parse_capture_views(valid).unwrap().len(), 1);
        let invalid = String::from_utf8_lossy(valid).replace("\"width\":1280", "\"width\":0");
        assert!(parse_capture_views(invalid.as_bytes()).is_err());
        let capture = TerrainCapture {
            started: web_time::Instant::now(),
            timeout: std::time::Duration::ZERO,
        };
        assert!(capture.ready(None).is_err());
        let mut status = CesiumStatus {
            selection_complete: true,
            visible_tiles: 1,
            stable_frames: 8,
            ..default()
        };
        assert_eq!(capture.ready(Some(&status)), Ok(true));
        status.load_failed = true;
        assert!(capture.ready(Some(&status)).is_err());
    }

    #[test]
    fn new_tile_is_placed_before_its_first_visible_frame() {
        let origin = GeoOrigin::new_from_degrees(40.0, -75.0, 300.0);
        let geo = GeoContext::from(origin).with_present(Present::Sphere);
        let ecef_from_mesh = GeoFrame::ECEF._M_(&GeoFrame::ENU, &geo)
            * DMat4::from_translation(DVec3::new(12000.0, -8000.0, 70.0));
        let expected = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo) * ecef_from_mesh;
        let mut app = App::new();
        let root = app.world_mut().spawn_empty().id();
        app.insert_resource(geo)
            .insert_resource(FloatingOriginSettings::new(2000.0, 100.0))
            .insert_resource(BigSpaceRootEntity(root))
            .add_systems(Update, place_tiles);
        let tile = app
            .world_mut()
            .spawn((
                CesiumPrimitive { ecef_from_mesh },
                Transform::IDENTITY,
                Visibility::Visible,
            ))
            .id();
        // No PreUpdate geographic systems: the adapter must initialize the pose.
        app.update();
        let transform = app.world().get::<Transform>(tile).unwrap();
        let cell = app.world().get::<GridCell>(tile).unwrap();
        let grid = app.world().resource::<FloatingOriginSettings>();
        assert!(
            (grid.grid_position_double(cell, transform) - expected.w_axis.truncate()).length()
                < 0.001
        );
        let (_, expected_rotation, _) = expected.to_scale_rotation_translation();
        assert!(
            transform
                .rotation
                .as_dquat()
                .angle_between(expected_rotation)
                < 1e-5
        );
        assert_eq!(app.world().get::<ChildOf>(tile).unwrap().parent(), root);
        assert!(app.world().get::<Pickable>(tile).is_some());
    }

    #[test]
    fn cesium_camera_matches_ecef_across_grid_cells_and_earth_scale() {
        let origin =
            GeoOrigin::new_from_degrees(40.03879587543084, -75.59670696255716, 297.25783960334957);
        let grid = FloatingOriginSettings::new(2000.0, 100.0);
        for presentation in [Present::Plane, Present::Sphere] {
            let geo = GeoContext::from(origin.clone()).with_present(presentation);
            let bevy_from_ecef = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo);
            let ecef_from_enu = GeoFrame::ECEF._M_(&GeoFrame::ENU, &geo);
            for local in [
                DVec3::new(25.0, -80.0, 70.0),
                DVec3::new(12345.678, -54321.987, 200000.123),
                DVec3::new(10000000.123, -20000000.456, 30000000.789),
            ] {
                let expected = ecef_from_enu.transform_point3(local);
                let (cell, translation) =
                    grid.translation_to_grid(bevy_from_ecef.transform_point3(expected));
                let actual = camera_position_ecef(
                    bevy_from_ecef.inverse(),
                    &grid,
                    &cell,
                    &Transform::from_translation(translation),
                );
                assert!(
                    (actual - expected).length() < 0.001,
                    "ECEF camera reconstruction error: {} m",
                    (actual - expected).length()
                );
            }
        }
    }
}

impl Plugin for CesiumPlugin {
    fn build(&self, app: &mut App) {
        super::terrain_playback::install(app);
        app.init_resource::<RetryTerrain>();
        if let Ok(path) = std::env::var("ELODIN_CESIUM_CAPTURE_VIEWS") {
            let bytes = std::fs::read(path).expect("Cannot read ELODIN_CESIUM_CAPTURE_VIEWS");
            let views = parse_capture_views(&bytes).expect("Invalid ECEF capture view file");
            info!(
                reference_views = views.len(),
                "Cesium fixed capture selection enabled"
            );
            app.insert_resource(CesiumDetailViews(views))
                .insert_resource(CesiumSelectionMode::Fixed)
                .insert_resource(TerrainCapture {
                    started: web_time::Instant::now(),
                    timeout: std::time::Duration::from_secs_f64(env_positive(
                        "ELODIN_CESIUM_CAPTURE_TIMEOUT",
                        120.0,
                        false,
                    )),
                });
        }
        app.add_plugins(
            StreamingPlugin::deferred()
                .with_external_coordinates()
                .with_initial_load_wait()
                .with_lod_retention_margin(if cfg!(target_family = "wasm") {
                    0.0
                } else {
                    500.0
                })
                .with_maximum_screen_error(24.0),
        )
        .add_systems(
            Update,
            request_terrain_retry.before(configure_terrain).run_if(
                resource_exists::<Messages<crate::ui::schematic::DocumentLoaded>>
                    .and_then(resource_exists::<Messages<crate::ui::schematic::DocumentReloaded>>),
            ),
        )
        .add_systems(Update, configure_terrain.before(CesiumSet::Stream))
        .add_systems(Update, collect_views.in_set(CesiumSet::Views))
        .add_systems(Update, place_tiles.in_set(CesiumSet::Placement))
        .add_systems(
            Update,
            update_startup
                .after(CesiumSet::Publish)
                .run_if(resource_exists::<TerrainStartup>),
        )
        .add_systems(
            Update,
            release_simulation
                .after(configure_terrain)
                .after(update_startup)
                .run_if(resource_exists::<Messages<crate::ui::schematic::DocumentLoaded>>),
        )
        .add_systems(
            Update,
            show_startup
                .run_if(resource_exists::<TerrainStartup>)
                .after(update_startup)
                .after(crate::ui::render_layout),
        );
        app.add_systems(Update, log_streaming_status.after(CesiumSet::Publish));
    }
}

/// The V1 demo account is shared by scenery and height preparation.
/// This read-only token is intentionally distributed with the application.
pub fn ion_access_token() -> &'static str {
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6Ii1XcElsai05enF6U2lUTEQiLCJqdGkiOiJhMGEyMjBlNy00ZjFjLTQ2YTctYWNkYi0wZWNlZWJmYjc3NzciLCJpZCI6NTMyMjI0LCJzdWIiOiJ6eHE4MmxtIiwiaXNzIjoiaHR0cHM6Ly9hcGkuY2VzaXVtLmNvbSIsImF1ZCI6IkVsb2RpbiBUZXJyYWluIGRlbW8gwrcgZGVza3RvcCIsImlhdCI6MTc5MzY2NDI0OX0.0aUiiGh55YtAMAe3hUrBLqlasel0qXoALs_zzSyrzpE"
}

fn terrain_source(asset_id: i64) -> CesiumSource {
    let access_token = ion_access_token().to_string();
    if asset_id == 1 {
        CesiumSource::IonTerrain {
            asset_id,
            imagery_asset_id: 2,
            access_token,
        }
    } else {
        CesiumSource::Ion {
            asset_id,
            access_token,
        }
    }
}

fn configure_terrain(
    world: &mut World,
    mut previous: Local<
        Option<(
            Option<impeller_wkt::TerrainConfig>,
            Option<impeller_bevy::DbEndpoint>,
            super::terrain_playback::TerrainPlayback,
        )>,
    >,
) {
    use super::terrain_playback::{TerrainMode, TerrainPlayback};
    let policy = world
        .get_resource::<TerrainPlayback>()
        .cloned()
        .unwrap_or_default();
    let retry_requested = world
        .get_resource_mut::<RetryTerrain>()
        .is_some_and(|mut retry| std::mem::take(&mut retry.0));
    let Some(schematic) = world.get_resource::<crate::ui::schematic::CurrentSchematic>() else {
        return;
    };
    let endpoint = world
        .get_resource::<impeller_bevy::ConnectionAddr>()
        .map(|connection| connection.endpoint());
    let retry_failed = retry_requested
        && schematic.terrain.is_some()
        && world
            .get_resource::<TerrainStartup>()
            .is_some_and(|startup| startup.error.is_some());
    if previous
        .as_ref()
        .is_some_and(|(old, old_endpoint, old_policy)| {
            old == &schematic.terrain && old_endpoint.as_ref() == endpoint && old_policy == &policy
        })
        && !retry_failed
    {
        return;
    }
    let terrain = schematic.terrain.clone();
    let endpoint = endpoint.cloned();
    *previous = Some((terrain.clone(), endpoint.clone(), policy.clone()));
    let Some(terrain) = terrain else {
        world.insert_resource(CesiumSourceUpdate(None));
        world.remove_resource::<TerrainStartup>();
        return;
    };
    let mut startup = TerrainStartup::waiting();
    let source = if !terrain.is_valid() {
        startup.error = Some("Invalid terrain source configuration");
        None
    } else {
        match terrain {
            impeller_wkt::TerrainConfig::Ion { .. }
                if policy.mode == Some(TerrainMode::Offline) =>
            {
                startup.error = Some(
                    "Offline mode requires prepared terrain: run elodin terrain prepare on the recording",
                );
                None
            }
            impeller_wkt::TerrainConfig::Ion { asset_id } => Some(terrain_source(asset_id)),
            impeller_wkt::TerrainConfig::Asset { path } => endpoint.as_ref().map(|endpoint| {
                let key = path.strip_prefix("db:").unwrap_or(&path);
                CesiumSource::Prepared {
                    url: endpoint.assets_url(key).to_string(),
                    ion_access_token: (policy.mode != Some(TerrainMode::Offline))
                        .then(|| ion_access_token().to_owned()),
                }
            }),
        }
    };
    let simulation_held = world
        .get_resource::<impeller_wkt::DbConfig>()
        .is_some_and(|c| {
            c.metadata
                .contains_key(impeller_wkt::EDITOR_STARTUP_GATE_METADATA_KEY)
        });
    if matches!(source, Some(CesiumSource::Prepared { .. }))
        && !simulation_held
        && !world.contains_resource::<crate::sensor_camera::HeadlessMode>()
    {
        startup.ready = true;
    }
    world.insert_resource(CesiumSourceUpdate(source));
    world.insert_resource(startup);
}

fn request_terrain_retry(
    mut loaded: MessageReader<crate::ui::schematic::DocumentLoaded>,
    mut reloaded: MessageReader<crate::ui::schematic::DocumentReloaded>,
    mut retry: ResMut<RetryTerrain>,
) {
    let loaded = loaded.read().count() > 0;
    let mut reloaded_full = false;
    for event in reloaded.read() {
        reloaded_full |= event.changed_window_indices.is_empty();
    }
    if loaded || reloaded_full {
        retry.0 = true;
    }
}

fn update_startup(
    status: Res<CesiumStatus>,
    mut startup: ResMut<TerrainStartup>,
    mut updates: Local<u32>,
    mut commands: Commands,
) {
    if startup.ready || startup.error.is_some() {
        return;
    }
    if (*updates).is_multiple_of(120) {
        debug!(
            pending = status.pending,
            preparing = status.pending_render_tiles,
            visible = status.visible_tiles,
            resident = status.resident_tiles,
            retained = status.retained_tiles,
            complete = status.selection_complete,
            stable = status.stable_frames,
            "Waiting for initial terrain"
        );
    }
    *updates = updates.wrapping_add(1);
    startup.update(&status);
    if startup.error.is_some() {
        commands.insert_resource(CesiumSourceUpdate(None));
    }
    if startup.ready {
        info!(
            seconds = startup.started.elapsed().as_secs_f64(),
            "Initial terrain ready"
        );
    }
}

fn show_startup(startup: Res<TerrainStartup>, mut contexts: Query<&mut bevy_egui::EguiContext>) {
    if startup.ready {
        return;
    }
    for mut context in &mut contexts {
        let context = context.get_mut();
        if let Some(error) = startup.error {
            egui::Window::new("Terrain unavailable")
                .collapsible(false)
                .resizable(false)
                .show(context, |ui| {
                    ui.label(error);
                    ui.label("Continuing without streamed terrain. Reload the schematic to retry.");
                });
        } else {
            egui::Modal::new(egui::Id::new("cesium_startup"))
                .backdrop_color(context.global_style().visuals.panel_fill.to_opaque())
                .show(context, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Loading terrain and aerial images…");
                    });
                });
        }
    }
}

#[allow(clippy::type_complexity)]
fn collect_views(
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    sensor_configs: Option<Res<SensorCameraConfigs>>,
    cameras: Query<
        (
            &Camera,
            &Projection,
            &Transform,
            Option<&GridCell>,
            Option<&ChildOf>,
            Option<&SensorCamera>,
        ),
        Or<(With<MainCamera>, With<SensorCamera>)>,
    >,
    parents: Query<(&Transform, &GridCell)>,
    mut output: ResMut<CesiumViews>,
) {
    let ecef_from_bevy = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo).inverse();
    let views: Vec<_> = cameras
        .iter()
        .filter_map(|(camera, projection, transform, cell, parent, sensor)| {
            // Headless scheduling disables sensors between emissions. Their terrain
            // requirements must survive those gaps and include every sensor.
            if !camera.is_active && sensor.is_none() {
                return None;
            }
            if let Some(configs) = sensor_configs.as_ref()
                && let Some(config) = sensor.and_then(|sensor| configs.0.get(sensor.config_index))
                && has_wider_sensor_view(config, &configs.0)
            {
                // Co-located, aligned sensors share the wider view's terrain.
                // Independent zoom refinement continuously changes the GPU-ready
                // selection during flight and can strand both streams on old tiles.
                // This trades zoomed terrain detail for stable preview coverage;
                // cameras at different poses still contribute their own views.
                return None;
            }
            let Projection::Perspective(p) = projection else {
                return None;
            };
            let size = camera.physical_viewport_size().or_else(|| {
                let config = sensor_configs.as_ref()?.0.get(sensor?.config_index)?;
                Some(UVec2::new(config.width, config.height))
            })?;
            if size.x == 0 || size.y == 0 {
                return None;
            }
            // Sensor cameras own their grid cell; Editor cameras inherit the
            // viewport anchor's cell and compose their local transform with it.
            let (transform, cell) = cell
                .map(|cell| (*transform, *cell))
                .or_else(|| {
                    parent
                        .and_then(|p| parents.get(p.parent()).ok())
                        .map(|(t, c)| (t.mul_transform(*transform), *c))
                })
                .unwrap_or((*transform, GridCell::default()));
            Some(CesiumView {
                position: camera_position_ecef(ecef_from_bevy, &grid, &cell, &transform).to_array(),
                direction: ecef_from_bevy
                    .transform_vector3(transform.forward().as_dvec3())
                    .normalize()
                    .to_array(),
                up: ecef_from_bevy
                    .transform_vector3(transform.up().as_dvec3())
                    .normalize()
                    .to_array(),
                width: f64::from(size.x),
                height: f64::from(size.y),
                vertical_fov: f64::from(p.fov),
                horizontal_fov: 2.0
                    * ((f64::from(p.fov) * 0.5).tan() * f64::from(size.x) / f64::from(size.y))
                        .atan(),
            })
        })
        .collect();
    output.0 = views;
}

fn has_wider_sensor_view(config: &SensorCameraConfig, configs: &[SensorCameraConfig]) -> bool {
    let horizontal_scale = |c: &SensorCameraConfig| {
        (f64::from(c.fov_degrees).to_radians() * 0.5).tan() * f64::from(c.width)
            / f64::from(c.height.max(1))
    };
    configs.iter().any(|other| {
        other.entity_name == config.entity_name
            && other.pos_offset == config.pos_offset
            && other.rot_offset == config.rot_offset
            && other.near <= config.near
            && other.far >= config.far
            && other.fov_degrees > config.fov_degrees
            && horizontal_scale(other) >= horizontal_scale(config)
    })
}

fn place_tiles(
    mut commands: Commands,
    root: Res<BigSpaceRootEntity>,
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    tiles: Query<(Entity, &CesiumPrimitive), Added<CesiumPrimitive>>,
) {
    for (entity, tile) in &tiles {
        let (scale, rotation, translation) = tile.ecef_from_mesh.to_scale_rotation_translation();
        // Streaming happens after PreUpdate's geographic synchronization. Set
        // the local pose now so the first visible frame is already georeferenced.
        let local = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo) * tile.ecef_from_mesh;
        let (_, local_rotation, local_position) = local.to_scale_rotation_translation();
        let (cell, local_translation) = grid.translation_to_grid(local_position);
        commands.entity(entity).insert((
            Transform {
                translation: local_translation,
                rotation: local_rotation.as_quat(),
                scale: scale.as_vec3(),
            },
            // Cinematic sunlight uses its own layer; streamed scenery must
            // receive it just like the existing object_3d meshes.
            bevy::camera::visibility::RenderLayers::default()
                .with(crate::plugins::render_layer_alloc::CINEMATIC_EARTH_RENDER_LAYER),
            GeoPosition(GeoFrame::ECEF, translation),
            GeoRotation::absolute(GeoFrame::ECEF, rotation),
            cell,
            ChildOf(root.0),
            // Mesh picking is opt-in in the editor. Without this marker, zoom
            // falls back to the aircraft anchor instead of the ground hit.
            Pickable::default(),
        ));
    }
}

fn log_streaming_status(status: Res<CesiumStatus>, time: Res<Time>, mut last: Local<f64>) {
    let now = time.elapsed_secs_f64();
    if now - *last < 10.0 {
        return;
    }
    *last = now;
    debug!(
        visible = status.visible_tiles,
        resident = status.resident_tiles,
        preparing = status.pending_render_tiles,
        pending = status.pending,
        complete = status.selection_complete,
        failed = status.load_failed,
        "Cesium streaming status"
    );
}
