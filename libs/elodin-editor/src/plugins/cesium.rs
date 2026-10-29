//! Editor adapter for the independently maintained bevy-cesium plugin.
mod map_overlay;
use crate::{
    MainCamera,
    sensor_camera::{SensorCamera, SensorCameraConfig, SensorCameraConfigs},
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
};
use bevy::{
    math::DMat4, picking::Pickable, prelude::*, render::render_asset::RenderAssetBytesPerFrame,
};
use bevy_cesium::{
    CesiumDetailViews, CesiumPlugin as StreamingPlugin, CesiumPrimitive, CesiumSelectionMode,
    CesiumSet, CesiumSource, CesiumStatus, CesiumView, CesiumViews,
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

pub struct CesiumPlugin;

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
    started: std::time::Instant,
    timeout: std::time::Duration,
}
impl TerrainCapture {
    pub(crate) fn ready(&self, status: Option<&CesiumStatus>) -> Result<bool, &'static str> {
        if status.is_some_and(|s| s.load_failed) {
            return Err("Terrain capture failed to load");
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
struct TerrainStartup {
    gate: TerrainCapture,
    ready: bool,
    error: Option<&'static str>,
}

impl TerrainStartup {
    fn update(&mut self, status: &CesiumStatus) {
        if self.ready || self.error.is_some() {
            return;
        }
        match self.gate.ready(Some(status)) {
            Ok(ready) => self.ready = ready,
            Err(error) => self.error = Some(error),
        }
    }

    fn result(&self) -> Result<bool, &'static str> {
        self.error.map_or(Ok(self.ready), Err)
    }
}

pub(crate) fn capture_ready(world: &World) -> Result<bool, &'static str> {
    if let Some(startup) = world.get_resource::<TerrainStartup>()
        && !startup.result()?
    {
        return Ok(false);
    }
    world
        .get_resource::<TerrainCapture>()
        .map_or(Ok(true), |capture| {
            capture.ready(world.get_resource::<CesiumStatus>())
        })
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
    fn initial_terrain_wait_is_bounded_and_does_not_restart_during_flight() {
        let mut startup = TerrainStartup {
            gate: TerrainCapture {
                started: std::time::Instant::now(),
                timeout: std::time::Duration::from_secs(120),
            },
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
        startup.ready = false;
        startup.gate.timeout = std::time::Duration::ZERO;
        startup.update(&CesiumStatus::default());
        assert!(startup.result().is_err());
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
            gate: TerrainCapture {
                started: std::time::Instant::now(),
                timeout: std::time::Duration::from_secs(120),
            },
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
            started: std::time::Instant::now(),
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
        let source = match CesiumSource::from_env() {
            Ok(Some(source)) => source,
            Ok(None) => {
                assert!(
                    std::env::var_os("ELODIN_CESIUM_CAPTURE_VIEWS").is_none(),
                    "Fixed terrain capture requires a Cesium source"
                );
                return;
            }
            Err(error) => {
                assert!(
                    std::env::var_os("ELODIN_CESIUM_CAPTURE_VIEWS").is_none(),
                    "Fixed terrain capture requires a valid Cesium source"
                );
                error!("{error}");
                return;
            }
        };
        // Share Bevy's existing upload budget across meshes and images. A burst
        // of streamed tiles should not monopolize the sensor/render frame.
        // The limit is soft (one large asset may exceed it). Respect a budget
        // already chosen by the host, and leave non-Cesium sessions unchanged.
        app.init_resource::<RenderAssetBytesPerFrame>();
        app.world_mut()
            .resource_mut::<RenderAssetBytesPerFrame>()
            .max_bytes
            .get_or_insert(8 * 1024 * 1024);
        let margin = env_positive("CESIUM_LOD_RETENTION_METERS", 500.0, true);
        let screen_error = env_positive("CESIUM_MAXIMUM_SCREEN_ERROR", 16.0, false);
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
                    started: std::time::Instant::now(),
                    timeout: std::time::Duration::from_secs_f64(env_positive(
                        "ELODIN_CESIUM_CAPTURE_TIMEOUT",
                        120.0,
                        false,
                    )),
                });
        }
        info!(
            margin_meters = margin,
            screen_error_pixels = screen_error,
            "Cesium terrain selection configured"
        );
        app.add_plugins(
            StreamingPlugin::new(source)
                .with_external_coordinates()
                .with_initial_load_wait()
                .with_lod_retention_margin(margin)
                .with_maximum_screen_error(screen_error),
        )
        .insert_resource(TerrainStartup {
            gate: TerrainCapture {
                started: std::time::Instant::now(),
                timeout: std::time::Duration::from_secs_f64(env_positive(
                    "ELODIN_CESIUM_CAPTURE_TIMEOUT",
                    120.0,
                    false,
                )),
            },
            ready: false,
            error: None,
        })
        .add_systems(Update, collect_views.in_set(CesiumSet::Views))
        .add_systems(Update, place_tiles.in_set(CesiumSet::Placement))
        .add_systems(Update, update_startup.after(CesiumSet::Publish))
        .add_systems(
            Update,
            show_startup
                .after(update_startup)
                .after(crate::ui::render_layout),
        );
        app.add_plugins(map_overlay::MapOverlayPlugin);
        app.add_systems(Update, log_streaming_status.after(CesiumSet::Publish));
        if std::env::var("ELODIN_RC_JET_TERRAIN").as_deref() == Ok("cesium") {
            app.add_systems(Update, style_rc_jet_terrain.after(CesiumSet::Placement));
        }
    }
}

fn style_rc_jet_terrain(
    tiles: Query<&MeshMaterial3d<StandardMaterial>, Added<CesiumPrimitive>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for handle in &tiles {
        if let Some(mut material) = materials.get_mut(&handle.0)
            && material.base_color_texture.is_none()
        {
            // Display tint for bare terrain geometry, not satellite imagery.
            material.base_color = Color::srgb(0.58, 0.44, 0.27);
            material.perceptual_roughness = 1.0;
            material.metallic = 0.0;
        }
    }
}

fn update_startup(
    status: Res<CesiumStatus>,
    mut startup: ResMut<TerrainStartup>,
    mut updates: Local<u32>,
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
    if startup.ready {
        info!(
            seconds = startup.gate.started.elapsed().as_secs_f64(),
            "Initial terrain ready"
        );
    }
}

fn show_startup(startup: Res<TerrainStartup>, mut contexts: Query<&mut bevy_egui::EguiContext>) {
    if startup.ready {
        return;
    }
    for mut context in &mut contexts {
        egui::Window::new("Terrain")
            .id(egui::Id::new("cesium_startup"))
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .show(context.get_mut(), |ui| {
                if startup.error.is_some() {
                    ui.label("Terrain could not finish loading.");
                    ui.label("Check the connection and data access, then restart.");
                } else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Loading terrain and aerial images…");
                    });
                }
            });
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
