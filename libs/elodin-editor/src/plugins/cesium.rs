//! Editor adapter for the independently maintained bevy-cesium plugin.
use crate::{
    MainCamera,
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
};
use bevy::{math::DMat4, prelude::*};
use bevy_cesium::{
    CesiumCredits, CesiumPlugin as StreamingPlugin, CesiumPrimitive, CesiumSet, CesiumSource,
    CesiumStatus, CesiumView, CesiumViews,
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

pub struct CesiumPlugin;
#[derive(Resource)]
struct SourceLabel(String);

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
            Ok(None) => return,
            Err(error) => {
                error!("{error}");
                return;
            }
        };
        app.add_plugins(StreamingPlugin::new(source).with_external_coordinates())
            .insert_resource(SourceLabel(
                std::env::var("CESIUM_CREDIT").unwrap_or_default(),
            ))
            .add_systems(Update, collect_views.in_set(CesiumSet::Views))
            .add_systems(Update, place_tiles.in_set(CesiumSet::Placement))
            .add_systems(Update, show_credits.after(crate::ui::render_layout));
    }
}

#[allow(clippy::type_complexity)]
fn collect_views(
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    cameras: Query<(&Camera, &Projection, &Transform, Option<&ChildOf>), With<MainCamera>>,
    parents: Query<(&Transform, &GridCell), Without<MainCamera>>,
    mut output: ResMut<CesiumViews>,
) {
    let ecef_from_bevy = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo).inverse();
    let views: Vec<_> = cameras
        .iter()
        .filter_map(|(camera, projection, transform, parent)| {
            if !camera.is_active {
                return None;
            }
            let Projection::Perspective(p) = projection else {
                return None;
            };
            let size = camera.physical_viewport_size()?;
            if size.x == 0 || size.y == 0 {
                return None;
            }
            let (transform, cell) = parent
                .and_then(|p| parents.get(p.parent()).ok())
                .map(|(t, c)| (t.mul_transform(*transform), *c))
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
        ));
    }
}

fn show_credits(
    mut contexts: Query<(&crate::ui::tiles::WindowId, &mut bevy_egui::EguiContext)>,
    credits: Res<CesiumCredits>,
    status: Res<CesiumStatus>,
    label: Res<SourceLabel>,
) {
    // The Editor draws each window in Update, using its own EguiContext.
    for (window, mut context) in &mut contexts {
        egui::Area::new(egui::Id::new("cesium_credits"))
            .anchor(
                egui::Align2::RIGHT_BOTTOM,
                egui::vec2(-12.0, if window.is_primary() { -112.0 } else { -12.0 }),
            )
            .show(context.get_mut(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.hyperlink_to("Cesium", "https://cesium.com/");
                        if !label.0.is_empty() {
                            ui.add_space(6.0);
                            ui.label(&label.0);
                        }
                    });
                    if status.load_failed {
                        ui.colored_label(
                            egui::Color32::LIGHT_RED,
                            "Tileset unavailable. Check URL, asset access and token.",
                        );
                    }
                    for credit in &credits.html {
                        ui.label(credit);
                    }
                });
            });
    }
}
