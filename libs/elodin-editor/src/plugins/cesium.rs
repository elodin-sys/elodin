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
        let source = if let Ok(asset) = std::env::var("CESIUM_ION_ASSET_ID") {
            let Ok(asset_id) = asset.parse::<i64>() else {
                error!("CESIUM_ION_ASSET_ID must be a positive integer");
                return;
            };
            let Ok(access_token) = std::env::var("CESIUM_ION_TOKEN") else {
                error!("Set CESIUM_ION_TOKEN to load a Cesium ion asset");
                return;
            };
            if asset_id <= 0 || access_token.is_empty() {
                error!("Invalid Cesium ion configuration");
                return;
            }
            CesiumSource::Ion {
                asset_id,
                access_token,
            }
        } else if let Ok(url) = std::env::var("CESIUM_TILESET_URL") {
            CesiumSource::Url(url)
        } else {
            return;
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
    tiles: Query<(Entity, &CesiumPrimitive), Added<CesiumPrimitive>>,
) {
    for (entity, tile) in &tiles {
        let (scale, rotation, translation) = tile.ecef_from_mesh.to_scale_rotation_translation();
        commands.entity(entity).insert((
            Transform::from_scale(scale.as_vec3()),
            GeoPosition(GeoFrame::ECEF, translation),
            GeoRotation::absolute(GeoFrame::ECEF, rotation),
            GridCell::default(),
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
