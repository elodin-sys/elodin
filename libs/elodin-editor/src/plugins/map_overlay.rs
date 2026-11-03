//! Geographic place labels from the schematic, independent of terrain providers.
use crate::{
    MainCamera,
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
    ui::{inspector::viewport::Viewport, schematic::CurrentSchematic},
};
use bevy::{
    camera::{RenderTarget, visibility::RenderLayers},
    math::DVec3,
    picking::prelude::{MeshRayCast, MeshRayCastSettings},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoOrigin};

#[derive(Resource, Default)]
struct MapOverlay {
    config: Option<impeller_wkt::PlaceNamesConfig>,
    places: Vec<(String, DVec3)>,
}

#[derive(Resource, Default)]
pub(crate) struct OverlayVisibility(pub bool);

#[derive(Resource, Default)]
pub(crate) struct OverlayPresentation {
    pub colors: std::collections::HashMap<String, egui::Color32>,
    pub attribution: Option<(String, String)>,
}

#[derive(Component)]
pub(crate) struct OverlayOccluder {
    pub center: Vec3,
    pub half_extents: Vec3,
}

pub(crate) struct MapOverlayPlugin;
impl Plugin for MapOverlayPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<bevy_egui::EguiPlugin>() {
            return;
        }
        app.init_resource::<MapOverlay>()
            .init_resource::<OverlayVisibility>()
            .add_systems(
                Update,
                (
                    sync_names,
                    draw.before(crate::plugins::editor_cam_input::gated_camera_inputs),
                )
                    .chain()
                    .after(crate::ui::render_layout),
            );
    }
}

fn sync_names(
    schematic: Option<Res<CurrentSchematic>>,
    mut overlay: ResMut<MapOverlay>,
    mut visibility: ResMut<OverlayVisibility>,
) {
    let config = schematic.as_ref().and_then(|s| s.0.place_names.as_ref());
    if config == overlay.config.as_ref() {
        return;
    }
    if let Some(config) = config
        && let Err(error) = config.validate()
    {
        warn!(error, "Invalid schematic place names");
        return;
    }
    overlay.places = config.map_or_else(Vec::new, |c| {
        c.places
            .iter()
            .map(|p| {
                // Lift the label above the named ground point; use the existing frame primitives.
                let origin =
                    GeoOrigin::new_from_degrees(p.latitude, p.longitude, p.altitude + 12.0);
                (p.name.clone(), GeoFrame::origin_ecef(&origin))
            })
            .collect()
    });
    visibility.0 = config.is_some_and(|c| c.visible);
    overlay.config = config.cloned();
}

fn visible(
    ray_cast: &mut MeshRayCast,
    tiles: &Query<(), With<Mesh3d>>,
    origin: Vec3,
    point: Vec3,
) -> bool {
    let delta = point - origin;
    let Ok(direction) = Dir3::new(delta) else {
        return false;
    };
    let filter = |entity| tiles.contains(entity);
    let settings = MeshRayCastSettings::default()
        .with_filter(&filter)
        .always_early_exit();
    ray_cast
        .cast_ray(Ray3d::new(origin, direction), &settings)
        .first()
        .is_none_or(|(_, hit)| hit.distance >= delta.length() - 1.0)
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw(
    overlay: Res<MapOverlay>,
    mut visibility: ResMut<OverlayVisibility>,
    presentation: Option<Res<OverlayPresentation>>,
    occluders: Query<(&GlobalTransform, &OverlayOccluder)>,
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    root: Res<BigSpaceRootEntity>,
    globals: Query<&GlobalTransform>,
    parents: Query<(&Transform, &GridCell)>,
    cameras: Query<
        (
            Entity,
            &Camera,
            &RenderTarget,
            &Transform,
            &GlobalTransform,
            &ChildOf,
            &crate::ui::ViewportRect,
            &RenderLayers,
        ),
        (With<MainCamera>, With<Viewport>),
    >,
    windows: Query<Entity, With<PrimaryWindow>>,
    mut contexts: Query<(&crate::ui::tiles::WindowId, &mut bevy_egui::EguiContext)>,
    tiles: Query<(), With<Mesh3d>>,
    mut ray_cast: MeshRayCast,
) {
    if overlay.places.is_empty() {
        return;
    }
    let Ok(primary) = windows.single() else {
        return;
    };
    let Ok(root_transform) = globals.get(root.0) else {
        return;
    };
    let matrix = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo);
    for (window, mut context) in &mut contexts {
        if !window.is_primary() {
            continue;
        }
        let ctx = context.get_mut();
        for (entity, camera, target, local, global, parent, panel_rect, layers) in &cameras {
            if !layers.intersects(&RenderLayers::layer(
                crate::plugins::render_layer_alloc::SCENE_LINE_RENDER_LAYER,
            )) {
                continue;
            }
            if panel_rect.0.is_none() {
                continue;
            }
            if !camera.is_active
                || crate::ui::window::window_entity_from_target(target, primary) != Some(primary)
            {
                continue;
            }
            let Some(viewport) = camera.logical_viewport_rect() else {
                continue;
            };
            let rect = egui::Rect::from_min_max(
                egui::pos2(viewport.min.x, viewport.min.y),
                egui::pos2(viewport.max.x, viewport.max.y),
            );
            if rect.width() < 50.0 || rect.height() < 50.0 {
                continue;
            }
            let Ok((anchor, cell)) = parents.get(parent.parent()) else {
                continue;
            };
            let camera_position = grid.grid_position_double(cell, &anchor.mul_transform(*local));
            let view_from_root = global.affine().inverse() * root_transform.affine();
            let clip_from_view = camera.clip_from_view();
            let world = |ecef: DVec3| {
                global.translation()
                    + root_transform.affine().transform_vector3(
                        (matrix.transform_point3(ecef) - camera_position).as_vec3(),
                    )
            };
            let project = |ecef: DVec3| {
                let delta = matrix.transform_point3(ecef) - camera_position;
                let clip = clip_from_view
                    * view_from_root
                        .transform_vector3(delta.as_vec3())
                        .extend(1.0);
                if clip.w <= 0.0 {
                    return None;
                }
                let ndc = clip.truncate() / clip.w;
                if !ndc.is_finite() || !(0.0..=1.0).contains(&ndc.z) {
                    return None;
                }
                Some(
                    rect.min
                        + egui::vec2(
                            (ndc.x + 1.0) * 0.5 * rect.width(),
                            (1.0 - ndc.y) * 0.5 * rect.height(),
                        ),
                )
            };
            let painter = ctx
                .layer_painter(egui::LayerId::new(
                    egui::Order::Middle,
                    egui::Id::new(("place_names", entity)),
                ))
                .with_clip_rect(rect);
            egui::Area::new(egui::Id::new(("place_names_toggle", entity)))
                .fixed_pos(rect.min + egui::vec2(8.0, 8.0))
                .show(ctx, |ui| {
                    let label = if visibility.0 {
                        "Overlay on"
                    } else {
                        "Overlay off"
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(label).font(egui::FontId::proportional(13.0)),
                            )
                            .wrap_mode(egui::TextWrapMode::Extend)
                            .min_size(egui::vec2(94.0, 26.0))
                            .fill(egui::Color32::from_black_alpha(65)),
                        )
                        .on_hover_ui(|ui| {
                            ui.label("Show or hide map overlays");
                            if let Some(presentation) = presentation.as_deref()
                                && let Some((label, url)) = &presentation.attribution
                            {
                                ui.hyperlink_to(label, url);
                            }
                        })
                        .clicked()
                    {
                        visibility.0 = !visibility.0;
                    }
                });
            if !visibility.0 {
                continue;
            }
            let mut occupied: Vec<egui::Rect> = Vec::new();
            // Skinned meshes are not reliably represented by CPU mesh raycasts.
            // Reserve their projected bounds so labels never cover the vehicle.
            for (transform, bounds) in &occluders {
                let mut projected = Vec::new();
                for x in [-1.0, 1.0] {
                    for y in [-1.0, 1.0] {
                        for z in [-1.0, 1.0] {
                            let local = bounds.center + bounds.half_extents * Vec3::new(x, y, z);
                            if let Ok(p) = camera.world_to_viewport(
                                global,
                                transform.affine().transform_point3(local),
                            ) {
                                projected.push(egui::pos2(p.x, p.y));
                            }
                        }
                    }
                }
                if let Some(bounds) = projected
                    .into_iter()
                    .map(|p| egui::Rect::from_min_max(p, p))
                    .reduce(|a, b| a.union(b))
                {
                    occupied.push(bounds.expand(6.0));
                }
            }
            let mut candidates: Vec<_> = overlay
                .places
                .iter()
                .map(|(name, ecef)| {
                    (
                        name,
                        *ecef,
                        (matrix.transform_point3(*ecef) - camera_position).length_squared(),
                    )
                })
                .collect();
            candidates.sort_by(|a, b| a.2.total_cmp(&b.2));
            let mut rays = 0;
            for (name, ecef, _) in candidates {
                let Some(point) = project(ecef) else { continue };
                if !rect.shrink(12.0).contains(point) {
                    continue;
                }
                let color = presentation
                    .as_ref()
                    .and_then(|p| p.colors.get(name))
                    .copied()
                    .unwrap_or(egui::Color32::from_rgb(231, 235, 230));
                let galley =
                    painter.layout_no_wrap(name.clone(), egui::FontId::proportional(13.0), color);
                let label = egui::Rect::from_min_size(point + egui::vec2(9.0, -8.0), galley.size())
                    .expand(4.0);
                if !rect.contains_rect(label) || occupied.iter().any(|r| r.intersects(label)) {
                    continue;
                }
                if rays >= 12 {
                    break;
                }
                rays += 1;
                if !visible(&mut ray_cast, &tiles, global.translation(), world(ecef)) {
                    continue;
                }
                occupied.push(label);
                painter.circle_filled(point, 2.5, color);
                painter.rect_filled(label, 3.0, egui::Color32::from_black_alpha(150));
                painter.galley(label.min + egui::vec2(4.0, 4.0), galley, color);
            }
        }
    }
}
