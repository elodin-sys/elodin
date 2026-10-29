//! Offline OSM demonstration. Roads use the normal depth-tested 3D pipeline;
//! screen-space labels and hover cards are shown only after a terrain ray test.
use crate::{
    MainCamera,
    object_3d::EditableEQL,
    plugins::render_layer_alloc::RenderLayerAllocator,
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
    ui::inspector::viewport::Viewport,
};
use bevy::ecs::system::SystemParam;
use bevy::gizmos::{
    config::{GizmoConfigGroup, GizmoConfigStore},
    gizmos::Gizmos,
};
use bevy::{
    camera::{RenderTarget, visibility::RenderLayers},
    math::{DMat4, DQuat, DVec3},
    picking::prelude::{MeshRayCast, MeshRayCastSettings},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_editor_cam::prelude::{EditorCam, OrbitConstraint};

#[derive(Default, Reflect, GizmoConfigGroup)]
struct MapRoadLines;
#[derive(Default, Reflect, GizmoConfigGroup)]
struct MapBuildingLines;
#[derive(Default, Reflect, GizmoConfigGroup)]
struct MapWaterLines;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

#[derive(serde::Deserialize)]
struct Feature {
    osm_id: u64,
    osm_type: String,
    kind: String,
    name: String,
    tags: BTreeMap<String, String>,
    parts: Vec<Vec<[f64; 3]>>,
    #[serde(default)]
    outline_height_m: Option<f64>,
    #[serde(default)]
    outline_height_source: Option<String>,
    #[serde(skip)]
    center: DVec3,
    #[serde(skip)]
    radius: f64,
}
struct FollowBackup {
    pos: EditableEQL,
}
#[derive(Resource, serde::Deserialize)]
struct MapOverlay {
    version: u32,
    attribution: String,
    copyright_url: String,
    features: Vec<Feature>,
    #[serde(skip)]
    enabled: [bool; 5],
    #[serde(skip)]
    shown: bool,
    #[serde(skip)]
    road_entity: Option<Entity>,
    #[serde(skip)]
    road_layer: Option<RenderLayers>,
    #[serde(skip)]
    follow: HashMap<Entity, FollowBackup>,
    #[serde(skip)]
    road_lines: Vec<(Vec3, Vec3)>,
    #[serde(skip)]
    building_lines: Vec<(Vec3, Vec3)>,
    #[serde(skip)]
    water_lines: Vec<(Vec3, Vec3)>,
}
pub(super) struct MapOverlayPlugin;
impl Plugin for MapOverlayPlugin {
    fn build(&self, app: &mut App) {
        // Overlays belong to the Editor, never to recorded sensor images.
        if !app.is_plugin_added::<bevy_egui::EguiPlugin>() {
            return;
        }
        let Ok(path) = std::env::var("ELODIN_CESIUM_OVERLAY_FILE") else {
            return;
        };
        match load(Path::new(&path)) {
            Ok(mut overlay) => {
                overlay.enabled = [true; 5];
                info!(
                    features = overlay.features.len(),
                    "Offline map overlay loaded"
                );
                app.init_gizmo_group::<MapRoadLines>()
                    .init_gizmo_group::<MapBuildingLines>()
                    .init_gizmo_group::<MapWaterLines>()
                    .insert_resource(overlay)
                    .add_systems(
                        PostUpdate,
                        draw_map_lines.after(bevy::transform::TransformSystems::Propagate),
                    )
                    .add_systems(
                        PreUpdate,
                        restore_follow_on_cube_reset
                            .before(crate::ui::inspector::viewport::set_viewport_pos),
                    )
                    .add_systems(
                        Update,
                        (
                            prepare_roads,
                            draw.before(crate::plugins::editor_cam_input::gated_camera_inputs),
                        )
                            .chain()
                            .after(crate::ui::render_layout),
                    );
            }
            Err(error) => warn!(%error,"Map overlay disabled"),
        }
    }
}
fn load(path: &Path) -> Result<MapOverlay, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() > 16_000_000 {
        return Err("Map overlay exceeds 16 MB".into());
    }
    let mut overlay: MapOverlay = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if overlay.version != 1
        || overlay.features.len() > 4000
        || overlay
            .features
            .iter()
            .flat_map(|f| &f.parts)
            .map(Vec::len)
            .sum::<usize>()
            > 250000
        || overlay
            .features
            .iter()
            .flat_map(|f| &f.parts)
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err("Invalid map geometry or version".into());
    }
    for f in &mut overlay.features {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for p in f.parts.iter().flatten() {
            let p = DVec3::from_array(*p);
            min = min.min(p);
            max = max.max(p);
        }
        if min.is_finite() {
            f.center = (min + max) * 0.5;
            f.radius = (max - min).length() * 0.5;
        }
    }
    Ok(overlay)
}
#[allow(clippy::too_many_arguments)]
fn prepare_roads(
    mut commands: Commands,
    mut overlay: ResMut<MapOverlay>,
    root: Res<BigSpaceRootEntity>,
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    mut allocator: ResMut<RenderLayerAllocator>,
    mut configs: ResMut<GizmoConfigStore>,
    cameras: Query<(Entity, &RenderLayers), With<Viewport>>,
) {
    if overlay.road_entity.is_none() {
        let Some(base) = overlay
            .features
            .iter()
            .filter(|f| matches!(f.kind.as_str(), "road" | "building" | "water"))
            .flat_map(|f| &f.parts)
            .flatten()
            .next()
            .copied()
        else {
            return;
        };
        let base = DVec3::from_array(base);
        let segments = |kind: &str| {
            overlay
                .features
                .iter()
                .filter(|f| f.kind == kind)
                .flat_map(|f| &f.parts)
                .flat_map(|part| part.windows(2))
                .map(|p| {
                    (
                        (DVec3::from_array(p[0]) - base).as_vec3(),
                        (DVec3::from_array(p[1]) - base).as_vec3(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let roads = segments("road");
        let buildings = segments("building");
        let water = segments("water");
        overlay.road_lines = roads;
        overlay.building_lines = buildings;
        overlay.water_lines = water;
        let Some(lease) = allocator.alloc() else {
            return;
        };
        let layers = lease.render_layers();
        let (config, _) = configs.config_mut::<MapRoadLines>();
        config.line.width = 2.2;
        config.line.perspective = false;
        config.depth_bias = 0.0;
        config.render_layers = layers.clone();
        let (config, _) = configs.config_mut::<MapBuildingLines>();
        config.line.width = 2.0;
        config.line.perspective = false;
        config.depth_bias = 0.0;
        config.render_layers = layers.clone();
        let (config, _) = configs.config_mut::<MapWaterLines>();
        config.line.width = 2.2;
        config.line.perspective = false;
        config.depth_bias = 0.0;
        config.render_layers = layers.clone();
        let local = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo) * DMat4::from_translation(base);
        let (_, rotation, position) = local.to_scale_rotation_translation();
        let (cell, translation) = grid.translation_to_grid(position);
        let entity = commands
            .spawn((
                Name::new("OSM map anchor"),
                Transform {
                    translation,
                    rotation: rotation.as_quat(),
                    ..default()
                },
                GeoPosition(GeoFrame::ECEF, base),
                GeoRotation::absolute(GeoFrame::ECEF, DQuat::IDENTITY),
                cell,
                ChildOf(root.0),
                lease,
                layers.clone(),
                Visibility::Visible,
            ))
            .id();
        overlay.road_entity = Some(entity);
        overlay.road_layer = Some(layers);
    }
    if let Some(layer) = &overlay.road_layer {
        for (entity, layers) in &cameras {
            if !layers.intersects(layer) {
                commands.entity(entity).insert(layers.union(layer));
            }
        }
    }
}

fn draw_map_lines(
    overlay: Res<MapOverlay>,
    anchors: Query<&GlobalTransform>,
    mut roads: Gizmos<MapRoadLines>,
    mut buildings: Gizmos<MapBuildingLines>,
    mut water: Gizmos<MapWaterLines>,
) {
    if !overlay.shown {
        return;
    }
    let Some(anchor) = overlay.road_entity.and_then(|e| anchors.get(e).ok()) else {
        return;
    };
    let transform = anchor.affine();
    if overlay.enabled[2] {
        for (a, b) in &overlay.road_lines {
            roads.line(
                transform.transform_point3(*a),
                transform.transform_point3(*b),
                Color::srgba(0.96, 0.51, 0.71, 0.65),
            );
        }
    }
    if overlay.enabled[3] {
        for (a, b) in &overlay.building_lines {
            buildings.line(
                transform.transform_point3(*a),
                transform.transform_point3(*b),
                Color::srgba(0.97, 0.59, 0.77, 0.72),
            );
        }
    }
    if overlay.enabled[4] {
        for (a, b) in &overlay.water_lines {
            water.line(
                transform.transform_point3(*a),
                transform.transform_point3(*b),
                Color::srgba(0.45, 0.80, 0.96, 0.72),
            );
        }
    }
}

#[derive(SystemParam)]
struct MapNavigation<'w, 's> {
    viewports: Query<'w, 's, &'static mut Viewport>,
    controls: Query<'w, 's, &'static mut EditorCam>,
}

fn segment_t(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    ((p - a).dot(b - a) / (b - a).length_sq().max(1e-8)).clamp(0.0, 1.0)
}
fn visible(
    ray_cast: &mut MeshRayCast,
    tiles: &Query<(), With<bevy_cesium::CesiumPrimitive>>,
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
    mut overlay: ResMut<MapOverlay>,
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
        ),
        (With<MainCamera>, With<Viewport>),
    >,
    mut navigation: MapNavigation,
    mut commands: Commands,
    windows: Query<Entity, With<PrimaryWindow>>,
    mut contexts: Query<(&crate::ui::tiles::WindowId, &mut bevy_egui::EguiContext)>,
    tiles: Query<(), With<bevy_cesium::CesiumPrimitive>>,
    mut ray_cast: MeshRayCast,
) {
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
        for (entity, camera, target, local, global, parent, panel_rect) in &cameras {
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
            let pointer = ctx.pointer_hover_pos().filter(|p| rect.contains(*p));
            let painter = ctx
                .layer_painter(egui::LayerId::new(
                    egui::Order::Middle,
                    egui::Id::new(("map_overlay", entity)),
                ))
                .with_clip_rect(rect);
            let mut hovered = None;
            let mut nearest = 6.0_f32;
            let mut tested_segments = 0;
            if overlay.shown
                && (overlay.enabled[2] || overlay.enabled[3] || overlay.enabled[4])
                && let Some(pointer) = pointer
            {
                'roads: for (index, f) in overlay.features.iter().enumerate().filter(|(_, f)| {
                    (f.kind == "road" && overlay.enabled[2])
                        || (f.kind == "building" && overlay.enabled[3])
                        || (f.kind == "water" && overlay.enabled[4])
                }) {
                    // Do not project distant context highways merely to service a local hover.
                    let center = matrix.transform_point3(f.center);
                    if (center - camera_position).length() > 20000.0 + f.radius {
                        continue;
                    }
                    for part in &f.parts {
                        for pair in part.windows(2) {
                            let a3 = DVec3::from_array(pair[0]);
                            let b3 = DVec3::from_array(pair[1]);
                            let (Some(a), Some(b)) = (project(a3), project(b3)) else {
                                continue;
                            };
                            if !rect.intersects(egui::Rect::from_two_pos(a, b).expand(2.0)) {
                                continue;
                            }
                            let t = segment_t(pointer, a, b);
                            let distance = pointer.distance(a + (b - a) * t);
                            if distance < nearest {
                                nearest = distance;
                                hovered = Some((index, world(a3.lerp(b3, f64::from(t)))));
                            }
                            tested_segments += 1;
                            if tested_segments >= 5000 {
                                break 'roads;
                            }
                        }
                    }
                }
            }
            if hovered.is_some_and(|(_, point)| {
                !visible(&mut ray_cast, &tiles, global.translation(), point)
            }) {
                hovered = None;
            }
            let mut occupied: Vec<egui::Rect> = Vec::new();
            let mut candidates: Vec<_> = overlay
                .features
                .iter()
                .enumerate()
                .filter(|(_, f)| overlay.shown && (f.kind == "peak" || f.kind == "place"))
                .map(|(i, f)| {
                    (
                        i,
                        (matrix.transform_point3(f.center) - camera_position).length_squared(),
                    )
                })
                .collect();
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut rays = 0;
            for (index, _) in candidates {
                let feature = &overlay.features[index];
                let peak = feature.kind == "peak";
                if !overlay.enabled[if peak { 0 } else { 1 }] {
                    continue;
                }
                let Some(ecef) = feature
                    .parts
                    .first()
                    .and_then(|p| p.first())
                    .map(|p| DVec3::from_array(*p))
                else {
                    continue;
                };
                let Some(point) = project(ecef) else { continue };
                if !rect.shrink(12.0).contains(point) {
                    continue;
                }
                let text = if peak {
                    feature.tags.get("ele").map_or_else(
                        || feature.name.clone(),
                        |e| format!("{} · {} m", feature.name, e),
                    )
                } else {
                    feature.name.clone()
                };
                let color = if peak {
                    egui::Color32::from_rgb(248, 213, 155)
                } else {
                    egui::Color32::from_rgb(231, 235, 230)
                };
                let galley = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), color);
                let label_rect =
                    egui::Rect::from_min_size(point + egui::vec2(9.0, -8.0), galley.size())
                        .expand(4.0);
                if !rect.contains_rect(label_rect)
                    || occupied.iter().any(|r| r.intersects(label_rect))
                {
                    continue;
                }
                if rays >= 12 {
                    break;
                }
                rays += 1;
                if !visible(&mut ray_cast, &tiles, global.translation(), world(ecef)) {
                    continue;
                }
                occupied.push(label_rect);
                painter.circle_filled(point, if peak { 3.5 } else { 2.5 }, color);
                painter.rect_filled(label_rect, 3.0, egui::Color32::from_black_alpha(150));
                painter.galley(label_rect.min + egui::vec2(4.0, 4.0), galley, color);
                if pointer.is_some_and(|p| label_rect.expand(5.0).contains(p)) {
                    hovered = Some((index, world(ecef)));
                }
            }
            egui::Area::new(egui::Id::new(("map_layers", entity)))
                .fixed_pos(rect.min + egui::vec2(8.0, 8.0))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_black_alpha(65))
                        .inner_margin(egui::Margin::same(4))
                        .corner_radius(egui::CornerRadius::same(5))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let label = if overlay.shown {
                                    "Overlay on"
                                } else {
                                    "Overlay off"
                                };
                                let fill = if overlay.shown {
                                    egui::Color32::from_rgba_unmultiplied(65, 95, 100, 110)
                                } else {
                                    egui::Color32::from_black_alpha(45)
                                };
                                if ui
                                    .add(egui::Button::new(label).fill(fill))
                                    .on_hover_ui(|ui| {
                                        ui.label("Show or hide map overlays");
                                        ui.hyperlink_to(
                                            &overlay.attribution,
                                            &overlay.copyright_url,
                                        );
                                    })
                                    .clicked()
                                {
                                    overlay.shown = !overlay.shown;
                                }
                                if !overlay.follow.contains_key(&entity) {
                                    ui.small("Following RC-jet");
                                    if ui.button("Explore terrain").clicked()
                                        && let Ok(mut viewport) =
                                            navigation.viewports.get_mut(entity)
                                    {
                                        overlay.follow.insert(
                                            entity,
                                            FollowBackup {
                                                pos: std::mem::take(&mut viewport.pos),
                                            },
                                        );
                                        if let (Ok(mut control), Ok(parent_global)) = (
                                            navigation.controls.get_mut(entity),
                                            globals.get(parent.parent()),
                                        ) {
                                            control.end_move();
                                            let up = parent_global
                                                .affine()
                                                .inverse()
                                                .transform_vector3(
                                                    root_transform
                                                        .affine()
                                                        .transform_vector3(Vec3::Y),
                                                )
                                                .normalize()
                                                .as_dvec3();
                                            control.orbit_constraint = OrbitConstraint::Fixed {
                                                up,
                                                can_pass_tdc: false,
                                            };
                                        }
                                    }
                                }
                                if ui.button("Recenter RC-jet").clicked()
                                    && let Ok(mut viewport) = navigation.viewports.get_mut(entity)
                                {
                                    if let Some(backup) = overlay.follow.remove(&entity) {
                                        viewport.pos = backup.pos;
                                    }
                                    commands.entity(entity).insert(Transform::IDENTITY);
                                    if let Ok(mut control) = navigation.controls.get_mut(entity) {
                                        control.end_move();
                                        control.orbit_constraint = OrbitConstraint::Fixed {
                                            up: DVec3::Y,
                                            can_pass_tdc: false,
                                        };
                                    }
                                }
                            });
                        });
                });
            if let (Some((index, _)), Some(pointer)) = (hovered, pointer) {
                let feature = &overlay.features[index];
                egui::Area::new(egui::Id::new(("map_info", entity)))
                    .order(egui::Order::Tooltip)
                    .fixed_pos(pointer + egui::vec2(14.0, 18.0))
                    .interactable(false)
                    .show(ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_max_width(260.0);
                            ui.label(&feature.name);
                            for (key, value) in &feature.tags {
                                if key != "name" {
                                    ui.small(format!("{key}: {value}"));
                                }
                            }
                            if let Some(height) = feature.outline_height_m {
                                let note = if feature.outline_height_source.as_deref()
                                    == Some("illustrative_default")
                                {
                                    "illustrative"
                                } else {
                                    "OSM-derived"
                                };
                                ui.small(format!("Outline: {height:.1} m ({note})"));
                            }
                            ui.small(format!(
                                "OpenStreetMap {}/{}",
                                feature.osm_type, feature.osm_id
                            ));
                        });
                    });
            }
        }
    }
}

// Resetting the child camera alone returns to the frozen free-camera anchor.
// Restore the live follow expression before the next pose update as well.
fn restore_follow_on_cube_reset(
    mut events: MessageReader<crate::plugins::view_cube::ViewCubeEvent>,
    links: Query<&crate::plugins::view_cube::ViewCubeLink>,
    mut overlay: ResMut<MapOverlay>,
    mut viewports: Query<&mut Viewport>,
    mut controls: Query<&mut EditorCam>,
    mut commands: Commands,
) {
    use crate::plugins::view_cube::{ViewCubeEvent, ViewportActionButton};
    for event in events.read() {
        let ViewCubeEvent::ViewportActionClicked {
            action: ViewportActionButton::Reset,
            source,
        } = event
        else {
            continue;
        };
        let Ok(link) = links.get(*source) else {
            continue;
        };
        let Some(backup) = overlay.follow.remove(&link.main_camera) else {
            continue;
        };
        if let Ok(mut viewport) = viewports.get_mut(link.main_camera) {
            viewport.pos = backup.pos;
        }
        if let Ok(mut control) = controls.get_mut(link.main_camera) {
            control.current_motion = bevy_editor_cam::controller::motion::CurrentMotion::Stationary;
            control.last_anchor_depth = -2.0;
            control.orbit_constraint = OrbitConstraint::Fixed {
                up: DVec3::Y,
                can_pass_tdc: false,
            };
        }
        commands
            .entity(link.main_camera)
            .insert(Transform::IDENTITY);
    }
}
