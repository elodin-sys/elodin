//! Geographic JSON assets, shared by native and browser editors.
use crate::{
    MainCamera,
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
    ui::{inspector::viewport::Viewport, schematic::CurrentSchematic},
};
use bevy::{
    asset::{AssetLoader, RenderAssetUsages, io::Reader},
    camera::{RenderTarget, visibility::RenderLayers},
    math::{DQuat, DVec3},
    picking::prelude::{MeshRayCast, MeshRayCastSettings},
    prelude::*,
    render::render_resource::PrimitiveTopology,
    window::PrimaryWindow,
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoOrigin, GeoPosition, GeoRotation};
use impeller_wkt::MapOverlayConfig;
use serde::Deserialize;

#[derive(Deserialize)]
struct Place {
    name: String,
    latitude: f64,
    longitude: f64,
    altitude: f64,
}

#[derive(Deserialize)]
struct Line {
    kind: String,
    points: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct Area {
    triangles: Vec<[[f64; 3]; 3]>,
}

/// Accept the two data-only JSON formats used by the original terrain demo.
#[derive(Deserialize, Default)]
#[serde(default)]
struct Document {
    labels: Vec<(String, f64, f64, f64)>,
    places: Vec<Place>,
    lines: Vec<Line>,
    areas: Vec<Area>,
    water_names: Vec<String>,
    excluded_names: Vec<String>,
    source: String,
    source_url: String,
    sources: Vec<String>,
}

struct Label {
    name: String,
    ecef: DVec3,
    water: bool,
}

#[derive(Asset, TypePath)]
struct MapOverlayAsset {
    labels: Vec<Label>,
    lines: Vec<(String, Vec<DVec3>)>,
    triangles: Vec<DVec3>,
    attribution: Vec<(String, String)>,
}

#[derive(Debug, thiserror::Error)]
enum OverlayError {
    #[error("cannot read overlay: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid overlay JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid geographic overlay: {0}")]
    Invalid(&'static str),
}

fn geographic_point(p: [f64; 3]) -> Result<DVec3, OverlayError> {
    if !p.iter().all(|v| v.is_finite())
        || !(-90.0..=90.0).contains(&p[0])
        || !(-180.0..=180.0).contains(&p[1])
    {
        return Err(OverlayError::Invalid(
            "coordinates must be finite latitude, longitude, altitude",
        ));
    }
    Ok(GeoFrame::origin_ecef(&GeoOrigin::new_from_degrees(
        p[0], p[1], p[2],
    )))
}

impl MapOverlayAsset {
    fn parse(bytes: &[u8]) -> Result<Self, OverlayError> {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(OverlayError::Invalid("JSON exceeds 16 MiB"));
        }
        let doc: Document = serde_json::from_slice(bytes)?;
        let count = doc.lines.iter().map(|l| l.points.len()).sum::<usize>()
            + doc
                .areas
                .iter()
                .map(|a| a.triangles.len() * 3)
                .sum::<usize>();
        if count > 100_000 || doc.labels.len() + doc.places.len() > 512 {
            return Err(OverlayError::Invalid("too many features"));
        }
        let mut labels = Vec::new();
        for p in doc.places.into_iter().chain(doc.labels.into_iter().map(
            |(name, latitude, longitude, altitude)| Place {
                name,
                latitude,
                longitude,
                altitude,
            },
        )) {
            if p.name.trim().is_empty() || p.name.chars().count() > 160 {
                return Err(OverlayError::Invalid(
                    "label name must contain 1–160 characters",
                ));
            }
            if doc
                .excluded_names
                .iter()
                .any(|n| n.eq_ignore_ascii_case(&p.name))
            {
                continue;
            }
            labels.push(Label {
                water: doc.water_names.contains(&p.name),
                ecef: geographic_point([p.latitude, p.longitude, p.altitude + 12.0])?,
                name: p.name,
            });
        }
        let lines = doc
            .lines
            .into_iter()
            .map(|l| {
                if l.points.len() < 2 {
                    return Err(OverlayError::Invalid("line needs two points"));
                }
                Ok((
                    l.kind,
                    l.points
                        .into_iter()
                        .map(geographic_point)
                        .collect::<Result<_, _>>()?,
                ))
            })
            .collect::<Result<_, OverlayError>>()?;
        let triangles = doc
            .areas
            .into_iter()
            .flat_map(|a| a.triangles)
            .flatten()
            .map(geographic_point)
            .collect::<Result<_, _>>()?;
        let mut attribution = Vec::new();
        if !doc.source.is_empty() {
            attribution.push((doc.source, doc.source_url));
        }
        attribution.extend(doc.sources.into_iter().map(|url| (url.clone(), url)));
        Ok(Self {
            labels,
            lines,
            triangles,
            attribution,
        })
    }

    fn is_empty(&self) -> bool {
        self.labels.is_empty() && self.lines.is_empty() && self.triangles.is_empty()
    }
}

#[derive(Default, TypePath)]
struct OverlayLoader;
impl AssetLoader for OverlayLoader {
    type Asset = MapOverlayAsset;
    type Settings = ();
    type Error = OverlayError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        _: &(),
        _: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        MapOverlayAsset::parse(&bytes)
    }
    fn extensions(&self) -> &[&str] {
        &["json"]
    }
}

#[derive(Component)]
struct OverlayGeometry;

struct Layer {
    handle: Handle<MapOverlayAsset>,
    visible: bool,
    geometry: Vec<Entity>,
    loaded: bool,
}

#[derive(Resource, Default)]
struct Overlays {
    selection: Vec<(MapOverlayConfig, String)>,
    layers: Vec<Layer>,
}

pub struct MapOverlayPlugin;
impl Plugin for MapOverlayPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<bevy_egui::EguiPlugin>() {
            return;
        }
        app.init_asset::<MapOverlayAsset>()
            .init_asset_loader::<OverlayLoader>()
            .init_resource::<Overlays>()
            .add_systems(
                Update,
                (
                    sync_assets,
                    draw.before(crate::plugins::editor_cam_input::gated_camera_inputs),
                )
                    .chain()
                    .after(crate::ui::render_layout),
            );
    }
}

#[allow(clippy::too_many_arguments)]
fn sync_assets(
    mut commands: Commands,
    schematic: Option<Res<CurrentSchematic>>,
    connection: Option<Res<impeller_bevy::ConnectionAddr>>,
    server: Res<AssetServer>,
    assets: Res<Assets<MapOverlayAsset>>,
    mut overlays: ResMut<Overlays>,
    root: Res<BigSpaceRootEntity>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut changes: MessageReader<AssetEvent<MapOverlayAsset>>,
) {
    let endpoint = connection.as_ref().map(|c| c.endpoint());
    let selection: Vec<_> = schematic
        .iter()
        .flat_map(|s| &s.map_overlays)
        .filter(|o| o.is_valid())
        .map(|o| {
            (
                o.clone(),
                crate::object_3d::resolve_db_endpoint(&o.path, endpoint),
            )
        })
        .collect();
    if overlays.selection != selection {
        for layer in &overlays.layers {
            for e in &layer.geometry {
                commands.entity(*e).despawn();
            }
        }
        overlays.layers = selection
            .iter()
            .map(|(o, url)| Layer {
                handle: server.load(url.clone()),
                visible: o.visible,
                geometry: Vec::new(),
                loaded: false,
            })
            .collect();
        overlays.selection = selection;
    }
    for event in changes.read() {
        if let AssetEvent::Modified { id } = event {
            for layer in &mut overlays.layers {
                if layer.handle.id() == *id {
                    for e in layer.geometry.drain(..) {
                        commands.entity(e).despawn();
                    }
                    layer.loaded = false;
                }
            }
        }
    }
    for layer in &mut overlays.layers {
        if layer.loaded {
            continue;
        }
        let Some(data) = assets.get(&layer.handle) else {
            continue;
        };
        layer.loaded = true;
        // Anchor each mesh in ECEF using double precision; vertices stay local.
        for (kind, topology, points) in [
            (
                "road",
                PrimitiveTopology::LineList,
                data.lines
                    .iter()
                    .filter(|(k, _)| k == "road")
                    .flat_map(|(_, p)| p.windows(2).flat_map(|w| [w[0], w[1]]))
                    .collect::<Vec<_>>(),
            ),
            (
                "water",
                PrimitiveTopology::LineList,
                data.lines
                    .iter()
                    .filter(|(k, _)| k == "water")
                    .flat_map(|(_, p)| p.windows(2).flat_map(|w| [w[0], w[1]]))
                    .collect(),
            ),
            (
                "airport",
                PrimitiveTopology::LineList,
                data.lines
                    .iter()
                    .filter(|(k, _)| k != "water" && k != "road")
                    .flat_map(|(_, p)| p.windows(2).flat_map(|w| [w[0], w[1]]))
                    .collect(),
            ),
            (
                "area",
                PrimitiveTopology::TriangleList,
                data.triangles.clone(),
            ),
        ] {
            let Some(anchor) = points.first().copied() else {
                continue;
            };
            let color = match kind {
                "road" => Color::srgba(0.91, 0.84, 0.68, 0.58),
                "water" => Color::srgba(0.46, 0.75, 0.84, 0.65),
                "area" => Color::srgba(0.43, 0.68, 0.73, 0.16),
                _ => Color::srgba(0.64, 0.83, 0.85, 0.66),
            };
            let positions: Vec<_> = points
                .iter()
                .map(|p| (*p - anchor).as_vec3().to_array())
                .collect();
            let mesh = Mesh::new(topology, RenderAssetUsages::default())
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    vec![[0., 1., 0.]; positions.len()],
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            let entity = commands
                .spawn((
                    OverlayGeometry,
                    Name::new("Map overlay geometry"),
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        unlit: true,
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        ..default()
                    })),
                    RenderLayers::layer(super::render_layer_alloc::SCENE_LINE_RENDER_LAYER),
                    bevy::light::NotShadowCaster,
                    GeoPosition(GeoFrame::ECEF, anchor),
                    GeoRotation::absolute(GeoFrame::ECEF, DQuat::IDENTITY),
                    GridCell::default(),
                    ChildOf(root.0),
                    Transform::default(),
                    if layer.visible {
                        Visibility::Inherited
                    } else {
                        Visibility::Hidden
                    },
                ))
                .id();
            layer.geometry.push(entity);
        }
    }
}

fn unobstructed(
    ray_cast: &mut MeshRayCast,
    meshes: &Query<(), (With<Mesh3d>, Without<OverlayGeometry>)>,
    origin: Vec3,
    point: Vec3,
) -> bool {
    let delta = point - origin;
    let Ok(direction) = Dir3::new(delta) else {
        return false;
    };
    let filter = |entity| meshes.contains(entity);
    ray_cast
        .cast_ray(
            Ray3d::new(origin, direction),
            &MeshRayCastSettings::default()
                .with_filter(&filter)
                .always_early_exit(),
        )
        .first()
        .is_none_or(|(_, hit)| hit.distance >= delta.length() - 1.0)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw(
    mut overlays: ResMut<Overlays>,
    assets: Res<Assets<MapOverlayAsset>>,
    mut geometry: Query<&mut Visibility, With<OverlayGeometry>>,
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
    mut contexts: Query<
        (&RenderTarget, &mut bevy_egui::EguiContext),
        With<crate::ui::tiles::WindowId>,
    >,
    meshes: Query<(), (With<Mesh3d>, Without<OverlayGeometry>)>,
    mut ray_cast: MeshRayCast,
) {
    // No declared, successfully loaded, nonempty asset means no button.
    if !overlays
        .layers
        .iter()
        .any(|l| assets.get(&l.handle).is_some_and(|a| !a.is_empty()))
    {
        return;
    }
    let Ok(primary) = windows.single() else {
        return;
    };
    let Ok(root_transform) = globals.get(root.0) else {
        return;
    };
    let matrix = GeoFrame::bevy_M_(&GeoFrame::ECEF, &geo);
    for (context_target, mut context) in &mut contexts {
        let Some(window) = crate::ui::window::window_entity_from_target(context_target, primary)
        else {
            continue;
        };
        let ctx = context.get_mut();
        for (entity, camera, target, local, global, parent, panel_rect, layers) in &cameras {
            if !camera.is_active
                || panel_rect.0.is_none()
                || !layers.intersects(&RenderLayers::layer(
                    super::render_layer_alloc::SCENE_LINE_RENDER_LAYER,
                ))
                || crate::ui::window::window_entity_from_target(target, primary) != Some(window)
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
            if rect.width() < 100.0 || rect.height() < 60.0 {
                continue;
            }
            let Ok((anchor, cell)) = parents.get(parent.parent()) else {
                continue;
            };
            let camera_position = grid.grid_position_double(cell, &anchor.mul_transform(*local));
            let view_from_root = global.affine().inverse() * root_transform.affine();
            let world = |ecef: DVec3| {
                global.translation()
                    + root_transform.affine().transform_vector3(
                        (matrix.transform_point3(ecef) - camera_position).as_vec3(),
                    )
            };
            let project = |ecef: DVec3| {
                let delta = matrix.transform_point3(ecef) - camera_position;
                let clip = camera.clip_from_view()
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
                            (ndc.x + 1.) * 0.5 * rect.width(),
                            (1. - ndc.y) * 0.5 * rect.height(),
                        ),
                )
            };
            let any_visible = overlays
                .layers
                .iter()
                .any(|l| l.visible && assets.get(&l.handle).is_some_and(|a| !a.is_empty()));
            let response = egui::Area::new(egui::Id::new(("map_overlay_toggle", entity)))
                .constrain_to(rect)
                .fixed_pos(rect.min + egui::vec2(8., 8.))
                .show(ctx, |ui| {
                    ui.add(
                        egui::Button::new(if any_visible {
                            "Overlays on"
                        } else {
                            "Overlays off"
                        })
                        .wrap_mode(egui::TextWrapMode::Extend),
                    )
                    .on_hover_ui(|ui| {
                        ui.label("Show or hide geographic overlays");
                        for layer in &overlays.layers {
                            if let Some(asset) = assets.get(&layer.handle) {
                                for (label, url) in &asset.attribution {
                                    if url.starts_with("https://") || url.starts_with("http://") {
                                        ui.hyperlink_to(label, url);
                                    } else {
                                        ui.label(label);
                                    }
                                }
                            }
                        }
                    })
                });
            if response.inner.clicked() {
                for layer in &mut overlays.layers {
                    layer.visible = !any_visible;
                    for entity in &layer.geometry {
                        if let Ok(mut v) = geometry.get_mut(*entity) {
                            *v = if layer.visible {
                                Visibility::Inherited
                            } else {
                                Visibility::Hidden
                            };
                        }
                    }
                }
            }
            let painter = ctx
                .layer_painter(egui::LayerId::new(
                    egui::Order::Middle,
                    egui::Id::new(("map_overlay_labels", entity)),
                ))
                .with_clip_rect(rect);
            let mut occupied = vec![response.response.rect.expand(4.)];
            let mut candidates: Vec<_> = overlays
                .layers
                .iter()
                .filter(|l| l.visible)
                .filter_map(|l| assets.get(&l.handle))
                .flat_map(|a| &a.labels)
                .map(|label| {
                    (
                        label,
                        (matrix.transform_point3(label.ecef) - camera_position).length_squared(),
                    )
                })
                .collect();
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut rays = 0;
            let mut names = std::collections::HashSet::new();
            for (label, _) in candidates {
                if names.contains(label.name.as_str()) {
                    continue;
                }
                let Some(point) = project(label.ecef) else {
                    continue;
                };
                if !rect.shrink(12.).contains(point) {
                    continue;
                }
                let color = if label.water {
                    egui::Color32::from_rgb(141, 188, 207)
                } else {
                    egui::Color32::from_rgb(231, 235, 230)
                };
                let galley = painter.layout_no_wrap(
                    label.name.clone(),
                    egui::FontId::proportional(13.),
                    color,
                );
                let bounds = egui::Rect::from_min_size(point + egui::vec2(9., -8.), galley.size())
                    .expand(4.);
                if !rect.contains_rect(bounds) || occupied.iter().any(|r| r.intersects(bounds)) {
                    continue;
                }
                if rays >= 12 {
                    break;
                }
                rays += 1;
                if !unobstructed(
                    &mut ray_cast,
                    &meshes,
                    global.translation(),
                    world(label.ecef),
                ) {
                    continue;
                }
                occupied.push(bounds);
                names.insert(label.name.as_str());
                painter.circle_filled(point, 2.5, color);
                painter.rect_filled(bounds, 3., egui::Color32::from_black_alpha(150));
                painter.galley(bounds.min + egui::vec2(4., 4.), galley, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn demo_assets_are_loaded_as_data_not_code() {
        let local = MapOverlayAsset::parse(include_bytes!(
            "../../../../examples/falcon9/assets/overlays/cape-canaveral.json"
        ))
        .unwrap();
        assert_eq!(local.labels.len(), 28);
        assert_eq!(local.lines.len(), 2128);
        assert!(!local.triangles.is_empty());
        assert_eq!(local.attribution[0].0, "© OpenStreetMap contributors");
        let sites = MapOverlayAsset::parse(include_bytes!(
            "../../../../examples/falcon9/assets/overlays/falcon9-sites.json"
        ))
        .unwrap();
        assert_eq!(sites.labels.len(), 2);
        let orbital = MapOverlayAsset::parse(include_bytes!(
            "../../../../examples/falcon9/assets/overlays/orbital-context.json"
        ))
        .unwrap();
        assert_eq!(orbital.labels.len(), 20);
        assert!(
            orbital
                .labels
                .iter()
                .any(|l| l.name == "Atlantic Ocean" && l.water)
        );
    }
    #[test]
    fn invalid_coordinates_fail_without_partial_overlay() {
        assert!(MapOverlayAsset::parse(br#"{"labels":[["x",91,0,0]]}"#).is_err());
        assert!(MapOverlayAsset::parse(br#"{"labels":[["",0,0,0]]}"#).is_err());
        assert!(MapOverlayAsset::parse(br#"{}"#).unwrap().is_empty());
    }
}
