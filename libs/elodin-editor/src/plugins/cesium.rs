//! Experimental Cesium Native streaming. Opt-in at build time and runtime.
use std::collections::{HashMap, HashSet};

use bevy::{
    asset::RenderAssetUsages,
    math::DMat4,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use cesium_native_bridge::ffi;

use crate::{
    MainCamera,
    spatial::{BigSpaceRootEntity, FloatingOriginSettings, GridCell},
};

pub struct CesiumPlugin;

struct NativeSession {
    session: cesium_native_bridge::SessionHandle,
    tiles: HashMap<u64, Vec<Entity>>,
}

#[derive(Resource, Default)]
struct Credits {
    html: Vec<String>,
    source: String,
    load_failed: bool,
}

impl Plugin for CesiumPlugin {
    fn build(&self, app: &mut App) {
        let result = if let Ok(asset) = std::env::var("CESIUM_ION_ASSET_ID") {
            let Ok(asset) = asset.parse::<i64>() else {
                error!("CESIUM_ION_ASSET_ID must be a positive integer");
                return;
            };
            let Ok(token) = std::env::var("CESIUM_ION_TOKEN") else {
                error!("Set CESIUM_ION_TOKEN to load a Cesium ion asset");
                return;
            };
            if asset <= 0 || token.is_empty() {
                error!("Invalid Cesium ion configuration");
                return;
            }
            ffi::open_ion(asset, &token)
        } else if let Ok(url) = std::env::var("CESIUM_TILESET_URL") {
            ffi::open_url(&url)
        } else {
            return;
        };
        match result {
            Ok(session) => {
                app.insert_non_send(NativeSession {
                    session,
                    tiles: HashMap::new(),
                })
                .insert_resource(Credits {
                    source: std::env::var("CESIUM_CREDIT").unwrap_or_default(),
                    ..default()
                })
                .add_systems(Update, stream)
                .add_systems(Update, show_credits.after(crate::ui::render_layout));
            }
            Err(_) => error!("Could not initialize Cesium Native"),
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn stream(
    mut commands: Commands,
    mut native: NonSendMut<NativeSession>,
    geo: Res<GeoContext>,
    grid: Res<FloatingOriginSettings>,
    root: Res<BigSpaceRootEntity>,
    time: Res<Time>,
    cameras: Query<(&Camera, &Projection, &Transform, Option<&ChildOf>), With<MainCamera>>,
    parents: Query<(&Transform, &GridCell), Without<MainCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut visibility: Query<&mut Visibility>,
    mut credits: ResMut<Credits>,
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
            let position = grid.grid_position_double(&cell, &transform);
            Some(ffi::View {
                position: ecef_from_bevy.transform_point3(position).to_array(),
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
    let Ok(frame) = native.session.pin_mut().update(&views, time.delta_secs()) else {
        // Native diagnostics can contain signed URLs: do not print exceptions.
        error!("Cesium tile update failed");
        return;
    };
    credits.html = frame.credits;
    credits.load_failed = frame.load_failed;
    for tile in frame.loaded {
        let mut entities = Vec::new();
        for p in tile.primitives {
            let matrix = DMat4::from_cols_array(&p.transform);
            let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
            let positions = p.positions.as_chunks::<3>().0.to_vec();
            let normals = p.normals.as_chunks::<3>().0.to_vec();
            let uvs = p.uvs.as_chunks::<2>().0.to_vec();
            let mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(p.indices));
            let texture = if p.width > 0 && p.height > 0 {
                Some(images.add(Image::new(
                    Extent3d {
                        width: p.width,
                        height: p.height,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    p.pixels,
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::default(),
                )))
            } else {
                None
            };
            let material = StandardMaterial {
                base_color: Color::linear_rgba(
                    p.base_color[0],
                    p.base_color[1],
                    p.base_color[2],
                    p.base_color[3],
                ),
                base_color_texture: texture,
                double_sided: p.double_sided,
                cull_mode: if p.double_sided {
                    None
                } else {
                    Some(bevy::render::render_resource::Face::Back)
                },
                perceptual_roughness: 1.0,
                ..default()
            };
            let entity = commands
                .spawn((
                    Name::new("Cesium tile"),
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(materials.add(material)),
                    Transform::from_scale(scale.as_vec3()),
                    GeoPosition(GeoFrame::ECEF, translation),
                    GeoRotation::absolute(GeoFrame::ECEF, rotation),
                    GridCell::default(),
                    ChildOf(root.0),
                    Visibility::Hidden,
                ))
                .id();
            entities.push(entity);
        }
        native.tiles.insert(tile.id, entities);
    }
    // A tile can finish loading and be evicted in the same update.
    for id in frame.removed {
        if let Some(entities) = native.tiles.remove(&id) {
            for entity in entities {
                commands.entity(entity).despawn();
            }
        }
    }
    let visible: HashSet<_> = frame.visible.into_iter().collect();
    for (id, entities) in &native.tiles {
        let desired = if visible.contains(id) {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        for entity in entities {
            if let Ok(mut current) = visibility.get_mut(*entity) {
                if *current != desired {
                    *current = desired;
                }
            } else {
                commands.entity(*entity).insert(desired);
            }
        }
    }
}

fn show_credits(
    mut contexts: Query<(&crate::ui::tiles::WindowId, &mut bevy_egui::EguiContext)>,
    credits: Res<Credits>,
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
                        if !credits.source.is_empty() {
                            ui.add_space(6.0);
                            ui.label(&credits.source);
                        }
                    });
                    if credits.load_failed {
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
