use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    reflect::TypePath,
    render::render_resource::AsBindGroup,
};

use crate::terrain::{TerrainConfig, TerrainRoot};

/// No-op extension to [`StandardMaterial`].
///
/// This gives us a distinct material type (`WorldMeshMaterial`) without needing
/// custom shaders for the initial integration.
///
/// Note: fields without `#[uniform]` / `#[texture]` / etc are ignored by the
/// `AsBindGroup` derive, so this struct currently contributes **zero** GPU
/// bindings.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct WorldMeshMaterialExt {
    _ignored: (),
}

impl MaterialExtension for WorldMeshMaterialExt {}

/// Default material used by the planar world-mesh scene.
pub type WorldMeshMaterial = ExtendedMaterial<StandardMaterial, WorldMeshMaterialExt>;

/// Spawns a simple planar terrain backdrop.
///
/// This is *not* the full world_mesh renderer yet; it is a small stand-in that
/// lets the editor render a stable "world mesh" surface while the real renderer
/// is brought in incrementally.
pub fn spawn_planar_backdrop(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<WorldMeshMaterial>,
    config: &TerrainConfig,
    transform: Transform,
    visible: bool,
) -> Entity {
    let mesh = meshes.add(
        bevy::math::primitives::Plane3d::default()
            .mesh()
            .size(config.width_m, config.depth_m),
    );

    let material = materials.add(WorldMeshMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.22, 0.24, 0.22),
            perceptual_roughness: 1.0,
            metallic: 0.0,
            double_sided: true,
            cull_mode: None,
            ..Default::default()
        },
        extension: WorldMeshMaterialExt::default(),
    });

    let visibility = if visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    commands
        .spawn((
            TerrainRoot,
            Mesh3d(mesh),
            MeshMaterial3d(material),
            transform,
            GlobalTransform::default(),
            visibility,
            InheritedVisibility::default(),
            ViewVisibility::default(),
            Name::new(format!("world_mesh planar ({})", config.region)),
        ))
        .id()
}

/// Spawn a planar heightmap mesh using an on-disk world-mesh atlas (height + albedo).
///
/// This is a *minimal* stopgap renderer intended purely to prove that the
/// preprocessed planar atlases (e.g. Death Valley) can be consumed by the Elodin
/// editor.
///
/// Returns `None` if the atlas files are not present.
/// On wasm targets this always returns `None`.
#[cfg(target_family = "wasm")]
pub fn try_spawn_planar_from_atlas(
    _commands: &mut Commands,
    _meshes: &mut Assets<Mesh>,
    _materials: &mut Assets<WorldMeshMaterial>,
    _images: &mut Assets<Image>,
    _region: &str,
    _transform: Transform,
    _visible: bool,
) -> Option<Entity> {
    None
}

#[cfg(not(target_family = "wasm"))]
pub fn try_spawn_planar_from_atlas(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<WorldMeshMaterial>,
    images: &mut Assets<Image>,
    region: &str,
    transform: Transform,
    visible: bool,
) -> Option<Entity> {
    use bevy::{
        asset::RenderAssetUsages,
        image::ImageSampler,
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    };

    let assets_root = resolve_assets_root();
    let region_root = crate::terrain::atlas::region_root_from_assets_dir(&assets_root, region);

    // Metadata is optional (region.toml).
    let meta = crate::terrain::atlas::load_region_metadata(&region_root).unwrap_or_else(|| {
        // Death Valley defaults from the issue write-up.
        let (w, d) = if region == "death_valley" {
            (22_000.0, 22_000.0)
        } else {
            (2_000.0, 2_000.0)
        };
        crate::terrain::atlas::RegionMetadata {
            width_m: w,
            depth_m: d,
            min_height_m: -100.0,
            max_height_m: 2_000.0,
        }
    });

    let tile = crate::terrain::atlas::pick_coarsest_tile(&region_root)
        .unwrap_or_else(|| "0_0_0_0".to_string());

    // Prefer the upstream bevy_terrain-style layout:
    //   data/height/<tile>.bin + data/albedo/<tile>.bin
    // but tolerate a flat layout under data/ by probing for matching files.
    let height_path = region_root.join("data/height").join(format!("{tile}.bin"));
    let albedo_path = region_root.join("data/albedo").join(format!("{tile}.bin"));

    let (height_bytes, albedo_bytes) = match (
        std::fs::read(&height_path).ok(),
        std::fs::read(&albedo_path).ok(),
    ) {
        (Some(h), Some(a)) => (h, a),
        _ => {
            let data_dir = region_root.join("data");
            let (h, a) = find_height_and_albedo_fallback(&data_dir, &tile)?;
            (h, a)
        }
    };

    let (height_tex_size, height_u16) = decode_r16_square(&height_bytes)?;
    let (albedo_tex_size, albedo_rgba8) = decode_rgba8_square(&albedo_bytes)?;

    let tex_size = height_tex_size.min(albedo_tex_size);
    if tex_size == 0 {
        return None;
    }

    if height_tex_size != albedo_tex_size {
        warn!(
            height = height_tex_size,
            albedo = albedo_tex_size,
            "world_mesh atlas: height/albedo tile sizes differ"
        );
    }

    // Build a bevy Image for albedo.
    let image_handle = {
        let size = Extent3d {
            width: tex_size,
            height: tex_size,
            depth_or_array_layers: 1,
        };
        let expected_len = (tex_size * tex_size * 4) as usize;
        if albedo_rgba8.len() < expected_len {
            return None;
        }

        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &albedo_rgba8[..expected_len],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        image.sampler = ImageSampler::linear();
        images.add(image)
    };

    // Downsample aggressively to keep vertex counts reasonable.
    let step = (tex_size / 256).max(1);

    let mesh = meshes.add(build_height_mesh(
        tex_size,
        step,
        &height_u16,
        meta.width_m,
        meta.depth_m,
        meta.min_height_m,
        meta.max_height_m,
    ));

    let material = materials.add(WorldMeshMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(image_handle),
            perceptual_roughness: 1.0,
            metallic: 0.0,
            double_sided: true,
            cull_mode: None,
            ..Default::default()
        },
        extension: WorldMeshMaterialExt::default(),
    });

    let visibility = if visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    Some(
        commands
            .spawn((
                TerrainRoot,
                Mesh3d(mesh),
                MeshMaterial3d(material),
                transform,
                GlobalTransform::default(),
                visibility,
                InheritedVisibility::default(),
                ViewVisibility::default(),
                Name::new(format!("world_mesh atlas ({region})")),
            ))
            .id(),
    )
}

#[cfg(not(target_family = "wasm"))]
fn resolve_assets_root() -> std::path::PathBuf {
    // Keep this in sync with `env_asset_source` in elodin-editor.
    let dir = std::env::var_os("ELODIN_ASSETS_DIR").unwrap_or_else(|| "assets".into());
    let mut p = std::path::PathBuf::from(dir);
    if p.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            p = cwd.join(p);
        }
    }
    p
}

#[cfg(not(target_family = "wasm"))]
fn find_height_and_albedo_fallback(
    data_dir: &std::path::Path,
    tile: &str,
) -> Option<(Vec<u8>, Vec<u8>)> {
    use std::fs;

    // Layout variant: data/<attachment>/<tile>.bin (attachment names not standardized).
    if let Ok(entries) = fs::read_dir(data_dir) {
        let mut height: Option<(u32, Vec<u8>)> = None;
        let mut albedo: Option<(u32, Vec<u8>)> = None;

        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let p = dir.join(format!("{tile}.bin"));
            if !p.is_file() {
                continue;
            }

            let bytes = fs::read(&p).ok()?;

            // Classify by raw pixel size.
            if height.is_none() && bytes.len() % 2 == 0 {
                let px = bytes.len() / 2;
                let side = (px as f64).sqrt() as u32;
                if (side as usize) * (side as usize) == px {
                    height = Some((side, bytes));
                    continue;
                }
            }
            if albedo.is_none() && bytes.len() % 4 == 0 {
                let px = bytes.len() / 4;
                let side = (px as f64).sqrt() as u32;
                if (side as usize) * (side as usize) == px {
                    albedo = Some((side, bytes));
                    continue;
                }
            }
        }

        if let (Some((h_side, h)), Some((a_side, a))) = (height, albedo) {
            if h_side == a_side {
                return Some((h, a));
            }
        }
    }

    // Layout variant: data/<tile>_*.bin (flat directory).
    find_height_and_albedo_flat(data_dir, tile)
}

#[cfg(not(target_family = "wasm"))]
fn find_height_and_albedo_flat(
    data_dir: &std::path::Path,
    tile: &str,
) -> Option<(Vec<u8>, Vec<u8>)> {
    use std::fs;

    let entries = fs::read_dir(data_dir).ok()?;

    let mut height: Option<(u32, Vec<u8>)> = None;
    let mut albedo: Option<(u32, Vec<u8>)> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.contains(tile) {
            continue;
        }

        let bytes = fs::read(&path).ok()?;

        // Height candidate: R16.
        if height.is_none() && bytes.len() % 2 == 0 {
            let px = bytes.len() / 2;
            let side = (px as f64).sqrt() as u32;
            if (side as usize) * (side as usize) == px {
                height = Some((side, bytes));
                continue;
            }
        }

        // Albedo candidate: RGBA8.
        if albedo.is_none() && bytes.len() % 4 == 0 {
            let px = bytes.len() / 4;
            let side = (px as f64).sqrt() as u32;
            if (side as usize) * (side as usize) == px {
                albedo = Some((side, bytes));
                continue;
            }
        }
    }

    let (h_side, h) = height?;
    let (a_side, a) = albedo?;
    if h_side != a_side {
        return None;
    }
    Some((h, a))
}

#[cfg(not(target_family = "wasm"))]
fn decode_r16_square(bytes: &[u8]) -> Option<(u32, Vec<u16>)> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let px = bytes.len() / 2;
    let size = (px as f64).sqrt() as u32;
    if (size as usize) * (size as usize) != px {
        return None;
    }
    let mut out = Vec::with_capacity(px);
    for chunk in bytes.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some((size, out))
}

#[cfg(not(target_family = "wasm"))]
fn decode_rgba8_square(bytes: &[u8]) -> Option<(u32, Vec<u8>)> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let px = bytes.len() / 4;
    let size = (px as f64).sqrt() as u32;
    if (size as usize) * (size as usize) != px {
        return None;
    }
    Some((size, bytes.to_vec()))
}

#[cfg(not(target_family = "wasm"))]
fn build_height_mesh(
    tex_size: u32,
    step: u32,
    height_u16: &[u16],
    width_m: f32,
    depth_m: f32,
    min_height_m: f32,
    max_height_m: f32,
) -> Mesh {
    use bevy::{
        asset::RenderAssetUsages,
        mesh::{Indices, PrimitiveTopology},
    };

    let sample_w = ((tex_size - 1) / step) + 1;
    let sample_h = sample_w;

    let dx = width_m / (sample_w - 1) as f32;
    let dz = depth_m / (sample_h - 1) as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((sample_w * sample_h) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((sample_w * sample_h) as usize);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity((sample_w * sample_h) as usize);

    let height_at = |x: i32, y: i32| -> f32 {
        let x = x.clamp(0, (sample_w - 1) as i32) as u32;
        let y = y.clamp(0, (sample_h - 1) as i32) as u32;
        let src_x = (x * step).min(tex_size - 1);
        let src_y = (y * step).min(tex_size - 1);
        let idx = (src_y * tex_size + src_x) as usize;
        let h_norm = height_u16.get(idx).copied().unwrap_or(0) as f32 / u16::MAX as f32;
        min_height_m + h_norm * (max_height_m - min_height_m)
    };

    for y in 0..sample_h {
        for x in 0..sample_w {
            let u = x as f32 / (sample_w - 1) as f32;
            let v = y as f32 / (sample_h - 1) as f32;

            let world_x = (u - 0.5) * width_m;
            let world_z = (v - 0.5) * depth_m;
            let h = height_at(x as i32, y as i32);

            // Estimate normals from central differences.
            let h_l = height_at(x as i32 - 1, y as i32);
            let h_r = height_at(x as i32 + 1, y as i32);
            let h_d = height_at(x as i32, y as i32 - 1);
            let h_u = height_at(x as i32, y as i32 + 1);

            let ddx = (h_r - h_l) / (2.0 * dx.max(1e-6));
            let ddz = (h_u - h_d) / (2.0 * dz.max(1e-6));
            let n = Vec3::new(-ddx, 1.0, -ddz).normalize_or_zero();

            positions.push([world_x, h, world_z]);
            normals.push(n.to_array());
            // Texture origin is top-left in most image conventions; flip V.
            uvs.push([u, 1.0 - v]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(((sample_w - 1) * (sample_h - 1) * 6) as usize);
    for y in 0..(sample_h - 1) {
        for x in 0..(sample_w - 1) {
            let i0 = (y * sample_w + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + sample_w as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
