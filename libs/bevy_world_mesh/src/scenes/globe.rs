//! Minimal spherical scene helpers.

use bevy::prelude::*;

use super::planar::{WorldMeshMaterial, WorldMeshMaterialExt};
use crate::terrain::TerrainRoot;

/// Spawns a simple sphere as a stand-in for the full globe renderer.
pub fn spawn_globe_backdrop(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<WorldMeshMaterial>,
    radius_m: f32,
    transform: Transform,
    visible: bool,
) -> Entity {
    let mesh = meshes.add(bevy::math::primitives::Sphere { radius: radius_m });

    let material = materials.add(WorldMeshMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.10, 0.13, 0.20),
            perceptual_roughness: 1.0,
            metallic: 0.0,
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
            Name::new("world_mesh globe"),
        ))
        .id()
}
