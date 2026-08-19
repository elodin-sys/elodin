use bevy::{
    asset::embedded_asset,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use super::curves::{CITY_NIGHT_FULL, CITY_NIGHT_START};

pub struct EarthNightMaterialPlugin;

impl Plugin for EarthNightMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "earth_night.wgsl");
        app.add_plugins(MaterialPlugin::<EarthNightMaterial>::default());
    }
}

#[derive(ShaderType, Copy, Clone, Debug)]
pub struct EarthNightParams {
    /// xyz = world-space direction toward the sun.
    pub to_sun_world: Vec4,
    /// x = [`CITY_NIGHT_START`], y = [`CITY_NIGHT_FULL`].
    pub band: Vec4,
}

impl Default for EarthNightParams {
    fn default() -> Self {
        Self {
            to_sun_world: Vec3::Y.extend(0.0),
            band: Vec4::new(CITY_NIGHT_START, CITY_NIGHT_FULL, 0.0, 0.0),
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct EarthNightExt {
    #[uniform(100)]
    pub params: EarthNightParams,
}

impl MaterialExtension for EarthNightExt {
    fn fragment_shader() -> ShaderRef {
        "embedded://elodin_editor/plugins/cinematic_earth/earth_night.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
}

pub type EarthNightMaterial = ExtendedMaterial<StandardMaterial, EarthNightExt>;
