//! The rendering thing not the beer
use serde::{Deserialize, Serialize};

use crate::{Asset, AssetId, Handle};

pub struct Shape {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Glb(pub String);

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Mesh {
    pub inner: MeshInner,
}

impl Mesh {
    /// Create a new box, bachs since box is a reserved word
    pub fn bachs(x: f32, y: f32, z: f32) -> Self {
        Self::cuboid(x, y, z)
    }

    pub fn cuboid(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: MeshInner::Box { x, y, z },
        }
    }

    pub fn sphere(radius: f32, sectors: usize, stacks: usize) -> Self {
        Self {
            inner: MeshInner::Sphere {
                radius,
                sectors,
                stacks,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MeshInner {
    Sphere {
        radius: f32,
        sectors: usize,
        stacks: usize,
    },
    Box {
        x: f32,
        y: f32,
        z: f32,
    },
    Cylinder {
        radius: f32,
        height: f32,
        resolution: u32,
        segments: u32,
    },
    Data(MeshData),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MeshData {
    pub mesh_type: u8,
    pub positions: Option<Vec<[f32; 3]>>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub joint_weights: Option<Vec<[f32; 4]>>,
    pub joint_indices: Option<Vec<[u16; 4]>>,
    pub indices: Option<Vec<u32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Material {
    pub base_color: Color,
    pub base_color_texture: Option<Image>,
    pub emissive: Color,
    pub emissive_texture: Option<Image>,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub metallic_roughness_texture: Option<Image>,
    pub reflectance: f32,
    pub diffuse_transmission: f32,
    pub specular_transmission: f32,
    pub thickness: f32,
    pub ior: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: Color,
    pub normal_map_texture: Option<Image>,
    pub flip_normal_map_y: bool,
    pub occlusion_texture: Option<Image>,
    pub double_sided: bool,
    pub cull_mode: Option<Face>,
    pub unlit: bool,
    pub fog_enabled: bool,
    pub alpha_mode: AlphaMode,
    pub depth_bias: f32,
    pub depth_map: Option<Image>,
    pub parallax_depth_scale: f32,
    pub parallax_mapping_method: ParallaxMappingMethod,
    pub max_parallax_layer_count: f32,
    pub opaque_render_method: OpaqueRendererMethod,
    pub deferred_lighting_pass_id: u8,
}

impl Material {
    pub fn color(r: f32, g: f32, b: f32) -> Self {
        Material {
            base_color: Color { r, g, b },

            ..Default::default()
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::WHITE,
            base_color_texture: None,
            emissive: Color::BLACK,
            emissive_texture: None,
            perceptual_roughness: 0.5,
            metallic: 0.0,
            metallic_roughness_texture: None,
            reflectance: 0.5,
            diffuse_transmission: 0.0,
            specular_transmission: 0.0,
            thickness: 0.0,
            ior: 1.5,
            attenuation_color: Color::WHITE,
            attenuation_distance: f32::INFINITY,
            occlusion_texture: None,
            normal_map_texture: None,
            flip_normal_map_y: false,
            double_sided: false,
            cull_mode: Some(Face::Back),
            unlit: false,
            fog_enabled: true,
            alpha_mode: AlphaMode::Opaque,
            depth_bias: 0.0,
            depth_map: None,
            parallax_depth_scale: 0.1,
            max_parallax_layer_count: 16.0,
            parallax_mapping_method: ParallaxMappingMethod::Occlusion,
            opaque_render_method: OpaqueRendererMethod::Auto,
            deferred_lighting_pass_id: 1,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Image {
    pub data: Vec<u8>,
    pub size: Extent3d,
    pub texture_dimension: TextureDimension,
    pub format: TextureFormat,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone, Hash, Eq, PartialEq)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,

    R16Uint,
    R16Sint,
    R16Unorm,
    R16Snorm,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,

    R32Uint,
    R32Sint,
    R32Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,

    Rgb9e5Ufloat,
    Rgb10a2Unorm,
    Rg11b10Float,

    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Float,

    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,

    Stencil8,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,

    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Etc2Rgb8Unorm,
    Etc2Rgb8UnormSrgb,
    Etc2Rgb8A1Unorm,
    Etc2Rgb8A1UnormSrgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8UnormSrgb,
    EacR11Unorm,
    EacR11Snorm,
    EacRg11Unorm,
    EacRg11Snorm,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
    Premultiplied,
    Add,
    Multiply,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ParallaxMappingMethod {
    Occlusion,
    Relief { max_steps: u32 },
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum OpaqueRendererMethod {
    Forward,
    Deferred,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Face {
    Front = 0,
    Back = 1,
}

impl Asset for Mesh {
    const ASSET_ID: AssetId = AssetId(2240);
    fn asset_id(&self) -> AssetId {
        Self::ASSET_ID
    }
}

impl Asset for Material {
    const ASSET_ID: AssetId = AssetId(2241);
    fn asset_id(&self) -> AssetId {
        Self::ASSET_ID
    }
}

impl Asset for Glb {
    const ASSET_ID: AssetId = AssetId(2242);
    fn asset_id(&self) -> AssetId {
        Self::ASSET_ID
    }
}
