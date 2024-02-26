pub use super::{
    AlphaMode, Color, Extent3d, Face, Image, Material, Mesh, MeshData, MeshInner,
    OpaqueRendererMethod, ParallaxMappingMethod, TextureDimension, TextureFormat,
};
use bevy::prelude::Assets;
impl From<MeshData> for bevy::prelude::Mesh {
    fn from(data: MeshData) -> Self {
        use bevy::prelude::Mesh as BevyMesh;
        use bevy::render::mesh::VertexAttributeValues::Uint16x4;
        use bevy::render::{mesh::Indices, render_resource::PrimitiveTopology};
        let mesh_type_enum = match data.mesh_type {
            0 => PrimitiveTopology::PointList,
            1 => PrimitiveTopology::LineList,
            2 => PrimitiveTopology::LineStrip,
            3 => PrimitiveTopology::TriangleList,
            4 => PrimitiveTopology::TriangleStrip,
            _ => PrimitiveTopology::TriangleList,
        };

        let mut mesh = BevyMesh::new(mesh_type_enum);

        if let Some(positions) = data.positions {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);
        }

        if let Some(normals) = data.normals {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
        }

        if let Some(uvs) = data.uvs {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_UV_0, uvs);
        }

        if let Some(tangents) = data.tangents {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_TANGENT, tangents);
        }

        if let Some(colors) = data.colors {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_COLOR, colors);
        }

        if let Some(joint_weights) = data.joint_weights {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_JOINT_WEIGHT, joint_weights);
        }

        if let Some(joint_indices) = data.joint_indices {
            mesh.insert_attribute(BevyMesh::ATTRIBUTE_JOINT_INDEX, Uint16x4(joint_indices));
        }

        if let Some(indices) = data.indices {
            mesh.set_indices(Some(Indices::U32(indices)));
        }

        mesh
    }
}

impl From<bevy::prelude::Mesh> for Mesh {
    fn from(value: bevy::prelude::Mesh) -> Self {
        Mesh {
            inner: MeshInner::Data(value.into()),
        }
    }
}

impl From<bevy::prelude::Mesh> for MeshData {
    fn from(mesh: bevy::prelude::Mesh) -> Self {
        use bevy::prelude::Mesh as BevyMesh;
        use bevy::render::mesh::VertexAttributeValues::Uint16x4;
        use bevy::render::mesh::VertexAttributeValues::{Float32x2, Float32x3, Float32x4};
        use bevy::render::{mesh::Indices, render_resource::PrimitiveTopology};

        let positions = if let Some(Float32x3(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_POSITION) {
            Some(t.clone())
        } else {
            None
        };

        let tangents = if let Some(Float32x4(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_TANGENT) {
            Some(t.clone())
        } else {
            None
        };

        let normals = if let Some(Float32x3(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_NORMAL) {
            Some(t.clone())
        } else {
            None
        };

        let uvs = if let Some(Float32x2(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_UV_0) {
            Some(t.clone())
        } else {
            None
        };

        let colors = if let Some(Float32x4(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_COLOR) {
            Some(t.clone())
        } else {
            None
        };

        let joint_weights =
            if let Some(Float32x4(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_JOINT_WEIGHT) {
                Some(t.clone())
            } else {
                None
            };

        let joint_indices =
            if let Some(Uint16x4(t)) = mesh.attribute(BevyMesh::ATTRIBUTE_JOINT_INDEX) {
                Some(t.clone())
            } else {
                None
            };

        let indices = if let Some(Indices::U32(t)) = mesh.indices() {
            Some(t.clone())
        } else {
            None
        };

        let mesh_type_num = match mesh.primitive_topology() {
            PrimitiveTopology::PointList => 0,
            PrimitiveTopology::LineList => 1,
            PrimitiveTopology::LineStrip => 2,
            PrimitiveTopology::TriangleList => 3,
            PrimitiveTopology::TriangleStrip => 4,
        };

        MeshData {
            mesh_type: mesh_type_num,
            positions,
            normals,
            uvs,
            tangents,
            colors,
            joint_weights,
            joint_indices,
            indices,
        }
    }
}

impl From<Mesh> for bevy::prelude::Mesh {
    fn from(val: Mesh) -> Self {
        use bevy::prelude::shape;
        match val.inner {
            MeshInner::Sphere {
                radius,
                sectors,
                stacks,
            } => shape::UVSphere {
                radius,
                sectors,
                stacks,
            }
            .into(),
            MeshInner::Box { x, y, z } => shape::Box::new(x, y, z).into(),
            MeshInner::Cylinder {
                radius,
                height,
                resolution,
                segments,
            } => shape::Cylinder {
                radius,
                height,
                resolution,
                segments,
            }
            .into(),
            MeshInner::Data(d) => d.into(),
        }
    }
}

impl From<Color> for bevy::prelude::Color {
    fn from(val: Color) -> Self {
        bevy::prelude::Color::rgb(val.r, val.g, val.b)
    }
}

impl From<bevy::prelude::Color> for Color {
    fn from(value: bevy::prelude::Color) -> Self {
        Self {
            r: value.r(),
            g: value.g(),
            b: value.b(),
        }
    }
}

impl From<AlphaMode> for bevy::prelude::AlphaMode {
    fn from(val: AlphaMode) -> Self {
        match val {
            AlphaMode::Opaque => bevy::prelude::AlphaMode::Opaque,
            AlphaMode::Mask(m) => bevy::prelude::AlphaMode::Mask(m),
            AlphaMode::Blend => bevy::prelude::AlphaMode::Blend,
            AlphaMode::Premultiplied => bevy::prelude::AlphaMode::Premultiplied,
            AlphaMode::Add => bevy::prelude::AlphaMode::Add,
            AlphaMode::Multiply => bevy::prelude::AlphaMode::Multiply,
        }
    }
}

impl From<bevy::prelude::AlphaMode> for AlphaMode {
    fn from(value: bevy::prelude::AlphaMode) -> Self {
        match value {
            bevy::prelude::AlphaMode::Opaque => AlphaMode::Opaque,
            bevy::prelude::AlphaMode::Mask(m) => AlphaMode::Mask(m),
            bevy::prelude::AlphaMode::Blend => AlphaMode::Blend,
            bevy::prelude::AlphaMode::Premultiplied => AlphaMode::Premultiplied,
            bevy::prelude::AlphaMode::Add => AlphaMode::Add,
            bevy::prelude::AlphaMode::Multiply => AlphaMode::Multiply,
        }
    }
}

impl From<ParallaxMappingMethod> for bevy::prelude::ParallaxMappingMethod {
    fn from(val: ParallaxMappingMethod) -> Self {
        match val {
            ParallaxMappingMethod::Occlusion => bevy::prelude::ParallaxMappingMethod::Occlusion,
            ParallaxMappingMethod::Relief { max_steps } => {
                bevy::prelude::ParallaxMappingMethod::Relief { max_steps }
            }
        }
    }
}

impl From<bevy::prelude::ParallaxMappingMethod> for ParallaxMappingMethod {
    fn from(value: bevy::prelude::ParallaxMappingMethod) -> Self {
        match value {
            bevy::prelude::ParallaxMappingMethod::Occlusion => ParallaxMappingMethod::Occlusion,
            bevy::prelude::ParallaxMappingMethod::Relief { max_steps } => {
                ParallaxMappingMethod::Relief { max_steps }
            }
        }
    }
}

impl From<OpaqueRendererMethod> for bevy::pbr::OpaqueRendererMethod {
    fn from(val: OpaqueRendererMethod) -> Self {
        match val {
            OpaqueRendererMethod::Forward => bevy::pbr::OpaqueRendererMethod::Forward,
            OpaqueRendererMethod::Deferred => bevy::pbr::OpaqueRendererMethod::Deferred,
            OpaqueRendererMethod::Auto => bevy::pbr::OpaqueRendererMethod::Auto,
        }
    }
}

impl From<bevy::pbr::OpaqueRendererMethod> for OpaqueRendererMethod {
    fn from(value: bevy::pbr::OpaqueRendererMethod) -> Self {
        match value {
            bevy::pbr::OpaqueRendererMethod::Forward => OpaqueRendererMethod::Forward,
            bevy::pbr::OpaqueRendererMethod::Deferred => OpaqueRendererMethod::Deferred,
            bevy::pbr::OpaqueRendererMethod::Auto => OpaqueRendererMethod::Auto,
        }
    }
}

impl From<Face> for bevy::render::render_resource::Face {
    fn from(val: Face) -> Self {
        match val {
            Face::Front => bevy::render::render_resource::Face::Front,
            Face::Back => bevy::render::render_resource::Face::Back,
        }
    }
}

impl From<bevy::render::render_resource::Face> for Face {
    fn from(value: bevy::render::render_resource::Face) -> Self {
        match value {
            bevy::render::render_resource::Face::Front => Face::Front,
            bevy::render::render_resource::Face::Back => Face::Back,
        }
    }
}

impl From<Extent3d> for bevy::render::render_resource::Extent3d {
    fn from(val: Extent3d) -> Self {
        bevy::render::render_resource::Extent3d {
            width: val.width,
            height: val.height,
            depth_or_array_layers: val.depth,
        }
    }
}

impl From<bevy::render::render_resource::Extent3d> for Extent3d {
    fn from(extent: bevy::render::render_resource::Extent3d) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
            depth: extent.depth_or_array_layers,
        }
    }
}

impl From<TextureDimension> for bevy::render::render_resource::TextureDimension {
    fn from(val: TextureDimension) -> Self {
        match val {
            TextureDimension::D1 => bevy::render::render_resource::TextureDimension::D1,
            TextureDimension::D2 => bevy::render::render_resource::TextureDimension::D2,
            TextureDimension::D3 => bevy::render::render_resource::TextureDimension::D3,
        }
    }
}

impl From<bevy::render::render_resource::TextureDimension> for TextureDimension {
    fn from(value: bevy::render::render_resource::TextureDimension) -> Self {
        match value {
            bevy::render::render_resource::TextureDimension::D1 => TextureDimension::D1,
            bevy::render::render_resource::TextureDimension::D2 => TextureDimension::D2,
            bevy::render::render_resource::TextureDimension::D3 => TextureDimension::D3,
        }
    }
}

impl From<TextureFormat> for bevy::render::render_resource::TextureFormat {
    fn from(val: TextureFormat) -> Self {
        match val {
            TextureFormat::R8Unorm => bevy::render::render_resource::TextureFormat::R8Unorm,
            TextureFormat::R8Snorm => bevy::render::render_resource::TextureFormat::R8Snorm,
            TextureFormat::R8Uint => bevy::render::render_resource::TextureFormat::R8Uint,
            TextureFormat::R8Sint => bevy::render::render_resource::TextureFormat::R8Sint,
            TextureFormat::R16Uint => bevy::render::render_resource::TextureFormat::R16Uint,
            TextureFormat::R16Sint => bevy::render::render_resource::TextureFormat::R16Sint,
            TextureFormat::R16Unorm => bevy::render::render_resource::TextureFormat::R16Unorm,
            TextureFormat::R16Snorm => bevy::render::render_resource::TextureFormat::R16Snorm,
            TextureFormat::R16Float => bevy::render::render_resource::TextureFormat::R16Float,
            TextureFormat::Rg8Unorm => bevy::render::render_resource::TextureFormat::Rg8Unorm,
            TextureFormat::Rg8Snorm => bevy::render::render_resource::TextureFormat::Rg8Snorm,
            TextureFormat::Rg8Uint => bevy::render::render_resource::TextureFormat::Rg8Uint,
            TextureFormat::Rg8Sint => bevy::render::render_resource::TextureFormat::Rg8Sint,
            TextureFormat::R32Uint => bevy::render::render_resource::TextureFormat::R32Uint,
            TextureFormat::R32Sint => bevy::render::render_resource::TextureFormat::R32Sint,
            TextureFormat::R32Float => bevy::render::render_resource::TextureFormat::R32Float,
            TextureFormat::Rg16Uint => bevy::render::render_resource::TextureFormat::Rg16Uint,
            TextureFormat::Rg16Sint => bevy::render::render_resource::TextureFormat::Rg16Sint,
            TextureFormat::Rg16Unorm => bevy::render::render_resource::TextureFormat::Rg16Unorm,
            TextureFormat::Rg16Snorm => bevy::render::render_resource::TextureFormat::Rg16Snorm,
            TextureFormat::Rg16Float => bevy::render::render_resource::TextureFormat::Rg16Float,
            TextureFormat::Rgba8Unorm => bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb => {
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb
            }
            TextureFormat::Rgba8Snorm => bevy::render::render_resource::TextureFormat::Rgba8Snorm,
            TextureFormat::Rgba8Uint => bevy::render::render_resource::TextureFormat::Rgba8Uint,
            TextureFormat::Rgba8Sint => bevy::render::render_resource::TextureFormat::Rgba8Sint,
            TextureFormat::Bgra8Unorm => bevy::render::render_resource::TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb => {
                bevy::render::render_resource::TextureFormat::Bgra8UnormSrgb
            }
            TextureFormat::Rgb9e5Ufloat => {
                bevy::render::render_resource::TextureFormat::Rgb9e5Ufloat
            }
            TextureFormat::Rgb10a2Unorm => {
                bevy::render::render_resource::TextureFormat::Rgb10a2Unorm
            }
            TextureFormat::Rg11b10Float => {
                bevy::render::render_resource::TextureFormat::Rg11b10Float
            }
            TextureFormat::Rg32Uint => bevy::render::render_resource::TextureFormat::Rg32Uint,
            TextureFormat::Rg32Sint => bevy::render::render_resource::TextureFormat::Rg32Sint,
            TextureFormat::Rg32Float => bevy::render::render_resource::TextureFormat::Rg32Float,
            TextureFormat::Rgba16Uint => bevy::render::render_resource::TextureFormat::Rgba16Uint,
            TextureFormat::Rgba16Sint => bevy::render::render_resource::TextureFormat::Rgba16Sint,
            TextureFormat::Rgba16Unorm => bevy::render::render_resource::TextureFormat::Rgba16Unorm,
            TextureFormat::Rgba16Snorm => bevy::render::render_resource::TextureFormat::Rgba16Snorm,
            TextureFormat::Rgba16Float => bevy::render::render_resource::TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Uint => bevy::render::render_resource::TextureFormat::Rgba32Uint,
            TextureFormat::Rgba32Sint => bevy::render::render_resource::TextureFormat::Rgba32Sint,
            TextureFormat::Rgba32Float => bevy::render::render_resource::TextureFormat::Rgba32Float,
            TextureFormat::Stencil8 => bevy::render::render_resource::TextureFormat::Stencil8,
            TextureFormat::Depth16Unorm => {
                bevy::render::render_resource::TextureFormat::Depth16Unorm
            }
            TextureFormat::Depth24Plus => bevy::render::render_resource::TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8 => {
                bevy::render::render_resource::TextureFormat::Depth24PlusStencil8
            }
            TextureFormat::Depth32Float => {
                bevy::render::render_resource::TextureFormat::Depth32Float
            }
            TextureFormat::Depth32FloatStencil8 => {
                bevy::render::render_resource::TextureFormat::Depth32FloatStencil8
            }
            TextureFormat::Bc1RgbaUnorm => {
                bevy::render::render_resource::TextureFormat::Bc1RgbaUnorm
            }
            TextureFormat::Bc1RgbaUnormSrgb => {
                bevy::render::render_resource::TextureFormat::Bc1RgbaUnormSrgb
            }
            TextureFormat::Bc2RgbaUnorm => {
                bevy::render::render_resource::TextureFormat::Bc2RgbaUnorm
            }
            TextureFormat::Bc2RgbaUnormSrgb => {
                bevy::render::render_resource::TextureFormat::Bc2RgbaUnormSrgb
            }
            TextureFormat::Bc3RgbaUnorm => {
                bevy::render::render_resource::TextureFormat::Bc3RgbaUnorm
            }
            TextureFormat::Bc3RgbaUnormSrgb => {
                bevy::render::render_resource::TextureFormat::Bc3RgbaUnormSrgb
            }
            TextureFormat::Bc4RUnorm => bevy::render::render_resource::TextureFormat::Bc4RUnorm,
            TextureFormat::Bc4RSnorm => bevy::render::render_resource::TextureFormat::Bc4RSnorm,
            TextureFormat::Bc5RgUnorm => bevy::render::render_resource::TextureFormat::Bc5RgUnorm,
            TextureFormat::Bc5RgSnorm => bevy::render::render_resource::TextureFormat::Bc5RgSnorm,
            TextureFormat::Bc6hRgbUfloat => {
                bevy::render::render_resource::TextureFormat::Bc6hRgbUfloat
            }
            TextureFormat::Bc6hRgbFloat => {
                bevy::render::render_resource::TextureFormat::Bc6hRgbFloat
            }
            TextureFormat::Bc7RgbaUnorm => {
                bevy::render::render_resource::TextureFormat::Bc7RgbaUnorm
            }
            TextureFormat::Bc7RgbaUnormSrgb => {
                bevy::render::render_resource::TextureFormat::Bc7RgbaUnormSrgb
            }
            TextureFormat::Etc2Rgb8Unorm => {
                bevy::render::render_resource::TextureFormat::Etc2Rgb8Unorm
            }
            TextureFormat::Etc2Rgb8UnormSrgb => {
                bevy::render::render_resource::TextureFormat::Etc2Rgb8UnormSrgb
            }
            TextureFormat::Etc2Rgb8A1Unorm => {
                bevy::render::render_resource::TextureFormat::Etc2Rgb8A1Unorm
            }
            TextureFormat::Etc2Rgb8A1UnormSrgb => {
                bevy::render::render_resource::TextureFormat::Etc2Rgb8A1UnormSrgb
            }
            TextureFormat::Etc2Rgba8Unorm => {
                bevy::render::render_resource::TextureFormat::Etc2Rgba8Unorm
            }
            TextureFormat::Etc2Rgba8UnormSrgb => {
                bevy::render::render_resource::TextureFormat::Etc2Rgba8UnormSrgb
            }
            TextureFormat::EacR11Unorm => bevy::render::render_resource::TextureFormat::EacR11Unorm,
            TextureFormat::EacR11Snorm => bevy::render::render_resource::TextureFormat::EacR11Snorm,
            TextureFormat::EacRg11Unorm => {
                bevy::render::render_resource::TextureFormat::EacRg11Unorm
            }
            TextureFormat::EacRg11Snorm => {
                bevy::render::render_resource::TextureFormat::EacRg11Snorm
            }
        }
    }
}

impl From<bevy::render::render_resource::TextureFormat> for TextureFormat {
    fn from(val: bevy::render::render_resource::TextureFormat) -> Self {
        use bevy::render::render_resource::TextureFormat as BevyTextureFormat;
        match val {
            BevyTextureFormat::R8Unorm => TextureFormat::R8Unorm,
            BevyTextureFormat::R8Snorm => TextureFormat::R8Snorm,
            BevyTextureFormat::R8Uint => TextureFormat::R8Uint,
            BevyTextureFormat::R8Sint => TextureFormat::R8Sint,
            BevyTextureFormat::R16Uint => TextureFormat::R16Uint,
            BevyTextureFormat::R16Sint => TextureFormat::R16Sint,
            BevyTextureFormat::R16Unorm => TextureFormat::R16Unorm,
            BevyTextureFormat::R16Snorm => TextureFormat::R16Snorm,
            BevyTextureFormat::R16Float => TextureFormat::R16Float,
            BevyTextureFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
            BevyTextureFormat::Rg8Snorm => TextureFormat::Rg8Snorm,
            BevyTextureFormat::Rg8Uint => TextureFormat::Rg8Uint,
            BevyTextureFormat::Rg8Sint => TextureFormat::Rg8Sint,
            BevyTextureFormat::R32Uint => TextureFormat::R32Uint,
            BevyTextureFormat::R32Sint => TextureFormat::R32Sint,
            BevyTextureFormat::R32Float => TextureFormat::R32Float,
            BevyTextureFormat::Rg16Uint => TextureFormat::Rg16Uint,
            BevyTextureFormat::Rg16Sint => TextureFormat::Rg16Sint,
            BevyTextureFormat::Rg16Unorm => TextureFormat::Rg16Unorm,
            BevyTextureFormat::Rg16Snorm => TextureFormat::Rg16Snorm,
            BevyTextureFormat::Rg16Float => TextureFormat::Rg16Float,
            BevyTextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            BevyTextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
            BevyTextureFormat::Rgba8Snorm => TextureFormat::Rgba8Snorm,
            BevyTextureFormat::Rgba8Uint => TextureFormat::Rgba8Uint,
            BevyTextureFormat::Rgba8Sint => TextureFormat::Rgba8Sint,
            BevyTextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            BevyTextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
            BevyTextureFormat::Rgb9e5Ufloat => TextureFormat::Rgb9e5Ufloat,
            BevyTextureFormat::Rgb10a2Unorm => TextureFormat::Rgb10a2Unorm,
            BevyTextureFormat::Rg11b10Float => TextureFormat::Rg11b10Float,
            BevyTextureFormat::Rg32Uint => TextureFormat::Rg32Uint,
            BevyTextureFormat::Rg32Sint => TextureFormat::Rg32Sint,
            BevyTextureFormat::Rg32Float => TextureFormat::Rg32Float,
            BevyTextureFormat::Rgba16Uint => TextureFormat::Rgba16Uint,
            BevyTextureFormat::Rgba16Sint => TextureFormat::Rgba16Sint,
            BevyTextureFormat::Rgba16Unorm => TextureFormat::Rgba16Unorm,
            BevyTextureFormat::Rgba16Snorm => TextureFormat::Rgba16Snorm,
            BevyTextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
            BevyTextureFormat::Rgba32Uint => TextureFormat::Rgba32Uint,
            BevyTextureFormat::Rgba32Sint => TextureFormat::Rgba32Sint,
            BevyTextureFormat::Rgba32Float => TextureFormat::Rgba32Float,
            BevyTextureFormat::Stencil8 => TextureFormat::Stencil8,
            BevyTextureFormat::Depth16Unorm => TextureFormat::Depth16Unorm,
            BevyTextureFormat::Depth24Plus => TextureFormat::Depth24Plus,
            BevyTextureFormat::Depth24PlusStencil8 => TextureFormat::Depth24PlusStencil8,
            BevyTextureFormat::Depth32Float => TextureFormat::Depth32Float,
            BevyTextureFormat::Depth32FloatStencil8 => TextureFormat::Depth32FloatStencil8,
            BevyTextureFormat::Bc1RgbaUnorm => TextureFormat::Bc1RgbaUnorm,
            BevyTextureFormat::Bc1RgbaUnormSrgb => TextureFormat::Bc1RgbaUnormSrgb,
            BevyTextureFormat::Bc2RgbaUnorm => TextureFormat::Bc2RgbaUnorm,
            BevyTextureFormat::Bc2RgbaUnormSrgb => TextureFormat::Bc2RgbaUnormSrgb,
            BevyTextureFormat::Bc3RgbaUnorm => TextureFormat::Bc3RgbaUnorm,
            BevyTextureFormat::Bc3RgbaUnormSrgb => TextureFormat::Bc3RgbaUnormSrgb,
            BevyTextureFormat::Bc4RUnorm => TextureFormat::Bc4RUnorm,
            BevyTextureFormat::Bc4RSnorm => TextureFormat::Bc4RSnorm,
            BevyTextureFormat::Bc5RgUnorm => TextureFormat::Bc5RgUnorm,
            BevyTextureFormat::Bc5RgSnorm => TextureFormat::Bc5RgSnorm,
            BevyTextureFormat::Bc6hRgbUfloat => TextureFormat::Bc6hRgbUfloat,
            BevyTextureFormat::Bc6hRgbFloat => TextureFormat::Bc6hRgbFloat,
            BevyTextureFormat::Bc7RgbaUnorm => TextureFormat::Bc7RgbaUnorm,
            BevyTextureFormat::Bc7RgbaUnormSrgb => TextureFormat::Bc7RgbaUnormSrgb,
            BevyTextureFormat::Etc2Rgb8Unorm => TextureFormat::Etc2Rgb8Unorm,
            BevyTextureFormat::Etc2Rgb8UnormSrgb => TextureFormat::Etc2Rgb8UnormSrgb,
            BevyTextureFormat::Etc2Rgb8A1Unorm => TextureFormat::Etc2Rgb8A1Unorm,
            BevyTextureFormat::Etc2Rgb8A1UnormSrgb => TextureFormat::Etc2Rgb8A1UnormSrgb,
            BevyTextureFormat::Etc2Rgba8Unorm => TextureFormat::Etc2Rgba8Unorm,
            BevyTextureFormat::Etc2Rgba8UnormSrgb => TextureFormat::Etc2Rgba8UnormSrgb,
            BevyTextureFormat::EacR11Unorm => TextureFormat::EacR11Unorm,
            BevyTextureFormat::EacR11Snorm => TextureFormat::EacR11Snorm,
            BevyTextureFormat::EacRg11Unorm => TextureFormat::EacRg11Unorm,
            BevyTextureFormat::EacRg11Snorm => TextureFormat::EacRg11Snorm,
            BevyTextureFormat::Astc { .. } => todo!(),
        }
    }
}

impl From<Image> for bevy::prelude::Image {
    fn from(val: Image) -> Self {
        bevy::prelude::Image::new(
            val.size.into(),
            val.texture_dimension.into(),
            val.data,
            val.format.into(),
        )
    }
}

impl From<bevy::prelude::Image> for Image {
    fn from(value: bevy::prelude::Image) -> Self {
        Self {
            data: value.data,
            size: value.texture_descriptor.size.into(),
            texture_dimension: value.texture_descriptor.dimension.into(),
            format: value.texture_descriptor.format.into(),
        }
    }
}

impl Material {
    pub fn into_material(
        self,
        images: &mut Assets<bevy::prelude::Image>,
    ) -> bevy::prelude::StandardMaterial {
        bevy::prelude::StandardMaterial {
            base_color: self.base_color.into(),
            base_color_texture: self.base_color_texture.map(|x| images.add(x.into())),
            emissive: self.emissive.into(),
            emissive_texture: self.emissive_texture.map(|x| images.add(x.into())),
            perceptual_roughness: self.perceptual_roughness,
            metallic: self.metallic,
            metallic_roughness_texture: self
                .metallic_roughness_texture
                .map(|x| images.add(x.into())),
            reflectance: self.reflectance,
            diffuse_transmission: self.diffuse_transmission,
            specular_transmission: self.specular_transmission,
            thickness: self.thickness,
            ior: self.ior,
            attenuation_distance: self.attenuation_distance,
            attenuation_color: self.attenuation_color.into(),
            normal_map_texture: self.normal_map_texture.map(|x| images.add(x.into())),
            flip_normal_map_y: self.flip_normal_map_y,
            occlusion_texture: self.occlusion_texture.map(|x| images.add(x.into())),
            double_sided: self.double_sided,
            cull_mode: self.cull_mode.map(Face::into),
            unlit: self.unlit,
            fog_enabled: self.fog_enabled,
            alpha_mode: self.alpha_mode.into(),
            depth_bias: self.depth_bias,
            depth_map: self.depth_map.map(|x| images.add(x.into())),
            parallax_depth_scale: self.parallax_depth_scale,
            parallax_mapping_method: self.parallax_mapping_method.into(),
            max_parallax_layer_count: self.max_parallax_layer_count,
            opaque_render_method: self.opaque_render_method.into(),
            deferred_lighting_pass_id: self.deferred_lighting_pass_id,
        }
    }
}

impl Material {
    pub fn from_bevy(
        value: bevy::prelude::StandardMaterial,
        images: &Assets<bevy::prelude::Image>,
    ) -> Self {
        Self {
            base_color: value.base_color.into(),
            base_color_texture: value
                .base_color_texture
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            emissive: value.emissive.into(),
            emissive_texture: value
                .emissive_texture
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            perceptual_roughness: value.perceptual_roughness,
            metallic: value.metallic,
            metallic_roughness_texture: value
                .metallic_roughness_texture
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            reflectance: value.reflectance,
            diffuse_transmission: value.diffuse_transmission,
            specular_transmission: value.specular_transmission,
            thickness: value.thickness,
            ior: value.ior,
            attenuation_distance: value.attenuation_distance,
            attenuation_color: value.attenuation_color.into(),
            normal_map_texture: value
                .normal_map_texture
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            flip_normal_map_y: value.flip_normal_map_y,
            occlusion_texture: value
                .occlusion_texture
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            double_sided: value.double_sided,
            cull_mode: value.cull_mode.map(Face::from),
            unlit: value.unlit,
            fog_enabled: value.fog_enabled,
            alpha_mode: value.alpha_mode.into(),
            depth_bias: value.depth_bias,
            depth_map: value
                .depth_map
                .and_then(|id| images.get(id))
                .cloned()
                .map(Image::from),
            parallax_depth_scale: value.parallax_depth_scale,
            parallax_mapping_method: value.parallax_mapping_method.into(),
            max_parallax_layer_count: value.max_parallax_layer_count,
            opaque_render_method: value.opaque_render_method.into(),
            deferred_lighting_pass_id: value.deferred_lighting_pass_id,
        }
    }
}
