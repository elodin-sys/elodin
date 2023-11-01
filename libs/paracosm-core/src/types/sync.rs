use bevy::{
    prelude::{
        AlphaMode, Color, Deref, DerefMut, Handle, Image, Mesh, ParallaxMappingMethod,
        StandardMaterial,
    },
    reflect::{
        serde::{TypedReflectDeserializer, TypedReflectSerializer},
        FromReflect, GetTypeRegistration, Reflect, TypeRegistryInternal,
    },
    render::{mesh::Indices, render_resource::PrimitiveTopology},
};
use bevy_ecs::component::Component;
use nalgebra::Vector3;
use serde::{de::DeserializeSeed, Deserialize, Serialize};

use crate::spatial::SpatialPos;

pub enum ServerMsg {
    Exit,
    RequestModel(Uuid),
}

pub enum ClientMsg {
    Clear,
    SyncWorldPos(SyncWorldPos),
    ModelDataResp(ModelData),
}

pub struct SyncWorldPos {
    pub body_id: Uuid,
    pub pos: SpatialPos,
}

pub enum ModelData {
    Pbr {
        body_id: Uuid,
        material: ReflectSerde<StandardMaterial>,
        mesh: MeshData,
    },
    Glb {
        body_id: Uuid,
        data: Vec<u8>,
    },
}

pub struct MeshData {
    mesh_type: u8,
    positions: Option<Vec<[f32; 3]>>,
    normals: Option<Vec<[f32; 3]>>,
    uvs: Option<Vec<[f32; 2]>>,
    tangents: Option<Vec<[f32; 4]>>,
    colors: Option<Vec<[f32; 4]>>,
    joint_weights: Option<Vec<[f32; 4]>>,
    joint_indices: Option<Vec<[u16; 4]>>,
    indices: Option<Vec<u32>>,
}

impl From<MeshData> for bevy::prelude::Mesh {
    fn from(data: MeshData) -> Self {
        use bevy::render::mesh::VertexAttributeValues::{
            Float32x2, Float32x3, Float32x4, Uint16x4,
        };
        let mesh_type_enum = match data.mesh_type {
            0 => PrimitiveTopology::PointList,
            1 => PrimitiveTopology::LineList,
            2 => PrimitiveTopology::LineStrip,
            3 => PrimitiveTopology::TriangleList,
            4 => PrimitiveTopology::TriangleStrip,
            _ => PrimitiveTopology::TriangleList,
        };

        let mut mesh = Mesh::new(mesh_type_enum);

        if let Some(positions) = data.positions {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        }

        if let Some(normals) = data.normals {
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        }

        if let Some(uvs) = data.uvs {
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        }

        if let Some(tangents) = data.tangents {
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
        }

        if let Some(colors) = data.colors {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }

        if let Some(joint_weights) = data.joint_weights {
            mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, joint_weights);
        }

        if let Some(joint_indices) = data.joint_indices {
            mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_INDEX, Uint16x4(joint_indices));
        }

        if let Some(indices) = data.indices {
            mesh.set_indices(Some(Indices::U32(indices)));
        }

        mesh
    }
}

impl From<bevy::prelude::Mesh> for MeshData {
    fn from(mesh: bevy::prelude::Mesh) -> Self {
        use bevy::render::mesh::VertexAttributeValues::{
            Float32x2, Float32x3, Float32x4, Uint16x4,
        };

        let positions = if let Some(Float32x3(t)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(t.clone())
        } else {
            None
        };

        let tangents = if let Some(Float32x4(t)) = mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
            Some(t.clone())
        } else {
            None
        };

        let normals = if let Some(Float32x3(t)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(t.clone())
        } else {
            None
        };

        let uvs = if let Some(Float32x2(t)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(t.clone())
        } else {
            None
        };

        let colors = if let Some(Float32x4(t)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(t.clone())
        } else {
            None
        };

        let joint_weights = if let Some(Float32x4(t)) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
        {
            Some(t.clone())
        } else {
            None
        };

        let joint_indices = if let Some(Uint16x4(t)) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Component, Clone, Copy, Debug)]
pub struct Uuid(pub u128);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Component, Clone, Copy, DerefMut, Deref, Debug)]
pub struct Synced(pub bool);

pub trait RecursiveReg {
    fn register(register: &mut TypeRegistryInternal);
}

impl RecursiveReg for StandardMaterial {
    fn register(register: &mut TypeRegistryInternal) {
        register.register::<Self>();
        register.register::<Color>();
        register.register::<Option<Handle<Image>>>();
        register.register::<AlphaMode>();
        register.register::<ParallaxMappingMethod>();
    }
}

#[derive(PartialEq, Eq)]
pub struct ReflectSerde<T>(pub T);

impl<T: Reflect + GetTypeRegistration + RecursiveReg> Serialize for ReflectSerde<T> {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut reg = TypeRegistryInternal::default();
        T::register(&mut reg);
        let serializer = TypedReflectSerializer::new(self.0.as_reflect(), &reg);

        serializer.serialize(s)
    }
}

impl<'de, T: Reflect + GetTypeRegistration + RecursiveReg + FromReflect> Deserialize<'de>
    for ReflectSerde<T>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut reg = TypeRegistryInternal::default();
        T::register(&mut reg);

        let type_reg = T::get_type_registration();
        let typed_deserialize = TypedReflectDeserializer::new(&type_reg, &reg);
        let reflect = typed_deserialize.deserialize(deserializer)?;
        Ok(ReflectSerde(T::from_reflect(reflect.as_ref()).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::{Color, StandardMaterial};

    use super::*;

    #[test]
    fn test_serialize_mesh() {
        //let mesh = Mesh::from(shape::Box::new(0.2, 1.0, 0.2));
        let mat = bevy::prelude::StandardMaterial {
            base_color: Color::hex("38ACFF").unwrap(),
            metallic: 0.6,
            perceptual_roughness: 0.1,
            ..Default::default()
        };
        let mat = ReflectSerde(mat);
        let data = serde_json::to_string(&mat).unwrap();
        println!("{:?}", data);
        let mat_b: ReflectSerde<StandardMaterial> = serde_json::from_str(&data).unwrap();
        assert_eq!(mat.0.base_color, mat_b.0.base_color)
    }
}
