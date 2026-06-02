//! This module contains the two fundamental data structures of the terrain:
//! the [`TileTree`] and the [`TileAtlas`].
//!
//! # Explanation
//! Each terrain possesses one [`TileAtlas`], which can be configured
//! to store any [`AtlasAttachment`](tile_atlas::AtlasAttachment) required (eg. height, density, albedo, splat, edc.)
//! These attachments can vary in resolution and texture format.
//!
//! To decide which tiles should be currently loaded you can create multiple
//! [`TileTree`] views that correspond to one tile atlas.
//! These tile_trees request and release tiles from the tile atlas based on their quality
//! setting (`load_distance`).
//! Additionally they are then used to access the best loaded data at any position.
//!
//! Both the tile atlas and the tile_trees also have a corresponding GPU representation,
//! which can be used to access the terrain data in shaders.

use crate::terrain::{
    terrain_data::{tile_atlas::TileAtlas, tile_tree::TileTree},
    util::CollectArray,
};
use bevy::{math::DVec3, prelude::*, render::render_resource::*};
use bincode::{Decode, Encode};
use bytemuck::cast_slice;
use itertools::iproduct;
use std::iter;

pub mod gpu_tile_atlas;
pub mod gpu_tile_tree;
pub mod tile_atlas;
pub mod tile_tree;

pub const INVALID_ATLAS_INDEX: u32 = u32::MAX;
pub const INVALID_LOD: u32 = u32::MAX;

/// The data format of an attachment.
#[derive(Encode, Decode, Clone, Copy, Debug)]
pub enum AttachmentFormat {
    /// Three channels  8 bit
    Rgb8,
    /// Four  channels  8 bit
    Rgba8,
    /// One   channel  16 bit
    R16,
    /// Two   channels 16 bit
    Rg16,
}

impl AttachmentFormat {
    pub(crate) fn id(self) -> u32 {
        match self {
            AttachmentFormat::Rgb8 => 5,
            AttachmentFormat::Rgba8 => 0,
            AttachmentFormat::R16 => 1,
            AttachmentFormat::Rg16 => 3,
        }
    }
    pub(crate) fn render_format(self) -> TextureFormat {
        match self {
            AttachmentFormat::Rgb8 => TextureFormat::Rgba8UnormSrgb,
            AttachmentFormat::Rgba8 => TextureFormat::Rgba8UnormSrgb,
            AttachmentFormat::R16 => TextureFormat::R16Unorm,
            AttachmentFormat::Rg16 => TextureFormat::Rg16Unorm,
        }
    }

    pub(crate) fn processing_format(self) -> TextureFormat {
        match self {
            AttachmentFormat::Rgb8 => TextureFormat::Rgba8Unorm,
            AttachmentFormat::Rgba8 => TextureFormat::Rgba8Unorm,
            AttachmentFormat::R16 => TextureFormat::R16Unorm,
            AttachmentFormat::Rg16 => TextureFormat::Rg16Unorm,
        }
    }

    pub(crate) fn pixel_size(self) -> u32 {
        match self {
            AttachmentFormat::Rgb8 => 3,
            AttachmentFormat::Rgba8 => 4,
            AttachmentFormat::R16 => 2,
            AttachmentFormat::Rg16 => 4,
        }
    }
}

/// Configures an attachment.
#[derive(Encode, Decode, Clone, Debug)]
pub struct AttachmentConfig {
    /// The name of the attachment.
    pub name: String,
    pub texture_size: u32,
    /// The overlapping border size around the tile, used to prevent sampling artifacts.
    pub border_size: u32,
    pub mip_level_count: u32,
    /// The format of the attachment.
    pub format: AttachmentFormat,
}

impl Default for AttachmentConfig {
    fn default() -> Self {
        Self {
            name: "".to_string(),
            texture_size: 512,
            border_size: 1,
            mip_level_count: 1,
            format: AttachmentFormat::R16,
        }
    }
}

#[derive(Clone)]
pub(crate) enum AttachmentData {
    None,
    /// Three channels  8 bit
    // Rgb8(Vec<(u8, u8, u8)>), Can not be represented currently
    /// Four  channels  8 bit
    Rgba8(Vec<[u8; 4]>),
    /// One   channel  16 bit
    R16(Vec<u16>),
    /// Two   channels 16 bit
    Rg16(Vec<[u16; 2]>),
}

impl AttachmentData {
    pub(crate) fn from_bytes(data: &[u8], format: AttachmentFormat) -> Self {
        match format {
            AttachmentFormat::Rgb8 => unimplemented!(),
            AttachmentFormat::Rgba8 => Self::Rgba8(cast_slice(data).to_vec()),
            AttachmentFormat::R16 => Self::R16(cast_slice(data).to_vec()),
            AttachmentFormat::Rg16 => Self::Rg16(cast_slice(data).to_vec()),
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            AttachmentData::Rgba8(data) => cast_slice(data),
            AttachmentData::R16(data) => cast_slice(data),
            AttachmentData::Rg16(data) => cast_slice(data),
            AttachmentData::None => panic!("Attachment has no data."),
        }
    }

    pub(crate) fn generate_mipmaps(&mut self, texture_size: u32, mip_level_count: u32) {
        fn generate_mipmap_rgba8(
            data: &mut Vec<[u8; 4]>,
            parent_size: usize,
            child_size: usize,
            start: usize,
        ) {
            for (child_y, child_x) in iproduct!(0..child_size, 0..child_size) {
                let mut value = [0u64; 4];

                for i in 0..4 {
                    let parent_x = (child_x << 1) + (i >> 1);
                    let parent_y = (child_y << 1) + (i & 1);

                    let index = start + parent_y * parent_size + parent_x;

                    iter::zip(&mut value, data[index]).for_each(|(value, v)| *value += v as u64);
                }

                let value = value.iter().map(|value| (value / 4) as u8).collect_array();

                data.push(value);
            }
        }

        fn generate_mipmap_r16(
            data: &mut Vec<u16>,
            parent_size: usize,
            child_size: usize,
            start: usize,
        ) {
            for (child_y, child_x) in iproduct!(0..child_size, 0..child_size) {
                let mut value = 0;
                let mut count = 0;

                for (parent_x, parent_y) in
                    iproduct!(0..2, 0..2).map(|(x, y)| ((child_x << 1) + x, (child_y << 1) + y))
                {
                    let index = start + parent_y * parent_size + parent_x;
                    let data = data[index] as u32;

                    if data != 0 {
                        value += data;
                        count += 1;
                    }
                }

                let value = if count == 0 {
                    0
                } else {
                    (value / count) as u16
                };

                data.push(value);
            }
        }

        let mut start = 0;
        let mut parent_size = texture_size as usize;

        for _mip_level in 1..mip_level_count {
            let child_size = parent_size >> 1;

            match self {
                AttachmentData::Rgba8(data) => {
                    generate_mipmap_rgba8(data, parent_size, child_size, start)
                }
                AttachmentData::R16(data) => {
                    generate_mipmap_r16(data, parent_size, child_size, start)
                }
                _ => {}
            }

            start += parent_size * parent_size;
            parent_size = child_size;
        }
    }

    pub(crate) fn sample(&self, uv: Vec2, size: u32) -> Vec4 {
        let uv = uv * size as f32 - 0.5;

        let remainder = uv % 1.0;
        let uv = uv.as_ivec2();

        let mut values = [[Vec4::ZERO; 2]; 2];

        for (x, y) in iproduct!(0..2, 0..2) {
            let index = (uv.y + y) * size as i32 + (uv.x + x);

            values[x as usize][y as usize] = match self {
                AttachmentData::None => Vec4::splat(0.0),
                AttachmentData::Rgba8(data) => {
                    let value = data[index as usize];
                    Vec4::new(
                        value[0] as f32 / u8::MAX as f32,
                        value[1] as f32 / u8::MAX as f32,
                        value[2] as f32 / u8::MAX as f32,
                        value[3] as f32 / u8::MAX as f32,
                    )
                }
                AttachmentData::R16(data) => {
                    let value = data[index as usize];
                    Vec4::new(value as f32 / u16::MAX as f32, 0.0, 0.0, 0.0)
                }
                AttachmentData::Rg16(data) => {
                    let value = data[index as usize];
                    Vec4::new(
                        value[0] as f32 / u16::MAX as f32,
                        value[1] as f32 / u16::MAX as f32,
                        0.0,
                        0.0,
                    )
                }
            };
        }

        Vec4::lerp(
            Vec4::lerp(values[0][0], values[0][1], remainder.y),
            Vec4::lerp(values[1][0], values[1][1], remainder.y),
            remainder.x,
        )
    }
}

pub fn sample_attachment(
    tile_tree: &TileTree,
    tile_atlas: &TileAtlas,
    attachment_index: u32,
    sample_world_position: DVec3,
) -> Vec4 {
    let model = &tile_atlas.model;

    // translate the sample position onto the terrain's surface
    // this is necessary to compute a valid blend LOD and ratio
    let surface_position =
        model.surface_position(sample_world_position, tile_tree.approximate_height as f64);

    let (lod, blend_ratio) = tile_tree.compute_blend(surface_position);

    let lookup = tile_tree.lookup_tile(surface_position, lod, model);
    let mut value = tile_atlas.sample_attachment(lookup, attachment_index);

    if blend_ratio > 0.0 {
        let lookup2 = tile_tree.lookup_tile(surface_position, lod - 1, model);
        value = Vec4::lerp(
            value,
            tile_atlas.sample_attachment(lookup2, attachment_index),
            blend_ratio,
        );
    }

    value
}

pub(crate) fn sample_height(
    tile_tree: &TileTree,
    tile_atlas: &TileAtlas,
    sample_world_position: DVec3,
) -> f32 {
    f32::lerp(
        tile_atlas.model.min_height,
        tile_atlas.model.max_height,
        sample_attachment(tile_tree, tile_atlas, 0, sample_world_position).x,
    )
}
