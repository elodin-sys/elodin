use crate::terrain::{
    render::instantiate_layout,
    terrain::TerrainComponents,
    terrain_data::{
        tile_atlas::{
            AtlasAttachment, AtlasTileAttachment, AtlasTileAttachmentWithData, TileAtlas,
        },
        AttachmentData, AttachmentFormat,
    },
    util::StaticBuffer,
};
use bevy::{
    prelude::*,
    render::{
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        Extract, MainWorld,
    },
    tasks::{AsyncComputeTaskPool, Task},
};
use itertools::Itertools;
use std::{iter, mem};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

fn align_byte_size(value: u32) -> u32 {
    // only works for non zero values
    value - 1 - (value - 1) % COPY_BYTES_PER_ROW_ALIGNMENT + COPY_BYTES_PER_ROW_ALIGNMENT
}

pub(crate) fn create_attachment_layout() -> BindGroupLayoutDescriptor {
    // Bevy 0.18 (PR#21205) switched `RenderPipelineDescriptor::layout` from
    // `Vec<BindGroupLayout>` to `Vec<BindGroupLayoutDescriptor>` so the
    // `PipelineCache` can materialise concrete layouts lazily. Return a
    // descriptor; callers that need a real `BindGroupLayout` (e.g. for
    // `create_bind_group`) go through `instantiate_layout(device, &desc)`.
    BindGroupLayoutDescriptor::new(
        "attachment_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // atlas_write_section
                texture_2d_array(TextureSampleType::Float { filterable: true }), // atlas
                sampler(SamplerBindingType::Filtering), // atlas sampler
                uniform_buffer::<AttachmentMeta>(false), // attachment meta
            ),
        ),
    )
}

#[derive(Default, ShaderType)]
pub(crate) struct AttachmentMeta {
    pub(crate) format_id: u32,
    pub(crate) lod_count: u32,
    pub(crate) texture_size: u32,
    pub(crate) border_size: u32,
    pub(crate) center_size: u32,
    pub(crate) pixels_per_entry: u32,
    pub(crate) entries_per_side: u32,
    pub(crate) entries_per_tile: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AtlasBufferInfo {
    lod_count: u32,
    pub(crate) texture_size: u32,
    pub(crate) border_size: u32,
    pub(crate) center_size: u32,
    format: AttachmentFormat,
    mip_level_count: u32,

    pixels_per_entry: u32,

    entries_per_side: u32,
    entries_per_tile: u32,

    actual_side_size: u32,
    aligned_side_size: u32,
    actual_tile_size: u32,
    aligned_tile_size: u32,

    pub(crate) workgroup_count: UVec3,
}

impl AtlasBufferInfo {
    fn new(attachment: &AtlasAttachment, lod_count: u32) -> Self {
        // Todo: adjust this code for pixel sizes larger than 4 byte
        // This approach is currently limited to 1, 2, and 4 byte sized pixels
        // Extending it to 8 and 16 sized pixels should be quite easy.
        // However 3, 6, 12 sized pixels do and will not work!
        // For them to work properly we will need to write into a texture instead of buffer.

        let format = attachment.format;
        let texture_size = attachment.texture_size;
        let border_size = attachment.border_size;
        let center_size = attachment.center_size;
        let mip_level_count = attachment.mip_level_count;

        let pixel_size = format.pixel_size();
        let entry_size = mem::size_of::<u32>() as u32;
        let pixels_per_entry = entry_size / pixel_size;

        let actual_side_size = texture_size * pixel_size;
        let aligned_side_size = align_byte_size(actual_side_size);
        let actual_tile_size = texture_size * actual_side_size;
        let aligned_tile_size = texture_size * aligned_side_size;

        let entries_per_side = aligned_side_size / entry_size;
        let entries_per_tile = texture_size * entries_per_side;

        let workgroup_count = UVec3::new(entries_per_side / 8, texture_size / 8, 1);

        Self {
            lod_count,
            border_size,
            center_size,
            texture_size,
            mip_level_count,
            pixels_per_entry,
            entries_per_side,
            entries_per_tile,
            actual_side_size,
            aligned_side_size,
            actual_tile_size,
            aligned_tile_size,
            format,
            workgroup_count,
        }
    }

    fn image_copy_texture<'a>(
        &'a self,
        texture: &'a Texture,
        index: u32,
        mip_level: u32,
    ) -> TexelCopyTextureInfo<'a> {
        TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: Origin3d {
                z: index,
                ..default()
            },
            aspect: TextureAspect::All,
        }
    }

    fn image_copy_buffer<'a>(&'a self, buffer: &'a Buffer, index: u32) -> TexelCopyBufferInfo<'a> {
        TexelCopyBufferInfo {
            buffer,
            layout: TexelCopyBufferLayout {
                bytes_per_row: Some(self.aligned_side_size),
                rows_per_image: Some(self.texture_size),
                offset: self.buffer_size(index) as BufferAddress,
            },
        }
    }

    fn image_copy_size(&self, mip_level: u32) -> Extent3d {
        Extent3d {
            width: self.texture_size >> mip_level,
            height: self.texture_size >> mip_level,
            depth_or_array_layers: 1,
        }
    }

    fn buffer_size(&self, slots: u32) -> u32 {
        slots * self.aligned_tile_size
    }

    fn attachment_meta(&self) -> AttachmentMeta {
        AttachmentMeta {
            format_id: self.format.id(),
            lod_count: self.lod_count,
            texture_size: self.texture_size,
            border_size: self.border_size,
            center_size: self.center_size,
            pixels_per_entry: self.pixels_per_entry,
            entries_per_side: self.entries_per_side,
            entries_per_tile: self.entries_per_tile,
        }
    }
}

pub(crate) struct GpuAtlasAttachment {
    pub(crate) name: String,
    pub(crate) buffer_info: AtlasBufferInfo,

    pub(crate) atlas_texture: Texture,
    pub(crate) atlas_write_section: StaticBuffer<()>,
    pub(crate) download_buffers: Vec<StaticBuffer<()>>,
    pub(crate) bind_group: BindGroup,

    pub(crate) max_atlas_write_slots: u32,
    pub(crate) atlas_write_slots: Vec<AtlasTileAttachment>,
    pub(crate) upload_tiles: Vec<AtlasTileAttachmentWithData>,
    pub(crate) download_tiles: Vec<Task<AtlasTileAttachmentWithData>>,
}

impl GpuAtlasAttachment {
    pub(crate) fn new(
        device: &RenderDevice,
        attachment: &AtlasAttachment,
        tile_atlas: &TileAtlas,
    ) -> Self {
        let name = attachment.name.clone();
        let max_atlas_write_slots = tile_atlas.state.max_atlas_write_slots;
        let atlas_write_slots = Vec::with_capacity(max_atlas_write_slots as usize);

        let buffer_info = AtlasBufferInfo::new(attachment, tile_atlas.lod_count);

        // dbg!(&buffer_info);

        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some(&format!("{name}_attachment")),
            size: Extent3d {
                width: buffer_info.texture_size,
                height: buffer_info.texture_size,
                depth_or_array_layers: tile_atlas.atlas_size,
            },
            mip_level_count: attachment.mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: buffer_info.format.render_format(),
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[buffer_info.format.processing_format()],
        });

        let atlas_view = atlas_texture.create_view(&TextureViewDescriptor {
            format: Some(buffer_info.format.processing_format()),
            ..default()
        });

        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..default()
        });

        let atlas_write_section = StaticBuffer::empty_sized(
            format!("{name}_atlas_write_section").as_str(),
            device,
            buffer_info.buffer_size(max_atlas_write_slots) as BufferAddress,
            BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        );

        let attachment_meta_buffer = StaticBuffer::create(
            format!("{name}_attachment_meta").as_str(),
            device,
            &buffer_info.attachment_meta(),
            BufferUsages::UNIFORM,
        );

        let bind_group = device.create_bind_group(
            format!("{name}attachment_bind_group").as_str(),
            &instantiate_layout(device, &create_attachment_layout()),
            &BindGroupEntries::sequential((
                &atlas_write_section,
                &atlas_view,
                &atlas_sampler,
                &attachment_meta_buffer,
            )),
        );

        Self {
            name,
            buffer_info,
            atlas_texture,
            atlas_write_section,
            download_buffers: default(),
            bind_group,
            max_atlas_write_slots,
            atlas_write_slots,
            upload_tiles: default(),
            download_tiles: default(),
        }
    }

    pub(crate) fn reserve_write_slot(&mut self, tile: AtlasTileAttachment) -> Option<u32> {
        if self.atlas_write_slots.len() < self.max_atlas_write_slots as usize {
            self.atlas_write_slots.push(tile);
            Some(self.atlas_write_slots.len() as u32 - 1)
        } else {
            None
        }
    }

    pub(crate) fn copy_tiles_to_write_section(&self, command_encoder: &mut CommandEncoder) {
        for (section_index, tile) in self.atlas_write_slots.iter().enumerate() {
            command_encoder.copy_texture_to_buffer(
                self.buffer_info
                    .image_copy_texture(&self.atlas_texture, tile.atlas_index, 0),
                self.buffer_info
                    .image_copy_buffer(&self.atlas_write_section, section_index as u32),
                self.buffer_info.image_copy_size(0),
            );
        }
    }

    pub(crate) fn copy_tiles_from_write_section(&self, command_encoder: &mut CommandEncoder) {
        for (section_index, tile) in self.atlas_write_slots.iter().enumerate() {
            command_encoder.copy_buffer_to_texture(
                self.buffer_info
                    .image_copy_buffer(&self.atlas_write_section, section_index as u32),
                self.buffer_info
                    .image_copy_texture(&self.atlas_texture, tile.atlas_index, 0),
                self.buffer_info.image_copy_size(0),
            );
        }
    }

    fn upload_tiles(&mut self, queue: &RenderQueue) {
        for tile in self.upload_tiles.drain(..) {
            let mut start = 0;

            for mip_level in 0..self.buffer_info.mip_level_count {
                let side_size = self.buffer_info.actual_side_size >> mip_level;
                let texture_size = self.buffer_info.texture_size >> mip_level;
                let end = start + (side_size * texture_size) as usize;

                queue.write_texture(
                    self.buffer_info.image_copy_texture(
                        &self.atlas_texture,
                        tile.tile.atlas_index,
                        mip_level,
                    ),
                    &tile.data.bytes()[start..end],
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(side_size),
                        rows_per_image: Some(texture_size),
                    },
                    self.buffer_info.image_copy_size(mip_level),
                );

                start = end;
            }
        }
    }

    pub(crate) fn download_tiles(&self, command_encoder: &mut CommandEncoder) {
        for (tile, download_buffer) in iter::zip(&self.atlas_write_slots, &self.download_buffers) {
            command_encoder.copy_texture_to_buffer(
                self.buffer_info
                    .image_copy_texture(&self.atlas_texture, tile.atlas_index, 0),
                self.buffer_info.image_copy_buffer(download_buffer, 0),
                self.buffer_info.image_copy_size(0),
            );
        }
    }

    fn create_download_buffers(&mut self, device: &RenderDevice) {
        self.download_buffers = (0..self.atlas_write_slots.len())
            .map(|i| {
                StaticBuffer::empty_sized(
                    format!("{}_download_buffer_{i}", self.name).as_str(),
                    device,
                    self.buffer_info.aligned_tile_size as BufferAddress,
                    BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                )
            })
            .collect_vec();
    }

    fn start_downloading_tiles(&mut self) {
        let buffer_info = self.buffer_info;
        let download_buffers = mem::take(&mut self.download_buffers);
        let atlas_write_slots = mem::take(&mut self.atlas_write_slots);

        self.download_tiles = iter::zip(atlas_write_slots, download_buffers)
            .map(|(tile, download_buffer)| {
                AsyncComputeTaskPool::get().spawn(async move {
                    let (tx, rx) = async_channel::bounded(1);

                    let buffer_slice = download_buffer.slice(..);

                    buffer_slice.map_async(MapMode::Read, move |_| {
                        tx.try_send(()).unwrap();
                    });

                    rx.recv().await.unwrap();

                    let mut data = buffer_slice.get_mapped_range().to_vec();

                    download_buffer.unmap();
                    drop(download_buffer);

                    if data.len() != buffer_info.actual_tile_size as usize {
                        let actual_side_size = buffer_info.actual_side_size as usize;
                        let aligned_side_size = buffer_info.aligned_side_size as usize;

                        let mut take_offset = aligned_side_size;
                        let mut place_offset = actual_side_size;

                        for _ in 1..buffer_info.texture_size {
                            data.copy_within(
                                take_offset..take_offset + aligned_side_size,
                                place_offset,
                            );
                            take_offset += aligned_side_size;
                            place_offset += actual_side_size;
                        }

                        data.truncate(buffer_info.actual_tile_size as usize);
                    }

                    AtlasTileAttachmentWithData {
                        tile,
                        data: AttachmentData::from_bytes(&data, buffer_info.format),
                        texture_size: buffer_info.texture_size,
                    }
                })
            })
            .collect_vec();
    }
}

/// Stores the GPU representation of the [`TileAtlas`] (array textures)
/// alongside the data to update it.
///
/// All attachments of newly loaded tiles are copied into their according atlas attachment.
#[derive(Component)]
pub struct GpuTileAtlas {
    /// Stores the atlas attachments of the terrain.
    pub(crate) attachments: Vec<GpuAtlasAttachment>,
    pub(crate) is_spherical: bool,
}

impl GpuTileAtlas {
    /// Creates a new gpu tile atlas and initializes its attachment textures.
    fn new(device: &RenderDevice, tile_atlas: &TileAtlas) -> Self {
        let attachments = tile_atlas
            .attachments
            .iter()
            .map(|attachment| GpuAtlasAttachment::new(device, attachment, tile_atlas))
            .collect_vec();

        Self {
            attachments,
            is_spherical: tile_atlas.model.is_spherical(),
        }
    }

    /// Initializes the [`GpuTileAtlas`] of newly created terrains.
    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
        mut tile_atlases: Extract<Query<(Entity, &TileAtlas), Added<TileAtlas>>>,
    ) {
        for (terrain, tile_atlas) in tile_atlases.iter_mut() {
            gpu_tile_atlases.insert(terrain, GpuTileAtlas::new(&device, tile_atlas));
        }
    }

    /// Extracts the tiles that have finished loading from all [`TileAtlas`]es into the
    /// corresponding [`GpuTileAtlas`]es.
    pub(crate) fn extract(
        mut main_world: ResMut<MainWorld>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
    ) {
        let mut tile_atlases = main_world.query::<(Entity, &mut TileAtlas)>();

        for (terrain, mut tile_atlas) in tile_atlases.iter_mut(&mut main_world) {
            let Some(gpu_tile_atlas) = gpu_tile_atlases.get_mut(&terrain) else {
                continue;
            };

            for (attachment, gpu_attachment) in
                iter::zip(&mut tile_atlas.attachments, &mut gpu_tile_atlas.attachments)
            {
                mem::swap(
                    &mut attachment.uploading_tiles,
                    &mut gpu_attachment.upload_tiles,
                );

                attachment
                    .downloading_tiles
                    .extend(mem::take(&mut gpu_attachment.download_tiles));
            }
        }
    }

    /// Queues the attachments of the tiles that have finished loading to be copied into the
    /// corresponding atlas attachments.
    pub(crate) fn prepare(
        device: Res<RenderDevice>,
        queue: Res<RenderQueue>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
    ) {
        for gpu_tile_atlas in gpu_tile_atlases.values_mut() {
            for attachment in &mut gpu_tile_atlas.attachments {
                attachment.create_download_buffers(&device);
                attachment.upload_tiles(&queue);
            }
        }
    }

    pub(crate) fn cleanup(mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>) {
        for gpu_tile_atlas in gpu_tile_atlases.values_mut() {
            for attachment in &mut gpu_tile_atlas.attachments {
                attachment.start_downloading_tiles();
            }
        }
    }
}
