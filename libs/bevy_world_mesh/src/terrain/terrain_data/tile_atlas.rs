use crate::terrain::{
    formats::TC,
    math::{TerrainModel, TileCoordinate},
    terrain::TerrainConfig,
    terrain_data::{
        tile_tree::{TileLookup, TileTree, TileTreeEntry},
        AttachmentConfig, AttachmentData, AttachmentFormat, INVALID_ATLAS_INDEX, INVALID_LOD,
    },
    terrain_view::TerrainViewComponents,
    util::asset_path_string,
};
use anyhow::Result;
use bevy::{
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::render_resource::*,
    tasks::{futures_lite::future, AsyncComputeTaskPool, Task},
};
use image::{DynamicImage, ImageBuffer, ImageReader, Luma, LumaA, Rgb, Rgba};
use itertools::Itertools;
use std::{collections::VecDeque, fs, mem, ops::DerefMut};

pub type Rgb8Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
pub type Rgba8Image = ImageBuffer<Rgba<u8>, Vec<u8>>;
pub type R16Image = ImageBuffer<Luma<u16>, Vec<u16>>;
pub type Rg16Image = ImageBuffer<LumaA<u16>, Vec<u16>>;

const STORE_PNG: bool = false;

#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub struct AtlasTile {
    pub(crate) coordinate: TileCoordinate,
    // encase 0.12 (Bevy 0.18) renamed `#[size(N)]` to `#[shader(size(N))]`.
    #[shader(size(16))]
    pub(crate) atlas_index: u32,
}

impl AtlasTile {
    pub fn new(tile_coordinate: TileCoordinate, atlas_index: u32) -> Self {
        Self {
            coordinate: tile_coordinate,
            atlas_index,
        }
    }
    pub fn attachment(self, attachment_index: u32) -> AtlasTileAttachment {
        AtlasTileAttachment {
            coordinate: self.coordinate,
            atlas_index: self.atlas_index,
            attachment_index,
        }
    }
}

impl From<AtlasTileAttachment> for AtlasTile {
    fn from(tile: AtlasTileAttachment) -> Self {
        Self {
            coordinate: tile.coordinate,
            atlas_index: tile.atlas_index,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AtlasTileAttachment {
    pub(crate) coordinate: TileCoordinate,
    pub(crate) atlas_index: u32,
    pub(crate) attachment_index: u32,
}

#[derive(Clone)]
pub(crate) struct AtlasTileAttachmentWithData {
    pub(crate) tile: AtlasTileAttachment,
    pub(crate) data: AttachmentData,
    pub(crate) texture_size: u32,
}

impl AtlasTileAttachmentWithData {
    pub(crate) fn start_saving(self, path: String) -> Task<AtlasTileAttachment> {
        AsyncComputeTaskPool::get().spawn(async move {
            if STORE_PNG {
                let path = self.tile.coordinate.path(&path, "png");

                let image = match self.data {
                    AttachmentData::Rgba8(data) => {
                        let data = data.into_iter().flatten().collect_vec();
                        DynamicImage::from(
                            Rgba8Image::from_raw(self.texture_size, self.texture_size, data)
                                .unwrap(),
                        )
                    }
                    AttachmentData::R16(data) => DynamicImage::from(
                        R16Image::from_raw(self.texture_size, self.texture_size, data).unwrap(),
                    ),
                    AttachmentData::Rg16(data) => {
                        let data = data.into_iter().flatten().collect_vec();
                        DynamicImage::from(
                            Rg16Image::from_raw(self.texture_size, self.texture_size, data)
                                .unwrap(),
                        )
                    }
                    AttachmentData::None => panic!("Attachment has not data."),
                };

                image.save(&path).unwrap();

                println!("Finished saving tile: {path}");
            } else {
                let path = self.tile.coordinate.path(&path, "bin");

                fs::write(path, self.data.bytes()).unwrap();

                // println!("Finished saving tile: {path}");
            }

            self.tile
        })
    }

    pub(crate) fn start_loading(
        tile: AtlasTileAttachment,
        path: String,
        texture_size: u32,
        format: AttachmentFormat,
        mip_level_count: u32,
    ) -> Task<Result<Self>> {
        AsyncComputeTaskPool::get().spawn(async move {
            let mut data = if STORE_PNG {
                let path = tile.coordinate.path(&path, "png");

                let mut reader = ImageReader::open(path)?;
                reader.no_limits();
                let image = reader.decode().unwrap();
                AttachmentData::from_bytes(image.as_bytes(), format)
            } else {
                let path = tile.coordinate.path(&path, "bin");

                let bytes = fs::read(path)?;

                AttachmentData::from_bytes(&bytes, format)
            };

            data.generate_mipmaps(texture_size, mip_level_count);

            Ok(Self {
                tile,
                data,
                texture_size: 0,
            })
        })
    }
}

/// An attachment of a [`TileAtlas`].
pub struct AtlasAttachment {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) texture_size: u32,
    pub(crate) center_size: u32,
    pub(crate) border_size: u32,
    scale: f32,
    offset: f32,
    pub(crate) mip_level_count: u32,
    pub(crate) format: AttachmentFormat,
    pub(crate) data: Vec<AttachmentData>,

    pub(crate) saving_tiles: Vec<Task<AtlasTileAttachment>>,
    pub(crate) loading_tiles: Vec<Task<Result<AtlasTileAttachmentWithData>>>,
    pub(crate) uploading_tiles: Vec<AtlasTileAttachmentWithData>,
    pub(crate) downloading_tiles: Vec<Task<AtlasTileAttachmentWithData>>,
}

impl AtlasAttachment {
    fn new(config: &AttachmentConfig, tile_atlas_size: u32, path: &str) -> Self {
        let name = config.name.clone();
        let path = asset_path_string(format!("{path}/data/{name}"));
        let center_size = config.texture_size - 2 * config.border_size;

        Self {
            name,
            path,
            texture_size: config.texture_size,
            center_size,
            border_size: config.border_size,
            scale: center_size as f32 / config.texture_size as f32,
            offset: config.border_size as f32 / config.texture_size as f32,
            mip_level_count: config.mip_level_count,
            format: config.format,
            data: vec![AttachmentData::None; tile_atlas_size as usize],
            saving_tiles: default(),
            loading_tiles: default(),
            uploading_tiles: default(),
            downloading_tiles: default(),
        }
    }

    fn update(&mut self, atlas_state: &mut TileAtlasState) {
        self.loading_tiles.retain_mut(|tile| {
            future::block_on(future::poll_once(tile)).is_none_or(|tile| {
                if let Ok(tile) = tile {
                    atlas_state.loaded_tile_attachment(tile.tile);
                    self.uploading_tiles.push(tile.clone());
                    self.data[tile.tile.atlas_index as usize] = tile.data;
                } else {
                    atlas_state.load_slots += 1;
                }

                false
            })
        });

        self.downloading_tiles.retain_mut(|tile| {
            future::block_on(future::poll_once(tile)).is_none_or(|tile| {
                atlas_state.downloaded_tile_attachment(tile.tile);
                self.data[tile.tile.atlas_index as usize] = tile.data;
                false
            })
        });

        self.saving_tiles.retain_mut(|task| {
            future::block_on(future::poll_once(task)).is_none_or(|tile| {
                atlas_state.saved_tile_attachment(tile);
                false
            })
        });
    }

    fn load(&mut self, tile: AtlasTileAttachment) {
        // Todo: build customizable loader abstraction
        self.loading_tiles
            .push(AtlasTileAttachmentWithData::start_loading(
                tile,
                self.path.clone(),
                self.texture_size,
                self.format,
                self.mip_level_count,
            ));
    }

    fn save(&mut self, tile: AtlasTileAttachment) {
        self.saving_tiles.push(
            AtlasTileAttachmentWithData {
                tile,
                data: self.data[tile.atlas_index as usize].clone(),
                texture_size: self.texture_size,
            }
            .start_saving(self.path.clone()),
        );
    }

    fn sample(&self, lookup: TileLookup) -> Vec4 {
        if lookup.atlas_index == INVALID_ATLAS_INDEX {
            return Vec4::splat(0.0); // Todo: Handle this better
        }

        let data = &self.data[lookup.atlas_index as usize];
        let uv = lookup.atlas_uv * self.scale + self.offset;

        data.sample(uv, self.texture_size)
    }
}

/// The current state of a tile of a [`TileAtlas`].
///
/// This indicates, whether the tile is loading or loaded and ready to be used.
#[derive(Clone, Copy)]
enum LoadingState {
    /// The tile is loading, but can not be used yet.
    Loading(u32),
    /// The tile is loaded and can be used.
    Loaded,
}

/// The internal representation of a present tile in a [`TileAtlas`].
struct TileState {
    /// Indicates whether or not the tile is loading or loaded.
    state: LoadingState,
    /// The index of the tile inside the atlas.
    atlas_index: u32,
    /// The count of [`TileTrees`] that have requested this tile.
    requests: u32,
}

pub(crate) struct TileAtlasState {
    tile_states: HashMap<TileCoordinate, TileState>,
    unused_tiles: VecDeque<AtlasTile>,
    pub(crate) existing_tiles: HashSet<TileCoordinate>,

    attachment_count: u32,

    to_load: VecDeque<AtlasTileAttachment>,
    load_slots: u32,
    to_save: VecDeque<AtlasTileAttachment>,
    pub(crate) save_slots: u32,
    pub(crate) max_save_slots: u32,

    pub(crate) download_slots: u32,
    pub(crate) max_download_slots: u32,

    pub(crate) max_atlas_write_slots: u32,
}

impl TileAtlasState {
    fn new(
        atlas_size: u32,
        attachment_count: u32,
        existing_tiles: HashSet<TileCoordinate>,
    ) -> Self {
        let unused_tiles = (0..atlas_size)
            .map(|atlas_index| AtlasTile::new(TileCoordinate::INVALID, atlas_index))
            .collect();

        Self {
            tile_states: default(),
            unused_tiles,
            existing_tiles,
            attachment_count,
            to_save: default(),
            to_load: default(),
            save_slots: 64,
            max_save_slots: 64,
            load_slots: 64,
            download_slots: 128,
            max_download_slots: 128,
            max_atlas_write_slots: 32,
        }
    }

    fn update(&mut self, attachments: &mut [AtlasAttachment]) {
        while self.save_slots > 0 {
            if let Some(tile) = self.to_save.pop_front() {
                attachments[tile.attachment_index as usize].save(tile);
                self.save_slots -= 1;
            } else {
                break;
            }
        }

        while self.load_slots > 0 {
            if let Some(tile) = self.to_load.pop_front() {
                attachments[tile.attachment_index as usize].load(tile);
                self.load_slots -= 1;
            } else {
                break;
            }
        }
    }

    fn loaded_tile_attachment(&mut self, tile: AtlasTileAttachment) {
        self.load_slots += 1;

        let tile_state = self.tile_states.get_mut(&tile.coordinate).unwrap();

        tile_state.state = match tile_state.state {
            LoadingState::Loading(1) => LoadingState::Loaded,
            LoadingState::Loading(n) => LoadingState::Loading(n - 1),
            LoadingState::Loaded => {
                panic!("Loaded more attachments, than registered with the tile atlas.")
            }
        };
    }

    fn saved_tile_attachment(&mut self, _tile: AtlasTileAttachment) {
        self.save_slots += 1;
    }

    fn downloaded_tile_attachment(&mut self, _tile: AtlasTileAttachment) {
        self.download_slots += 1;
    }

    fn get_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        if tile_coordinate == TileCoordinate::INVALID {
            return AtlasTile::new(TileCoordinate::INVALID, INVALID_ATLAS_INDEX);
        }

        let atlas_index = if self.existing_tiles.contains(&tile_coordinate) {
            self.tile_states.get(&tile_coordinate).unwrap().atlas_index
        } else {
            INVALID_ATLAS_INDEX
        };

        AtlasTile::new(tile_coordinate, atlas_index)
    }

    fn allocate_tile(&mut self) -> u32 {
        let unused_tile = self.unused_tiles.pop_front().expect("Atlas out of indices");

        self.tile_states.remove(&unused_tile.coordinate);

        unused_tile.atlas_index
    }

    fn get_or_allocate_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        if tile_coordinate == TileCoordinate::INVALID {
            return AtlasTile::new(TileCoordinate::INVALID, INVALID_ATLAS_INDEX);
        }

        self.existing_tiles.insert(tile_coordinate);

        let atlas_index = if let Some(tile) = self.tile_states.get(&tile_coordinate) {
            tile.atlas_index
        } else {
            let atlas_index = self.allocate_tile();

            self.tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loaded,
                    atlas_index,
                },
            );

            atlas_index
        };

        AtlasTile::new(tile_coordinate, atlas_index)
    }

    fn request_tile(&mut self, tile_coordinate: TileCoordinate) {
        if !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }

        let mut tile_states = mem::take(&mut self.tile_states);

        // check if the tile is already present else start loading it
        if let Some(tile) = tile_states.get_mut(&tile_coordinate) {
            if tile.requests == 0 {
                // the tile is now used again
                self.unused_tiles
                    .retain(|unused_tile| tile.atlas_index != unused_tile.atlas_index);
            }

            tile.requests += 1;
        } else {
            // Todo: implement better loading strategy
            let atlas_index = self.allocate_tile();

            tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loading(self.attachment_count),
                    atlas_index,
                },
            );

            for attachment_index in 0..self.attachment_count {
                self.to_load.push_back(AtlasTileAttachment {
                    coordinate: tile_coordinate,
                    atlas_index,
                    attachment_index,
                });
            }
        }

        self.tile_states = tile_states;
    }

    fn release_tile(&mut self, tile_coordinate: TileCoordinate) {
        if !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }

        let tile = self
            .tile_states
            .get_mut(&tile_coordinate)
            .expect("Tried releasing a tile, which is not present.");
        tile.requests -= 1;

        if tile.requests == 0 {
            // the tile is not used anymore
            self.unused_tiles
                .push_back(AtlasTile::new(tile_coordinate, tile.atlas_index));
        }
    }

    fn get_best_tile(&self, tile_coordinate: TileCoordinate) -> TileTreeEntry {
        let mut best_tile_coordinate = tile_coordinate;

        loop {
            if best_tile_coordinate == TileCoordinate::INVALID
                || best_tile_coordinate.lod == INVALID_LOD
            {
                // highest lod is not loaded
                return TileTreeEntry {
                    atlas_index: INVALID_ATLAS_INDEX,
                    atlas_lod: INVALID_LOD,
                };
            }

            if let Some(atlas_tile) = self.tile_states.get(&best_tile_coordinate) {
                if matches!(atlas_tile.state, LoadingState::Loaded) {
                    // found best loaded tile
                    return TileTreeEntry {
                        atlas_index: atlas_tile.atlas_index,
                        atlas_lod: best_tile_coordinate.lod,
                    };
                }
            }

            best_tile_coordinate = best_tile_coordinate.parent();
        }
    }
}

/// A sparse storage of all terrain attachments, which streams data in and out of memory
/// depending on the decisions of the corresponding [`TileTree`]s.
///
/// A tile is considered present and assigned an [`u32`] as soon as it is
/// requested by any tile_tree. Then the tile atlas will start loading all of its attachments
/// by storing the [`TileCoordinate`] (for one frame) in `load_events` for which
/// attachment-loading-systems can listen.
/// Tiles that are not being used by any tile_tree anymore are cached (LRU),
/// until new atlas indices are required.
///
/// The [`u32`] can be used for accessing the attached data in systems by the CPU
/// and in shaders by the GPU.
#[derive(Component)]
pub struct TileAtlas {
    pub(crate) attachments: Vec<AtlasAttachment>,
    // stores the attachment data
    pub(crate) state: TileAtlasState,
    pub(crate) path: String,
    pub(crate) atlas_size: u32,
    pub(crate) lod_count: u32,
    pub(crate) model: TerrainModel,
}

impl TileAtlas {
    /// Creates a new tile_tree from a terrain config.
    pub fn new(config: &TerrainConfig) -> Self {
        let attachments = config
            .attachments
            .iter()
            .map(|attachment| AtlasAttachment::new(attachment, config.atlas_size, &config.path))
            .collect_vec();

        let existing_tiles = Self::load_tile_config(&config.path);

        let state =
            TileAtlasState::new(config.atlas_size, attachments.len() as u32, existing_tiles);

        Self {
            model: config.model.clone(),
            attachments,
            state,
            path: config.path.to_string(),
            atlas_size: config.atlas_size,
            lod_count: config.lod_count,
        }
    }

    pub fn get_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        self.state.get_tile(tile_coordinate)
    }

    pub fn get_or_allocate_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        self.state.get_or_allocate_tile(tile_coordinate)
    }

    pub fn save(&mut self, tile: AtlasTileAttachment) {
        self.state.to_save.push_back(tile);
    }

    pub(super) fn get_best_tile(&self, tile_coordinate: TileCoordinate) -> TileTreeEntry {
        self.state.get_best_tile(tile_coordinate)
    }

    pub(super) fn sample_attachment(&self, tile_lookup: TileLookup, attachment_index: u32) -> Vec4 {
        self.attachments[attachment_index as usize].sample(tile_lookup)
    }

    /// Updates the tile atlas according to all corresponding tile_trees.
    pub(crate) fn update(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        mut tile_atlases: Query<&mut TileAtlas>,
    ) {
        for mut tile_atlas in tile_atlases.iter_mut() {
            let TileAtlas {
                state, attachments, ..
            } = tile_atlas.deref_mut();

            state.update(attachments);

            for attachment in attachments {
                attachment.update(state);
            }
        }

        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            let Ok(mut tile_atlas) = tile_atlases.get_mut(terrain) else {
                continue;
            };

            for tile_coordinate in tile_tree.released_tiles.drain(..) {
                tile_atlas.state.release_tile(tile_coordinate);
            }

            for tile_coordinate in tile_tree.requested_tiles.drain(..) {
                tile_atlas.state.request_tile(tile_coordinate);
            }
        }
    }

    /// Saves the tile configuration of the terrain, which stores the [`TileCoordinate`]s of all the tiles
    /// of the terrain.
    pub(crate) fn save_tile_config(&self) {
        let tc = TC {
            tiles: self.state.existing_tiles.iter().copied().collect_vec(),
        };

        tc.save_file(asset_path_string(format!("{}/config.tc", &self.path)))
            .unwrap();
    }

    /// Loads the tile configuration of the terrain, which stores the [`TileCoordinate`]s of all the tiles
    /// of the terrain.
    pub(crate) fn load_tile_config(path: &str) -> HashSet<TileCoordinate> {
        if let Ok(tc) = TC::load_file(asset_path_string(format!("{path}/config.tc"))) {
            tc.tiles.into_iter().collect()
        } else {
            println!("Tile config not found.");
            HashSet::default()
        }
    }
}
