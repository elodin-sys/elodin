use crate::terrain::{
    math::TileCoordinate,
    terrain_data::{
        tile_atlas::{AtlasTile, AtlasTileAttachment, TileAtlas},
        AttachmentFormat,
    },
    util::CollectArray,
};
use bevy::{image::ImageSampler, prelude::*};
use itertools::{iproduct, Itertools};
use std::{
    collections::VecDeque,
    fs,
    ops::{DerefMut, Range},
    time::Instant,
};

pub fn reset_directory(directory: &str) {
    let _ = fs::remove_file(format!("{directory}/../../config.tc"));
    let _ = fs::remove_dir_all(directory);
    fs::create_dir_all(directory).unwrap();
}

pub(crate) struct LoadingTile {
    id: AssetId<Image>,
    format: AttachmentFormat,
}

pub struct SphericalDataset {
    pub attachment_index: u32,
    pub paths: Vec<String>,
    pub lod_range: Range<u32>,
}

pub struct PreprocessDataset {
    pub attachment_index: u32,
    pub path: String,
    pub side: u32,
    pub top_left: Vec2,
    pub bottom_right: Vec2,
    pub lod_range: Range<u32>,
}

impl Default for PreprocessDataset {
    fn default() -> Self {
        Self {
            attachment_index: 0,
            path: "".to_string(),
            side: 0,
            top_left: Vec2::splat(0.0),
            bottom_right: Vec2::splat(1.0),
            lod_range: 0..1,
        }
    }
}

impl PreprocessDataset {
    fn overlapping_tiles(&self, lod: u32) -> impl Iterator<Item = TileCoordinate> + '_ {
        let tile_count = TileCoordinate::count(lod);

        let lower = (self.top_left * tile_count as f32).as_uvec2();
        let upper = (self.bottom_right * tile_count as f32).ceil().as_uvec2();

        iproduct!(lower.x..upper.x, lower.y..upper.y)
            .map(move |(x, y)| TileCoordinate::new(self.side, lod, x, y))
    }
}

#[derive(Clone)]
pub(crate) enum PreprocessTaskType {
    Split {
        tile_data: Handle<Image>,
        top_left: Vec2,
        bottom_right: Vec2,
    },
    Stitch {
        neighbour_tiles: [AtlasTile; 8],
    },
    Downsample {
        child_tiles: [AtlasTile; 4],
    },
    Save,
    Barrier,
}

// Todo: store tile_coordinate, task_type, tile_dependencies and tile dependencies
// loop over all tasks, take n, allocate/load tile and its dependencies, process task
#[derive(Clone)]
pub(crate) struct PreprocessTask {
    pub(crate) tile: AtlasTileAttachment,
    pub(crate) task_type: PreprocessTaskType,
}

impl PreprocessTask {
    fn is_ready(&self, asset_server: &AssetServer, tile_atlas: &TileAtlas) -> bool {
        match &self.task_type {
            PreprocessTaskType::Split { tile_data, .. } => {
                asset_server.is_loaded_with_dependencies(tile_data)
            }
            PreprocessTaskType::Stitch { .. } => true,
            PreprocessTaskType::Downsample { .. } => true,
            PreprocessTaskType::Barrier => {
                tile_atlas.state.download_slots == tile_atlas.state.max_download_slots
            }
            PreprocessTaskType::Save => true,
        }
    }

    #[allow(dead_code)]
    fn debug(&self) {
        match &self.task_type {
            PreprocessTaskType::Split { .. } => {
                println!("Splitting tile: {}", self.tile.coordinate)
            }
            PreprocessTaskType::Stitch { .. } => {
                println!("Stitching tile: {}", self.tile.coordinate)
            }
            PreprocessTaskType::Downsample { .. } => {
                println!("Downsampling tile: {}", self.tile.coordinate)
            }
            PreprocessTaskType::Save => {
                println!("Started saving tile: {}", self.tile.coordinate)
            }
            PreprocessTaskType::Barrier => println!("Barrier"),
        }
    }

    fn barrier() -> Self {
        Self {
            tile: default(),
            task_type: PreprocessTaskType::Barrier,
        }
    }

    fn save(
        tile_coordinate: TileCoordinate,
        tile_atlas: &mut TileAtlas,
        dataset: &PreprocessDataset,
    ) -> Self {
        let tile = tile_atlas
            .get_or_allocate_tile(tile_coordinate)
            .attachment(dataset.attachment_index);

        Self {
            tile,
            task_type: PreprocessTaskType::Save,
        }
    }

    fn split(
        tile_coordinate: TileCoordinate,
        tile_atlas: &mut TileAtlas,
        dataset: &PreprocessDataset,
        tile_data: Handle<Image>,
    ) -> Self {
        let tile = tile_atlas
            .get_or_allocate_tile(tile_coordinate)
            .attachment(dataset.attachment_index);

        Self {
            tile,
            task_type: PreprocessTaskType::Split {
                tile_data,
                top_left: dataset.top_left,
                bottom_right: dataset.bottom_right,
            },
        }
    }

    fn stitch(
        tile_coordinate: TileCoordinate,
        tile_atlas: &mut TileAtlas,
        dataset: &PreprocessDataset,
    ) -> Self {
        let tile = tile_atlas
            .get_or_allocate_tile(tile_coordinate)
            .attachment(dataset.attachment_index);

        let neighbour_tiles = tile
            .coordinate
            .neighbours(tile_atlas.model.is_spherical())
            .map(|coordinate| tile_atlas.get_tile(coordinate))
            .collect_array();

        Self {
            tile,
            task_type: PreprocessTaskType::Stitch { neighbour_tiles },
        }
    }

    fn downsample(
        tile_coordinate: TileCoordinate,
        tile_atlas: &mut TileAtlas,
        dataset: &PreprocessDataset,
    ) -> Self {
        let tile = tile_atlas
            .get_or_allocate_tile(tile_coordinate)
            .attachment(dataset.attachment_index);

        let child_tiles = tile
            .coordinate
            .children()
            .map(|coordinate| tile_atlas.get_tile(coordinate))
            .collect_array();

        Self {
            tile,
            task_type: PreprocessTaskType::Downsample { child_tiles },
        }
    }
}

#[derive(Component)]
pub struct Preprocessor {
    pub(crate) loading_tiles: Vec<LoadingTile>,
    pub(crate) task_queue: VecDeque<PreprocessTask>,
    pub(crate) ready_tasks: Vec<PreprocessTask>,

    pub(crate) start_time: Option<Instant>,
    loaded: bool,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            loading_tiles: default(),
            task_queue: default(),
            ready_tasks: default(),
            start_time: None,
            loaded: false,
        }
    }

    fn split_and_downsample(
        &mut self,
        dataset: &PreprocessDataset,
        asset_server: &AssetServer,
        tile_atlas: &mut TileAtlas,
    ) {
        let tile_handle = asset_server.load(&dataset.path);

        self.loading_tiles.push(LoadingTile {
            id: tile_handle.id(),
            format: tile_atlas.attachments[dataset.attachment_index as usize].format,
        });

        let mut lods = dataset.lod_range.clone().rev();

        for tile_coordinate in dataset.overlapping_tiles(lods.next().unwrap()) {
            self.task_queue.push_back(PreprocessTask::split(
                tile_coordinate,
                tile_atlas,
                dataset,
                tile_handle.clone(),
            ));
        }

        for lod in lods {
            self.task_queue.push_back(PreprocessTask::barrier());

            for tile_coordinate in dataset.overlapping_tiles(lod) {
                self.task_queue.push_back(PreprocessTask::downsample(
                    tile_coordinate,
                    tile_atlas,
                    dataset,
                ));
            }
        }
    }

    fn stitch_and_save_layer(
        &mut self,
        dataset: &PreprocessDataset,
        tile_atlas: &mut TileAtlas,
        lod: u32,
    ) {
        for tile_coordinate in dataset.overlapping_tiles(lod) {
            self.task_queue
                .push_back(PreprocessTask::stitch(tile_coordinate, tile_atlas, dataset));
        }

        self.task_queue.push_back(PreprocessTask::barrier());

        for tile_coordinate in dataset.overlapping_tiles(lod) {
            self.task_queue
                .push_back(PreprocessTask::save(tile_coordinate, tile_atlas, dataset));
        }
    }

    pub fn clear_attachment(self, attachment_index: u32, tile_atlas: &mut TileAtlas) -> Self {
        let attachment = &mut tile_atlas.attachments[attachment_index as usize];
        tile_atlas.state.existing_tiles.clear();
        reset_directory(&attachment.path);

        self
    }

    pub fn preprocess_tile(
        mut self,
        dataset: PreprocessDataset,
        asset_server: &AssetServer,
        tile_atlas: &mut TileAtlas,
    ) -> Self {
        self.split_and_downsample(&dataset, asset_server, tile_atlas);
        self.task_queue.push_back(PreprocessTask::barrier());

        for lod in dataset.lod_range.clone() {
            self.stitch_and_save_layer(&dataset, tile_atlas, lod);
        }

        self
    }

    pub fn preprocess_spherical(
        mut self,
        dataset: SphericalDataset,
        asset_server: &AssetServer,
        tile_atlas: &mut TileAtlas,
    ) -> Self {
        let side_datasets = (0..6)
            .map(|side| PreprocessDataset {
                attachment_index: dataset.attachment_index,
                path: dataset.paths[side as usize].clone(),
                side,
                lod_range: dataset.lod_range.clone(),
                ..default()
            })
            .collect_vec();

        for dataset in &side_datasets {
            self.split_and_downsample(dataset, asset_server, tile_atlas);
        }

        self.task_queue.push_back(PreprocessTask::barrier());

        for lod in dataset.lod_range {
            for dataset in &side_datasets {
                self.stitch_and_save_layer(dataset, tile_atlas, lod);
            }
        }

        self
    }
}

pub(crate) fn select_ready_tasks(
    asset_server: Res<AssetServer>,
    mut terrains: Query<(&mut Preprocessor, &mut TileAtlas)>,
) {
    for (mut preprocessor, mut tile_atlas) in terrains.iter_mut() {
        let Preprocessor {
            task_queue,
            ready_tasks,
            start_time,
            ..
        } = preprocessor.deref_mut();

        if let Some(time) = start_time {
            if task_queue.is_empty()
                && tile_atlas.state.download_slots == tile_atlas.state.max_download_slots
                && tile_atlas.state.save_slots == tile_atlas.state.max_save_slots
            {
                println!("Preprocessing took {:?}", time.elapsed());

                tile_atlas.save_tile_config();
                // tile_atlas.state.existing_tiles.iter().for_each(|tile| {
                //     println!("{tile}");
                // });

                *start_time = None;
            }
        } else {
            break;
        }

        ready_tasks.clear();

        loop {
            if (tile_atlas.state.download_slots > 0)
                && task_queue
                    .front()
                    .is_some_and(|task| task.is_ready(&asset_server, &tile_atlas))
            {
                let task = task_queue.pop_front().unwrap();

                // task.debug();

                if matches!(task.task_type, PreprocessTaskType::Save) {
                    tile_atlas.save(task.tile);
                } else {
                    ready_tasks.push(task);
                    tile_atlas.state.download_slots -= 1;
                }
            } else {
                break;
            }
        }
    }
}

pub(crate) fn preprocessor_load_tile(
    mut preprocessors: Query<&mut Preprocessor>,
    mut images: ResMut<Assets<Image>>,
) {
    for mut preprocessor in preprocessors.iter_mut() {
        preprocessor.loading_tiles.retain_mut(|tile| {
            if let Some(image) = images.get_mut(tile.id) {
                image.texture_descriptor.format = tile.format.processing_format();
                image.sampler = ImageSampler::linear();
                false
            } else {
                true
            }
        });

        if !preprocessor.loaded && preprocessor.loading_tiles.is_empty() {
            println!("finished loading all tiles");
            preprocessor.loaded = true;
            preprocessor.start_time = Some(Instant::now());
        }
    }
}
