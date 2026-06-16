//! Region presets and the `RegionManifest` on-disk format.
//!
//! Each preset describes a square footprint on Earth (centre lon/lat, side
//! length in km) plus the metadata the rendering pipeline needs downstream:
//! the elevation range to normalise the heightmap against and a normalised
//! camera offset/target to frame the scene against the dominant feature.
//!
//! [`Region`] is the compile-time preset; [`RegionManifest`] is the
//! serde-serialisable snapshot that gets written to
//! `assets/terrains/planar/<region>/region.toml` so the runtime scene
//! plugin (under the `scenes` feature) can re-frame the camera and
//! rescale `TERRAIN_SIZE` / `HEIGHT` without recompiling. The active
//! region is chosen at runtime via the `WORLD_MESH_REGION` env var
//! (default: `death_valley`).

use crate::terrain::util::asset_path;
use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

/// On-disk manifest written by the fetch binaries and consumed by the
/// planar scene plugin at startup. Serialises to
/// `assets/terrains/planar/<region>/region.toml`, where `<region>` is the
/// value of the `WORLD_MESH_REGION` env var (default: `death_valley`).
/// All fields are in real-world units (degrees lon/lat, kilometres for
/// the side, metres for elevations); the camera offsets are normalised
/// so the renderer can multiply them by `side_km * 1000.0` regardless of
/// region scale.
///
/// The [`Default`] impl mirrors the synthetic `synthesize_height`
/// binary's footprint (1 km square, 250 m-tall procedural terrain). This
/// is what the scene plugin falls back to when no `region.toml` is
/// present.
#[derive(Debug, Clone, Resource, Serialize, Deserialize)]
pub struct RegionManifest {
    pub name: String,
    pub center_lon: f64,
    pub center_lat: f64,
    pub side_km: f64,
    pub min_height_m: f32,
    pub max_height_m: f32,
    pub camera_offset_xyz_norm: [f32; 3],
    pub camera_target_xyz_norm: [f32; 3],
}

impl Default for RegionManifest {
    fn default() -> Self {
        Self {
            name: "synthetic".to_string(),
            center_lon: 0.0,
            center_lat: 0.0,
            side_km: 1.0,
            min_height_m: 0.0,
            max_height_m: 250.0,
            camera_offset_xyz_norm: [0.38, 0.22, 0.38],
            camera_target_xyz_norm: [0.0, -0.04, 0.0],
        }
    }
}

impl RegionManifest {
    /// Read the active region's manifest. The active region name is taken
    /// from `WORLD_MESH_REGION` (default: `death_valley`) and the file is
    /// expected at `assets/terrains/planar/<region>/region.toml` — the same
    /// path the fetch binary writes.
    pub fn try_load() -> Option<Self> {
        let region =
            std::env::var("WORLD_MESH_REGION").unwrap_or_else(|_| "death_valley".to_string());
        let path = asset_path(format!("terrains/planar/{region}/region.toml"));
        let text = std::fs::read_to_string(path).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn load_or_default() -> Self {
        Self::try_load().unwrap_or_default()
    }

    /// Side length in world-space metres.
    pub fn terrain_size_m(&self) -> f64 {
        self.side_km * 1000.0
    }

    /// Vertical extent in metres (used as `HEIGHT` in
    /// [`crate::terrain::math::TerrainModel::planar`]).
    pub fn height_m(&self) -> f32 {
        (self.max_height_m - self.min_height_m).max(1.0)
    }
}

/// Compile-time region preset. Stored in [`PRESETS`] and looked up by name
/// via [`lookup`].
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub name: &'static str,
    /// Centre longitude in EPSG:4326 degrees.
    pub center_lon: f64,
    /// Centre latitude in EPSG:4326 degrees.
    pub center_lat: f64,
    /// Square footprint side length (kilometres). The fetcher computes the
    /// bbox as a square in latitude-degrees × cos(centre_lat)-corrected
    /// longitude-degrees, so the rendered region is approximately
    /// `side_km × side_km` on the ground.
    pub side_km: f64,
    /// Lower bound of the elevation range we normalise the heightmap against.
    pub min_height_m: f32,
    /// Upper bound of the elevation range.
    pub max_height_m: f32,
    /// Camera position relative to the terrain centre, expressed in normalised
    /// units (multiplied by `side_km * 1000.0` at use site). Y is up.
    pub camera_offset_xyz_norm: [f32; 3],
    /// Camera target relative to the terrain centre, normalised (same scale).
    pub camera_target_xyz_norm: [f32; 3],
}

impl From<&Region> for RegionManifest {
    fn from(r: &Region) -> Self {
        Self {
            name: r.name.to_string(),
            center_lon: r.center_lon,
            center_lat: r.center_lat,
            side_km: r.side_km,
            min_height_m: r.min_height_m,
            max_height_m: r.max_height_m,
            camera_offset_xyz_norm: r.camera_offset_xyz_norm,
            camera_target_xyz_norm: r.camera_target_xyz_norm,
        }
    }
}

/// Brienz, Bernese Alps. Lake Brienz at the centre, with the Bernese Oberland
/// peaks to the south. ~12 km square; vertical relief ~560 m → ~2 400 m.
pub const BRIENZ: Region = Region {
    name: "brienz",
    center_lon: 8.0340,
    center_lat: 46.7240,
    side_km: 12.0,
    min_height_m: 560.0,
    max_height_m: 2400.0,
    // Drone camera, NE corner, low orbit, looking at lake centre.
    camera_offset_xyz_norm: [0.38, 0.22, 0.38],
    camera_target_xyz_norm: [0.0, -0.01, 0.0],
};

/// Death Valley + Panamint Range. Centred so the bbox covers Badwater Basin
/// (-86 m, the lowest point in North America) and Telescope Peak (3 366 m,
/// the local high point of the Panamint Range) in a 22 km square.
pub const DEATH_VALLEY: Region = Region {
    name: "death_valley",
    center_lon: -117.085,
    center_lat: 36.230,
    side_km: 22.0,
    min_height_m: -86.0,
    max_height_m: 3400.0,
    // Camera looking west across the valley toward the Panamint Range.
    camera_offset_xyz_norm: [0.42, 0.18, 0.0],
    camera_target_xyz_norm: [-0.20, -0.01, 0.0],
};

/// Mojave Desert basin-and-range. 500 km square centred near Ridgecrest, CA,
/// designed for aircraft-altitude (3 km+) flight with a realistic horizon
/// (~200 km in each direction). Captures Badwater Basin (-86 m) through the
/// Mt Whitney foothills (~4 421 m) and the surrounding basin-and-range
/// topography. Cold fetch is 30-60 min (~12 000 tiles each for height +
/// imagery at z=14); subsequent runs hit the on-disk tile cache.
pub const MOJAVE_DESERT: Region = Region {
    name: "mojave_desert",
    center_lon: -117.84383,
    center_lat: 35.33146,
    side_km: 500.0,
    min_height_m: -100.0,
    max_height_m: 4500.0,
    // Camera at ~3 km altitude (3 km / 500 km = 0.006 normalised Y), looking
    // toward the horizon with a slight downward pitch. The 0.3 X offset puts
    // the eye well off-centre so the view sweeps across the basin-and-range
    // landscape rather than staring straight down.
    camera_offset_xyz_norm: [0.0, 0.006, 0.0],
    camera_target_xyz_norm: [0.3, -0.001, 0.0],
};

pub const PRESETS: &[&Region] = &[&BRIENZ, &DEATH_VALLEY, &MOJAVE_DESERT];

pub fn lookup(name: &str) -> Option<&'static Region> {
    PRESETS.iter().copied().find(|r| r.name == name)
}
