//! Minimal helpers for consuming preprocessed world-mesh atlases on disk.
//!
//! This is intentionally *not* the full upstream world-mesh runtime. It's a
//! small adapter layer that lets the Elodin editor load the existing planar
//! atlas layout (e.g. Death Valley) and render it via a simple heightmap mesh.

#![cfg(not(target_family = "wasm"))]

use bincode::{Decode, Encode, config};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy)]
pub struct RegionMetadata {
    pub width_m: f32,
    pub depth_m: f32,
    pub min_height_m: f32,
    pub max_height_m: f32,
}

/// Tile coordinates encoded in `config.tc`.
#[derive(Encode, Decode, Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct TileCoordinate {
    pub side: u32,
    pub lod: u32,
    pub x: u32,
    pub y: u32,
}

impl fmt::Display for TileCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}_{}_{}", self.side, self.lod, self.x, self.y)
    }
}

#[derive(Encode, Decode, Debug)]
struct TC {
    tiles: Vec<TileCoordinate>,
}

fn decode_tc(bytes: &[u8]) -> Option<TC> {
    let cfg = config::standard();
    let (tc, _len): (TC, usize) = bincode::decode_from_slice(bytes, cfg).ok()?;
    Some(tc)
}

fn value_as_f64(v: &toml::Value) -> Option<f64> {
    v.as_float().or_else(|| v.as_integer().map(|i| i as f64))
}

fn find_number(v: &toml::Value, key: &str) -> Option<f64> {
    match v {
        toml::Value::Table(t) => {
            if let Some(v) = t.get(key).and_then(value_as_f64) {
                return Some(v);
            }
            for child in t.values() {
                if let Some(n) = find_number(child, key) {
                    return Some(n);
                }
            }
            None
        }
        toml::Value::Array(arr) => arr.iter().find_map(|x| find_number(x, key)),
        _ => None,
    }
}

/// Loads `region.toml` if present and extracts a few common numeric keys.
///
/// This is intentionally tolerant: it searches recursively for known keys.
pub fn load_region_metadata(region_root: &Path) -> Option<RegionMetadata> {
    let text = fs::read_to_string(region_root.join("region.toml")).ok()?;
    let value: toml::Value = text.parse().ok()?;

    let mut meta = RegionMetadata {
        width_m: 2_000.0,
        depth_m: 2_000.0,
        min_height_m: -100.0,
        max_height_m: 2_000.0,
    };

    if let Some(half_extent) =
        find_number(&value, "half_extent_m").or_else(|| find_number(&value, "half_extent"))
    {
        meta.width_m = (half_extent * 2.0) as f32;
        meta.depth_m = (half_extent * 2.0) as f32;
    } else if let Some(side_m) = find_number(&value, "side_length_m")
        .or_else(|| find_number(&value, "side_m"))
        .or_else(|| find_number(&value, "side"))
    {
        meta.width_m = side_m as f32;
        meta.depth_m = side_m as f32;
    } else if let Some(side_km) = find_number(&value, "side_km") {
        meta.width_m = (side_km * 1000.0) as f32;
        meta.depth_m = (side_km * 1000.0) as f32;
    }

    if let Some(min_h) =
        find_number(&value, "min_height_m").or_else(|| find_number(&value, "min_height"))
    {
        meta.min_height_m = min_h as f32;
    }

    if let Some(max_h) =
        find_number(&value, "max_height_m").or_else(|| find_number(&value, "max_height"))
    {
        meta.max_height_m = max_h as f32;
    }

    Some(meta)
}

/// Picks a tile coordinate string (e.g. "0_0_0_0") suitable for coarse preview.
///
/// Strategy:
/// - Prefer the smallest `lod` present in `config.tc`.
/// - Then prefer the smallest (side, y, x) for stability.
///
/// Returns `None` if `config.tc` is missing or undecodable.
pub fn pick_coarsest_tile(region_root: &Path) -> Option<String> {
    let bytes = fs::read(region_root.join("config.tc")).ok()?;
    let tc = decode_tc(&bytes)?;
    let min_lod = tc.tiles.iter().map(|t| t.lod).min()?;

    let mut tiles: Vec<_> = tc.tiles.into_iter().filter(|t| t.lod == min_lod).collect();
    tiles.sort_by_key(|t| (t.side, t.y, t.x));
    tiles.first().map(|t| t.to_string())
}

pub fn region_root_from_assets_dir(assets_dir: &Path, region: &str) -> PathBuf {
    assets_dir.join("terrains/planar").join(region)
}
