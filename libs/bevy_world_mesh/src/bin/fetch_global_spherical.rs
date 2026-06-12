// Fetch real-world DEM (AWS Terrain Tiles, terrarium-encoded) and imagery
// (EOX Sentinel-2 Cloudless WMTS) for the entire globe by sampling the cube
// faces that bevy_terrain's spherical preprocessor consumes. Writes 12
// outputs:
//
//   assets/terrains/spherical/source/height/face{0..5}.tif    16-bit u16
//   assets/terrains/spherical/source/albedo/face{0..5}.png    8-bit RGB
//
// (TIFF for height because bevy_terrain's `TiffLoader` is hardcoded to
// `R16Unorm`; PNG for albedo because Bevy's built-in PNG loader auto-expands
// RGB to RGBA8UnormSrgb which is what the `Rgba8` attachment needs.)
//
// Plus a `globe.toml` manifest. The same `TileFetcher` (disk + decoded
// caches) backs both the planar and spherical fetchers, so re-runs are
// instantaneous and adjacent face pixels almost always hit the warm
// in-memory tile.
//
// Usage:
//   cargo run --release --bin fetch_global_spherical
//   cargo run --release --bin fetch_global_spherical -- --zoom 6
//   cargo run --release --bin fetch_global_spherical -- --face-size 1024

use anyhow::{anyhow, Context, Result};
use bevy_world_mesh::fetch::{
    cube_face_to_dir, dir_to_lonlat, lonlat_to_tile, tile_to_lonlat, FetchStats, TileFetcher,
    TILE_PX,
};
use bevy_world_mesh::scenes::globe::{
    GlobeManifest, DEFAULT_CAMERA_DISTANCE_RADII, DEFAULT_LOD_COUNT, MAX_HEIGHT_M, MIN_HEIGHT_M,
};
use image::{ImageBuffer, Luma, RgbImage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};

const CACHE_DIR: &str = "target/tile_cache";
/// Default worker count per provider fetch pool. Keep this conservative so
/// public tile servers don't throttle us.
const DEFAULT_FETCH_WORKERS: usize = 8;
/// Fallback face-generation worker count if CPU parallelism can't be detected.
const FALLBACK_FACE_WORKERS: usize = 8;
/// Per-worker decoded-tile cache cap used while generating cube-face pixels.
const LOCAL_DECODED_CACHE_CAP: usize = 256;
/// Default zoom level for both elevation and imagery. z=8 gives ≈ 39 m/px at
/// the equator before resampling — 4x richer than z=7, at the cost of ≈ 4 GB
/// total one-time download (65 536 unique tiles globally per source).
const DEFAULT_ZOOM: u8 = 8;
/// Default cube-face TIFF resolution (pixels per side). 8192 matches z=8
/// source native per cube face (64 tiles × 256 px → 8192) with only ~2x
/// downsampling so face TIFFs preserve most of the z=8 information content.
const DEFAULT_FACE_SIZE: u32 = 8192;

const OUT_DIR_HEIGHT: &str = "assets/terrains/spherical/source/height";
const OUT_DIR_ALBEDO: &str = "assets/terrains/spherical/source/albedo";
const MANIFEST_PATH: &str = "assets/terrains/spherical/globe.toml";

fn main() -> Result<()> {
    let args = parse_args()?;
    let tiles_per_axis = tile_axis_len(args.zoom)?;
    let tiles_per_source = u64::from(tiles_per_axis) * u64::from(tiles_per_axis);

    eprintln!(
        "==> Fetching global spherical Earth (z={}, face_size={}\u{00B2}, ~{} tile{} per source globally)",
        args.zoom,
        args.face_size,
        tiles_per_source,
        if tiles_per_source == 1 { "" } else { "s" }
    );

    fs::create_dir_all(OUT_DIR_HEIGHT)?;
    fs::create_dir_all(OUT_DIR_ALBEDO)?;

    let fetch_workers = parse_fetch_workers();
    let face_workers = parse_face_workers();
    let fetcher = TileFetcher::new(CACHE_DIR);
    let reuse_existing_faces = existing_manifest_matches(args.zoom, args.face_size);
    if !reuse_existing_faces && any_source_face_exists() {
        eprintln!(
            "    existing spherical sources do not match requested z/face_size; regenerating faces"
        );
    }
    let missing_height =
        !reuse_existing_faces || (0..6u8).any(|face| !Path::new(&height_face_path(face)).exists());
    let missing_albedo =
        !reuse_existing_faces || (0..6u8).any(|face| !Path::new(&albedo_face_path(face)).exists());
    if missing_height || missing_albedo {
        prefetch_global_tiles(
            &fetcher,
            args.zoom,
            tiles_per_axis,
            tiles_per_source,
            missing_height,
            missing_albedo,
            fetch_workers,
        )?;
    } else {
        eprintln!("    all source faces already exist; skipping tile prefetch");
    }

    eprintln!(
        "    generating cube faces with {face_workers} worker(s) \
         (override via WORLD_MESH_FACE_WORKERS)"
    );
    let face_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(face_workers)
        .build()
        .context("build face generation rayon pool")?;

    for face in 0..6u8 {
        face_pool.install(|| {
            fetch_face_height(
                &fetcher,
                face,
                args.zoom,
                args.face_size,
                reuse_existing_faces,
            )
        })?;
        face_pool.install(|| {
            fetch_face_albedo(
                &fetcher,
                face,
                args.zoom,
                args.face_size,
                reuse_existing_faces,
            )
        })?;
    }

    write_manifest(args.zoom, args.face_size)?;

    eprintln!(
        "==> Done. Run `cargo run --release --bin preprocess_global` to rebuild the spherical atlas."
    );
    Ok(())
}

#[derive(Debug)]
struct Args {
    zoom: u8,
    face_size: u32,
}

fn parse_args() -> Result<Args> {
    let mut zoom = DEFAULT_ZOOM;
    let mut face_size = DEFAULT_FACE_SIZE;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--zoom" | "-z" => {
                zoom = iter
                    .next()
                    .context("--zoom needs a value")?
                    .parse()
                    .context("--zoom must be a u8")?;
            }
            other if other.starts_with("--zoom=") => {
                zoom = other.trim_start_matches("--zoom=").parse()?;
            }
            "--face-size" => {
                face_size = iter
                    .next()
                    .context("--face-size needs a value")?
                    .parse()
                    .context("--face-size must be a u32")?;
            }
            other if other.starts_with("--face-size=") => {
                face_size = other.trim_start_matches("--face-size=").parse()?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg {other:?}");
                print_usage();
                return Err(anyhow!("unknown arg {other:?}"));
            }
        }
    }
    Ok(Args { zoom, face_size })
}

fn print_usage() {
    eprintln!("usage: fetch_global_spherical [--zoom N] [--face-size N]");
    eprintln!("  defaults: --zoom {DEFAULT_ZOOM} --face-size {DEFAULT_FACE_SIZE}");
    eprintln!("  z=6 \u{2192} ~150 MB total, faster.  z=7 \u{2192} ~600 MB total, sharper.");
}

fn tile_axis_len(zoom: u8) -> Result<u32> {
    1u32.checked_shl(u32::from(zoom))
        .filter(|&n| n > 0)
        .ok_or_else(|| anyhow!("--zoom {zoom} is too large for u32 tile coordinates"))
}

fn height_face_path(face: u8) -> String {
    format!("{OUT_DIR_HEIGHT}/face{face}.tif")
}

fn albedo_face_path(face: u8) -> String {
    format!("{OUT_DIR_ALBEDO}/face{face}.png")
}

fn existing_manifest_matches(zoom: u8, face_size: u32) -> bool {
    fs::read_to_string(MANIFEST_PATH)
        .ok()
        .and_then(|text| toml::from_str::<GlobeManifest>(&text).ok())
        .map(|manifest| manifest.zoom == zoom && manifest.face_size == face_size)
        .unwrap_or(false)
}

fn any_source_face_exists() -> bool {
    (0..6u8).any(|face| {
        Path::new(&height_face_path(face)).exists() || Path::new(&albedo_face_path(face)).exists()
    })
}

fn prefetch_global_tiles(
    fetcher: &TileFetcher,
    zoom: u8,
    tiles_per_axis: u32,
    tiles_per_source: u64,
    missing_height: bool,
    missing_albedo: bool,
    workers: usize,
) -> Result<()> {
    eprintln!(
        "    prefetching z={zoom} tile cache with {workers} worker(s) per provider pool \
         (override via WORLD_MESH_FETCH_WORKERS)"
    );

    let mp = MultiProgress::new();
    match (missing_height, missing_albedo) {
        (true, true) => {
            print_phase_banner("terrain-pref", zoom, tiles_per_axis, tiles_per_source);
            print_phase_banner("imagery-pref", zoom, tiles_per_axis, tiles_per_source);
            let terrain_pb = mp.add(make_progress_bar(tiles_per_source, "terrain-pref"));
            let imagery_pb = mp.add(make_progress_bar(tiles_per_source, "imagery-pref"));
            let terrain_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build terrain prefetch rayon pool")?;
            let imagery_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build imagery prefetch rayon pool")?;

            std::thread::scope(|scope| -> Result<()> {
                let imagery_handle = scope.spawn(|| -> Result<()> {
                    imagery_pool.install(|| {
                        prefetch_imagery_tiles(
                            fetcher,
                            zoom,
                            tiles_per_axis,
                            tiles_per_source,
                            &imagery_pb,
                        )
                    })
                });

                terrain_pool.install(|| {
                    prefetch_terrain_tiles(
                        fetcher,
                        zoom,
                        tiles_per_axis,
                        tiles_per_source,
                        &terrain_pb,
                    )
                })?;

                imagery_handle
                    .join()
                    .map_err(|_| anyhow!("imagery prefetch thread panicked"))??;

                Ok(())
            })?;
        }
        (true, false) => {
            print_phase_banner("terrain-pref", zoom, tiles_per_axis, tiles_per_source);
            let terrain_pb = mp.add(make_progress_bar(tiles_per_source, "terrain-pref"));
            let terrain_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build terrain prefetch rayon pool")?;
            terrain_pool.install(|| {
                prefetch_terrain_tiles(fetcher, zoom, tiles_per_axis, tiles_per_source, &terrain_pb)
            })?;
        }
        (false, true) => {
            print_phase_banner("imagery-pref", zoom, tiles_per_axis, tiles_per_source);
            let imagery_pb = mp.add(make_progress_bar(tiles_per_source, "imagery-pref"));
            let imagery_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build imagery prefetch rayon pool")?;
            imagery_pool.install(|| {
                prefetch_imagery_tiles(fetcher, zoom, tiles_per_axis, tiles_per_source, &imagery_pb)
            })?;
        }
        (false, false) => {}
    }

    Ok(())
}

fn parse_fetch_workers() -> usize {
    std::env::var("WORLD_MESH_FETCH_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_FETCH_WORKERS)
}

fn parse_face_workers() -> usize {
    std::env::var("WORLD_MESH_FACE_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(FALLBACK_FACE_WORKERS)
        })
}

fn print_phase_banner(label: &'static str, zoom: u8, tiles_per_axis: u32, tiles_per_source: u64) {
    eprintln!(
        "    {label:12} z={zoom} {tiles_per_axis}x{tiles_per_axis} = {tiles_per_source} tile(s)"
    );
}

fn make_progress_bar(total: u64, label: &'static str) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "    {prefix:12} [{elapsed_precise}] [{bar:40.cyan/blue}] \
         {pos}/{len} {msg} {per_sec} ETA {eta}",
    )
    .expect("static progress template parses")
    .progress_chars("#>-");
    let pb = ProgressBar::new(total);
    pb.set_style(style);
    pb.set_prefix(label);
    pb
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn format_stats_message(s: FetchStats) -> String {
    if s.retries == 0 {
        format!("net={} cached={}", s.network_hits, s.cache_hits)
    } else {
        format!(
            "net={} cached={} retries={}",
            s.network_hits, s.cache_hits, s.retries
        )
    }
}

fn print_phase_summary(
    pb: &ProgressBar,
    label: &'static str,
    zoom: u8,
    total: u64,
    stats: FetchStats,
) {
    let elapsed = pb.elapsed();
    pb.finish_and_clear();
    let rate = total as f64 / elapsed.as_secs_f64().max(1e-6);
    let retries_suffix = if stats.retries == 0 {
        String::new()
    } else {
        format!(", {} retries", stats.retries)
    };
    eprintln!(
        "    {label:12} z={zoom} done in {} — {} net + {} cached{} ({rate:.1} tile/s avg)",
        format_duration(elapsed),
        stats.network_hits,
        stats.cache_hits,
        retries_suffix,
    );
}

fn prefetch_terrain_tiles(
    fetcher: &TileFetcher,
    zoom: u8,
    tiles_per_axis: u32,
    tiles_per_source: u64,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_aws_terrain_stats();
    let axis = u64::from(tiles_per_axis);
    let result = (0..tiles_per_source)
        .into_par_iter()
        .try_for_each(|idx| -> Result<()> {
            let tx = (idx % axis) as u32;
            let ty = (idx / axis) as u32;
            fetcher
                .prefetch_aws_terrain(tx, ty, zoom)
                .with_context(|| format!("prefetch AWS terrain {zoom}/{tx}/{ty}"))?;
            pb.set_message(format_stats_message(fetcher.aws_terrain_stats()));
            pb.inc(1);
            Ok(())
        });

    print_phase_summary(
        pb,
        "terrain-pref",
        zoom,
        tiles_per_source,
        fetcher.aws_terrain_stats(),
    );
    result
}

fn prefetch_imagery_tiles(
    fetcher: &TileFetcher,
    zoom: u8,
    tiles_per_axis: u32,
    tiles_per_source: u64,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_eox_imagery_stats();
    let axis = u64::from(tiles_per_axis);
    let result = (0..tiles_per_source)
        .into_par_iter()
        .try_for_each(|idx| -> Result<()> {
            let tx = (idx % axis) as u32;
            let ty = (idx / axis) as u32;
            fetcher
                .prefetch_eox_imagery(tx, ty, zoom)
                .with_context(|| format!("prefetch EOX imagery {zoom}/{tx}/{ty}"))?;
            pb.set_message(format_stats_message(fetcher.eox_imagery_stats()));
            pb.inc(1);
            Ok(())
        });

    print_phase_summary(
        pb,
        "imagery-pref",
        zoom,
        tiles_per_source,
        fetcher.eox_imagery_stats(),
    );
    result
}

type TileKey = (u8, u32, u32);

struct TerrainFaceSampler<'a> {
    fetcher: &'a TileFetcher,
    zoom: u8,
    cache: HashMap<TileKey, Vec<f32>>,
}

impl<'a> TerrainFaceSampler<'a> {
    fn new(fetcher: &'a TileFetcher, zoom: u8) -> Self {
        Self {
            fetcher,
            zoom,
            cache: HashMap::new(),
        }
    }

    fn sample(&mut self, lon: f64, lat: f64) -> Result<f32> {
        let (tx, ty) = lonlat_to_tile(lon, lat, self.zoom);
        let (u, v) = tile_subpixel_local(lon, lat, tx, ty, self.zoom);
        let tile = self.tile(tx, ty)?;
        Ok(sample_f32_bilinear_local(tile, TILE_PX, u, v))
    }

    fn tile(&mut self, tx: u32, ty: u32) -> Result<&[f32]> {
        let key = (self.zoom, tx, ty);
        if !self.cache.contains_key(&key) {
            let tile = self
                .fetcher
                .fetch_aws_terrain(tx, ty, self.zoom)
                .with_context(|| format!("AWS terrain tile {}/{}/{}", self.zoom, tx, ty))?;
            evict_local_if_full(&mut self.cache);
            self.cache.insert(key, tile);
        }
        Ok(self.cache.get(&key).expect("tile inserted").as_slice())
    }
}

struct ImageryFaceSampler<'a> {
    fetcher: &'a TileFetcher,
    zoom: u8,
    cache: HashMap<TileKey, RgbImage>,
}

impl<'a> ImageryFaceSampler<'a> {
    fn new(fetcher: &'a TileFetcher, zoom: u8) -> Self {
        Self {
            fetcher,
            zoom,
            cache: HashMap::new(),
        }
    }

    fn sample(&mut self, lon: f64, lat: f64) -> Result<[u8; 3]> {
        let (tx, ty) = lonlat_to_tile(lon, lat, self.zoom);
        let (u, v) = tile_subpixel_local(lon, lat, tx, ty, self.zoom);
        let tile = self.tile(tx, ty)?;
        Ok(sample_rgb_bilinear_local(tile, u, v))
    }

    fn tile(&mut self, tx: u32, ty: u32) -> Result<&RgbImage> {
        let key = (self.zoom, tx, ty);
        if !self.cache.contains_key(&key) {
            let tile = self
                .fetcher
                .fetch_eox_imagery(tx, ty, self.zoom)
                .with_context(|| format!("EOX imagery tile {}/{}/{}", self.zoom, tx, ty))?;
            evict_local_if_full(&mut self.cache);
            self.cache.insert(key, tile);
        }
        Ok(self.cache.get(&key).expect("tile inserted"))
    }
}

fn evict_local_if_full<V>(cache: &mut HashMap<TileKey, V>) {
    if cache.len() >= LOCAL_DECODED_CACHE_CAP {
        if let Some(k) = cache.keys().next().copied() {
            cache.remove(&k);
        }
    }
}

fn tile_subpixel_local(lon: f64, lat: f64, tx: u32, ty: u32, z: u8) -> (f64, f64) {
    let (nw_lon, nw_lat) = tile_to_lonlat(tx, ty, z);
    let (se_lon, se_lat) = tile_to_lonlat(tx + 1, ty + 1, z);
    let u = ((lon - nw_lon) / (se_lon - nw_lon)).clamp(0.0, 1.0);
    let v = ((nw_lat - lat) / (nw_lat - se_lat)).clamp(0.0, 1.0);
    (u, v)
}

fn sample_f32_bilinear_local(buf: &[f32], size: u32, u: f64, v: f64) -> f32 {
    let px_f = u * size as f64 - 0.5;
    let py_f = v * size as f64 - 0.5;
    let px0 = (px_f.floor() as i32).clamp(0, size as i32 - 1) as u32;
    let py0 = (py_f.floor() as i32).clamp(0, size as i32 - 1) as u32;
    let px1 = (px0 + 1).min(size - 1);
    let py1 = (py0 + 1).min(size - 1);
    let fx = (px_f - px_f.floor()).clamp(0.0, 1.0) as f32;
    let fy = (py_f - py_f.floor()).clamp(0.0, 1.0) as f32;
    let v00 = buf[(py0 * size + px0) as usize];
    let v10 = buf[(py0 * size + px1) as usize];
    let v01 = buf[(py1 * size + px0) as usize];
    let v11 = buf[(py1 * size + px1) as usize];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

fn sample_rgb_bilinear_local(img: &RgbImage, u: f64, v: f64) -> [u8; 3] {
    let size = TILE_PX;
    let px_f = u * size as f64 - 0.5;
    let py_f = v * size as f64 - 0.5;
    let px0 = (px_f.floor() as i32).clamp(0, size as i32 - 1) as u32;
    let py0 = (py_f.floor() as i32).clamp(0, size as i32 - 1) as u32;
    let px1 = (px0 + 1).min(size - 1);
    let py1 = (py0 + 1).min(size - 1);
    let fx = (px_f - px_f.floor()).clamp(0.0, 1.0) as f32;
    let fy = (py_f - py_f.floor()).clamp(0.0, 1.0) as f32;
    let p00 = img.get_pixel(px0, py0).0;
    let p10 = img.get_pixel(px1, py0).0;
    let p01 = img.get_pixel(px0, py1).0;
    let p11 = img.get_pixel(px1, py1).0;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
            + p10[c] as f32 * fx * (1.0 - fy)
            + p01[c] as f32 * (1.0 - fx) * fy
            + p11[c] as f32 * fx * fy;
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn fetch_face_height(
    fetcher: &TileFetcher,
    face: u8,
    zoom: u8,
    size: u32,
    reuse_existing_faces: bool,
) -> Result<()> {
    let out_path = height_face_path(face);
    if reuse_existing_faces && Path::new(&out_path).exists() {
        eprintln!("    skip {out_path} (already exists)");
        return Ok(());
    }

    eprintln!("    face {face} height ({size}\u{00D7}{size}, z={zoom})");
    let side = usize::try_from(size).context("face size doesn't fit usize")?;
    let pixel_count = side
        .checked_mul(side)
        .context("height face pixel count overflow")?;
    let span = (MAX_HEIGHT_M - MIN_HEIGHT_M).max(1e-6);
    let rows_done = AtomicU32::new(0);
    let mut buf = vec![0u16; pixel_count];

    buf.par_chunks_mut(side).enumerate().try_for_each_init(
        || TerrainFaceSampler::new(fetcher, zoom),
        |sampler, (y, row)| -> Result<()> {
            let y_u32 = u32::try_from(y).context("row index doesn't fit u32")?;
            for (x, out) in row.iter_mut().enumerate() {
                let x_u32 = u32::try_from(x).context("column index doesn't fit u32")?;
                let u = (x_u32 as f64 + 0.5) / size as f64;
                let v = (y_u32 as f64 + 0.5) / size as f64;
                let dir = cube_face_to_dir(face, u, v);
                let (lon, lat) = dir_to_lonlat(dir);
                let h = sampler.sample(lon, lat).with_context(|| {
                    format!("face={face} pixel=({x_u32},{y_u32}) lon={lon:.4} lat={lat:.4}")
                })?;
                let n = ((h - MIN_HEIGHT_M) / span).clamp(0.0, 1.0);
                *out = (n * u16::MAX as f32) as u16;
            }

            let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 128 == 0 || done == size {
                eprintln!("        face {face} height rows {done}/{size}");
            }
            Ok(())
        },
    )?;

    let img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(size, size, buf)
        .ok_or_else(|| anyhow!("height face buffer length doesn't match dimensions"))?;
    img.save(&out_path)
        .with_context(|| format!("save {out_path}"))?;
    eprintln!(
        "    wrote {out_path} ({size}\u{00D7}{size} u16, normalised against [{MIN_HEIGHT_M}, {MAX_HEIGHT_M}] m)"
    );
    Ok(())
}

fn fetch_face_albedo(
    fetcher: &TileFetcher,
    face: u8,
    zoom: u8,
    size: u32,
    reuse_existing_faces: bool,
) -> Result<()> {
    let out_path = albedo_face_path(face);
    if reuse_existing_faces && Path::new(&out_path).exists() {
        eprintln!("    skip {out_path} (already exists)");
        return Ok(());
    }

    eprintln!("    face {face} albedo ({size}\u{00D7}{size}, z={zoom})");
    let side = usize::try_from(size).context("face size doesn't fit usize")?;
    let row_bytes = side
        .checked_mul(3)
        .context("albedo face row byte count overflow")?;
    let buf_len = side
        .checked_mul(row_bytes)
        .context("albedo face byte count overflow")?;
    let rows_done = AtomicU32::new(0);
    let mut buf = vec![0u8; buf_len];

    buf.par_chunks_mut(row_bytes)
        .enumerate()
        .try_for_each_init(
            || ImageryFaceSampler::new(fetcher, zoom),
            |sampler, (y, row)| -> Result<()> {
                let y_u32 = u32::try_from(y).context("row index doesn't fit u32")?;
                for x in 0..side {
                    let x_u32 = u32::try_from(x).context("column index doesn't fit u32")?;
                    let u = (x_u32 as f64 + 0.5) / size as f64;
                    let v = (y_u32 as f64 + 0.5) / size as f64;
                    let dir = cube_face_to_dir(face, u, v);
                    let (lon, lat) = dir_to_lonlat(dir);
                    let rgb = sampler.sample(lon, lat).with_context(|| {
                        format!("face={face} pixel=({x_u32},{y_u32}) lon={lon:.4} lat={lat:.4}")
                    })?;
                    let off = x * 3;
                    row[off..off + 3].copy_from_slice(&rgb);
                }

                let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 128 == 0 || done == size {
                    eprintln!("        face {face} albedo rows {done}/{size}");
                }
                Ok(())
            },
        )?;

    let img: RgbImage = ImageBuffer::from_raw(size, size, buf)
        .ok_or_else(|| anyhow!("albedo face buffer length doesn't match dimensions"))?;
    img.save(&out_path)
        .with_context(|| format!("save {out_path}"))?;
    eprintln!("    wrote {out_path} ({size}\u{00D7}{size} 8-bit RGB)");
    Ok(())
}

fn write_manifest(zoom: u8, face_size: u32) -> Result<()> {
    let manifest = GlobeManifest {
        name: "earth".to_string(),
        zoom,
        face_size,
        min_height_m: MIN_HEIGHT_M,
        max_height_m: MAX_HEIGHT_M,
        lod_count: DEFAULT_LOD_COUNT,
        camera_distance_radii: DEFAULT_CAMERA_DISTANCE_RADII,
    };
    let toml = toml::to_string_pretty(&manifest).context("serialise globe.toml")?;
    if let Some(parent) = Path::new(MANIFEST_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(MANIFEST_PATH, toml).with_context(|| format!("write {MANIFEST_PATH}"))?;
    eprintln!("    wrote {MANIFEST_PATH}");
    Ok(())
}
