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
    cube_face_to_dir, dir_to_lonlat, ImageryLonLatSampler, TerrainLonLatSampler, TileFetcher,
};
use bevy_world_mesh::fetch_pipeline::{
    format_stats_message, workers_from_env_or, workers_from_env_or_available, TilePhase,
};
use bevy_world_mesh::scenes::globe::{
    GlobeManifest, DEFAULT_CAMERA_DISTANCE_RADII, DEFAULT_LOD_COUNT, MAX_HEIGHT_M, MIN_HEIGHT_M,
};
use image::{ImageBuffer, Luma, RgbImage};
use indicatif::{MultiProgress, ProgressBar};
use rayon::prelude::*;
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
};

const CACHE_DIR: &str = "target/tile_cache";
/// Default worker count per provider fetch pool. Keep this conservative so
/// public tile servers don't throttle us.
const DEFAULT_FETCH_WORKERS: usize = 8;
/// Fallback face-generation worker count if CPU parallelism can't be detected.
const FALLBACK_FACE_WORKERS: usize = 8;
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
    missing_height: bool,
    missing_albedo: bool,
    workers: usize,
) -> Result<()> {
    eprintln!(
        "    prefetching z={zoom} tile cache with {workers} worker(s) per provider pool \
         (override via WORLD_MESH_FETCH_WORKERS)"
    );

    let mp = MultiProgress::new();
    let terrain_phase = TilePhase::square("terrain-pref", zoom, tiles_per_axis);
    let imagery_phase = TilePhase::square("imagery-pref", zoom, tiles_per_axis);

    match (missing_height, missing_albedo) {
        (true, true) => {
            terrain_phase.print_banner();
            imagery_phase.print_banner();
            let terrain_pb = terrain_phase.progress_bar(&mp);
            let imagery_pb = imagery_phase.progress_bar(&mp);
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
                        prefetch_imagery_tiles(fetcher, imagery_phase, tiles_per_axis, &imagery_pb)
                    })
                });

                terrain_pool.install(|| {
                    prefetch_terrain_tiles(fetcher, terrain_phase, tiles_per_axis, &terrain_pb)
                })?;

                imagery_handle
                    .join()
                    .map_err(|_| anyhow!("imagery prefetch thread panicked"))??;

                Ok(())
            })?;
        }
        (true, false) => {
            terrain_phase.print_banner();
            let terrain_pb = terrain_phase.progress_bar(&mp);
            let terrain_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build terrain prefetch rayon pool")?;
            terrain_pool.install(|| {
                prefetch_terrain_tiles(fetcher, terrain_phase, tiles_per_axis, &terrain_pb)
            })?;
        }
        (false, true) => {
            imagery_phase.print_banner();
            let imagery_pb = imagery_phase.progress_bar(&mp);
            let imagery_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .context("build imagery prefetch rayon pool")?;
            imagery_pool.install(|| {
                prefetch_imagery_tiles(fetcher, imagery_phase, tiles_per_axis, &imagery_pb)
            })?;
        }
        (false, false) => {}
    }

    Ok(())
}

fn parse_fetch_workers() -> usize {
    workers_from_env_or("WORLD_MESH_FETCH_WORKERS", DEFAULT_FETCH_WORKERS)
}

fn parse_face_workers() -> usize {
    workers_from_env_or_available("WORLD_MESH_FACE_WORKERS", FALLBACK_FACE_WORKERS)
}

fn prefetch_terrain_tiles(
    fetcher: &TileFetcher,
    phase: TilePhase,
    tiles_per_axis: u32,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_aws_terrain_stats();
    let axis = u64::from(tiles_per_axis);
    let result = (0..phase.total)
        .into_par_iter()
        .try_for_each(|idx| -> Result<()> {
            let tx = (idx % axis) as u32;
            let ty = (idx / axis) as u32;
            fetcher
                .prefetch_aws_terrain(tx, ty, phase.z)
                .with_context(|| format!("prefetch AWS terrain {}/{tx}/{ty}", phase.z))?;
            pb.set_message(format_stats_message(fetcher.aws_terrain_stats()));
            pb.inc(1);
            Ok(())
        });

    phase.print_summary(pb, fetcher.aws_terrain_stats());
    result
}

fn prefetch_imagery_tiles(
    fetcher: &TileFetcher,
    phase: TilePhase,
    tiles_per_axis: u32,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_eox_imagery_stats();
    let axis = u64::from(tiles_per_axis);
    let result = (0..phase.total)
        .into_par_iter()
        .try_for_each(|idx| -> Result<()> {
            let tx = (idx % axis) as u32;
            let ty = (idx / axis) as u32;
            fetcher
                .prefetch_eox_imagery(tx, ty, phase.z)
                .with_context(|| format!("prefetch EOX imagery {}/{tx}/{ty}", phase.z))?;
            pb.set_message(format_stats_message(fetcher.eox_imagery_stats()));
            pb.inc(1);
            Ok(())
        });

    phase.print_summary(pb, fetcher.eox_imagery_stats());
    result
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
        || TerrainLonLatSampler::new(fetcher, zoom),
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
            || ImageryLonLatSampler::new(fetcher, zoom),
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
