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
use bevy_world_mesh::fetch::{cube_face_to_dir, dir_to_lonlat, TileFetcher};
use bevy_world_mesh::scenes::globe::{
    GlobeManifest, DEFAULT_CAMERA_DISTANCE_RADII, DEFAULT_LOD_COUNT, MAX_HEIGHT_M, MIN_HEIGHT_M,
};
use image::{ImageBuffer, Luma, Rgb, RgbImage};
use std::{fs, path::Path};

const CACHE_DIR: &str = "target/tile_cache";
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

    eprintln!(
        "==> Fetching global spherical Earth (z={}, face_size={}\u{00B2}, ~{} tile{} per source globally)",
        args.zoom,
        args.face_size,
        1u32 << (2 * args.zoom),
        if args.zoom == 0 { "" } else { "s" }
    );

    fs::create_dir_all(OUT_DIR_HEIGHT)?;
    fs::create_dir_all(OUT_DIR_ALBEDO)?;

    let fetcher = TileFetcher::new(CACHE_DIR);

    for face in 0..6u8 {
        fetch_face_height(&fetcher, face, args.zoom, args.face_size)?;
        fetch_face_albedo(&fetcher, face, args.zoom, args.face_size)?;
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

fn fetch_face_height(fetcher: &TileFetcher, face: u8, zoom: u8, size: u32) -> Result<()> {
    let out_path = format!("{OUT_DIR_HEIGHT}/face{face}.tif");
    if Path::new(&out_path).exists() {
        eprintln!("    skip {out_path} (already exists)");
        return Ok(());
    }

    eprintln!("    face {face} height ({size}\u{00D7}{size}, z={zoom})");
    let span = (MAX_HEIGHT_M - MIN_HEIGHT_M).max(1e-6);
    let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f64 + 0.5) / size as f64;
            let v = (y as f64 + 0.5) / size as f64;
            let dir = cube_face_to_dir(face, u, v);
            let (lon, lat) = dir_to_lonlat(dir);
            let h = fetcher
                .sample_terrain_at_lonlat(lon, lat, zoom)
                .with_context(|| {
                    format!("face={face} pixel=({x},{y}) lon={lon:.4} lat={lat:.4}")
                })?;
            let n = ((h - MIN_HEIGHT_M) / span).clamp(0.0, 1.0);
            img.put_pixel(x, y, Luma([(n * u16::MAX as f32) as u16]));
        }
        if y % 128 == 127 {
            eprintln!("        face {face} height row {}/{size}", y + 1);
        }
    }
    img.save(&out_path)
        .with_context(|| format!("save {out_path}"))?;
    eprintln!(
        "    wrote {out_path} ({size}\u{00D7}{size} u16, normalised against [{MIN_HEIGHT_M}, {MAX_HEIGHT_M}] m)"
    );
    Ok(())
}

fn fetch_face_albedo(fetcher: &TileFetcher, face: u8, zoom: u8, size: u32) -> Result<()> {
    let out_path = format!("{OUT_DIR_ALBEDO}/face{face}.png");
    if Path::new(&out_path).exists() {
        eprintln!("    skip {out_path} (already exists)");
        return Ok(());
    }

    eprintln!("    face {face} albedo ({size}\u{00D7}{size}, z={zoom})");
    let mut img: RgbImage = ImageBuffer::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f64 + 0.5) / size as f64;
            let v = (y as f64 + 0.5) / size as f64;
            let dir = cube_face_to_dir(face, u, v);
            let (lon, lat) = dir_to_lonlat(dir);
            let rgb = fetcher
                .sample_imagery_at_lonlat(lon, lat, zoom)
                .with_context(|| {
                    format!("face={face} pixel=({x},{y}) lon={lon:.4} lat={lat:.4}")
                })?;
            img.put_pixel(x, y, Rgb(rgb));
        }
        if y % 128 == 127 {
            eprintln!("        face {face} albedo row {}/{size}", y + 1);
        }
    }
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
