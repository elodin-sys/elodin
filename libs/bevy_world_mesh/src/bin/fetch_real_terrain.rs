// Fetch real-world DEM (AWS Terrain Tiles, terrarium-encoded) and imagery
// (EOX Sentinel-2 Cloudless WMTS) for a named region preset, stitch + crop
// + resize, and write the
// `assets/terrains/planar/<region>/source/{height,albedo}.png` pair the
// `preprocess` binary consumes. Also emits a
// `assets/terrains/planar/<region>/region.toml` manifest so
// `PlanarScenePlugin` can re-frame the camera and rescale `TERRAIN_SIZE` /
// `HEIGHT` without recompiling.
//
// Each region writes to its own subdirectory so multiple regions coexist on
// disk; the active region is selected by the `WORLD_MESH_REGION` env var
// (default: `death_valley`).
//
// Usage:
//   cargo run --release --bin fetch_real_terrain -- --region brienz
//   cargo run --release --bin fetch_real_terrain -- --region death_valley
//   cargo run --release --bin fetch_real_terrain -- --region mojave_desert

use anyhow::{anyhow, Context, Result};
use bevy_world_mesh::fetch::{
    bbox_to_tile_range, stream_stitch_imagery, stream_stitch_terrain, Bbox, TileFetcher, TileRange,
};
use bevy_world_mesh::fetch_pipeline::{
    format_stats_message, workers_from_env_or_available, TilePhase,
};
use bevy_world_mesh::regions::{self, Region, RegionManifest};
use indicatif::{MultiProgress, ProgressBar};
use rayon::prelude::*;
use std::{fs, path::Path};

/// Output PNG side length. Larger gives more detail at the cost of preprocess
/// time and disk; 16384 leaves the deepest atlas LOD (8192 px per terrain dim)
/// reading source at modest oversample for 100+ km regions, with the source
/// itself sampled at or above z=15 native resolution.
const OUTPUT_SIZE: u32 = 16384;
/// Zoom for AWS Terrain Tiles. z=15 ≈ 3-4 m/px before resize (mid-latitudes);
/// matches `IMAGERY_ZOOM` so heights and colour resolve the same features.
const TERRAIN_ZOOM: u8 = 15;
/// Zoom for EOX Sentinel-2 imagery. z=15 ≈ 3-4 m/px before resize.
const IMAGERY_ZOOM: u8 = 15;
/// Disk cache for tiles, persists between runs.
const CACHE_DIR: &str = "target/tile_cache";
/// Fallback worker count if [`std::thread::available_parallelism`] errors.
/// We never expect this branch to fire on a real OS, but it keeps the binary
/// from refusing to start.
const FALLBACK_WORKERS: usize = 8;

fn main() -> Result<()> {
    let region_name = parse_region_arg()?;
    let region = regions::lookup(&region_name).ok_or_else(|| {
        anyhow!(
            "unknown region {region_name:?}; known: {:?}",
            regions::PRESETS.iter().map(|r| r.name).collect::<Vec<_>>()
        )
    })?;

    eprintln!(
        "==> Fetching real terrain for {} ({:.4}°N, {:.4}°E, {} km square)",
        region.name, region.center_lat, region.center_lon, region.side_km
    );

    let bbox = Bbox::around(region.center_lon, region.center_lat, region.side_km);
    eprintln!(
        "    bbox: W={:.4} S={:.4} E={:.4} N={:.4}",
        bbox.west, bbox.south, bbox.east, bbox.north
    );

    let workers = parse_workers();
    eprintln!(
        "    using {workers} concurrent worker(s) per provider pool \
         (override via WORLD_MESH_FETCH_WORKERS)"
    );

    let terrain_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build terrain rayon pool")?;
    let imagery_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build imagery rayon pool")?;

    let fetcher = TileFetcher::new(CACHE_DIR);
    let terrain_range = bbox_to_tile_range(bbox, TERRAIN_ZOOM);
    let imagery_range = bbox_to_tile_range(bbox, IMAGERY_ZOOM);

    let mp = MultiProgress::new();

    // Phase 1 banners go up before the bars start drawing so they sit
    // statically above the live bars.
    let terrain_phase = TilePhase::from_range("terrain", terrain_range);
    let imagery_prefetch_phase = TilePhase::from_range("img-prefetch", imagery_range);
    terrain_phase.print_banner();
    imagery_prefetch_phase.print_banner();

    let terrain_pb = terrain_phase.progress_bar(&mp);
    let imagery_prefetch_pb = imagery_prefetch_phase.progress_bar(&mp);

    // Phase 1: terrain (fetch + stitch + resize + save) on the main thread,
    // imagery HTTP prefetch on a sibling scope-thread. By the time terrain
    // finishes, every imagery tile is on disk so phase 2 doesn't pay any
    // network latency.
    std::thread::scope(|scope| -> Result<()> {
        let imagery_handle = scope.spawn(|| -> Result<()> {
            imagery_pool.install(|| {
                prefetch_imagery_range(
                    &fetcher,
                    imagery_range,
                    imagery_prefetch_phase,
                    &imagery_prefetch_pb,
                )
            })
        });

        terrain_pool.install(|| {
            fetch_terrain(
                &fetcher,
                region,
                bbox,
                terrain_range,
                terrain_phase,
                &terrain_pb,
            )
        })?;

        imagery_handle
            .join()
            .map_err(|_| anyhow!("imagery prefetch thread panicked"))??;

        Ok(())
    })?;

    // Phase 2: imagery stitch + resize + save. Tiles are all disk-cached
    // from phase 1's prefetch; this phase just does the heavy local work.
    let imagery_phase = TilePhase::from_range("imagery", imagery_range);
    imagery_phase.print_banner();
    let imagery_pb = imagery_phase.progress_bar(&mp);
    imagery_pool.install(|| {
        fetch_imagery(
            &fetcher,
            region,
            bbox,
            imagery_range,
            imagery_phase,
            &imagery_pb,
        )
    })?;

    write_manifest(region)?;

    eprintln!("==> Done. Run `cargo run --release --bin preprocess` to rebuild the atlas.");
    Ok(())
}

/// Read `WORLD_MESH_FETCH_WORKERS` (defaulting to
/// [`std::thread::available_parallelism`]) and return the effective worker
/// count. The same number sizes both per-provider rayon pools, so the
/// total live worker count during phase 1 is `2 * workers`. Network-bound
/// work scales fine beyond CPU count, so oversubscription is intentional
/// and harmless.
fn parse_workers() -> usize {
    workers_from_env_or_available("WORLD_MESH_FETCH_WORKERS", FALLBACK_WORKERS)
}

fn parse_region_arg() -> Result<String> {
    let mut args = std::env::args().skip(1);
    let mut region: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--region" | "-r" => {
                region = args.next();
            }
            other if other.starts_with("--region=") => {
                region = Some(other.trim_start_matches("--region=").to_string());
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg {other:?}");
                print_usage();
                std::process::exit(2);
            }
        }
    }
    region.ok_or_else(|| {
        print_usage();
        anyhow!("missing --region <name>")
    })
}

fn print_usage() {
    eprintln!("usage: fetch_real_terrain --region <name>");
    eprintln!("known regions:");
    for r in regions::PRESETS {
        eprintln!(
            "  {:14}  {:.4}°N {:.4}°E  {} km",
            r.name, r.center_lat, r.center_lon, r.side_km
        );
    }
}

/// Phase 1's terrain side: fetch + stitch + resize + save the heightmap.
/// Reads from / writes to the AWS terrain stats only; the imagery prefetch
/// running on the sibling thread bumps its own counters independently.
fn fetch_terrain(
    fetcher: &TileFetcher,
    region: &Region,
    bbox: Bbox,
    range: TileRange,
    phase: TilePhase,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_aws_terrain_stats();

    let img = stream_stitch_terrain(
        fetcher,
        range,
        bbox,
        OUTPUT_SIZE,
        region.min_height_m,
        region.max_height_m,
        |_tx, _ty| {
            pb.set_message(format_stats_message(fetcher.aws_terrain_stats()));
            pb.inc(1);
        },
    )?;

    phase.print_summary(pb, fetcher.aws_terrain_stats());

    let out_path = format!("assets/terrains/planar/{}/source/height.png", region.name);
    let out = Path::new(&out_path);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    img.save(out)
        .with_context(|| format!("save {}", out.display()))?;
    eprintln!(
        "    wrote {} ({OUTPUT_SIZE}x{OUTPUT_SIZE} u16, normalised to [{},{}] m)",
        out.display(),
        region.min_height_m,
        region.max_height_m
    );
    Ok(())
}

/// Phase 1's imagery side: HTTP-only prefetch into the on-disk cache. No
/// decode, no stitch buffer — keeps memory pressure during phase 1 equal
/// to today's sequential ceiling. The sibling-thread that runs this is
/// supposed to finish before (or alongside) the terrain phase; phase 2's
/// `fetch_imagery` then sees an entirely warm cache and skips the
/// network.
fn prefetch_imagery_range(
    fetcher: &TileFetcher,
    range: TileRange,
    phase: TilePhase,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_eox_imagery_stats();

    let coords: Vec<(u32, u32)> = (range.y_min..=range.y_max)
        .flat_map(|ty| (range.x_min..=range.x_max).map(move |tx| (tx, ty)))
        .collect();

    let result = coords.par_iter().try_for_each(|(tx, ty)| -> Result<()> {
        fetcher
            .prefetch_eox_imagery(*tx, *ty, range.z)
            .with_context(|| format!("prefetch imagery {}/{}/{}", range.z, tx, ty))?;
        pb.set_message(format_stats_message(fetcher.eox_imagery_stats()));
        pb.inc(1);
        Ok(())
    });

    phase.print_summary(pb, fetcher.eox_imagery_stats());
    result
}

/// Phase 2: stitch + resize + save the imagery PNG. After phase 1's
/// prefetch every tile is on disk, so this phase's `eox_imagery_stats`
/// summary will normally read `0 net + N cached`.
fn fetch_imagery(
    fetcher: &TileFetcher,
    region: &Region,
    bbox: Bbox,
    range: TileRange,
    phase: TilePhase,
    pb: &ProgressBar,
) -> Result<()> {
    fetcher.reset_eox_imagery_stats();

    let img = stream_stitch_imagery(fetcher, range, bbox, OUTPUT_SIZE, |_tx, _ty| {
        pb.set_message(format_stats_message(fetcher.eox_imagery_stats()));
        pb.inc(1);
    })?;

    phase.print_summary(pb, fetcher.eox_imagery_stats());

    let out_path = format!("assets/terrains/planar/{}/source/albedo.png", region.name);
    let out = Path::new(&out_path);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    img.save(out)
        .with_context(|| format!("save {}", out.display()))?;
    eprintln!(
        "    wrote {} ({OUTPUT_SIZE}x{OUTPUT_SIZE} 8-bit RGB)",
        out.display()
    );
    Ok(())
}

fn write_manifest(region: &Region) -> Result<()> {
    let manifest = RegionManifest::from(region);
    let toml = toml::to_string_pretty(&manifest).context("serialise region.toml")?;
    let out_path = format!("assets/terrains/planar/{}/region.toml", region.name);
    let out = Path::new(&out_path);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, toml).with_context(|| format!("write {}", out.display()))?;
    eprintln!("    wrote {}", out.display());
    Ok(())
}
