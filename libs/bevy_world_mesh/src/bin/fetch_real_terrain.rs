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
    bbox_to_tile_range, stream_stitch_imagery, stream_stitch_terrain, Bbox, FetchStats,
    TileFetcher, TileRange,
};
use bevy_world_mesh::regions::{self, Region, RegionManifest};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{fs, path::Path, time::Duration};

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
    print_phase_banner(&mp, "terrain", terrain_range);
    print_phase_banner(&mp, "img-prefetch", imagery_range);

    let terrain_pb = mp.add(make_progress_bar(
        u64::from(terrain_range.count()),
        "terrain",
    ));
    let imagery_prefetch_pb = mp.add(make_progress_bar(
        u64::from(imagery_range.count()),
        "img-prefetch",
    ));

    // Phase 1: terrain (fetch + stitch + resize + save) on the main thread,
    // imagery HTTP prefetch on a sibling scope-thread. By the time terrain
    // finishes, every imagery tile is on disk so phase 2 doesn't pay any
    // network latency.
    std::thread::scope(|scope| -> Result<()> {
        let imagery_handle = scope.spawn(|| -> Result<()> {
            imagery_pool.install(|| {
                prefetch_imagery_range(&fetcher, imagery_range, &imagery_prefetch_pb, &mp)
            })
        });

        terrain_pool
            .install(|| fetch_terrain(&fetcher, region, bbox, terrain_range, &terrain_pb, &mp))?;

        imagery_handle
            .join()
            .map_err(|_| anyhow!("imagery prefetch thread panicked"))??;

        Ok(())
    })?;

    // Phase 2: imagery stitch + resize + save. Tiles are all disk-cached
    // from phase 1's prefetch; this phase just does the heavy local work.
    print_phase_banner(&mp, "imagery", imagery_range);
    let imagery_pb = mp.add(make_progress_bar(
        u64::from(imagery_range.count()),
        "imagery",
    ));
    imagery_pool
        .install(|| fetch_imagery(&fetcher, region, bbox, imagery_range, &imagery_pb, &mp))?;

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
    if let Some(parsed) = std::env::var("WORLD_MESH_FETCH_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return parsed;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(FALLBACK_WORKERS)
}

/// Print the standard `    label    z=Z RxC = N tile(s)` banner. We use
/// plain `eprintln!` (rather than `mp.println`) so the line shows up in
/// non-TTY runs too — `MultiProgress`'s draw target auto-hides when stderr
/// isn't a terminal, and `mp.println` goes silent with it. The (mild) cost
/// is that in TTY mode this can tear the live bar momentarily; that's
/// acceptable for our render-region pipeline, which mostly runs piped
/// through the screenshot harness anyway.
fn print_phase_banner(_mp: &MultiProgress, label: &'static str, range: TileRange) {
    eprintln!(
        "    {label:12} z={} {}x{} = {} tile(s)",
        range.z,
        range.x_max - range.x_min + 1,
        range.y_max - range.y_min + 1,
        range.count(),
    );
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

/// Build an [`indicatif::ProgressBar`] sized to `total` with our standard
/// template: prefix label, elapsed, ASCII bar, `pos/len`, free-form message
/// (used for the live `net=N cached=M` breakdown), throughput, and ETA.
/// Prefix width is 12 chars so all phase labels (`terrain`, `imagery`,
/// `img-prefetch`) align in the same column.
fn make_progress_bar(total: u64, label: &'static str) -> ProgressBar {
    // The template is well-formed at compile time; the unwrap is only
    // reachable on hand-edit typos in this string and is caught by the very
    // first run. Allowing it keeps the helper allocation-free at call site.
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

/// Format a [`Duration`] as `HH:MM:SS`. We don't pull in `humantime` for one
/// summary line; this is dependency-cheap and matches the bar's
/// `elapsed_precise` style.
fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Render a [`FetchStats`] snapshot for the progress bar's free-form message
/// slot. Keeps the line compact on healthy runs (no `retries=`) while
/// surfacing retry pressure the moment it appears.
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

/// Finalise a phase: stop the bar and emit the `done in HH:MM:SS — ...`
/// summary text. `eprintln!` (not `mp.println`) for the same reason as
/// [`print_phase_banner`] — non-TTY runs would otherwise lose this line.
fn print_phase_summary(
    _mp: &MultiProgress,
    pb: &ProgressBar,
    label: &'static str,
    range: TileRange,
    stats: FetchStats,
) {
    let elapsed = pb.elapsed();
    pb.finish_and_clear();
    let total = u64::from(range.count());
    let rate = total as f64 / elapsed.as_secs_f64().max(1e-6);
    let retries_suffix = if stats.retries == 0 {
        String::new()
    } else {
        format!(", {} retries", stats.retries)
    };
    eprintln!(
        "    {label:12} z={} done in {} — {} net + {} cached{} ({rate:.1} tile/s avg)",
        range.z,
        format_duration(elapsed),
        stats.network_hits,
        stats.cache_hits,
        retries_suffix,
    );
}

/// Phase 1's terrain side: fetch + stitch + resize + save the heightmap.
/// Reads from / writes to the AWS terrain stats only; the imagery prefetch
/// running on the sibling thread bumps its own counters independently.
fn fetch_terrain(
    fetcher: &TileFetcher,
    region: &Region,
    bbox: Bbox,
    range: TileRange,
    pb: &ProgressBar,
    mp: &MultiProgress,
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

    print_phase_summary(mp, pb, "terrain", range, fetcher.aws_terrain_stats());

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
    pb: &ProgressBar,
    mp: &MultiProgress,
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

    print_phase_summary(mp, pb, "img-prefetch", range, fetcher.eox_imagery_stats());
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
    pb: &ProgressBar,
    mp: &MultiProgress,
) -> Result<()> {
    fetcher.reset_eox_imagery_stats();

    let img = stream_stitch_imagery(fetcher, range, bbox, OUTPUT_SIZE, |_tx, _ty| {
        pb.set_message(format_stats_message(fetcher.eox_imagery_stats()));
        pb.inc(1);
    })?;

    print_phase_summary(mp, pb, "imagery", range, fetcher.eox_imagery_stats());

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
