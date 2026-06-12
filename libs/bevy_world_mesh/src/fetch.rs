//! Slippy-map tile fetching, decoding, stitching for both AWS Terrain Tiles
//! (terrarium PNG, elevation in metres) and EOX Sentinel-2 Cloudless (RGB
//! WMTS imagery). Bbox-based for the planar fetcher; the same primitives
//! are reusable from a future spherical / LEO cube-face generator that
//! samples this layer per pixel and lets the on-disk tile cache deduplicate
//! the network round-trips.
//!
//! Both providers expose tiles in the standard Web Mercator (EPSG:3857)
//! XYZ scheme so a single `lonlat_to_tile` helper covers both. The tile
//! cache lives at `target/tile_cache/<host>/<z>/<x>/<y>.<ext>` and is
//! agnostic to which binary populated it.

use anyhow::{anyhow, Context, Result};
use image::{
    imageops::{self, FilterType},
    ImageBuffer, Luma, RgbImage,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, SystemTime},
};

/// Side length of every XYZ tile (slippy / WMTS GoogleMapsCompatible).
pub const TILE_PX: u32 = 256;

/// EPSG:4326 axis-aligned bounding box in degrees.
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Bbox {
    /// Build a square-ish bbox centred at `(lon, lat)` covering `side_km` on
    /// each side. Latitude is converted directly; longitude is corrected by
    /// `cos(lat)` so the resulting bbox is approximately a square on the
    /// ground rather than a wide rectangle near the poles.
    pub fn around(lon: f64, lat: f64, side_km: f64) -> Self {
        const KM_PER_DEG_LAT: f64 = 111.32;
        let half_lat = side_km / 2.0 / KM_PER_DEG_LAT;
        let half_lon = half_lat / lat.to_radians().cos();
        Self {
            west: lon - half_lon,
            east: lon + half_lon,
            south: lat - half_lat,
            north: lat + half_lat,
        }
    }
}

/// Inclusive XYZ tile range covering a bbox at zoom `z`.
#[derive(Debug, Clone, Copy)]
pub struct TileRange {
    pub z: u8,
    pub x_min: u32,
    pub x_max: u32,
    pub y_min: u32,
    pub y_max: u32,
}

impl TileRange {
    pub fn count(&self) -> u32 {
        (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)
    }
}

/// Convert a lon/lat point to its containing slippy tile at zoom `z`.
///
/// Web Mercator clips at ±~85.0511 °. We clamp before the asinh to keep the
/// math finite at the poles.
pub fn lonlat_to_tile(lon: f64, lat: f64, z: u8) -> (u32, u32) {
    const MAX_LAT: f64 = 85.0511287798;
    let lat = lat.clamp(-MAX_LAT, MAX_LAT);
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let y = ((1.0 - (lat.to_radians().tan().asinh() / std::f64::consts::PI)) / 2.0 * n).floor();
    (
        (x as i64).clamp(0, n as i64 - 1) as u32,
        (y as i64).clamp(0, n as i64 - 1) as u32,
    )
}

/// Inverse of [`lonlat_to_tile`] — returns the lon/lat of the *NW* corner of
/// the given tile. (NW = top-left in slippy tile order.)
pub fn tile_to_lonlat(x: u32, y: u32, z: u8) -> (f64, f64) {
    let n = (1u64 << z) as f64;
    let lon = x as f64 / n * 360.0 - 180.0;
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * y as f64 / n))
        .sinh()
        .atan();
    (lon, lat_rad.to_degrees())
}

pub fn bbox_to_tile_range(bbox: Bbox, z: u8) -> TileRange {
    let (x_min, y_max) = lonlat_to_tile(bbox.west, bbox.south, z);
    let (x_max, y_min) = lonlat_to_tile(bbox.east, bbox.north, z);
    TileRange {
        z,
        x_min: x_min.min(x_max),
        x_max: x_min.max(x_max),
        y_min: y_min.min(y_max),
        y_max: y_min.max(y_max),
    }
}

/// HTTP fetcher with on-disk cache plus a small in-memory decoded-tile cache
/// for the per-pixel sphere sampler. The decoded cache holds the most recently
/// fetched terrain and imagery tiles keyed by `(z, x, y)`; capacity is small
/// (32 of each) since the sphere fetcher accesses pixels in row-major order
/// per face and so almost always re-uses the previously decoded tile.
///
/// `aws_terrain_stats` / `eox_imagery_stats` are bumped inside
/// [`Self::fetch_cached`] (the right one per call) so consumers can read
/// per-provider [`FetchStats`] via [`Self::aws_terrain_stats`] /
/// [`Self::eox_imagery_stats`]. The split matters when both providers are
/// being fetched concurrently — a single global counter would mix them and
/// the progress UI couldn't attribute hits to either phase. The inner
/// [`AtomicU64`]s keep the fetcher `Sync` across rayon workers.
pub struct TileFetcher {
    agent: ureq::Agent,
    cache_dir: PathBuf,
    terrain_cache: Mutex<HashMap<TileKey, Vec<f32>>>,
    imagery_cache: Mutex<HashMap<TileKey, RgbImage>>,
    aws_terrain_stats: AtomicTileStats,
    eox_imagery_stats: AtomicTileStats,
}

/// Atomic counterpart to [`FetchStats`] embedded in [`TileFetcher`] per
/// provider. `Sync` so a `&TileFetcher` shared across rayon workers can
/// bump the right counters concurrently.
#[derive(Debug, Default)]
struct AtomicTileStats {
    cache_hits: AtomicU64,
    network_hits: AtomicU64,
    retries: AtomicU64,
}

impl AtomicTileStats {
    fn snapshot(&self) -> FetchStats {
        FetchStats {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            network_hits: self.network_hits.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
        self.network_hits.store(0, Ordering::Relaxed);
        self.retries.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of how many tile reads the fetcher has served from the on-disk
/// cache vs the network since the last reset, plus a running count of HTTP
/// retries triggered by transient errors (5xx, 408, 429, transport
/// faults). The progress UI in `fetch_real_terrain` reads this on every
/// iteration to show a live `net=N cached=M [retries=R]` breakdown
/// per provider.
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchStats {
    pub cache_hits: u64,
    pub network_hits: u64,
    pub retries: u64,
}

impl FetchStats {
    pub fn total(&self) -> u64 {
        self.cache_hits + self.network_hits
    }

    /// Element-wise sum, used by [`TileFetcher::stats`] to combine per-
    /// provider counters into a backwards-compat aggregate.
    fn combine(self, other: FetchStats) -> FetchStats {
        FetchStats {
            cache_hits: self.cache_hits + other.cache_hits,
            network_hits: self.network_hits + other.network_hits,
            retries: self.retries + other.retries,
        }
    }
}

/// Cap for the per-tile decoded caches. Sized for the spherical globe fetcher
/// at face_size=8192 / z=8: each face row crosses 64 unique z=8 tiles, so
/// 256 entries hold ~4 rows of tile bands and keep adjacent rows mostly
/// cache-hot without thrashing. RAM cost is ~192 MB per attachment
/// (256 × 256×256×3 bytes for imagery and 256 × 256×256×4 bytes f32 for
/// terrarium-decoded terrain). Scale down to 64 if you revert to
/// face_size=2048 and care about RAM.
const DECODED_CACHE_CAP: usize = 256;

type TileKey = (u8, u32, u32);

/// AWS Terrain Tiles (terrarium PNG) URL for `(x, y, z)`. Hosted on S3 so
/// any number of concurrent requests is fine.
fn aws_terrain_url(x: u32, y: u32, z: u8) -> String {
    format!("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")
}

/// EOX Sentinel-2 Cloudless WMTS imagery URL for `(x, y, z)`.
///
/// Note: the year slug rotates each spring (`s2cloudless-2024_3857` →
/// `..._2025_3857` etc.). Bump as the latest mosaic appears.
fn eox_imagery_url(x: u32, y: u32, z: u8) -> String {
    const LAYER: &str = "s2cloudless-2024_3857";
    // WMTS REST URL template: `{TileMatrix}/{TileRow}/{TileCol}` = `{z}/{y}/{x}`.
    format!("https://tiles.maps.eox.at/wmts/1.0.0/{LAYER}/default/g/{z}/{y}/{x}.jpg")
}

impl TileFetcher {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .user_agent("world_mesh/0.1 (+https://github.com/world_mesh)")
            .timeout(std::time::Duration::from_secs(30))
            .build();
        Self {
            agent,
            cache_dir: cache_dir.into(),
            terrain_cache: Mutex::new(HashMap::new()),
            imagery_cache: Mutex::new(HashMap::new()),
            aws_terrain_stats: AtomicTileStats::default(),
            eox_imagery_stats: AtomicTileStats::default(),
        }
    }

    /// Snapshot the AWS terrain provider's counters since the last reset.
    /// Reads are `Relaxed` — fine for a UI counter, where a few-microsecond
    /// skew between threads is invisible to the user.
    pub fn aws_terrain_stats(&self) -> FetchStats {
        self.aws_terrain_stats.snapshot()
    }

    /// Snapshot the EOX imagery provider's counters since the last reset.
    pub fn eox_imagery_stats(&self) -> FetchStats {
        self.eox_imagery_stats.snapshot()
    }

    /// Zero out the AWS terrain counters. Call this at the start of the
    /// terrain fetch phase so the progress bar's totals reflect that phase
    /// only.
    pub fn reset_aws_terrain_stats(&self) {
        self.aws_terrain_stats.reset();
    }

    /// Zero out the EOX imagery counters. Call this at the start of the
    /// imagery prefetch (or stitch) phase.
    pub fn reset_eox_imagery_stats(&self) {
        self.eox_imagery_stats.reset();
    }

    /// Combined snapshot across both providers. Kept for backwards compat;
    /// new code that's aware of per-phase orchestration should prefer
    /// [`Self::aws_terrain_stats`] / [`Self::eox_imagery_stats`].
    pub fn stats(&self) -> FetchStats {
        self.aws_terrain_stats().combine(self.eox_imagery_stats())
    }

    /// Reset both providers' counters at once. Backwards-compat shim; the
    /// new orchestration resets each provider independently at the start
    /// of its phase.
    pub fn reset_stats(&self) {
        self.aws_terrain_stats.reset();
        self.eox_imagery_stats.reset();
    }

    /// Fetch a URL's bytes, hitting the on-disk cache first. The cache key is
    /// `<cache_dir>/<host>/<rest of URL path>`. Tiles are immutable so we
    /// never invalidate.
    ///
    /// `stats` is the per-provider counter set the call should bump:
    /// `&self.aws_terrain_stats` for terrain URLs, `&self.eox_imagery_stats`
    /// for imagery. The cache-hit branch bumps `cache_hits`; the network
    /// branch (after retries succeed) bumps `network_hits`. Transient HTTP
    /// failures (5xx, 408, 429, transport-level errors) bounce through
    /// [`Self::fetch_url_with_retries`] before surfacing as `Err`;
    /// permanent errors (404, 403, ...) fail fast.
    fn fetch_cached(&self, url: &str, stats: &AtomicTileStats) -> Result<Vec<u8>> {
        let cache_path = url_to_cache_path(&self.cache_dir, url)?;
        if cache_path.exists() {
            let bytes = fs::read(&cache_path)
                .with_context(|| format!("read cached tile {}", cache_path.display()))?;
            stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(bytes);
        }

        let bytes = self.fetch_url_with_retries(url, stats)?;

        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &bytes)
            .with_context(|| format!("write tile cache {}", cache_path.display()))?;
        stats.network_hits.fetch_add(1, Ordering::Relaxed);
        Ok(bytes)
    }

    /// Ensure a URL's bytes exist in the on-disk cache without decoding or
    /// reading an already-cached body back into memory. This is the fast path
    /// used by prefetchers: a warm cache should cost one metadata check per
    /// tile, not a full tile read that will be repeated by the stitcher or
    /// sphere sampler later.
    fn prefetch_cached(&self, url: &str, stats: &AtomicTileStats) -> Result<()> {
        let cache_path = url_to_cache_path(&self.cache_dir, url)?;
        if cache_path
            .metadata()
            .map(|metadata| metadata.is_file() && metadata.len() > 0)
            .unwrap_or(false)
        {
            stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let bytes = self.fetch_url_with_retries(url, stats)?;

        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &bytes)
            .with_context(|| format!("write tile cache {}", cache_path.display()))?;
        stats.network_hits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Fetch `url` over HTTP, retrying transient errors with exponential
    /// backoff (1s, 2s, 4s, ...; capped at 60s with ±500 ms jitter). Up to
    /// 8 attempts total; gives up after ~2 minutes of cumulative sleep.
    ///
    /// AWS S3 and EOX both return sporadic 5xx during sustained heavy fetch
    /// loads (the user observed an isolated `500` mid-Mojave); a single
    /// retry typically succeeds and the run continues. Each retry sleep
    /// bumps `stats.retries` so the progress UI can surface the count for
    /// the right provider.
    ///
    /// Permanent errors (4xx other than 408/425/429) short-circuit
    /// immediately: a 404 on a tile we asked for means we asked for the
    /// wrong tile, no amount of retrying fixes that.
    fn fetch_url_with_retries(&self, url: &str, stats: &AtomicTileStats) -> Result<Vec<u8>> {
        const MAX_ATTEMPTS: u32 = 8;
        let mut last_msg = String::new();

        for attempt in 1..=MAX_ATTEMPTS {
            match self.try_fetch_url_once(url) {
                FetchAttempt::Ok(bytes) => return Ok(bytes),
                FetchAttempt::Permanent(err) => {
                    return Err(err.context(format!("HTTP GET {url}")));
                }
                FetchAttempt::Transient(msg) => {
                    last_msg = msg;
                    if attempt < MAX_ATTEMPTS {
                        stats.retries.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(retry_backoff(attempt));
                    }
                }
            }
        }

        Err(anyhow!(
            "HTTP GET {url}: gave up after {MAX_ATTEMPTS} attempts (last error: {last_msg})"
        ))
    }

    /// One HTTP attempt, classified for the retry loop. We unwrap ureq's
    /// `Error::Status` here so we can decide retry-or-not based on the
    /// status code, and we treat body-read I/O errors as transient
    /// (timeouts, connection resets) since that's what they almost always
    /// are in a tile-fetch workload.
    fn try_fetch_url_once(&self, url: &str) -> FetchAttempt {
        match self.agent.get(url).call() {
            Ok(resp) => {
                let mut bytes = Vec::with_capacity(64 * 1024);
                match resp.into_reader().read_to_end(&mut bytes) {
                    Ok(_) => FetchAttempt::Ok(bytes),
                    Err(io_err) => FetchAttempt::Transient(format!("body read: {io_err}")),
                }
            }
            Err(ureq::Error::Status(code, resp)) => {
                let label = format!("HTTP {code} {}", resp.status_text());
                if is_transient_status(code) {
                    FetchAttempt::Transient(label)
                } else {
                    FetchAttempt::Permanent(anyhow!(label))
                }
            }
            Err(ureq::Error::Transport(t)) => FetchAttempt::Transient(format!("transport: {t}")),
        }
    }

    /// Fetch one AWS Terrain Tile (terrarium-encoded PNG) and decode the RGB
    /// channels back into metres. Returns a `TILE_PX × TILE_PX` row-major
    /// `Vec<f32>`.
    pub fn fetch_aws_terrain(&self, x: u32, y: u32, z: u8) -> Result<Vec<f32>> {
        let url = aws_terrain_url(x, y, z);
        let bytes = self.fetch_cached(&url, &self.aws_terrain_stats)?;
        let img = image::load_from_memory(&bytes)
            .with_context(|| format!("decode terrarium PNG {url}"))?;
        let rgba = img.to_rgba8();
        if rgba.width() != TILE_PX || rgba.height() != TILE_PX {
            return Err(anyhow!(
                "AWS terrain tile {url} unexpected size {}x{}",
                rgba.width(),
                rgba.height()
            ));
        }
        let mut out = Vec::with_capacity((TILE_PX * TILE_PX) as usize);
        for px in rgba.pixels() {
            let r = px.0[0] as f32;
            let g = px.0[1] as f32;
            let b = px.0[2] as f32;
            // AWS terrarium decoding (per docs.safe.com / nimbo.earth /
            // tilezen formats spec): height (m) = (R * 256 + G + B / 256) - 32768
            let height_m = (r * 256.0 + g + b / 256.0) - 32768.0;
            out.push(height_m);
        }
        Ok(out)
    }

    /// Fetch one EOX Sentinel-2 Cloudless WMTS tile as an RGB image.
    pub fn fetch_eox_imagery(&self, x: u32, y: u32, z: u8) -> Result<RgbImage> {
        let url = eox_imagery_url(x, y, z);
        let bytes = self.fetch_cached(&url, &self.eox_imagery_stats)?;
        let img = image::load_from_memory(&bytes)
            .with_context(|| format!("decode EOX imagery JPG {url}"))?;
        let rgb = img.to_rgb8();
        if rgb.width() != TILE_PX || rgb.height() != TILE_PX {
            return Err(anyhow!(
                "EOX imagery tile {url} unexpected size {}x{}",
                rgb.width(),
                rgb.height()
            ));
        }
        Ok(rgb)
    }

    /// Download an AWS Terrain Tile to the on-disk cache without decoding.
    /// Symmetric with [`Self::prefetch_eox_imagery`]; the imagery prefetch
    /// is the one wired into the parallel orchestration in
    /// `fetch_real_terrain`, but `prefetch_aws_terrain` is exposed so a
    /// future spherical-or-mixed fetcher gets the same cache-warming
    /// primitive for free.
    ///
    /// Bumps `aws_terrain_stats` so the prefetch progress UI can read its
    /// own counters via [`Self::aws_terrain_stats`].
    pub fn prefetch_aws_terrain(&self, x: u32, y: u32, z: u8) -> Result<()> {
        self.prefetch_cached(&aws_terrain_url(x, y, z), &self.aws_terrain_stats)
    }

    /// Download an EOX imagery tile to the on-disk cache without decoding.
    /// Used by the parallel orchestration in `fetch_real_terrain`: the
    /// imagery HTTP fetch runs alongside the terrain phase, so by the time
    /// the imagery stitcher starts, every tile is a disk-cache hit.
    ///
    /// Bumps `eox_imagery_stats` so the prefetch progress UI can read its
    /// own counters via [`Self::eox_imagery_stats`].
    pub fn prefetch_eox_imagery(&self, x: u32, y: u32, z: u8) -> Result<()> {
        self.prefetch_cached(&eox_imagery_url(x, y, z), &self.eox_imagery_stats)
    }

    /// Bilinear-sample the AWS Terrain elevation at `(lon, lat)` from the tile
    /// at zoom `z`. The decoded terrarium buffer is cached in memory so the
    /// sphere fetcher's per-pixel sampling re-uses it across the typical 8-32
    /// adjacent face pixels per source pixel.
    pub fn sample_terrain_at_lonlat(&self, lon: f64, lat: f64, z: u8) -> Result<f32> {
        let (tx, ty) = lonlat_to_tile(lon, lat, z);
        let (u, v) = tile_subpixel(lon, lat, tx, ty, z);
        self.with_terrain_tile(tx, ty, z, |buf| sample_f32_bilinear(buf, TILE_PX, u, v))
    }

    /// Bilinear-sample the EOX imagery at `(lon, lat)` and return an
    /// 8-bit RGB triplet. Uses the same in-memory decoded-tile cache.
    pub fn sample_imagery_at_lonlat(&self, lon: f64, lat: f64, z: u8) -> Result<[u8; 3]> {
        let (tx, ty) = lonlat_to_tile(lon, lat, z);
        let (u, v) = tile_subpixel(lon, lat, tx, ty, z);
        self.with_imagery_tile(tx, ty, z, |img| sample_rgb_bilinear(img, u, v))
    }

    fn with_terrain_tile<R>(
        &self,
        tx: u32,
        ty: u32,
        z: u8,
        f: impl FnOnce(&[f32]) -> R,
    ) -> Result<R> {
        let key: TileKey = (z, tx, ty);
        {
            let cache = self.terrain_cache.lock().unwrap();
            if let Some(buf) = cache.get(&key) {
                return Ok(f(buf));
            }
        }
        let buf = self.fetch_aws_terrain(tx, ty, z)?;
        let mut cache = self.terrain_cache.lock().unwrap();
        evict_one_if_full(&mut cache);
        cache.insert(key, buf);
        Ok(f(cache.get(&key).unwrap()))
    }

    fn with_imagery_tile<R>(
        &self,
        tx: u32,
        ty: u32,
        z: u8,
        f: impl FnOnce(&RgbImage) -> R,
    ) -> Result<R> {
        let key: TileKey = (z, tx, ty);
        {
            let cache = self.imagery_cache.lock().unwrap();
            if let Some(img) = cache.get(&key) {
                return Ok(f(img));
            }
        }
        let img = self.fetch_eox_imagery(tx, ty, z)?;
        let mut cache = self.imagery_cache.lock().unwrap();
        evict_one_if_full(&mut cache);
        cache.insert(key, img);
        Ok(f(cache.get(&key).unwrap()))
    }
}

/// Outcome of a single HTTP attempt inside [`TileFetcher::fetch_url_with_retries`].
/// Kept private so the public surface stays a plain `Result<Vec<u8>>`.
enum FetchAttempt {
    /// Body read successfully; bytes are ready to cache.
    Ok(Vec<u8>),
    /// Server returned a transient signal (5xx, 408, 429, transport
    /// fault). The retry loop should sleep and try again. Carries a
    /// human-readable label for the final "gave up" error message.
    Transient(String),
    /// Permanent failure (4xx other than 408/425/429, or a malformed
    /// response). Don't retry; surface immediately.
    Permanent(anyhow::Error),
}

/// HTTP status codes worth retrying. `408 Request Timeout`, `425 Too Early`,
/// `429 Too Many Requests`, and the standard 5xx server errors all indicate
/// "this isn't your fault, try again" semantics. Everything else in the 4xx
/// range is a permanent client error.
fn is_transient_status(code: u16) -> bool {
    matches!(code, 408 | 425 | 429) || (500..=599).contains(&code)
}

/// Exponential backoff with a 60s ceiling and ±500 ms jitter. `attempt`
/// starts at 1 (i.e. before the first retry sleep): 1s, 2s, 4s, 8s, 16s,
/// 32s, 60s, 60s. ~2 minutes of cumulative sleep before the 8th attempt.
fn retry_backoff(attempt: u32) -> Duration {
    const CAP_MS: u64 = 60_000;
    let exp = attempt.saturating_sub(1).min(10);
    let base_ms = 1000_u64.saturating_mul(1_u64 << exp).min(CAP_MS);
    Duration::from_millis(base_ms.saturating_add(jitter_ms()))
}

/// Cheap, dependency-free 0..500 ms jitter sourced from the wall clock's
/// nanosecond fragment. Doesn't need to be cryptographic — it just keeps a
/// fleet of rayon workers from synchronising their retries onto the same
/// recovery instant.
fn jitter_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()) % 500)
        .unwrap_or(0)
}

fn evict_one_if_full<V>(cache: &mut HashMap<TileKey, V>) {
    if cache.len() >= DECODED_CACHE_CAP {
        if let Some(k) = cache.keys().next().copied() {
            cache.remove(&k);
        }
    }
}

/// Sub-pixel (u, v) ∈ [0, 1] of `(lon, lat)` inside the slippy tile
/// `(tx, ty, z)`. Used by the per-pixel sphere sampler.
fn tile_subpixel(lon: f64, lat: f64, tx: u32, ty: u32, z: u8) -> (f64, f64) {
    let (nw_lon, nw_lat) = tile_to_lonlat(tx, ty, z);
    let (se_lon, se_lat) = tile_to_lonlat(tx + 1, ty + 1, z);
    let u = ((lon - nw_lon) / (se_lon - nw_lon)).clamp(0.0, 1.0);
    let v = ((nw_lat - lat) / (nw_lat - se_lat)).clamp(0.0, 1.0);
    (u, v)
}

fn sample_f32_bilinear(buf: &[f32], size: u32, u: f64, v: f64) -> f32 {
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

fn sample_rgb_bilinear(img: &RgbImage, u: f64, v: f64) -> [u8; 3] {
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

/// Cube-face (face index, u, v in [0, 1]²) → unit-sphere direction.
///
/// This MUST match `bevy_terrain`'s `Coordinate::world_position` mapping
/// (at `crates/bevy_terrain/src/math/coordinate.rs`): pixel-space `(u, v)` is
/// linearly remapped to `w ∈ [-1, +1]²` and then pushed through the inverse
/// GSRC projection `uv_3d = w / sqrt(1 + C² − C²·w²)` (with `C = 0.87`) that
/// the renderer uses to un-warp its unit-cube-sphere. The face index
/// convention is also bevy_terrain's (not the naive axis-labelling one):
///
/// - side 0: −X face (u → +Z edge, v → −Y edge)
/// - side 1: +Z face (u → +X,      v → −Y)
/// - side 2: +Y face (u → +X,      v → +Z)
/// - side 3: +X face (u → −Y,      v → +Z)
/// - side 4: −Z face (u → −Y,      v → −X)
/// - side 5: −Y face (u → +X,      v → −Z)
///
/// Using a naive linear-per-face mapping (the one the synthetic
/// `synthesize_spherical_faces` happened to ship with) yields *visually*
/// plausible results on abstract noise because the synthetic field has no
/// landmarks, but it scrambles real Earth: every face's content lands on
/// the wrong part of the sphere and you render an alpine close-up instead
/// of a marble.
pub fn cube_face_to_dir(face: u8, u: f64, v: f64) -> [f64; 3] {
    const C_SQR: f64 = 0.87 * 0.87;
    let wx = 2.0 * u - 1.0;
    let wy = 2.0 * v - 1.0;
    let ux = wx / (1.0 + C_SQR - C_SQR * wx * wx).sqrt();
    let uy = wy / (1.0 + C_SQR - C_SQR * wy * wy).sqrt();
    let d = match face {
        0 => [-1.0, -uy, ux],
        1 => [ux, -uy, 1.0],
        2 => [ux, 1.0, uy],
        3 => [1.0, -ux, uy],
        4 => [uy, -ux, -1.0],
        5 => [uy, -1.0, ux],
        _ => panic!("invalid cube face index {face}"),
    };
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

/// Unit direction → geographic (lon, lat) in degrees.
///
/// Convention: Y is up (north pole at +Y, south pole at -Y). The Greenwich
/// meridian sits along +Z so that the +Z cube face is centred on lon=0,
/// lat=0; +X is therefore at lon=+90 (Indian Ocean / SE Asia direction).
pub fn dir_to_lonlat(dir: [f64; 3]) -> (f64, f64) {
    let lat = dir[1].clamp(-1.0, 1.0).asin().to_degrees();
    let lon = dir[0].atan2(dir[2]).to_degrees();
    (lon, lat)
}

fn url_to_cache_path(base: &Path, url: &str) -> Result<PathBuf> {
    let trimmed = url
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let mut p = PathBuf::from(base);
    for seg in trimmed.split('/') {
        if seg.is_empty() {
            continue;
        }
        // sanitise — should never trigger for our tile URLs but cheap to guard.
        let safe: String = seg
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        p.push(safe);
    }
    Ok(p)
}

/// Stream-fetch and stitch terrarium-decoded terrain tiles into one 16-bit
/// grayscale heightmap, crop to the bbox, and resize to `output_size`.
///
/// Memory-aware version of the old `stitch_terrain(tiles: &[Vec<f32>], ...)`:
/// instead of accumulating every decoded tile into a `Vec<Vec<f32>>` (which
/// is `~66 GB` at z=15 over a 500 km bbox) and then stitching to a `Vec<f32>`
/// (another `~66 GB`), we pre-allocate a single `Vec<u16>` of size
/// `stitched_w × stitched_h × 2 B` (`~33 GB` at z=15 mojave) and let rayon
/// parallel-write tile-rows into disjoint slices via `par_chunks_mut`. The
/// `f32 → u16` normalisation against `[min_height_m, max_height_m]` happens
/// inline with each tile write, eliminating the f32 staging buffer entirely.
///
/// All buffer-size and indexing arithmetic runs in `usize` so dimensions
/// `> 2³²` (z=15 over a 500 km bbox is `128,512²`) work correctly without
/// wrap-around — the panic the old `(u32 * u32) as usize` path produced.
///
/// `on_tile_done(tx, ty)` is invoked from a rayon worker after each tile
/// finishes; the binary uses it to advance an `indicatif` progress bar.
pub fn stream_stitch_terrain<F>(
    fetcher: &TileFetcher,
    range: TileRange,
    bbox: Bbox,
    output_size: u32,
    min_height_m: f32,
    max_height_m: f32,
    on_tile_done: F,
) -> Result<ImageBuffer<Luma<u16>, Vec<u16>>>
where
    F: Fn(u32, u32) + Sync,
{
    let cols = (range.x_max - range.x_min) as usize + 1;
    let rows = (range.y_max - range.y_min) as usize + 1;
    let tile_px = TILE_PX as usize;
    let stitched_w = cols
        .checked_mul(tile_px)
        .context("stitched width overflow")?;
    let stitched_h = rows
        .checked_mul(tile_px)
        .context("stitched height overflow")?;
    let stitched_len = stitched_w
        .checked_mul(stitched_h)
        .context("stitched buffer length overflow")?;
    let band_len = stitched_w
        .checked_mul(tile_px)
        .context("band length overflow")?;

    let span = (max_height_m - min_height_m).max(1e-6);
    let mut stitched: Vec<u16> = vec![0u16; stitched_len];

    // Each rayon task owns one TILE_PX-row band. Inside the band the inner
    // tile loop is sequential, so a worker fetches `cols` tiles back-to-back
    // along its row before moving to the next assignment. With the default
    // 8-worker pool and ~500 row-bands per fetch, work-stealing keeps all
    // workers busy while preserving the disjoint-write invariant.
    stitched
        .par_chunks_mut(band_len)
        .enumerate()
        .try_for_each(|(iy, band)| -> Result<()> {
            let ty = range.y_min + iy as u32;
            for ix in 0..cols {
                let tx = range.x_min + ix as u32;
                let tile = fetcher
                    .fetch_aws_terrain(tx, ty, range.z)
                    .with_context(|| format!("AWS terrain tile {}/{}/{}", range.z, tx, ty))?;
                if tile.len() != tile_px * tile_px {
                    return Err(anyhow!(
                        "AWS terrain tile {}/{}/{} returned {} pixels, expected {}",
                        range.z,
                        tx,
                        ty,
                        tile.len(),
                        tile_px * tile_px
                    ));
                }

                let block_origin_x = ix * tile_px;
                for py in 0..tile_px {
                    let band_off = py * stitched_w + block_origin_x;
                    let tile_off = py * tile_px;
                    let band_row = &mut band[band_off..band_off + tile_px];
                    let tile_row = &tile[tile_off..tile_off + tile_px];
                    for (dst, src) in band_row.iter_mut().zip(tile_row.iter()) {
                        let n = ((src - min_height_m) / span).clamp(0.0, 1.0);
                        *dst = (n * u16::MAX as f32) as u16;
                    }
                }

                on_tile_done(tx, ty);
            }
            Ok(())
        })?;

    let stitched_w_u32 = u32::try_from(stitched_w).context("stitched_w doesn't fit u32")?;
    let stitched_h_u32 = u32::try_from(stitched_h).context("stitched_h doesn't fit u32")?;
    let (crop_x, crop_y, crop_w, crop_h) =
        bbox_crop_in_stitched(range, bbox, stitched_w, stitched_h);

    // Resize the full stitched buffer to a slightly oversize image such
    // that the bbox region inside it lands at exactly `output_size ×
    // output_size`, then crop the resized result. This avoids any
    // pre-resize crop allocation (which would be ~33 GB at z=15 mojave —
    // half a stitched buffer's worth) at the cost of resizing ~2 % more
    // pixels than strictly necessary. The post-resize crop materialises a
    // small buffer (`output_size² × bytes_per_pixel`).
    //
    // The intermediate `resized_full` is dropped before this function
    // returns, so peak memory during the resize stays bounded by stitched
    // + the small resized image.
    let (resize_w, resize_h) =
        oversize_for_crop(stitched_w_u32, stitched_h_u32, crop_w, crop_h, output_size);
    let resized_full = {
        let stitched_img: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(stitched_w_u32, stitched_h_u32, stitched)
                .ok_or_else(|| anyhow!("stitched buffer length doesn't match dimensions"))?;
        imageops::resize(&stitched_img, resize_w, resize_h, FilterType::Lanczos3)
    };
    let crop_x_resized = scale_crop_offset(crop_x, stitched_w_u32, resize_w);
    let crop_y_resized = scale_crop_offset(crop_y, stitched_h_u32, resize_h);
    let final_img = imageops::crop_imm(
        &resized_full,
        crop_x_resized,
        crop_y_resized,
        output_size,
        output_size,
    )
    .to_image();
    Ok(final_img)
}

/// Stream-fetch and stitch EOX imagery tiles into one RGB image, crop to
/// the bbox, and resize to `output_size`. Same shape and rationale as
/// [`stream_stitch_terrain`] — see that function's doc for the memory
/// trade-off — but the per-tile inner loop is a `copy_from_slice` of the
/// tile's RGB row into the destination band (no per-pixel work).
///
/// Stitched buffer size: `stitched_w × stitched_h × 3 B` (~50 GB at z=15
/// mojave). Still fits in 64 GB; we don't need a u8 → u16 narrowing trick
/// for imagery.
pub fn stream_stitch_imagery<F>(
    fetcher: &TileFetcher,
    range: TileRange,
    bbox: Bbox,
    output_size: u32,
    on_tile_done: F,
) -> Result<RgbImage>
where
    F: Fn(u32, u32) + Sync,
{
    let cols = (range.x_max - range.x_min) as usize + 1;
    let rows = (range.y_max - range.y_min) as usize + 1;
    let tile_px = TILE_PX as usize;
    let stitched_w = cols
        .checked_mul(tile_px)
        .context("stitched width overflow")?;
    let stitched_h = rows
        .checked_mul(tile_px)
        .context("stitched height overflow")?;
    let stitched_pixels = stitched_w
        .checked_mul(stitched_h)
        .context("stitched pixel count overflow")?;
    let stitched_bytes = stitched_pixels
        .checked_mul(3)
        .context("stitched RGB byte length overflow")?;
    let row_bytes = stitched_w
        .checked_mul(3)
        .context("stitched row bytes overflow")?;
    let band_bytes = row_bytes
        .checked_mul(tile_px)
        .context("band byte length overflow")?;
    let tile_row_bytes = tile_px * 3;

    let mut stitched: Vec<u8> = vec![0u8; stitched_bytes];

    stitched
        .par_chunks_mut(band_bytes)
        .enumerate()
        .try_for_each(|(iy, band)| -> Result<()> {
            let ty = range.y_min + iy as u32;
            for ix in 0..cols {
                let tx = range.x_min + ix as u32;
                let tile = fetcher
                    .fetch_eox_imagery(tx, ty, range.z)
                    .with_context(|| format!("EOX imagery tile {}/{}/{}", range.z, tx, ty))?;
                if tile.width() != TILE_PX || tile.height() != TILE_PX {
                    return Err(anyhow!(
                        "EOX imagery tile {}/{}/{} unexpected size {}x{}",
                        range.z,
                        tx,
                        ty,
                        tile.width(),
                        tile.height()
                    ));
                }
                let tile_raw = tile.as_raw();
                if tile_raw.len() != tile_row_bytes * tile_px {
                    return Err(anyhow!(
                        "EOX imagery tile {}/{}/{} backing buffer is {} bytes, expected {}",
                        range.z,
                        tx,
                        ty,
                        tile_raw.len(),
                        tile_row_bytes * tile_px
                    ));
                }

                let block_origin = ix * tile_row_bytes;
                for py in 0..tile_px {
                    let band_off = py * row_bytes + block_origin;
                    let tile_off = py * tile_row_bytes;
                    band[band_off..band_off + tile_row_bytes]
                        .copy_from_slice(&tile_raw[tile_off..tile_off + tile_row_bytes]);
                }

                on_tile_done(tx, ty);
            }
            Ok(())
        })?;

    let stitched_w_u32 = u32::try_from(stitched_w).context("stitched_w doesn't fit u32")?;
    let stitched_h_u32 = u32::try_from(stitched_h).context("stitched_h doesn't fit u32")?;
    let (crop_x, crop_y, crop_w, crop_h) =
        bbox_crop_in_stitched(range, bbox, stitched_w, stitched_h);

    // Same resize-then-crop trick as `stream_stitch_terrain` — see that
    // function for the rationale.
    let (resize_w, resize_h) =
        oversize_for_crop(stitched_w_u32, stitched_h_u32, crop_w, crop_h, output_size);
    let resized_full = {
        let stitched_img: RgbImage =
            ImageBuffer::from_raw(stitched_w_u32, stitched_h_u32, stitched)
                .ok_or_else(|| anyhow!("stitched RGB buffer length doesn't match dimensions"))?;
        imageops::resize(&stitched_img, resize_w, resize_h, FilterType::Lanczos3)
    };
    let crop_x_resized = scale_crop_offset(crop_x, stitched_w_u32, resize_w);
    let crop_y_resized = scale_crop_offset(crop_y, stitched_h_u32, resize_h);
    let final_img = imageops::crop_imm(
        &resized_full,
        crop_x_resized,
        crop_y_resized,
        output_size,
        output_size,
    )
    .to_image();
    Ok(final_img)
}

/// Map the geographic bbox onto the stitched image's pixel rectangle. The
/// stitched image's NW corner is the NW corner of tile `(x_min, y_min)`;
/// its SE corner is the SE corner of `(x_max, y_max)`. Returns
/// `(crop_x, crop_y, crop_w, crop_h)` in stitched-pixel space, clamped to
/// the stitched bounds.
fn bbox_crop_in_stitched(
    range: TileRange,
    bbox: Bbox,
    stitched_w: usize,
    stitched_h: usize,
) -> (u32, u32, u32, u32) {
    let (nw_lon, nw_lat) = tile_to_lonlat(range.x_min, range.y_min, range.z);
    let (se_lon, se_lat) = tile_to_lonlat(range.x_max + 1, range.y_max + 1, range.z);
    let pix_x = |lon: f64| (lon - nw_lon) / (se_lon - nw_lon) * stitched_w as f64;
    let pix_y = |lat: f64| (nw_lat - lat) / (nw_lat - se_lat) * stitched_h as f64;
    let crop_x0 = pix_x(bbox.west).round().max(0.0) as u32;
    let crop_x1 = pix_x(bbox.east).round().min(stitched_w as f64) as u32;
    let crop_y0 = pix_y(bbox.north).round().max(0.0) as u32;
    let crop_y1 = pix_y(bbox.south).round().min(stitched_h as f64) as u32;
    let crop_w = crop_x1.saturating_sub(crop_x0).max(1);
    let crop_h = crop_y1.saturating_sub(crop_y0).max(1);
    (crop_x0, crop_y0, crop_w, crop_h)
}

/// Compute the oversize resize dimensions such that the stitched buffer's
/// `(crop_w, crop_h)` sub-rectangle, when scaled by the same factor as the
/// rest of the stitched, lands at exactly `output_size × output_size` in
/// the resized image. Equivalently: pick the resize that would produce
/// `output_size` if applied to just the crop region, but apply it to the
/// whole stitched.
///
/// The crop ratio is `(crop_w / stitched_w, crop_h / stitched_h)`, both
/// close to 1.0 in practice (the stitched typically extends only a
/// fractional tile beyond the bbox). The resized image is therefore only
/// a few percent larger than `output_size × output_size`.
fn oversize_for_crop(
    stitched_w: u32,
    stitched_h: u32,
    crop_w: u32,
    crop_h: u32,
    output_size: u32,
) -> (u32, u32) {
    let crop_w = crop_w.max(1);
    let crop_h = crop_h.max(1);
    let resize_w = (output_size as f64 * stitched_w as f64 / crop_w as f64).round() as u32;
    let resize_h = (output_size as f64 * stitched_h as f64 / crop_h as f64).round() as u32;
    (resize_w.max(output_size), resize_h.max(output_size))
}

/// Translate a crop offset measured in stitched-pixel space into the
/// resized image's pixel space, using the linear scale factor between the
/// two. Used to find where the bbox's NW corner lands in the resized
/// image.
fn scale_crop_offset(crop: u32, stitched_dim: u32, resize_dim: u32) -> u32 {
    if stitched_dim == 0 {
        return 0;
    }
    (crop as f64 * resize_dim as f64 / stitched_dim as f64).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lonlat_tile_roundtrip() {
        // Brienz at z=12 should land in the well-known Bernese Alps tile cluster.
        let (x, y) = lonlat_to_tile(8.0340, 46.7240, 12);
        let (lon, lat) = tile_to_lonlat(x, y, 12);
        assert!(lon <= 8.0340 && 8.0340 < tile_to_lonlat(x + 1, y, 12).0);
        assert!(lat >= 46.7240 && 46.7240 > tile_to_lonlat(x, y + 1, 12).1);
    }

    #[test]
    fn bbox_around_brienz_is_near_square() {
        let b = Bbox::around(8.0340, 46.7240, 12.0);
        let lon_span_km = (b.east - b.west) * 111.32 * 46.7240_f64.to_radians().cos();
        let lat_span_km = (b.north - b.south) * 111.32;
        // Within 1 % of the 12 km target.
        assert!((lon_span_km - 12.0).abs() < 0.12);
        assert!((lat_span_km - 12.0).abs() < 0.12);
    }

    #[test]
    fn cube_face_centres_hit_axis_directions() {
        // (u=0.5, v=0.5) maps to w=(0,0) → u3d=(0,0), so each face's centre
        // direction is the pure unit axis for the face, using bevy_terrain's
        // side index convention: side 0=-X, 1=+Z, 2=+Y, 3=+X, 4=-Z, 5=-Y.
        let cases: [(u8, [f64; 3]); 6] = [
            (0, [-1.0, 0.0, 0.0]),
            (1, [0.0, 0.0, 1.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [1.0, 0.0, 0.0]),
            (4, [0.0, 0.0, -1.0]),
            (5, [0.0, -1.0, 0.0]),
        ];
        for (face, expected) in cases {
            let got = cube_face_to_dir(face, 0.5, 0.5);
            for c in 0..3 {
                assert!(
                    (got[c] - expected[c]).abs() < 1e-9,
                    "face {face} dir mismatch"
                );
            }
        }
    }

    #[test]
    fn dir_to_lonlat_axes() {
        // +Z → (lon=0, lat=0) — Greenwich at the equator.
        let (lon, lat) = dir_to_lonlat([0.0, 0.0, 1.0]);
        assert!(lon.abs() < 1e-9 && lat.abs() < 1e-9);
        // +X → (lon=90, lat=0).
        let (lon, lat) = dir_to_lonlat([1.0, 0.0, 0.0]);
        assert!((lon - 90.0).abs() < 1e-9 && lat.abs() < 1e-9);
        // +Y → (lat=90).
        let (_lon, lat) = dir_to_lonlat([0.0, 1.0, 0.0]);
        assert!((lat - 90.0).abs() < 1e-9);
        // -Y → (lat=-90).
        let (_lon, lat) = dir_to_lonlat([0.0, -1.0, 0.0]);
        assert!((lat + 90.0).abs() < 1e-9);
    }

    #[test]
    fn face_corners_lie_on_unit_sphere() {
        for face in 0..6u8 {
            for &(u, v) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let d = cube_face_to_dir(face, u, v);
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-9);
            }
        }
    }
}
