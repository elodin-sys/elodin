use bevy::asset::Assets;
use bevy::asset::RenderAssetUsages;
use bevy::ecs::query::QueryData;
use bevy::ecs::system::In;
use bevy::ecs::system::InRef;
use bevy::image::Image;
#[cfg(target_family = "wasm")]
use bevy::prelude::{With, Without};
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::TextureDimension;
use bevy::ui::Display;
use bevy::ui::Node;
use bevy::ui::widget::ImageNode;
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, Local, Query, Res, ResMut, World},
    ui::Val,
};
use egui::{self, Color32, Vec2};
use impeller::types::{OwnedPacket, Timestamp};
use impeller_bevy::{CommandsExt, PacketGrantR};
use impeller_wkt::{CurrentTimestamp, ErrorResponse, GetMsgs, MsgBatch, TimestampedMsgStream};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{self};
use web_time::Instant;

use super::{
    PaneName,
    colors::{ColorExt, EColor, get_scheme},
};
use crate::plugins::frustum_common::frustum_up_marker_color;
use crate::sensor_camera::SensorCameraConfigs;
use crate::ui::up_marker::paint_up_marker;
use crate::ui::widgets::SystemStateExt;
use impeller_wkt::FrustumUpMarker;

// ---------------------------------------------------------------------------
// Public pane types (unchanged API)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct VideoStreamPane {
    pub entity: Entity,
    pub name: PaneName,
}

#[derive(Clone, Copy)]
pub struct VideoStreamWidgetArgs {
    pub entity: Entity,
    pub window: Entity,
}

// ---------------------------------------------------------------------------
// VideoStream component
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawPixelFormat {
    Rgba8,
    Gray8,
}

impl RawPixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => 4,
            Self::Gray8 => 1,
        }
    }
}

#[derive(Component)]
pub struct VideoStream {
    pub msg_id: [u8; 2],
    pub msg_name: String,
    pub current_frame: Option<Image>,
    pub size: Vec2,
    pub state: StreamState,
    /// When set, incoming MsgBatch bytes bypass the H.264 decoder.
    pub raw_dims: Option<(u32, u32, RawPixelFormat)>,
}

#[derive(Component)]
pub struct IsTileVisible(pub bool);

impl Default for VideoStream {
    fn default() -> Self {
        Self {
            msg_id: [0, 0],
            msg_name: String::new(),
            current_frame: None,
            size: Vec2::ZERO,
            state: StreamState::default(),
            raw_dims: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of UI frames to wait before sending the stream request.
/// This prevents blocking during initial schematic loading.
const FRAMES_BEFORE_CONNECT: u32 = 5;

/// Maximum number of decoded RGBA frames to keep in the LRU cache.
/// At 640×512 RGBA each frame is ~1.3 MB, so 300 frames ≈ 390 MB.
const MAX_CACHED_FRAMES: usize = 300;

/// If no raw frame exists within this many microseconds of `CurrentTimestamp`,
/// show the "No video at this time" overlay.  500 ms.
const NO_DATA_THRESHOLD_US: i64 = 500_000;

/// Live follow can sit on `LastUpdated` while the last camera AU is older
/// (encode + IDR spacing). Keep showing that frame for one GOP.
const LIVE_HOLD_THRESHOLD_US: i64 = 2_000_000;

/// If the decoder's last position is within this many microseconds of
/// `current_ts` (in either direction), suppress seek retries.  This is
/// wider than `SEQUENTIAL_THRESHOLD_US` to prevent the seek loop where
/// the decoder is "close but not sequential" (e.g. OBS at 30fps with
/// wide frame spacing).  2 seconds.
const DECODER_NEAR_THRESHOLD_US: i64 = 2_000_000;

/// Raw frames fed to the wasm decoder per UI frame. A UI running below the
/// stream rate must feed several frames per tick or it falls into a seek.
#[cfg(target_family = "wasm")]
const MAX_SEQUENTIAL_PER_TICK: usize = 8;

/// Re-issue an in-flight wasm seek to the same keyframe after this long.
#[cfg(target_family = "wasm")]
const SEEK_TIMEOUT_SECS: f32 = 1.0;

/// Per-stream cap on cached raw bytes. A count cap sized for H.264 never
/// fires for raw RGBA (~1.2 MB/frame), which fills wasm linear memory.
#[cfg(target_family = "wasm")]
const MAX_RAW_BYTES: usize = 256 * 1024 * 1024;
/// About the old 50 000-frame H.264 ceiling, so encoded streams keep the
/// same headroom they had under the count cap.
#[cfg(not(target_family = "wasm"))]
const MAX_RAW_BYTES: usize = 1536 * 1024 * 1024;

/// How far behind and ahead of the playhead a fetch tries to keep raw frames.
const FETCH_WINDOW_US: i64 = 3_000_000;

/// Extra windows walked backward looking for an H.264 IDR.
const KEYFRAME_EXTEND_LIMIT: u8 = 4;

/// If the live-tail `TimestampedMsgStream` has not delivered a frame for this
/// many seconds, re-send the backfill and stream requests to recover from
/// a dropped DB connection.
const STREAM_RECOVERY_TIMEOUT_SECS: f32 = 30.0;

/// If backfill_in_flight has been true this long with no callback, reset it
/// (e.g. MsgPacketTx channel full or RequestId exhausted).
const BACKFILL_TIMEOUT_SECS: f32 = 10.0;

// ---------------------------------------------------------------------------
// H.264 keyframe detection
// ---------------------------------------------------------------------------

/// Check whether raw H.264 NAL unit data contains a keyframe (IDR, NAL type 5).
///
/// Scans for Annex B start codes (`00 00 01` or `00 00 00 01`) and checks the
/// NAL unit type in the first byte after the start code.  With
/// `h264parse config-interval=-1`, SPS/PPS are sent alongside every IDR, so
/// any packet containing a type-5 NAL is a valid seek point.
fn is_keyframe(nal_data: &[u8]) -> bool {
    let len = nal_data.len();
    let mut i = 0;
    while i < len.saturating_sub(3) {
        // 4-byte start code: 00 00 00 01
        if i + 3 < len
            && nal_data[i] == 0
            && nal_data[i + 1] == 0
            && nal_data[i + 2] == 0
            && nal_data[i + 3] == 1
        {
            let nal_byte_pos = i + 4;
            if nal_byte_pos < len && (nal_data[nal_byte_pos] & 0x1F) == 5 {
                return true;
            }
            i = nal_byte_pos;
            continue;
        }
        // 3-byte start code: 00 00 01
        if nal_data[i] == 0 && nal_data[i + 1] == 0 && nal_data[i + 2] == 1 {
            let nal_byte_pos = i + 3;
            if nal_byte_pos < len && (nal_data[nal_byte_pos] & 0x1F) == 5 {
                return true;
            }
            i = nal_byte_pos;
            continue;
        }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// StreamState — simplified to 3 variants
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum StreamState {
    /// Startup delay before connecting to the DB.
    WaitingToConnect { frames_waited: u32 },
    /// Connected.  Raw frame cache is being populated (historical backfill +
    /// live tail).  All playback is cache-driven.
    Active,
    /// Unrecoverable error.
    Error(String),
}

impl Default for StreamState {
    fn default() -> Self {
        Self::WaitingToConnect { frames_waited: 0 }
    }
}

// ---------------------------------------------------------------------------
// VideoFrameCache — the single source of truth for video data
// ---------------------------------------------------------------------------

/// Cache of raw H.264 data and decoded RGBA frames.
///
/// Two sources fill `raw_frames`:
/// - `GetMsgs` pages for the part of a window around the playhead that this
///   cache does not already cover, re-fetched after eviction
/// - A continuous live tail (`TimestampedMsgStream`) so H.264 sensor cameras
///   land on the same clock as gray8/rgba instead of the playback tick driver.
///
/// Neither source ever touches the decoder directly.  The playback loop reads
/// `CurrentTimestamp`, looks up the cache, and feeds controlled sequences to
/// the decoder thread.
#[derive(Component)]
pub struct VideoFrameCache {
    /// All raw frame data (H.264 NAL or raw RGBA), keyed by timestamp.
    /// Wrapped in Arc to avoid cloning large frame buffers on access.
    raw_frames: BTreeMap<Timestamp, Arc<Vec<u8>>>,
    /// Decoded RGBA frames (LRU-evicted to `max_frames`).
    decoded_frames: BTreeMap<Timestamp, Image>,
    /// Sorted timestamps of known keyframes (IDR NAL units).
    keyframe_index: Vec<Timestamp>,
    /// Maximum number of decoded frames to retain.
    max_frames: usize,
    /// Cap on [`Self::raw_bytes`].
    max_raw_bytes: usize,
    /// Sum of `raw_frames` payload lengths.
    raw_bytes: usize,
    /// DB ranges whose messages have been fetched or received live.
    /// Sorted and disjoint. Gaps of at most [`NO_DATA_THRESHOLD_US`] merge.
    coverage: Vec<Range<Timestamp>>,
    /// How many extra fetch windows have been walked back looking for an IDR.
    keyframe_extend_steps: u8,
    /// Timestamp of the last frame fed to the decoder (for sequential opt).
    last_decoded_ts: Option<Timestamp>,
    /// `true` while a `SeekBatch` is in-flight on the decoder thread.
    seeking: bool,
    /// Generation counter for `SeekBatch` coalescing.
    seek_generation: u64,
    /// Wall-clock time of the last frame received from the live-tail
    /// `TimestampedMsgStream`.  Used to detect DB disconnection and trigger
    /// recovery.  `None` until the first live frame arrives.
    last_stream_activity: Option<Instant>,
    /// `true` while a `GetMsgs` request is in-flight. Prevents duplicate
    /// requests from piling up.
    backfill_in_flight: bool,
    /// When the current `GetMsgs` request was sent; used to auto-reset
    /// `backfill_in_flight` if the reply never arrives (e.g. channel full).
    backfill_started_at: Option<Instant>,
    /// Timestamp of the raw RGBA frame currently displayed. Used to skip
    /// redundant texture uploads when the playback position hasn't changed.
    last_displayed_ts: Option<Timestamp>,
    /// When false, raw frames are RGBA (sensor camera); skip H.264 keyframe scan.
    pub is_h264: bool,
    /// Playhead seen on the previous wasm UI frame. A still playhead means paused.
    #[cfg(target_family = "wasm")]
    last_playhead: Option<Timestamp>,
    /// When the in-flight wasm seek was sent.
    #[cfg(target_family = "wasm")]
    seek_sent_at: Option<Instant>,
}

impl Default for VideoFrameCache {
    fn default() -> Self {
        Self {
            raw_frames: BTreeMap::new(),
            decoded_frames: BTreeMap::new(),
            keyframe_index: Vec::new(),
            max_frames: MAX_CACHED_FRAMES,
            max_raw_bytes: MAX_RAW_BYTES,
            raw_bytes: 0,
            coverage: Vec::new(),
            keyframe_extend_steps: 0,
            last_decoded_ts: None,
            seeking: false,
            seek_generation: 0,
            last_stream_activity: None,
            backfill_in_flight: false,
            backfill_started_at: None,
            last_displayed_ts: None,
            is_h264: true,
            #[cfg(target_family = "wasm")]
            last_playhead: None,
            #[cfg(target_family = "wasm")]
            seek_sent_at: None,
        }
    }
}

impl VideoFrameCache {
    pub fn for_raw() -> Self {
        Self {
            is_h264: false,
            ..Self::default()
        }
    }
}

impl VideoFrameCache {
    /// Insert a decoded frame and evict old entries if needed.
    pub fn insert_decoded(&mut self, timestamp: Timestamp, image: Image) {
        self.decoded_frames.insert(timestamp, image);
        self.evict_if_needed();
    }

    /// Record a keyframe timestamp (maintains sorted order, deduplicates).
    pub fn record_keyframe(&mut self, timestamp: Timestamp) {
        match self.keyframe_index.binary_search(&timestamp) {
            Ok(_) => {}
            Err(pos) => self.keyframe_index.insert(pos, timestamp),
        }
    }

    /// Find the nearest keyframe at or before `target`.
    pub fn nearest_keyframe_before(&self, target: Timestamp) -> Option<Timestamp> {
        match self.keyframe_index.binary_search(&target) {
            Ok(i) => Some(self.keyframe_index[i]),
            Err(0) => None,
            Err(i) => Some(self.keyframe_index[i - 1]),
        }
    }

    /// Find the first keyframe strictly after `target`.
    pub fn first_keyframe_after(&self, target: Timestamp) -> Option<Timestamp> {
        match self.keyframe_index.binary_search(&target) {
            Ok(i) => self.keyframe_index.get(i + 1).copied(),
            Err(i) => self.keyframe_index.get(i).copied(),
        }
    }

    /// Get the decoded frame at-or-before `target` (most recent frame that
    /// should be displayed at this playback position).
    pub fn decoded_at_or_before(&self, target: Timestamp) -> Option<(Timestamp, &Image)> {
        self.decoded_frames
            .range(..=target)
            .next_back()
            .map(|(t, img)| (*t, img))
    }

    /// Get the first decoded frame at-or-after `target` (used when data
    /// only exists ahead of the playback position, e.g. after a forward-seek).
    pub fn decoded_at_or_after(&self, target: Timestamp) -> Option<(Timestamp, &Image)> {
        self.decoded_frames
            .range(target..)
            .next()
            .map(|(t, img)| (*t, img))
    }

    /// Check whether raw data exists within `threshold_us` of `target`.
    pub fn has_raw_data_near(&self, target: Timestamp, threshold_us: i64) -> bool {
        if let Some((t, _)) = self.raw_frames.range(..=target).next_back()
            && (target.0 - t.0).abs() <= threshold_us
        {
            return true;
        }
        if let Some((t, _)) = self.raw_frames.range(target..).next()
            && (t.0 - target.0).abs() <= threshold_us
        {
            return true;
        }
        false
    }

    /// True when a tile should paint (or drop the "No video" overlay).
    ///
    /// Live follow uses `LastUpdated` as the playhead. Camera AUs can lag that
    /// by more than [`NO_DATA_THRESHOLD_US`] without being missing.
    pub fn has_video_at(&self, target: Timestamp) -> bool {
        if self.has_raw_data_near(target, NO_DATA_THRESHOLD_US) {
            return true;
        }
        self.raw_frames
            .iter()
            .next_back()
            .is_some_and(|(t, _)| target.0 >= t.0 && target.0 - t.0 <= LIVE_HOLD_THRESHOLD_US)
    }

    /// Extract raw frames in `[from, to]` for decoding (cloned out of cache).
    pub fn get_raw_sequence(&self, from: Timestamp, to: Timestamp) -> Vec<(Timestamp, Vec<u8>)> {
        self.raw_frames
            .range(from..=to)
            .map(|(t, d)| (*t, Arc::unwrap_or_clone(Arc::clone(d))))
            .collect()
    }

    /// Get the next raw frame strictly after `ts`.
    pub fn next_raw_frame_after(&self, ts: Timestamp) -> Option<(Timestamp, &[u8])> {
        self.raw_frames
            .range((std::ops::Bound::Excluded(ts), std::ops::Bound::Unbounded))
            .next()
            .map(|(t, d)| (*t, d.as_slice()))
    }

    /// Insert a raw frame, mark that instant covered, and evict away from `anchor`.
    pub fn insert_raw(&mut self, timestamp: Timestamp, data: Vec<u8>, anchor: Timestamp) {
        self.store_raw(timestamp, data);
        let end = Timestamp(timestamp.0.saturating_add(1));
        self.note_covered(timestamp..end);
        self.evict_raw_if_needed(anchor);
    }

    fn store_raw(&mut self, timestamp: Timestamp, data: Vec<u8>) {
        let keyframe = self.is_h264 && is_keyframe(&data);
        let len = data.len();
        if let Some(old) = self.raw_frames.insert(timestamp, Arc::new(data)) {
            self.raw_bytes = self.raw_bytes.saturating_sub(old.len());
        }
        self.raw_bytes = self.raw_bytes.saturating_add(len);
        if !self.is_h264 {
            return;
        }
        if keyframe {
            self.record_keyframe(timestamp);
        } else if let Ok(pos) = self.keyframe_index.binary_search(&timestamp) {
            self.keyframe_index.remove(pos);
        }
    }

    fn note_covered(&mut self, range: Range<Timestamp>) {
        if range.start.0 >= range.end.0 {
            return;
        }
        let mut start = range.start.0;
        let mut end = range.end.0;
        let mut kept = Vec::with_capacity(self.coverage.len() + 1);
        for existing in &self.coverage {
            let near = existing.end.0.saturating_add(NO_DATA_THRESHOLD_US) >= start
                && end.saturating_add(NO_DATA_THRESHOLD_US) >= existing.start.0;
            if near {
                start = start.min(existing.start.0);
                end = end.max(existing.end.0);
            } else {
                kept.push(existing.start..existing.end);
            }
        }
        kept.push(Timestamp(start)..Timestamp(end));
        kept.sort_by_key(|covered| covered.start.0);
        self.coverage = kept;
    }

    /// First sub-range of `want` that has not been fetched.
    fn first_uncovered(&self, want: &Range<Timestamp>) -> Option<Range<Timestamp>> {
        if want.start.0 >= want.end.0 {
            return None;
        }
        let mut cursor = want.start.0;
        for covered in &self.coverage {
            if covered.end.0 <= cursor {
                continue;
            }
            if covered.start.0 > cursor {
                let end = covered.start.0.min(want.end.0);
                if end > cursor {
                    return Some(Timestamp(cursor)..Timestamp(end));
                }
                return None;
            }
            cursor = covered.end.0;
            if cursor >= want.end.0 {
                return None;
            }
        }
        (cursor < want.end.0).then_some(Timestamp(cursor)..want.end)
    }

    fn uncover_before(&mut self, ts: Timestamp) {
        let mut kept = Vec::with_capacity(self.coverage.len());
        for range in self.coverage.drain(..) {
            if range.end.0 <= ts.0 {
                continue;
            }
            if range.start.0 < ts.0 {
                kept.push(Timestamp(ts.0)..range.end);
            } else {
                kept.push(range);
            }
        }
        self.coverage = kept;
    }

    fn uncover_from(&mut self, ts: Timestamp) {
        let mut kept = Vec::with_capacity(self.coverage.len());
        for range in self.coverage.drain(..) {
            if range.start.0 >= ts.0 {
                continue;
            }
            if range.end.0 > ts.0 {
                kept.push(range.start..Timestamp(ts.0));
            } else {
                kept.push(range);
            }
        }
        self.coverage = kept;
    }

    fn consider_keyframe_extend(&mut self, playhead: Timestamp, earliest: Timestamp) {
        if !self.is_h264 {
            self.keyframe_extend_steps = 0;
            return;
        }
        let no_keyframe = self.nearest_keyframe_before(playhead).is_none();
        let above_earliest = self
            .coverage
            .first()
            .is_some_and(|range| range.start.0 > earliest.0);
        if no_keyframe && above_earliest && self.keyframe_extend_steps < KEYFRAME_EXTEND_LIMIT {
            self.keyframe_extend_steps += 1;
        } else if !no_keyframe {
            self.keyframe_extend_steps = 0;
        }
    }

    fn clear_raw_keeping(&mut self, keep: Option<(Timestamp, Arc<Vec<u8>>)>) {
        self.raw_frames.clear();
        self.raw_bytes = 0;
        self.coverage.clear();
        self.keyframe_index.clear();
        self.decoded_frames.clear();
        self.keyframe_extend_steps = 0;
        if let Some((timestamp, frame)) = keep {
            self.raw_bytes = frame.len();
            self.raw_frames.insert(timestamp, frame);
            let end = Timestamp(timestamp.0.saturating_add(1));
            self.coverage.push(timestamp..end);
        }
    }

    fn evict_if_needed(&mut self) {
        while self.decoded_frames.len() > self.max_frames {
            // Evict the frame farthest from where the decoder is currently
            // working.  This prevents freshly-decoded frames from being
            // immediately evicted when scrubbing backward (where new frames
            // have lower timestamps than old cached frames).
            let pivot = self.last_decoded_ts.map(|t| t.0).unwrap_or(0);
            let first_ts = self.decoded_frames.keys().next().map(|t| t.0);
            let last_ts = self.decoded_frames.keys().next_back().map(|t| t.0);
            match (first_ts, last_ts) {
                (Some(first), Some(last)) => {
                    let dist_first = (pivot - first).abs();
                    let dist_last = (pivot - last).abs();
                    if dist_last >= dist_first {
                        self.decoded_frames.pop_last();
                    } else {
                        self.decoded_frames.pop_first();
                    }
                }
                _ => break,
            }
        }
    }

    fn evict_raw_if_needed(&mut self, anchor: Timestamp) {
        while self.raw_bytes > self.max_raw_bytes && self.raw_frames.len() > 1 {
            let first = *self.raw_frames.keys().next().unwrap();
            let last = *self.raw_frames.keys().next_back().unwrap();
            let drop_last = anchor.0.abs_diff(last.0) >= anchor.0.abs_diff(first.0);
            let Some((evicted_ts, data)) = (if drop_last {
                self.raw_frames.pop_last()
            } else {
                self.raw_frames.pop_first()
            }) else {
                break;
            };
            self.raw_bytes = self.raw_bytes.saturating_sub(data.len());
            if drop_last {
                self.uncover_from(evicted_ts);
            } else if let Some(&new_first) = self.raw_frames.keys().next() {
                self.uncover_before(new_first);
            }
        }
        self.trim_to_raw_span();
    }

    fn trim_to_raw_span(&mut self) {
        let Some(&first) = self.raw_frames.keys().next() else {
            self.keyframe_index.clear();
            self.decoded_frames.clear();
            return;
        };
        let &last = self.raw_frames.keys().next_back().unwrap();
        self.keyframe_index.retain(|ts| *ts >= first && *ts <= last);
        self.decoded_frames
            .retain(|ts, _| *ts >= first && *ts <= last);
    }
}

// ---------------------------------------------------------------------------
// Decoder thread interface
// ---------------------------------------------------------------------------

/// Input to the decoder background thread.
pub enum DecoderInput {
    /// Single frame for sequential playback (decoder state is warm).
    Frame(Vec<u8>, Timestamp),
    /// Seek batch: reset the decoder and decode all frames in order.
    SeekBatch {
        frames: Vec<(Timestamp, Vec<u8>)>,
        generation: u64,
    },
}

/// Output from the decoder background thread.
pub enum DecoderOutput {
    /// A decoded RGBA frame.
    Frame(Box<Image>, Timestamp),
    /// All frames in a seek batch have been decoded.
    SeekComplete(u64),
}

#[derive(Component)]
pub struct VideoDecoderHandle {
    #[cfg(not(target_family = "wasm"))]
    tx: flume::Sender<DecoderInput>,
    rx: flume::Receiver<DecoderOutput>,
    width: Arc<AtomicUsize>,
    /// Shared with the decoder thread so it can skip stale seek batches.
    latest_seek_generation: Arc<AtomicU64>,
    #[cfg(not(target_family = "wasm"))]
    _handle: std::thread::JoinHandle<()>,
}

// ---------------------------------------------------------------------------
// Decoder implementations (platform-specific)
// ---------------------------------------------------------------------------

#[cfg(all(not(target_os = "macos"), not(target_family = "wasm")))]
fn decode_one_frame(
    decoder: &mut openh264::decoder::Decoder,
    packet: &[u8],
    rgba: &mut Vec<u8>,
) -> Option<Image> {
    use openh264::formats::YUVSource;
    if let Ok(Some(yuv)) = decoder.decode(packet) {
        let (width, height) = yuv.dimensions();
        rgba.clear();
        rgba.resize(width * height * 4, 0);
        yuv.write_rgba8(rgba);
        Some(Image::new(
            Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba.clone(),
            bevy_render::render_resource::TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        ))
    } else {
        None
    }
}

#[cfg(all(not(target_os = "macos"), not(target_family = "wasm")))]
fn decode_video(
    _frame_width: Arc<AtomicUsize>,
    packet_rx: flume::Receiver<DecoderInput>,
    image_tx: flume::Sender<DecoderOutput>,
    latest_generation: Arc<AtomicU64>,
) {
    let mut decoder = openh264::decoder::Decoder::new().unwrap();
    let mut rgba = vec![];

    while let Ok(input) = packet_rx.recv() {
        match input {
            DecoderInput::Frame(packet, timestamp) => {
                if let Some(image) = decode_one_frame(&mut decoder, &packet, &mut rgba) {
                    let _ = image_tx.try_send(DecoderOutput::Frame(Box::new(image), timestamp));
                }
            }
            DecoderInput::SeekBatch { frames, generation } => {
                // If a newer seek has been requested, skip this stale batch
                // entirely — no point decoding 60 frames for a position the
                // user has already scrubbed past.
                if latest_generation.load(atomic::Ordering::Relaxed) != generation {
                    let _ = image_tx.send(DecoderOutput::SeekComplete(generation));
                    continue;
                }
                decoder = openh264::decoder::Decoder::new().unwrap();
                let mut last_image: Option<(Image, Timestamp)> = None;
                for (timestamp, packet) in &frames {
                    // Check mid-decode too — bail early if superseded.
                    if latest_generation.load(atomic::Ordering::Relaxed) != generation {
                        break;
                    }
                    if let Some(image) = decode_one_frame(&mut decoder, packet, &mut rgba) {
                        last_image = Some((image, *timestamp));
                    }
                }
                if let Some((image, ts)) = last_image {
                    let _ = image_tx.send(DecoderOutput::Frame(Box::new(image), ts));
                }
                let _ = image_tx.send(DecoderOutput::SeekComplete(generation));
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn decode_one_frame_vt(
    video_toolbox: &mut video_toolbox::VideoToolboxDecoder,
    packet: &[u8],
    frame_count: u64,
) -> Option<Image> {
    match video_toolbox.decode(packet, 0) {
        Ok(Some(frame)) => {
            if frame_count <= 3 {
                bevy::log::info!(
                    "VideoToolbox: frame {} decoded {}x{}",
                    frame_count,
                    frame.width,
                    frame.height
                );
            }
            Some(Image::new(
                Extent3d {
                    width: frame.width as u32,
                    height: frame.height as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                frame.rgba,
                bevy_render::render_resource::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            ))
        }
        Ok(None) => {
            if frame_count <= 3 {
                bevy::log::info!(
                    "VideoToolbox: packet {} produced no frame (parameter sets or empty)",
                    frame_count
                );
            }
            None
        }
        Err(e) => {
            if frame_count <= 10 {
                bevy::log::warn!("VideoToolbox decode error (frame {}): {}", frame_count, e);
            }
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn decode_video(
    frame_width: Arc<AtomicUsize>,
    packet_rx: flume::Receiver<DecoderInput>,
    image_tx: flume::Sender<DecoderOutput>,
    latest_generation: Arc<AtomicU64>,
) {
    let mut video_toolbox = video_toolbox::VideoToolboxDecoder::new(frame_width.clone()).unwrap();
    let mut frame_count = 0u64;

    while let Ok(input) = packet_rx.recv() {
        match input {
            DecoderInput::Frame(packet, timestamp) => {
                frame_count += 1;
                if frame_count <= 3 {
                    let nal_units = video_toolbox::find_nal_units(&packet);
                    let nal_info: Vec<String> = nal_units
                        .iter()
                        .map(|n| format!("{:?}({}B)", n.nal_type, n.data.len()))
                        .collect();
                    bevy::log::info!(
                        "VideoToolbox: packet {} size={} NALs=[{}]",
                        frame_count,
                        packet.len(),
                        nal_info.join(", ")
                    );
                }
                if let Some(image) = decode_one_frame_vt(&mut video_toolbox, &packet, frame_count) {
                    let _ = image_tx.try_send(DecoderOutput::Frame(Box::new(image), timestamp));
                }
            }
            DecoderInput::SeekBatch { frames, generation } => {
                // Skip stale seek batches.
                if latest_generation.load(atomic::Ordering::Relaxed) != generation {
                    let _ = image_tx.send(DecoderOutput::SeekComplete(generation));
                    continue;
                }
                video_toolbox =
                    video_toolbox::VideoToolboxDecoder::new(frame_width.clone()).unwrap();
                frame_count = 0;
                let mut last_image: Option<(Image, Timestamp)> = None;
                for (timestamp, packet) in &frames {
                    if latest_generation.load(atomic::Ordering::Relaxed) != generation {
                        break;
                    }
                    frame_count += 1;
                    if let Some(image) =
                        decode_one_frame_vt(&mut video_toolbox, packet, frame_count)
                    {
                        last_image = Some((image, *timestamp));
                    }
                }
                if let Some((image, ts)) = last_image {
                    let _ = image_tx.send(DecoderOutput::Frame(Box::new(image), ts));
                }
                let _ = image_tx.send(DecoderOutput::SeekComplete(generation));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VideoDecoderHandle
// ---------------------------------------------------------------------------

#[cfg(not(target_family = "wasm"))]
impl Default for VideoDecoderHandle {
    fn default() -> Self {
        let (packet_tx, packet_rx) = flume::bounded::<DecoderInput>(32);
        let (image_tx, image_rx) = flume::bounded::<DecoderOutput>(32);
        let width = Arc::new(AtomicUsize::new(0));
        let frame_width = width.clone();
        let latest_gen = Arc::new(AtomicU64::new(0));
        let decoder_gen = latest_gen.clone();
        let _handle =
            std::thread::spawn(move || decode_video(frame_width, packet_rx, image_tx, decoder_gen));
        VideoDecoderHandle {
            tx: packet_tx,
            rx: image_rx,
            _handle,
            width,
            latest_seek_generation: latest_gen,
        }
    }
}

#[cfg(target_family = "wasm")]
impl Default for VideoDecoderHandle {
    fn default() -> Self {
        let (_, rx) = flume::bounded::<DecoderOutput>(1);
        VideoDecoderHandle {
            rx,
            width: Arc::new(AtomicUsize::new(0)),
            latest_seek_generation: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl VideoDecoderHandle {
    /// Send a single raw H.264 frame for sequential decode (warm decoder).
    /// Returns `true` if the frame was successfully enqueued.
    #[cfg(not(target_family = "wasm"))]
    pub fn process_frame(&self, timestamp: Timestamp, frame_data: &[u8]) -> bool {
        self.tx
            .try_send(DecoderInput::Frame(frame_data.to_vec(), timestamp))
            .is_ok()
    }

    /// Wasm playback feeds `webcodecs_video::bevy::VideoStream` directly.
    #[cfg(target_family = "wasm")]
    pub fn process_frame(&self, _timestamp: Timestamp, _frame_data: &[u8]) -> bool {
        false
    }

    /// Send a batch of raw frames for seek-decode (resets the decoder).
    /// Returns `true` if the batch was successfully enqueued.
    /// Also publishes `generation` to the shared atomic so the decoder
    /// thread can skip any older (stale) batches still in the queue.
    #[cfg(not(target_family = "wasm"))]
    pub fn send_seek_batch(&self, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) -> bool {
        self.latest_seek_generation
            .store(generation, atomic::Ordering::Relaxed);
        self.tx
            .try_send(DecoderInput::SeekBatch { frames, generation })
            .is_ok()
    }

    /// Wasm playback feeds `webcodecs_video::bevy::VideoStream` directly.
    #[cfg(target_family = "wasm")]
    pub fn send_seek_batch(&self, _frames: Vec<(Timestamp, Vec<u8>)>, _generation: u64) -> bool {
        false
    }

    /// Drain decoded frames from the decoder thread into the cache.
    /// Returns `true` if a `SeekComplete` for the *current* generation
    /// was received (stale completions from older batches are ignored).
    pub fn drain_into_cache(&self, cache: &mut VideoFrameCache) -> bool {
        let mut seek_completed = false;
        while let Ok(output) = self.rx.try_recv() {
            match output {
                DecoderOutput::Frame(frame, timestamp) => {
                    cache.insert_decoded(timestamp, *frame);
                    cache.last_decoded_ts = Some(timestamp);
                }
                DecoderOutput::SeekComplete(completed_gen) => {
                    // Only acknowledge if this matches the current generation.
                    // A stale completion from an older batch must not clear
                    // `seeking` while a newer seek is still in flight.
                    if completed_gen == cache.seek_generation {
                        seek_completed = true;
                    }
                }
            }
        }
        seek_completed
    }
}

// ---------------------------------------------------------------------------
// Widget query & params
// ---------------------------------------------------------------------------

#[derive(QueryData)]
#[query_data(mutable)]
pub struct WidgetQuery {
    stream: &'static mut VideoStream,
    decoder: &'static mut VideoDecoderHandle,
    cache: &'static mut VideoFrameCache,
    ui_node: &'static mut Node,
    image_node: &'static mut ImageNode,
    #[cfg(target_family = "wasm")]
    wc: Option<&'static mut webcodecs_video::bevy::VideoStream>,
}

#[derive(SystemParam)]
pub struct VideoStreamWidget<'w, 's> {
    query: Query<'w, 's, WidgetQuery>,
    current_time: Res<'w, CurrentTimestamp>,
    images: ResMut<'w, Assets<Image>>,
    window_settings: Query<'w, 's, &'static bevy_egui::EguiContextSettings>,
    sensor_camera_configs: Option<Res<'w, SensorCameraConfigs>>,
}

// ---------------------------------------------------------------------------
// DB request helpers
// ---------------------------------------------------------------------------

/// Start the live-tail stream for cache population.
///
/// The callback **only** inserts raw frames into the cache and records
/// keyframe timestamps.  It never touches the decoder.
fn send_stream_request(commands: &mut Commands, entity: Entity, msg_id: [u8; 2]) {
    let handler = move |InRef(pkt): InRef<OwnedPacket<PacketGrantR>>,
                        mut caches: Query<&mut VideoFrameCache>,
                        current_ts: Res<CurrentTimestamp>| {
        if let OwnedPacket::Msg(msg_buf) = pkt
            && let Some(timestamp) = msg_buf.timestamp
            && let Ok(mut cache) = caches.get_mut(entity)
        {
            cache.insert_raw(timestamp, msg_buf.buf.to_vec(), current_ts.0);
            cache.last_stream_activity = Some(Instant::now());
        }
        false
    };

    commands.send_msg_req_reply_raw(TimestampedMsgStream { msg_id }, handler);
}

/// Maximum payload size (in bytes) for a single backfill `GetMsgs` batch.
/// Sized so that each batch processes within ~10 ms in the editor sink,
/// avoiding the 80-130 ms stalls that the old 32 MB limit caused when
/// raw RGBA sensor camera frames accumulated. The paginated backfill
/// callback requests the next page immediately, keeping total throughput
/// high while preventing any single response from dominating a frame.
/// Also kept low enough that two camera streams won't exceed the u8
/// RequestId space (256) during a full backfill.
const BACKFILL_MAX_BATCH_BYTES: usize = 5 * 1024 * 1024;

/// Assumed max raw RGBA frame size when dims unknown (640×480×4).
/// Keeps first backfill page within BACKFILL_MAX_BATCH_BYTES for replay DBs.
const BACKFILL_ASSUMED_MAX_RAW_FRAME: usize = 640 * 480 * 4;

/// Typical H.264 frame size when dims unknown (~25–50 KB); avoids 4-frame batch for video_stream.
const BACKFILL_ASSUMED_H264_FRAME: usize = 50 * 1024;

fn stream_frame_bytes(raw_dims: Option<(u32, u32, RawPixelFormat)>, is_h264: bool) -> usize {
    match raw_dims {
        Some((w, h, format)) => (w as usize * h as usize * format.bytes_per_pixel()).max(1),
        None if is_h264 => BACKFILL_ASSUMED_H264_FRAME,
        None => BACKFILL_ASSUMED_MAX_RAW_FRAME,
    }
}

fn stream_fps(configs: Option<&SensorCameraConfigs>, camera_name: &str) -> f32 {
    configs
        .and_then(|configs| {
            configs
                .0
                .iter()
                .find(|config| config.camera_name == camera_name)
        })
        .map(|config| config.fps)
        .filter(|fps| *fps > 0.0)
        .unwrap_or(30.0)
}

/// `2 * window * frame_bytes * fps <= budget / 2`, and never more than
/// [`FETCH_WINDOW_US`]. Keeps a fetched window from being evicted on arrival.
fn fetch_window_us(frame_bytes: usize, fps: f32, budget: usize) -> i64 {
    if frame_bytes == 0 || fps <= 0.0 || budget == 0 {
        return FETCH_WINDOW_US;
    }
    let denom = 4.0 * frame_bytes as f64 * f64::from(fps);
    if denom <= 0.0 {
        return FETCH_WINDOW_US;
    }
    let max_us = (budget as f64 * 1_000_000.0 / denom) as i64;
    FETCH_WINDOW_US.min(max_us.max(1))
}

fn clamped_fetch_window(
    playhead: Timestamp,
    behind_us: i64,
    ahead_us: i64,
    earliest: Timestamp,
    last_updated: Timestamp,
) -> Option<Range<Timestamp>> {
    if earliest.0 == i64::MAX || last_updated.0 == i64::MIN || earliest.0 > last_updated.0 {
        return None;
    }
    let start = playhead.0.saturating_sub(behind_us).max(earliest.0);
    let end = playhead.0.saturating_add(ahead_us).min(last_updated.0);
    (start < end).then_some(Timestamp(start)..Timestamp(end))
}

/// Compute the backfill frame limit from a byte budget.
fn backfill_frame_limit(raw_dims: Option<(u32, u32, RawPixelFormat)>, is_h264: bool) -> usize {
    let frame_bytes = stream_frame_bytes(raw_dims, is_h264);
    (BACKFILL_MAX_BATCH_BYTES / frame_bytes).max(1)
}

/// Fetch one page of `range`. A full page continues from the last timestamp;
/// a short page marks the whole range covered, including stretches with no frames.
fn send_window_request(
    commands: &mut Commands,
    entity: Entity,
    msg_id: [u8; 2],
    range: Range<Timestamp>,
    limit: usize,
) {
    let range_end = range.end;
    commands.send_msg_req_reply(
        GetMsgs {
            msg_id,
            range: range.start..range_end,
            limit: Some(limit),
        },
        move |In(result): In<Result<MsgBatch, ErrorResponse>>,
              mut caches: Query<&mut VideoFrameCache>,
              current_ts: Res<CurrentTimestamp>,
              earliest: Res<impeller_wkt::EarliestTimestamp>,
              mut cmds: Commands| {
            let Ok(mut cache) = caches.get_mut(entity) else {
                return true;
            };
            cache.backfill_in_flight = false;
            cache.backfill_started_at = None;
            let Ok(batch) = result else {
                return true;
            };
            let mut count = 0usize;
            let mut first_ts = range.start;
            let mut last_ts = range.start;
            for (timestamp, data) in batch.data {
                if count == 0 {
                    first_ts = timestamp;
                }
                last_ts = timestamp;
                count += 1;
                cache.store_raw(timestamp, data);
            }
            // `GetMsgs` snaps the start to the message at or before it, so the
            // first sample can sit just outside `range`.
            let covered_start = if count == 0 {
                range.start
            } else {
                Timestamp(first_ts.0.min(range.start.0))
            };
            let covered_end = if count >= limit {
                Timestamp(last_ts.0.saturating_add(1))
            } else {
                range_end
            };
            cache.note_covered(covered_start..covered_end);
            cache.evict_raw_if_needed(current_ts.0);
            cache.consider_keyframe_extend(current_ts.0, earliest.0);
            if count >= limit && covered_end.0 < range_end.0 {
                cache.backfill_in_flight = true;
                cache.backfill_started_at = Some(Instant::now());
                send_window_request(&mut cmds, entity, msg_id, covered_end..range_end, limit);
            }
            true
        },
    );
}

// ---------------------------------------------------------------------------
// GPU texture upload helper
// ---------------------------------------------------------------------------

/// Upload a decoded frame to the GPU, reusing the existing texture when
/// the resolution has not changed.
#[cfg(not(target_family = "wasm"))]
fn upload_frame(
    frame: Image,
    image_node: &mut bevy::prelude::Mut<'_, ImageNode>,
    images: &mut Assets<Image>,
) {
    let reused = if let Some(mut existing) = images.get_mut(&image_node.image) {
        if existing.width() == frame.width()
            && existing.height() == frame.height()
            && let Some(dst) = &mut existing.data
            && let Some(src) = &frame.data
        {
            dst.copy_from_slice(src);
            true
        } else {
            false
        }
    } else {
        false
    };
    if !reused {
        image_node.image = images.add(frame);
    }
}

fn copy_raw_frame_to_rgba(src: &[u8], dst: &mut [u8], format: RawPixelFormat) -> bool {
    match format {
        RawPixelFormat::Rgba8 if src.len() == dst.len() => {
            dst.copy_from_slice(src);
            true
        }
        RawPixelFormat::Gray8 if src.len().checked_mul(4) == Some(dst.len()) => {
            for (&gray, pixel) in src.iter().zip(dst.as_chunks_mut::<4>().0) {
                pixel.copy_from_slice(&[gray, gray, gray, 255]);
            }
            true
        }
        _ => false,
    }
}

/// Insert the library decoder beside a video tile. Spawn sites stay shared
/// with native; this runs once per entity on wasm.
#[cfg(target_family = "wasm")]
pub fn attach_webcodecs(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<VideoDecoderHandle>,
            Without<webcodecs_video::bevy::VideoStream>,
        ),
    >,
) {
    for entity in &query {
        commands
            .entity(entity)
            .insert(webcodecs_video::bevy::VideoStream::new());
    }
}

/// Feed the wasm decoder from the raw Annex-B cache and keep the playhead in sync.
#[cfg(target_family = "wasm")]
fn drive_webcodecs(
    current_ts: Timestamp,
    cache: &mut VideoFrameCache,
    wc: &mut webcodecs_video::bevy::VideoStream,
) {
    wc.set_time(current_ts.0);
    // A moving playhead is playback. A still one is a paused scrub, which
    // must flush the decoder so the frame under the playhead paints.
    wc.set_playing(cache.last_playhead.is_some_and(|last| last != current_ts));
    cache.last_playhead = Some(current_ts);

    let was_seeking = cache.seeking;
    cache.seeking = wc.is_seeking();
    if was_seeking && !cache.seeking {
        // `last_decoded_ts` already points at the last sample the seek
        // submitted, so sequential feeding resumes after it.
        cache.seek_sent_at = None;
    }

    let decoder_near = cache
        .last_decoded_ts
        .is_some_and(|last| (current_ts.0 - last.0).abs() < DECODER_NEAR_THRESHOLD_US);
    let is_sequential = cache.last_decoded_ts.is_some_and(|last| {
        current_ts.0 >= last.0 && (current_ts.0 - last.0) < DECODER_NEAR_THRESHOLD_US
    });
    let has_raw = cache.has_video_at(current_ts);

    let mut fed = false;
    if is_sequential && !cache.seeking && !wc.needs_keyframe() {
        // A UI slower than the stream falls behind feeding frame by frame. An
        // IDR resets the decoder's references, so skip to the newest keyframe
        // once the backlog is longer than the no-data threshold.
        if let Some(last) = cache.last_decoded_ts
            && current_ts.0 - last.0 > NO_DATA_THRESHOLD_US
            && let Some(kf) = cache.nearest_keyframe_before(current_ts)
            && kf > last
            && let Some((&prev, _)) = cache.raw_frames.range(..kf).next_back()
        {
            cache.last_decoded_ts = Some(prev);
        }
        for _ in 0..MAX_SEQUENTIAL_PER_TICK {
            let Some(last) = cache.last_decoded_ts else {
                break;
            };
            let Some((ts, data)) = cache.next_raw_frame_after(last) else {
                break;
            };
            if ts.0 > current_ts.0 + DECODER_NEAR_THRESHOLD_US {
                break;
            }
            let data = data.to_vec();
            if !wc.push(ts.0, &data) {
                break;
            }
            cache.last_decoded_ts = Some(ts);
            fed = true;
        }
    }

    // The native path can step backward inside `decoder_near` from its RGBA
    // cache. Wasm only has the texture, so a playhead that is not the frame
    // on screen seeks. Skip that seek when the texture is already there, or a
    // paused flush (`needs_keyframe`) would re-decode the same GOP forever.
    let at_playhead = wc
        .last_presented
        .is_some_and(|pts| (current_ts.0 - pts).abs() <= NO_DATA_THRESHOLD_US);
    let wants_seek =
        has_raw && !fed && !at_playhead && (wc.needs_keyframe() || !is_sequential || !decoder_near);
    if !wants_seek {
        return;
    }

    // A live stream lands a new IDR every GOP, so the nearest keyframe keeps
    // moving. Let an in-flight seek finish rather than restarting it.
    let in_flight = cache.seeking
        && cache
            .seek_sent_at
            .is_some_and(|sent| sent.elapsed().as_secs_f32() < SEEK_TIMEOUT_SECS);
    if in_flight {
        return;
    }

    let Some(kf) = cache
        .nearest_keyframe_before(current_ts)
        .or_else(|| cache.first_keyframe_after(current_ts))
    else {
        return;
    };
    // The decoder discards everything before the target and completes only on
    // an exact match, so the target has to be a real sample, not the playhead.
    let mut frames = cache.get_raw_sequence(kf, current_ts.max(kf));
    let Some(mut target) = frames.last().map(|(ts, _)| *ts) else {
        return;
    };
    // Decoders hold the last submitted picture until the next one arrives
    // (Safari always, Chrome for some streams), so submit one sample past the
    // target. At the live edge there is none yet: show the previous frame and
    // let the newest be the trailing sample.
    if let Some((ts, data)) = cache.next_raw_frame_after(target) {
        frames.push((ts, data.to_vec()));
    } else if frames.len() >= 2 {
        target = frames[frames.len() - 2].0;
    }
    let last_submitted = frames[frames.len() - 1].0;
    let samples = frames.into_iter().map(|(ts, data)| (ts.0, data)).collect();
    wc.seek_through(samples, target.0);
    cache.seeking = true;
    cache.last_decoded_ts = Some(last_submitted);
    cache.seek_sent_at = Some(Instant::now());
}

// ---------------------------------------------------------------------------
// Widget implementation — cache-first playback
// ---------------------------------------------------------------------------

impl super::widgets::WidgetSystem for VideoStreamWidget<'_, '_> {
    type Args = VideoStreamWidgetArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        VideoStreamWidgetArgs { entity, window }: Self::Args,
    ) -> Self::Output {
        let mut state = state.params_mut(world);

        let Ok(WidgetQueryItem {
            mut stream,
            decoder,
            mut cache,
            mut image_node,
            mut ui_node,
            #[cfg(target_family = "wasm")]
            mut wc,
        }) = state.query.get_mut(entity)
        else {
            return;
        };

        let current_ts = state.current_time.0;

        // ---------------------------------------------------------------
        // State machine
        // ---------------------------------------------------------------
        match &mut stream.state {
            StreamState::WaitingToConnect { .. } => {
                // Connection + drain handled by `connect_streams` system.
            }
            StreamState::Active => {
                if let Some((w, h, format)) = stream.raw_dims {
                    if let Some((found_ts, data)) =
                        cache.raw_frames.range(..=current_ts).next_back()
                    {
                        let found_ts = *found_ts;
                        if cache.last_displayed_ts != Some(found_ts) {
                            let expected = w as usize * h as usize * format.bytes_per_pixel();
                            if data.len() == expected {
                                let data = Arc::clone(data);
                                let mut wrote_direct = false;
                                if let Some(mut existing) = state.images.get_mut(&image_node.image)
                                    && existing.width() == w
                                    && existing.height() == h
                                    && let Some(dst) = &mut existing.data
                                {
                                    wrote_direct = copy_raw_frame_to_rgba(&data, dst, format);
                                }
                                if !wrote_direct {
                                    let mut rgba = vec![0; w as usize * h as usize * 4];
                                    let copied = copy_raw_frame_to_rgba(&data, &mut rgba, format);
                                    debug_assert!(copied);
                                    let img = Image::new(
                                        Extent3d {
                                            width: w,
                                            height: h,
                                            depth_or_array_layers: 1,
                                        },
                                        TextureDimension::D2,
                                        rgba,
                                        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                                        RenderAssetUsages::MAIN_WORLD
                                            | RenderAssetUsages::RENDER_WORLD,
                                    );
                                    image_node.image = state.images.add(img);
                                }
                                if stream.size == Vec2::ZERO {
                                    stream.size = Vec2::new(w as f32, h as f32);
                                }
                                cache.last_displayed_ts = Some(found_ts);
                            }
                        }
                    }
                } else {
                    #[cfg(not(target_family = "wasm"))]
                    {
                        // H.264 decoded video: existing decode pipeline.
                        let seek_done = decoder.drain_into_cache(&mut cache);
                        if seek_done {
                            cache.seeking = false;
                        }

                        if let Some((ts, img)) = cache.decoded_at_or_before(current_ts)
                            && ((current_ts.0 - ts.0).abs() <= NO_DATA_THRESHOLD_US
                                || (current_ts.0 >= ts.0
                                    && current_ts.0 - ts.0 <= LIVE_HOLD_THRESHOLD_US))
                        {
                            stream.current_frame = Some(img.clone());
                        } else if let Some((ts, img)) = cache.decoded_at_or_after(current_ts)
                            && (ts.0 - current_ts.0).abs() <= NO_DATA_THRESHOLD_US
                        {
                            stream.current_frame = Some(img.clone());
                        }

                        let decoder_near = if let Some(last) = cache.last_decoded_ts {
                            (current_ts.0 - last.0).abs() < DECODER_NEAR_THRESHOLD_US
                        } else {
                            false
                        };

                        let is_sequential = if let Some(last) = cache.last_decoded_ts {
                            current_ts.0 >= last.0
                                && (current_ts.0 - last.0) < DECODER_NEAR_THRESHOLD_US
                        } else {
                            false
                        };

                        let has_raw = cache.has_video_at(current_ts);

                        if is_sequential && !cache.seeking {
                            if let Some((ts, data)) =
                                cache.next_raw_frame_after(cache.last_decoded_ts.unwrap())
                                && ts.0 <= current_ts.0 + DECODER_NEAR_THRESHOLD_US
                                && decoder.process_frame(ts, data)
                            {
                                cache.last_decoded_ts = Some(ts);
                            }
                        } else if !decoder_near && has_raw {
                            let (seek_start, seek_end) =
                                if let Some(kf) = cache.nearest_keyframe_before(current_ts) {
                                    (kf, current_ts)
                                } else if let Some(kf) = cache.first_keyframe_after(current_ts) {
                                    let end = cache
                                        .raw_frames
                                        .range(kf..)
                                        .nth(1)
                                        .map(|(t, _)| *t)
                                        .unwrap_or(kf);
                                    (kf, end)
                                } else {
                                    (current_ts, current_ts)
                                };

                            let frames = cache.get_raw_sequence(seek_start, seek_end);
                            if !frames.is_empty() {
                                cache.seek_generation += 1;
                                if decoder.send_seek_batch(frames, cache.seek_generation) {
                                    cache.seeking = true;
                                }
                            }
                        }
                    }
                    #[cfg(target_family = "wasm")]
                    {
                        if let Some(wc) = wc.as_deref_mut() {
                            drive_webcodecs(current_ts, &mut cache, wc);
                        }
                    }
                }
            }
            StreamState::Error(_) => {}
        }

        // ---------------------------------------------------------------
        // Layout
        // ---------------------------------------------------------------
        let max_rect = ui.max_rect();

        // Guard: only lay out when the window still has an egui context.
        // bevy_egui 0.40 removed `EguiContextSettings::scale_factor` (zoom is
        // handled via egui's native pixels-per-point now); egui points map
        // 1:1 onto Bevy UI logical pixels.
        if state.window_settings.get(window).is_err() {
            return;
        };

        let viewport_pos = max_rect.left_top().to_vec2();
        let viewport_size = max_rect.size();

        // The wasm decoder's texture is render-world only, so it is absent
        // from `Assets<Image>`; fall back to the stream's reported size.
        let frame_size = state
            .images
            .get(&image_node.image)
            .map(|image| Vec2::new(image.width() as f32, image.height() as f32))
            .or_else(|| (stream.size != Vec2::ZERO).then_some(stream.size));
        let (width, height) = if let Some(frame_size) = frame_size {
            let aspect_ratio = frame_size.y / frame_size.x;
            let height = viewport_size.x * aspect_ratio;
            if height > viewport_size.y {
                let width = viewport_size.y / aspect_ratio;
                (width, viewport_size.y)
            } else {
                (viewport_size.x, height)
            }
        } else {
            (viewport_size.x, viewport_size.y)
        };

        let x_offset = (viewport_size.x - width) / 2.0;
        let y_offset = (viewport_size.y - height) / 2.0;
        ui_node.left = Val::Px(viewport_pos.x + x_offset);
        ui_node.top = Val::Px(viewport_pos.y + y_offset);
        ui_node.width = Val::Px(width);
        ui_node.height = Val::Px(height);
        ui_node.max_width = Val::Px(viewport_size.x);
        ui_node.max_height = Val::Px(viewport_size.y);

        // ---------------------------------------------------------------
        // Rendering
        // ---------------------------------------------------------------
        match &stream.state {
            StreamState::WaitingToConnect { frames_waited } => {
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);
                ui.centered_and_justified(|ui| {
                    let progress =
                        (*frames_waited as f32 / FRAMES_BEFORE_CONNECT as f32 * 100.0) as u32;
                    ui.label(format!("Initializing video stream... {}%", progress));
                });
            }
            StreamState::Active => {
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);

                // Native copies a decoded RGBA image into the tile. Wasm points the
                // tile at the decoder's persistent GPU texture.
                #[cfg(not(target_family = "wasm"))]
                if let Some(frame) = stream.current_frame.take() {
                    if stream.size == Vec2::ZERO {
                        stream.size = Vec2::new(frame.width() as f32, frame.height() as f32);
                    }
                    upload_frame(frame, &mut image_node, &mut state.images);
                }
                #[cfg(target_family = "wasm")]
                if stream.raw_dims.is_none()
                    && let Some(wc) = wc.as_ref()
                    && let Some(handle) = wc.image.as_ref()
                {
                    if stream.size == Vec2::ZERO
                        && let Some((w, h)) = wc.size
                    {
                        stream.size = Vec2::new(w as f32, h as f32);
                    }
                    if image_node.image != *handle {
                        image_node.image = handle.clone();
                    }
                }

                // "No video at this time" overlay when there is no raw data
                // near the current playback position.
                if !cache.has_video_at(current_ts) {
                    ui.painter()
                        .rect_filled(max_rect, 0, Color32::BLACK.opacity(0.75));
                    ui.put(
                        egui::Rect::from_center_size(
                            max_rect.center_top() + egui::vec2(0., 64.0),
                            egui::vec2(max_rect.width(), 20.0),
                        ),
                        egui::Label::new(
                            egui::RichText::new("No video at this time")
                                .size(16.0)
                                .color(get_scheme().highlight),
                        ),
                    );
                }

                if let Some(config) = state
                    .sensor_camera_configs
                    .as_ref()
                    .and_then(|configs| {
                        configs
                            .0
                            .iter()
                            .find(|config| config.camera_name == stream.msg_name)
                    })
                    // Gated on create_frustum like the 3D marker, so deleting the
                    // frustum cannot strand a bar the inspector no longer exposes.
                    .filter(|config| {
                        config.create_frustum
                            && config.frustums_up_marker_overlay
                            && config.frustums_up_marker != FrustumUpMarker::None
                    })
                {
                    let image_rect = egui::Rect::from_min_size(
                        egui::pos2(viewport_pos.x + x_offset, viewport_pos.y + y_offset),
                        Vec2::new(width, height),
                    );
                    paint_up_marker(
                        ui.painter(),
                        image_rect,
                        config.frustums_up_marker,
                        config.frustums_color.into_color32(),
                        frustum_up_marker_color(config.frustums_color).into_color32(),
                    );
                }
            }
            StreamState::Error(error) => {
                ui.centered_and_justified(|ui| {
                    ui.colored_label(get_scheme().error, format!("Error: {}", error));
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Eager connection system — runs for ALL video streams, even hidden ones
// ---------------------------------------------------------------------------

/// Bevy system that transitions every `VideoStream` from `WaitingToConnect`
/// to `Active` after the startup delay, regardless of whether the panel is
/// currently visible.  This ensures the raw-frame cache starts filling
/// immediately for all streams.
pub fn connect_streams(
    mut query: Query<(
        Entity,
        &mut VideoStream,
        &VideoDecoderHandle,
        &mut VideoFrameCache,
    )>,
    mut commands: Commands,
    last_updated: Res<impeller_wkt::LastUpdated>,
    current_ts: Res<CurrentTimestamp>,
    earliest: Res<impeller_wkt::EarliestTimestamp>,
    sensor_configs: Option<Res<SensorCameraConfigs>>,
) {
    for (entity, mut stream, decoder, mut cache) in &mut query {
        match &mut stream.state {
            StreamState::WaitingToConnect { frames_waited } => {
                *frames_waited += 1;
                if *frames_waited >= FRAMES_BEFORE_CONNECT {
                    let msg_id = stream.msg_id;
                    bevy::log::debug!(
                        "video stream connected: msg_name={}, raw_dims={:?}",
                        stream.msg_name,
                        stream.raw_dims
                    );
                    send_stream_request(&mut commands, entity, msg_id);
                    stream.state = StreamState::Active;
                }
            }
            StreamState::Active => {
                let seek_done = decoder.drain_into_cache(&mut cache);
                if seek_done {
                    cache.seeking = false;
                }

                if cache.backfill_in_flight
                    && let Some(started) = cache.backfill_started_at
                    && started.elapsed().as_secs_f32() >= BACKFILL_TIMEOUT_SECS
                {
                    bevy::log::warn!(
                        "backfill_in_flight stuck for {BACKFILL_TIMEOUT_SECS}s, resetting"
                    );
                    cache.backfill_in_flight = false;
                    cache.backfill_started_at = None;
                }

                if let Some(last) = cache.last_stream_activity
                    && last.elapsed().as_secs_f32() > STREAM_RECOVERY_TIMEOUT_SECS
                {
                    send_stream_request(&mut commands, entity, stream.msg_id);
                    cache.last_stream_activity = None;
                }

                if !cache.backfill_in_flight {
                    request_playhead_window(
                        &mut commands,
                        entity,
                        &stream,
                        &mut cache,
                        (current_ts.0, earliest.0, last_updated.0),
                        sensor_configs.as_deref(),
                    );
                }
            }
            StreamState::Error(_) => {}
        }
    }
}

fn request_playhead_window(
    commands: &mut Commands,
    entity: Entity,
    stream: &VideoStream,
    cache: &mut VideoFrameCache,
    (playhead, earliest, last_updated): (Timestamp, Timestamp, Timestamp),
    sensor_configs: Option<&SensorCameraConfigs>,
) {
    let frame_bytes = stream_frame_bytes(stream.raw_dims, cache.is_h264);
    let fps = stream_fps(sensor_configs, &stream.msg_name);
    let ahead_us = fetch_window_us(frame_bytes, fps, cache.max_raw_bytes);
    let behind_us = if cache.is_h264 {
        ahead_us.saturating_mul(1 + i64::from(cache.keyframe_extend_steps))
    } else {
        ahead_us
    };
    let Some(want) = clamped_fetch_window(playhead, behind_us, ahead_us, earliest, last_updated)
    else {
        return;
    };
    let Some(gap) = cache.first_uncovered(&want) else {
        return;
    };
    let limit = backfill_frame_limit(stream.raw_dims, cache.is_h264);
    cache.backfill_in_flight = true;
    cache.backfill_started_at = Some(Instant::now());
    send_window_request(commands, entity, stream.msg_id, gap, limit);
}

// ---------------------------------------------------------------------------
// Visibility system
// ---------------------------------------------------------------------------

pub fn set_visibility(mut query: Query<(&mut Node, &IsTileVisible)>) {
    for (mut ui_node, is_visible) in &mut query {
        if is_visible.0 {
            ui_node.display = Display::Flex;
        } else {
            ui_node.display = Display::None;
        }
    }
}

pub(crate) fn invalidate_sensor_frames_if_loaded_skybox_differs(
    loaded_skybox: Option<&str>,
    config: &impeller_wkt::DbConfig,
    caches: &mut Query<&mut VideoFrameCache>,
) {
    if loaded_skybox == config.skybox_active() {
        return;
    }
    clear_sensor_raw_frame_caches(caches);
}

pub(crate) fn invalidate_sensor_frames_on_db_skybox_change(
    config: Res<impeller_wkt::DbConfig>,
    #[cfg(not(target_family = "wasm"))] skybox_ui: Option<
        Res<bevy_ai_skybox::prelude::SkyboxGenerationUi>,
    >,
    #[cfg(not(target_family = "wasm"))] mut locally_pushed: ResMut<
        crate::skybox_generation::LocallyPushedSkyboxActive,
    >,
    mut last_active: Local<Option<String>>,
    mut caches: Query<&mut VideoFrameCache>,
) {
    let active = config.skybox_active().map(str::to_string);
    if *last_active == active {
        return;
    }
    #[cfg(not(target_family = "wasm"))]
    {
        if locally_pushed.consume_matching(active.as_deref()) {
            *last_active = active;
            return;
        }
        if let (Some(active), Some(skybox_ui)) = (active.as_deref(), skybox_ui.as_deref())
            && skybox_ui.is_busy()
            && skybox_ui.target_name.as_deref() == Some(active)
        {
            *last_active = Some(active.to_string());
            return;
        }
    }
    *last_active = active;
    clear_sensor_raw_frame_caches(&mut caches);
}

fn clear_sensor_raw_frame_caches(caches: &mut Query<&mut VideoFrameCache>) {
    for mut cache in caches {
        if cache.is_h264 {
            continue;
        }
        let displayed = cache
            .last_displayed_ts
            .and_then(|timestamp| cache.raw_frames.get(&timestamp).cloned());
        let keep = cache.last_displayed_ts.zip(displayed);
        cache.clear_raw_keeping(keep);
        if cache.raw_frames.is_empty() {
            cache.last_displayed_ts = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray8_expands_to_opaque_rgba() {
        let mut rgba = [0; 8];
        assert!(copy_raw_frame_to_rgba(
            &[16, 235],
            &mut rgba,
            RawPixelFormat::Gray8
        ));
        assert_eq!(rgba, [16, 16, 16, 255, 235, 235, 235, 255]);
    }

    #[test]
    fn gray8_backfill_uses_one_byte_per_pixel() {
        let gray = backfill_frame_limit(Some((640, 512, RawPixelFormat::Gray8)), false);
        let rgba = backfill_frame_limit(Some((640, 512, RawPixelFormat::Rgba8)), false);
        assert_eq!(gray, rgba * 4);
    }

    #[test]
    fn live_hold_keeps_video_when_playhead_is_ahead() {
        let mut cache = VideoFrameCache::default();
        cache.insert_raw(
            Timestamp(1_000_000),
            vec![0, 0, 0, 1, 0x65],
            Timestamp(1_000_000),
        );
        assert!(cache.has_video_at(Timestamp(1_000_000)));
        assert!(cache.has_video_at(Timestamp(2_500_000)));
        assert!(!cache.has_video_at(Timestamp(4_000_000)));
        assert!(!cache.has_video_at(Timestamp(0)));
    }

    #[test]
    fn raw_bytes_track_insert_replace_and_clear() {
        let mut cache = VideoFrameCache::for_raw();
        cache.max_raw_bytes = 1_000;
        let anchor = Timestamp(1);
        cache.insert_raw(Timestamp(1), vec![0; 4], anchor);
        cache.insert_raw(Timestamp(1), vec![0; 7], anchor);
        assert_eq!(cache.raw_bytes, 7);
        assert_eq!(cache.raw_frames.len(), 1);

        cache.insert_raw(Timestamp(2), vec![0; 3], anchor);
        assert_eq!(cache.raw_bytes, 10);
        let kept = cache.raw_frames.get(&Timestamp(1)).unwrap().clone();
        cache.clear_raw_keeping(Some((Timestamp(1), kept)));
        assert_eq!(cache.raw_bytes, 7);
        assert_eq!(cache.raw_frames.len(), 1);
        assert_eq!(cache.coverage.len(), 1);
    }

    #[test]
    fn eviction_drops_the_end_farthest_from_the_anchor() {
        let mut cache = VideoFrameCache::for_raw();
        cache.max_raw_bytes = 10;
        cache.insert_raw(Timestamp(1_000), vec![0; 6], Timestamp(1_000));
        cache.insert_raw(Timestamp(2_000), vec![0; 6], Timestamp(1_000));
        assert!(cache.raw_frames.contains_key(&Timestamp(1_000)));
        assert!(!cache.raw_frames.contains_key(&Timestamp(2_000)));
        assert_eq!(cache.raw_bytes, 6);
        assert!(
            cache
                .first_uncovered(&(Timestamp(1_500)..Timestamp(2_500)))
                .is_some()
        );
        assert!(
            cache
                .first_uncovered(&(Timestamp(1_000)..Timestamp(1_001)))
                .is_none()
        );

        let mut cache = VideoFrameCache::for_raw();
        cache.max_raw_bytes = 10;
        cache.insert_raw(Timestamp(1_000), vec![0; 6], Timestamp(2_000));
        cache.insert_raw(Timestamp(2_000), vec![0; 6], Timestamp(2_000));
        assert!(!cache.raw_frames.contains_key(&Timestamp(1_000)));
        assert!(cache.raw_frames.contains_key(&Timestamp(2_000)));
    }

    #[test]
    fn coverage_merges_within_tolerance_and_reports_the_gap() {
        let mut cache = VideoFrameCache::for_raw();
        cache.note_covered(Timestamp(0)..Timestamp(1_000));
        cache.note_covered(Timestamp(1_000 + NO_DATA_THRESHOLD_US)..Timestamp(2_000_000));
        assert_eq!(cache.coverage.len(), 1);
        assert_eq!(cache.coverage[0].end.0, 2_000_000);

        let gap_start = 2_000_000 + NO_DATA_THRESHOLD_US + 1;
        cache.note_covered(Timestamp(gap_start)..Timestamp(3_000_000));
        assert_eq!(cache.coverage.len(), 2);
        let gap = cache
            .first_uncovered(&(Timestamp(0)..Timestamp(3_000_000)))
            .unwrap();
        assert_eq!(gap.start.0, 2_000_000);
        assert_eq!(gap.end.0, gap_start);
    }

    #[test]
    fn fetch_window_clamps_to_db_bounds_and_skips_covered_ranges() {
        let want = clamped_fetch_window(
            Timestamp(1_000_000),
            100_000,
            100_000,
            Timestamp(950_000),
            Timestamp(1_050_000),
        )
        .unwrap();
        assert_eq!(want.start.0, 950_000);
        assert_eq!(want.end.0, 1_050_000);

        assert!(
            clamped_fetch_window(
                Timestamp(1_000),
                10,
                10,
                Timestamp(i64::MAX),
                Timestamp(2_000),
            )
            .is_none()
        );

        let mut cache = VideoFrameCache::for_raw();
        cache.note_covered(want.start..want.end);
        assert!(cache.first_uncovered(&want).is_none());

        let rgba = stream_frame_bytes(Some((640, 480, RawPixelFormat::Rgba8)), false);
        let window = fetch_window_us(rgba, 30.0, 256 * 1024 * 1024);
        assert!(window < FETCH_WINDOW_US);
        let h264 = fetch_window_us(BACKFILL_ASSUMED_H264_FRAME, 30.0, 256 * 1024 * 1024);
        assert_eq!(h264, FETCH_WINDOW_US);
    }
}
