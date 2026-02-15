use bevy::asset::Assets;
use bevy::asset::RenderAssetUsages;
use bevy::ecs::query::QueryData;
use bevy::ecs::system::In;
use bevy::ecs::system::InRef;
use bevy::image::Image;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::TextureDimension;
use bevy::ui::Display;
use bevy::ui::Node;
use bevy::ui::widget::ImageNode;
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, Query, Res, ResMut, World},
    ui::Val,
};
use egui::{self, Color32, Vec2};
use impeller2::types::{OwnedPacket, Timestamp};
use impeller2_bevy::{CommandsExt, CurrentStreamId, PacketGrantR};
use impeller2_wkt::{
    CurrentTimestamp, ErrorResponse, FixedRateMsgStream, FixedRateOp, GetMsgs, MsgBatch,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{self};
use std::time::Instant;

use super::{
    PaneName,
    colors::{ColorExt, get_scheme},
};

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

#[derive(Component)]
pub struct VideoStream {
    pub msg_id: [u8; 2],
    pub msg_name: String,
    pub current_frame: Option<Image>,
    pub size: Vec2,
    pub state: StreamState,
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

/// If the decoder's last position is within this many microseconds of
/// `current_ts` (in either direction), suppress seek retries.  This is
/// wider than `SEQUENTIAL_THRESHOLD_US` to prevent the seek loop where
/// the decoder is "close but not sequential" (e.g. OBS at 30fps with
/// wide frame spacing).  2 seconds.
const DECODER_NEAR_THRESHOLD_US: i64 = 2_000_000;

/// Maximum number of raw H.264 frames to keep in the cache.
/// At ~25 KB/frame (OBS 1080p), 50 000 frames ≈ 1.2 GB, covering ~28 min
/// at 30 fps.  Prevents unbounded memory growth during long live sessions.
const MAX_RAW_FRAMES: usize = 50_000;

/// If the live-tail `FixedRateMsgStream` has not delivered a frame for this
/// many seconds, re-send the backfill and stream requests to recover from
/// a dropped DB connection.
const STREAM_RECOVERY_TIMEOUT_SECS: f32 = 30.0;

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
/// - A one-shot `GetMsgs` backfill (historical data already in the DB)
/// - A continuous `FixedRateMsgStream` live tail (new data arriving in real
///   time from a GStreamer / OBS source)
///
/// Neither source ever touches the decoder directly.  The playback loop reads
/// `CurrentTimestamp`, looks up the cache, and feeds controlled sequences to
/// the decoder thread.
#[derive(Component)]
pub struct VideoFrameCache {
    /// All raw H.264 NAL data, keyed by timestamp.
    raw_frames: BTreeMap<Timestamp, Vec<u8>>,
    /// Decoded RGBA frames (LRU-evicted to `max_frames`).
    decoded_frames: BTreeMap<Timestamp, Image>,
    /// Sorted timestamps of known keyframes (IDR NAL units).
    keyframe_index: Vec<Timestamp>,
    /// Maximum number of decoded frames to retain.
    max_frames: usize,
    /// Maximum number of raw frames to retain.
    max_raw_frames: usize,
    /// Timestamp of the last frame fed to the decoder (for sequential opt).
    last_decoded_ts: Option<Timestamp>,
    /// `true` while a `SeekBatch` is in-flight on the decoder thread.
    seeking: bool,
    /// Generation counter for `SeekBatch` coalescing.
    seek_generation: u64,
    /// Wall-clock time of the last frame received from the live-tail
    /// `FixedRateMsgStream`.  Used to detect DB disconnection and trigger
    /// recovery.  `None` until the first live frame arrives.
    last_stream_activity: Option<Instant>,
}

impl Default for VideoFrameCache {
    fn default() -> Self {
        Self {
            raw_frames: BTreeMap::new(),
            decoded_frames: BTreeMap::new(),
            keyframe_index: Vec::new(),
            max_frames: MAX_CACHED_FRAMES,
            max_raw_frames: MAX_RAW_FRAMES,
            last_decoded_ts: None,
            seeking: false,
            seek_generation: 0,
            last_stream_activity: None,
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

    /// Extract raw frames in `[from, to]` for decoding (cloned out of cache).
    pub fn get_raw_sequence(&self, from: Timestamp, to: Timestamp) -> Vec<(Timestamp, Vec<u8>)> {
        self.raw_frames
            .range(from..=to)
            .map(|(t, d)| (*t, d.clone()))
            .collect()
    }

    /// Get the next raw frame strictly after `ts`.
    pub fn next_raw_frame_after(&self, ts: Timestamp) -> Option<(Timestamp, &Vec<u8>)> {
        self.raw_frames
            .range((std::ops::Bound::Excluded(ts), std::ops::Bound::Unbounded))
            .next()
            .map(|(t, d)| (*t, d))
    }

    /// Insert a raw frame and evict oldest entries if over the limit.
    pub fn insert_raw(&mut self, timestamp: Timestamp, data: Vec<u8>) {
        if is_keyframe(&data) {
            self.record_keyframe(timestamp);
        }
        self.raw_frames.insert(timestamp, data);
        self.evict_raw_if_needed();
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

    fn evict_raw_if_needed(&mut self) {
        while self.raw_frames.len() > self.max_raw_frames {
            if let Some((evicted_ts, _)) = self.raw_frames.pop_first() {
                // Also remove stale keyframe index entries that are now
                // before the earliest raw frame.
                if let Some(&first_raw_ts) = self.raw_frames.keys().next() {
                    self.keyframe_index.retain(|ts| *ts >= first_raw_ts);
                }
                // Also remove decoded frames that are before the evicted
                // raw frame since they can no longer be re-derived.
                while let Some((&decoded_ts, _)) = self.decoded_frames.iter().next() {
                    if decoded_ts <= evicted_ts {
                        self.decoded_frames.pop_first();
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }
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
    tx: flume::Sender<DecoderInput>,
    rx: flume::Receiver<DecoderOutput>,
    width: Arc<AtomicUsize>,
    /// Shared with the decoder thread so it can skip stale seek batches.
    latest_seek_generation: Arc<AtomicU64>,
    _handle: std::thread::JoinHandle<()>,
}

// ---------------------------------------------------------------------------
// Decoder implementations (platform-specific)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
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

impl VideoDecoderHandle {
    /// Send a single raw H.264 frame for sequential decode (warm decoder).
    /// Returns `true` if the frame was successfully enqueued.
    pub fn process_frame(&self, timestamp: Timestamp, frame_data: &[u8]) -> bool {
        self.tx
            .try_send(DecoderInput::Frame(frame_data.to_vec(), timestamp))
            .is_ok()
    }

    /// Send a batch of raw frames for seek-decode (resets the decoder).
    /// Returns `true` if the batch was successfully enqueued.
    /// Also publishes `generation` to the shared atomic so the decoder
    /// thread can skip any older (stale) batches still in the queue.
    pub fn send_seek_batch(&self, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) -> bool {
        self.latest_seek_generation
            .store(generation, atomic::Ordering::Relaxed);
        self.tx
            .try_send(DecoderInput::SeekBatch { frames, generation })
            .is_ok()
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
}

#[derive(SystemParam)]
pub struct VideoStreamWidget<'w, 's> {
    query: Query<'w, 's, WidgetQuery>,
    current_time: Res<'w, CurrentTimestamp>,
    images: ResMut<'w, Assets<Image>>,
    window_settings: Query<'w, 's, &'static bevy_egui::EguiContextSettings>,
}

// ---------------------------------------------------------------------------
// DB request helpers
// ---------------------------------------------------------------------------

/// Start the `FixedRateMsgStream` for live-tail cache population.
///
/// The callback **only** inserts raw frames into the cache and records
/// keyframe timestamps.  It never touches the decoder.
fn send_stream_request(commands: &mut Commands, entity: Entity, msg_id: [u8; 2], stream_id: u64) {
    commands.send_req_reply_raw(
        FixedRateMsgStream {
            msg_id,
            fixed_rate: FixedRateOp {
                stream_id,
                behavior: Default::default(),
            },
        },
        move |InRef(pkt): InRef<OwnedPacket<PacketGrantR>>,
              mut caches: Query<&mut VideoFrameCache>| {
            if let OwnedPacket::Msg(msg_buf) = pkt
                && let Some(timestamp) = msg_buf.timestamp
                && let Ok(mut cache) = caches.get_mut(entity)
            {
                cache.insert_raw(timestamp, msg_buf.buf.to_vec());
                cache.last_stream_activity = Some(Instant::now());
            }
            false
        },
    );
}

/// Maximum number of raw frames to request in a single backfill.
/// Keeps the DB response size manageable.  At ~25 KB/frame (OBS 1080p),
/// 500 frames ≈ 12.5 MB, well within tested limits (10 MB succeeded).
/// The live-tail `FixedRateMsgStream` fills the rest during playback.
const BACKFILL_FRAME_LIMIT: usize = 500;

/// Paginated `GetMsgs` backfill: fetch historical raw data from the DB
/// in small batches of `BACKFILL_FRAME_LIMIT` frames.  When a full batch
/// arrives, the callback immediately requests the next page.  This keeps
/// each individual DB response small (~6-12 MB) while progressively
/// filling the entire raw cache.
fn send_backfill_request(
    commands: &mut Commands,
    entity: Entity,
    msg_id: [u8; 2],
    start_from: Timestamp,
) {
    commands.send_req_reply(
        GetMsgs {
            msg_id,
            range: start_from..Timestamp(i64::MAX),
            limit: Some(BACKFILL_FRAME_LIMIT),
        },
        move |In(result): In<Result<MsgBatch, ErrorResponse>>,
              mut caches: Query<&mut VideoFrameCache>,
              mut cmds: Commands| {
            if let Ok(batch) = result
                && let Ok(mut cache) = caches.get_mut(entity)
            {
                let count = batch.data.len();
                let mut last_ts = start_from;
                for (timestamp, data) in batch.data {
                    last_ts = timestamp;
                    cache.insert_raw(timestamp, data);
                }
                // If we received a full page, there's likely more data.
                // Request the next page starting just after the last frame.
                if count >= BACKFILL_FRAME_LIMIT {
                    send_backfill_request(&mut cmds, entity, msg_id, Timestamp(last_ts.0 + 1));
                }
            }
            true // remove this handler; the continuation (if any) registers a new one
        },
    );
}

// ---------------------------------------------------------------------------
// GPU texture upload helper
// ---------------------------------------------------------------------------

/// Upload a decoded frame to the GPU, reusing the existing texture when
/// the resolution has not changed.
fn upload_frame(
    frame: Image,
    image_node: &mut bevy::prelude::Mut<'_, ImageNode>,
    images: &mut Assets<Image>,
) {
    if let Some(existing) = images.get_mut(&image_node.image) {
        if existing.width() == frame.width()
            && existing.height() == frame.height()
            && let Some(dst) = &mut existing.data
            && let Some(src) = &frame.data
        {
            dst.copy_from_slice(src);
        } else {
            image_node.image = images.add(frame);
        }
    } else {
        image_node.image = images.add(frame);
    }
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
        let mut state = state.get_mut(world);

        let Ok(WidgetQueryItem {
            mut stream,
            decoder,
            mut cache,
            mut image_node,
            mut ui_node,
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
                // Drain is also done by connect_streams, but drain again
                // here so the widget sees the latest decoded frames
                // within the same tick.
                let seek_done = decoder.drain_into_cache(&mut cache);
                if seek_done {
                    cache.seeking = false;
                }

                // --- 2. Pick the best frame to display ---
                // Check at-or-before first (normal case), then at-or-after
                // (handles forward-seek when data only exists ahead).
                if let Some((ts, img)) = cache.decoded_at_or_before(current_ts)
                    && (current_ts.0 - ts.0).abs() <= NO_DATA_THRESHOLD_US
                {
                    stream.current_frame = Some(img.clone());
                } else if let Some((ts, img)) = cache.decoded_at_or_after(current_ts)
                    && (ts.0 - current_ts.0).abs() <= NO_DATA_THRESHOLD_US
                {
                    stream.current_frame = Some(img.clone());
                }

                // --- 3. Feed the decoder from the raw cache ---
                let decoder_near = if let Some(last) = cache.last_decoded_ts {
                    (current_ts.0 - last.0).abs() < DECODER_NEAR_THRESHOLD_US
                } else {
                    false
                };

                // Sequential: decoder is behind current_ts but within the
                // near threshold.  Feed frames one at a time to catch up.
                // Uses the same wide threshold as decoder_near so there's
                // no dead zone where neither sequential nor seek fires.
                let is_sequential = if let Some(last) = cache.last_decoded_ts {
                    current_ts.0 >= last.0 && (current_ts.0 - last.0) < DECODER_NEAR_THRESHOLD_US
                } else {
                    false
                };

                let has_raw = cache.has_raw_data_near(current_ts, NO_DATA_THRESHOLD_US);

                if is_sequential && !cache.seeking {
                    // Sequential playback: decoder state is warm, feed next frame.
                    if let Some((ts, data)) =
                        cache.next_raw_frame_after(cache.last_decoded_ts.unwrap())
                        && ts.0 <= current_ts.0 + DECODER_NEAR_THRESHOLD_US
                    {
                        let data = data.clone();
                        if decoder.process_frame(ts, &data) {
                            cache.last_decoded_ts = Some(ts);
                        }
                    }
                } else if !decoder_near && has_raw {
                    // Jump / scrub / first frame: seek from nearest keyframe.
                    // If no data at-or-before current_ts, seek forward to the
                    // first available keyframe instead.
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

                // Live-stream recovery handled by `connect_streams`.
            }
            StreamState::Error(_) => {}
        }

        // ---------------------------------------------------------------
        // Layout
        // ---------------------------------------------------------------
        let max_rect = ui.max_rect();

        let Ok(egui_settings) = state.window_settings.get(window) else {
            return;
        };

        let scale_factor = egui_settings.scale_factor;
        let viewport_pos = max_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = max_rect.size() * scale_factor;

        let (width, height) = if let Some(image) = state.images.get(&image_node.image) {
            let aspect_ratio = image.height() as f32 / image.width() as f32;
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

                // Upload the current frame to the GPU.
                if let Some(frame) = stream.current_frame.take() {
                    if stream.size == Vec2::ZERO {
                        stream.size = Vec2::new(frame.width() as f32, frame.height() as f32);
                    }
                    upload_frame(frame, &mut image_node, &mut state.images);
                }

                // "No video at this time" overlay when there is no raw data
                // near the current playback position.
                if !cache.has_raw_data_near(current_ts, NO_DATA_THRESHOLD_US) {
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
    stream_id: Res<CurrentStreamId>,
) {
    for (entity, mut stream, decoder, mut cache) in &mut query {
        match &mut stream.state {
            StreamState::WaitingToConnect { frames_waited } => {
                *frames_waited += 1;
                if *frames_waited >= FRAMES_BEFORE_CONNECT {
                    let msg_id = stream.msg_id;
                    send_backfill_request(&mut commands, entity, msg_id, Timestamp(i64::MIN));
                    send_stream_request(&mut commands, entity, msg_id, stream_id.0);
                    stream.state = StreamState::Active;
                }
            }
            StreamState::Active => {
                // Drain decoder output even when the widget is not visible,
                // so seek completions are processed promptly.
                let seek_done = decoder.drain_into_cache(&mut cache);
                if seek_done {
                    cache.seeking = false;
                }

                // Live-stream recovery: if the FixedRateMsgStream was
                // previously delivering frames but has gone silent, re-send
                // both requests.  Reset to None so recovery only fires once
                // — it won't repeat until the stream actually delivers
                // another frame (which sets last_stream_activity back to
                // Some).  This prevents repeated re-fetches during normal
                // paused/idle periods on recorded data.
                if let Some(last) = cache.last_stream_activity
                    && last.elapsed().as_secs_f32() > STREAM_RECOVERY_TIMEOUT_SECS
                {
                    let msg_id = stream.msg_id;
                    send_backfill_request(&mut commands, entity, msg_id, Timestamp(i64::MIN));
                    send_stream_request(&mut commands, entity, msg_id, stream_id.0);
                    cache.last_stream_activity = None;
                }
            }
            StreamState::Error(_) => {}
        }
    }
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
