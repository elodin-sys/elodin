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

/// If `current_ts` is within this many microseconds *forward* of
/// `last_decoded_ts`, treat it as sequential playback and send a single
/// `DecoderInput::Frame` (no decoder reset).  200 ms.
const SEQUENTIAL_THRESHOLD_US: i64 = 200_000;

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

    /// Get the decoded frame at-or-before `target` (most recent frame that
    /// should be displayed at this playback position).
    pub fn decoded_at_or_before(&self, target: Timestamp) -> Option<(Timestamp, &Image)> {
        self.decoded_frames
            .range(..=target)
            .next_back()
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
            if self.decoded_frames.pop_first().is_none() {
                break;
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
                decoder = openh264::decoder::Decoder::new().unwrap();
                // Decode every frame to build H.264 reference state, but
                // only keep the LAST successfully decoded image.  This avoids
                // flooding the bounded output channel (and the heap) with
                // dozens of intermediate 1080p RGBA frames.
                let mut last_image: Option<(Image, Timestamp)> = None;
                for (timestamp, packet) in &frames {
                    if let Some(image) = decode_one_frame(&mut decoder, packet, &mut rgba) {
                        last_image = Some((image, *timestamp));
                    }
                }
                if let Some((image, ts)) = last_image {
                    // Blocking send — must not be lost.
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
                video_toolbox =
                    video_toolbox::VideoToolboxDecoder::new(frame_width.clone()).unwrap();
                frame_count = 0;
                // Decode every frame to build reference state, keep only
                // the last decoded image to avoid channel/memory congestion.
                let mut last_image: Option<(Image, Timestamp)> = None;
                for (timestamp, packet) in &frames {
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
        let _handle = std::thread::spawn(move || decode_video(frame_width, packet_rx, image_tx));
        VideoDecoderHandle {
            tx: packet_tx,
            rx: image_rx,
            _handle,
            width,
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
    pub fn send_seek_batch(&self, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) -> bool {
        self.tx
            .try_send(DecoderInput::SeekBatch { frames, generation })
            .is_ok()
    }

    /// Drain decoded frames from the decoder thread into the cache.
    /// Returns `true` if a `SeekComplete` was received.
    pub fn drain_into_cache(&self, cache: &mut VideoFrameCache) -> bool {
        let mut seek_completed = false;
        while let Ok(output) = self.rx.try_recv() {
            match output {
                DecoderOutput::Frame(frame, timestamp) => {
                    cache.insert_decoded(timestamp, *frame);
                    cache.last_decoded_ts = Some(timestamp);
                }
                DecoderOutput::SeekComplete(_) => {
                    seek_completed = true;
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
    commands: Commands<'w, 's>,
    stream_id: Res<'w, CurrentStreamId>,
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

/// One-shot `GetMsgs` backfill: fetch all historical raw data from the DB.
fn send_backfill_request(commands: &mut Commands, entity: Entity, msg_id: [u8; 2]) {
    commands.send_req_reply(
        GetMsgs {
            msg_id,
            range: Timestamp(i64::MIN)..Timestamp(i64::MAX),
            limit: None,
        },
        move |In(result): In<Result<MsgBatch, ErrorResponse>>,
              mut caches: Query<&mut VideoFrameCache>| {
            if let Ok(batch) = result
                && let Ok(mut cache) = caches.get_mut(entity)
            {
                for (timestamp, data) in batch.data {
                    cache.insert_raw(timestamp, data);
                }
            }
            true // one-shot
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

        let stream_id = state.stream_id.0;
        let msg_id = stream.msg_id;
        let current_ts = state.current_time.0;

        // ---------------------------------------------------------------
        // State machine
        // ---------------------------------------------------------------
        match &mut stream.state {
            StreamState::WaitingToConnect { frames_waited } => {
                *frames_waited += 1;
                if *frames_waited >= FRAMES_BEFORE_CONNECT {
                    // Fire both requests and transition immediately.
                    send_backfill_request(&mut state.commands, entity, msg_id);
                    send_stream_request(&mut state.commands, entity, msg_id, stream_id);
                    stream.state = StreamState::Active;
                }
            }
            StreamState::Active => {
                // --- 1. Drain decoder output into decoded cache ---
                let seek_done = decoder.drain_into_cache(&mut cache);
                if seek_done {
                    cache.seeking = false;
                }

                // --- 2. Pick the best frame to display ---
                // First check the decoded cache for a frame at-or-before
                // the current playback position.
                if let Some((ts, img)) = cache.decoded_at_or_before(current_ts)
                    && (current_ts.0 - ts.0).abs() <= NO_DATA_THRESHOLD_US
                {
                    stream.current_frame = Some(img.clone());
                }

                // --- 3. Feed the decoder from the raw cache ---
                if !cache.seeking {
                    let is_sequential = if let Some(last) = cache.last_decoded_ts {
                        current_ts.0 > last.0 && (current_ts.0 - last.0) < SEQUENTIAL_THRESHOLD_US
                    } else {
                        false
                    };

                    if is_sequential {
                        // Sequential playback: send the next raw frame after
                        // last_decoded_ts.  Decoder state is warm.
                        if let Some((ts, data)) =
                            cache.next_raw_frame_after(cache.last_decoded_ts.unwrap())
                            && ts.0 <= current_ts.0 + SEQUENTIAL_THRESHOLD_US
                        {
                            let data = data.clone();
                            if decoder.process_frame(ts, &data) {
                                cache.last_decoded_ts = Some(ts);
                            }
                        }
                    } else if cache.has_raw_data_near(current_ts, NO_DATA_THRESHOLD_US) {
                        // Jump / scrub / first frame: need a full seek from
                        // the nearest keyframe.  We seek even when `displayed`
                        // is true (stale decoded frame from a previous visit)
                        // because the decoder must be repositioned for
                        // sequential playback to resume from the new position.
                        let seek_start = cache
                            .nearest_keyframe_before(current_ts)
                            .unwrap_or(Timestamp(current_ts.0.saturating_sub(1_000_000)));
                        let seek_end = current_ts;

                        let frames = cache.get_raw_sequence(seek_start, seek_end);
                        if !frames.is_empty() {
                            cache.seek_generation += 1;
                            if decoder.send_seek_batch(frames, cache.seek_generation) {
                                cache.seeking = true;
                            }
                        }
                    }
                }

                // --- 4. Live-stream recovery ---
                // If the FixedRateMsgStream was previously delivering frames
                // but has gone silent (e.g. DB connection dropped and
                // reconnected), re-send both requests to recover.
                if let Some(last) = cache.last_stream_activity
                    && last.elapsed().as_secs_f32() > STREAM_RECOVERY_TIMEOUT_SECS
                {
                    send_backfill_request(&mut state.commands, entity, msg_id);
                    send_stream_request(&mut state.commands, entity, msg_id, stream_id);
                    cache.last_stream_activity = Some(Instant::now());
                }
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
// Visibility system (unchanged)
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
