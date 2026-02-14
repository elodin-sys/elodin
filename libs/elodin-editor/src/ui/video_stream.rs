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
use egui::{self, Color32, TextureHandle, Vec2};
use impeller2::types::{OwnedPacket, Timestamp};
use impeller2_bevy::{CommandsExt, CurrentStreamId, PacketGrantR};
use impeller2_wkt::{ErrorResponse, FixedRateMsgStream, FixedRateOp, GetMsgs, MsgBatch};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{self};
use std::time::{Duration, Instant};

use super::{
    PaneName,
    colors::{ColorExt, get_scheme},
};

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

#[derive(Component)]
pub struct VideoStream {
    pub msg_id: [u8; 2],
    pub msg_name: String,
    pub current_frame: Option<Image>,
    pub frame_timestamp: Option<Timestamp>,
    pub texture_handle: Option<TextureHandle>,
    pub size: Vec2,
    pub frame_count: usize,
    pub state: StreamState,
    pub last_update: Instant,
}

#[derive(Component)]
pub struct IsTileVisible(pub bool);

impl Default for VideoStream {
    fn default() -> Self {
        Self {
            msg_id: [0, 0],
            msg_name: String::new(),
            current_frame: None,
            texture_handle: None,
            size: Vec2::ZERO,
            frame_count: 0,
            state: StreamState::None,
            last_update: Instant::now(),
            frame_timestamp: None,
        }
    }
}

/// Number of frames to wait before sending the stream request.
/// This prevents blocking during initial schematic loading.
const FRAMES_BEFORE_CONNECT: u32 = 5;

/// Timeout in seconds before considering a stream disconnected.
const STREAM_TIMEOUT_SECS: f32 = 5.0;

/// Delay before attempting to reconnect after disconnect.
const RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// Timeout for the initial connection attempt.
/// If no frames arrive within this window the request is assumed lost
/// (e.g. sent during a TCP reconnection race) and we transition to
/// `Disconnected` so the retry logic re-sends `FixedRateMsgStream`.
const CONNECT_TIMEOUT_SECS: f32 = 30.0;

/// Maximum number of decoded RGBA frames to keep in the cache.
/// At 640x512 RGBA, each frame is ~1.3 MB, so 300 frames ~ 390 MB.
const MAX_CACHED_FRAMES: usize = 300;

/// Timestamp discontinuity threshold (in microseconds) that triggers a seek.
/// If the incoming frame timestamp differs from the last displayed timestamp
/// by more than this amount, we treat it as a scrub/seek event.
/// 500 ms = 500_000 us — generous enough to not trigger on normal jitter.
const SCRUB_THRESHOLD_US: i64 = 500_000;

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

#[derive(Default, Clone)]
pub enum StreamState {
    /// Initial state - waiting to start connection process
    #[default]
    None,
    /// Deferring connection to avoid blocking during initialization
    WaitingToConnect { frames_waited: u32 },
    /// Stream request sent, waiting for data
    Connecting { since: Instant },
    /// Actively receiving video frames
    Streaming,
    /// A seek/scrub is in flight — waiting for `GetMsgs` batch decode to complete.
    Seeking { generation: u64, target: Timestamp },
    /// Stream disconnected, will retry after specified time
    Disconnected { retry_after: Instant },
    /// Unrecoverable error
    Error(String),
}

/// Cached decoded video frames and keyframe index for seek support.
///
/// During normal playback, decoded frames and keyframe positions are recorded.
/// When the user scrubs the timeline, the cache serves previously-decoded
/// frames instantly.  On a cache miss the keyframe index is used to request
/// a batch of raw frames from the DB (via `GetMsgs`), which are then
/// fast-decoded to fill the cache.
#[derive(Component)]
pub struct VideoFrameCache {
    /// Decoded RGBA frames indexed by timestamp.
    decoded_frames: BTreeMap<Timestamp, Image>,
    /// Sorted timestamps of known keyframes (IDR NAL units).
    keyframe_index: Vec<Timestamp>,
    /// Maximum number of decoded frames to retain.
    max_frames: usize,
    /// Last timestamp that was displayed — used for discontinuity detection.
    last_displayed_ts: Option<Timestamp>,
    /// Generation counter for coalescing rapid seek requests.
    seek_generation: u64,
}

impl Default for VideoFrameCache {
    fn default() -> Self {
        Self {
            decoded_frames: BTreeMap::new(),
            keyframe_index: Vec::new(),
            max_frames: MAX_CACHED_FRAMES,
            last_displayed_ts: None,
            seek_generation: 0,
        }
    }
}

impl VideoFrameCache {
    /// Insert a decoded frame into the cache and evict old entries if needed.
    pub fn insert(&mut self, timestamp: Timestamp, image: Image) {
        self.decoded_frames.insert(timestamp, image);
        self.evict_if_needed();
    }

    /// Record a keyframe timestamp (maintains sorted order, deduplicates).
    pub fn record_keyframe(&mut self, timestamp: Timestamp) {
        match self.keyframe_index.binary_search(&timestamp) {
            Ok(_) => {} // already present
            Err(pos) => self.keyframe_index.insert(pos, timestamp),
        }
    }

    /// Find the nearest keyframe at or before `target`.
    /// Returns `None` if no keyframes are known before the target.
    pub fn nearest_keyframe_before(&self, target: Timestamp) -> Option<Timestamp> {
        match self.keyframe_index.binary_search(&target) {
            Ok(i) => Some(self.keyframe_index[i]),
            Err(0) => None,
            Err(i) => Some(self.keyframe_index[i - 1]),
        }
    }

    /// Look up the closest cached decoded frame to `target`.
    pub fn get_nearest(&self, target: Timestamp) -> Option<(Timestamp, &Image)> {
        // Check for exact match first
        if let Some(img) = self.decoded_frames.get(&target) {
            return Some((target, img));
        }
        // Look at the entry just before and just after, pick the closest
        let before = self
            .decoded_frames
            .range(..=target)
            .next_back()
            .map(|(t, img)| (*t, img));
        let after = self
            .decoded_frames
            .range(target..)
            .next()
            .map(|(t, img)| (*t, img));
        match (before, after) {
            (Some((bt, bi)), Some((at, ai))) => {
                if (target.0 - bt.0).abs() <= (at.0 - target.0).abs() {
                    Some((bt, bi))
                } else {
                    Some((at, ai))
                }
            }
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Evict frames furthest from the most recently displayed timestamp.
    fn evict_if_needed(&mut self) {
        while self.decoded_frames.len() > self.max_frames {
            // Remove the oldest frame (lowest timestamp) as a simple strategy.
            // A more sophisticated approach could evict based on distance from
            // last_displayed_ts, but FIFO works well for sequential playback.
            if self.decoded_frames.pop_first().is_none() {
                break;
            }
        }
    }
}

/// Input to the decoder background thread.
pub enum DecoderInput {
    /// Normal frame from `FixedRateMsgStream` playback.
    Frame(Vec<u8>, Timestamp),
    /// Seek batch: reset the decoder and decode all frames in order.
    /// The `generation` is used to match responses and discard stale seeks.
    SeekBatch {
        frames: Vec<(Timestamp, Vec<u8>)>,
        generation: u64,
    },
}

/// Output from the decoder background thread.
pub enum DecoderOutput {
    /// A single decoded frame (from normal playback or a seek batch).
    Frame(Box<Image>, Timestamp),
    /// Signals that all frames in a seek batch have been decoded.
    SeekComplete(u64),
}

#[derive(Component)]
pub struct VideoDecoderHandle {
    tx: flume::Sender<DecoderInput>,
    rx: flume::Receiver<DecoderOutput>,
    width: Arc<AtomicUsize>,
    _handle: std::thread::JoinHandle<()>,
}

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
                // Reset the decoder by creating a fresh instance — openh264
                // has no flush/reset API, but construction is cheap.
                decoder = openh264::decoder::Decoder::new().unwrap();
                for (timestamp, packet) in &frames {
                    if let Some(image) = decode_one_frame(&mut decoder, packet, &mut rgba) {
                        let _ =
                            image_tx.try_send(DecoderOutput::Frame(Box::new(image), *timestamp));
                    }
                }
                let _ = image_tx.try_send(DecoderOutput::SeekComplete(generation));
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
                // Reset by creating a fresh VideoToolbox decoder instance.
                video_toolbox =
                    video_toolbox::VideoToolboxDecoder::new(frame_width.clone()).unwrap();
                frame_count = 0;
                for (timestamp, packet) in &frames {
                    frame_count += 1;
                    if let Some(image) =
                        decode_one_frame_vt(&mut video_toolbox, packet, frame_count)
                    {
                        let _ =
                            image_tx.try_send(DecoderOutput::Frame(Box::new(image), *timestamp));
                    }
                }
                let _ = image_tx.try_send(DecoderOutput::SeekComplete(generation));
            }
        }
    }
}

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
    /// Send a single raw H.264 frame to the decoder thread.
    pub fn process_frame(&mut self, timestamp: Timestamp, frame_data: &[u8]) {
        let _ = self
            .tx
            .try_send(DecoderInput::Frame(frame_data.to_vec(), timestamp));
    }

    /// Send a batch of raw frames for seek-decode (resets the decoder first).
    pub fn send_seek_batch(&self, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) {
        let _ = self
            .tx
            .try_send(DecoderInput::SeekBatch { frames, generation });
    }

    /// Drain all decoded frames from the decoder thread into the stream and cache.
    ///
    /// Returns `true` if a `SeekComplete` signal was received (the seek batch
    /// has been fully decoded).
    pub fn render_frame(&mut self, stream: &mut VideoStream, cache: &mut VideoFrameCache) -> bool {
        let mut seek_completed = false;
        while let Ok(output) = self.rx.try_recv() {
            match output {
                DecoderOutput::Frame(frame, timestamp) => {
                    let frame = *frame;
                    // Cache every decoded frame for future seeks.
                    cache.insert(timestamp, frame.clone());
                    stream.current_frame = Some(frame);
                    stream.frame_timestamp = Some(timestamp);
                    stream.frame_count += 1;
                    if stream.size == Vec2::ZERO
                        && let Some(ref img) = stream.current_frame
                    {
                        stream.size = Vec2::new(img.width() as f32, img.height() as f32);
                    }
                }
                DecoderOutput::SeekComplete(completed_gen) => {
                    // Only acknowledge if this matches the current generation.
                    if completed_gen == cache.seek_generation {
                        seek_completed = true;
                    }
                }
            }
        }
        seek_completed
    }
}

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
    images: ResMut<'w, Assets<Image>>,
    window_settings: Query<'w, 's, &'static bevy_egui::EguiContextSettings>,
}

/// Send a FixedRateMsgStream request to start receiving video frames.
///
/// The callback also records keyframe timestamps in the `VideoFrameCache`
/// so seek operations can identify valid seek points.
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
              mut decoders: Query<&mut VideoDecoderHandle>,
              mut caches: Query<&mut VideoFrameCache>| {
            if let OwnedPacket::Msg(msg_buf) = pkt
                && let Ok(mut decoder) = decoders.get_mut(entity)
                && let Some(timestamp) = msg_buf.timestamp
            {
                // Record keyframe positions for seek support.
                if is_keyframe(&msg_buf.buf)
                    && let Ok(mut cache) = caches.get_mut(entity)
                {
                    cache.record_keyframe(timestamp);
                }
                decoder.process_frame(timestamp, &msg_buf.buf);
            }
            false
        },
    );
}

/// Request a batch of raw H.264 frames from the DB for seek-decode.
///
/// Sends a `GetMsgs` request covering `range` and forwards the resulting
/// `MsgBatch` to the decoder thread as a `SeekBatch`.  The `generation`
/// counter is used to discard stale responses when the user scrubs rapidly.
fn request_seek_batch(
    commands: &mut Commands,
    entity: Entity,
    msg_id: [u8; 2],
    range: std::ops::Range<Timestamp>,
    generation: u64,
) {
    commands.send_req_reply(
        GetMsgs {
            msg_id,
            range,
            limit: None,
        },
        move |In(result): In<Result<MsgBatch, ErrorResponse>>,
              mut decoders: Query<&mut VideoDecoderHandle>,
              mut caches: Query<&mut VideoFrameCache>| {
            if let Ok(batch) = result {
                // Check that this seek hasn't been superseded.
                if let Ok(cache) = caches.get_mut(entity)
                    && cache.seek_generation != generation
                {
                    // Stale seek — discard.
                    return true;
                }
                if let Ok(decoder) = decoders.get_mut(entity) {
                    decoder.send_seek_batch(batch.data, generation);
                }
            }
            true // one-shot request
        },
    );
}

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
            // Reuse the existing GPU texture — just overwrite pixel data.
            // The mutable borrow triggers Bevy change detection so the
            // texture is re-uploaded to the GPU automatically.
            dst.copy_from_slice(src);
        } else {
            // Resolution changed or missing data — must allocate a new texture.
            image_node.image = images.add(frame);
        }
    } else {
        // First frame — no existing texture yet.
        image_node.image = images.add(frame);
    }
}

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
            mut decoder,
            mut cache,
            mut image_node,
            mut ui_node,
        }) = state.query.get_mut(entity)
        else {
            return;
        };

        // State machine for video stream connection
        let stream_id = state.stream_id.0;
        let msg_id = stream.msg_id;

        match &mut stream.state {
            StreamState::None => {
                // Start the deferred connection process
                stream.state = StreamState::WaitingToConnect { frames_waited: 0 };
            }
            StreamState::WaitingToConnect { frames_waited } => {
                // Wait for the specified number of frames before connecting
                *frames_waited += 1;
                if *frames_waited >= FRAMES_BEFORE_CONNECT {
                    stream.state = StreamState::Connecting {
                        since: Instant::now(),
                    };
                    send_stream_request(&mut state.commands, entity, msg_id, stream_id);
                }
            }
            StreamState::Connecting { since } => {
                let since = *since; // copy to release borrow on stream.state
                // Check if we've received any frames (transition to Streaming)
                decoder.render_frame(&mut stream, &mut cache);
                if stream.frame_count > 0 {
                    stream.state = StreamState::Streaming;
                    stream.last_update = Instant::now();
                } else if since.elapsed().as_secs_f32() > CONNECT_TIMEOUT_SECS {
                    // The initial request may have been lost (e.g. sent
                    // during a TCP reconnection race) or the source may be
                    // unavailable.  Transition to Disconnected so the retry
                    // logic re-sends FixedRateMsgStream.
                    //
                    // The timeout is intentionally generous (30 s) to avoid
                    // unnecessary duplicate handlers when the DB handler is
                    // legitimately waiting for a slow source (e.g. OBS
                    // Studio that hasn't connected yet).
                    stream.state = StreamState::Disconnected {
                        retry_after: Instant::now() + RECONNECT_DELAY,
                    };
                }
            }
            StreamState::Streaming => {
                // Actively receiving - render frames and check for timeout
                let prev_frame_count = stream.frame_count;
                let prev_ts = stream.frame_timestamp;
                decoder.render_frame(&mut stream, &mut cache);
                if stream.frame_count > prev_frame_count {
                    stream.last_update = Instant::now();
                }

                // --- Scrub / seek detection ---
                // If the newly arrived frame's timestamp jumped significantly
                // from the previously displayed timestamp, the user has
                // scrubbed the timeline.  We need to seek to the correct
                // position by requesting raw frames starting from the nearest
                // keyframe and fast-decoding them.
                if let (Some(new_ts), Some(old_ts)) = (stream.frame_timestamp, prev_ts) {
                    let delta = (new_ts.0 - old_ts.0).abs();
                    if delta > SCRUB_THRESHOLD_US && stream.frame_count > prev_frame_count {
                        // Try the cache first — if we have a decoded frame
                        // near the target, display it instantly.
                        if let Some((_cached_ts, cached_frame)) = cache.get_nearest(new_ts) {
                            stream.current_frame = Some(cached_frame.clone());
                            cache.last_displayed_ts = Some(new_ts);
                        } else {
                            // Cache miss — initiate a seek via GetMsgs.
                            cache.seek_generation += 1;
                            let generation = cache.seek_generation;
                            let target = new_ts;

                            // Find the nearest keyframe at-or-before the target.
                            // Fall back to a fixed window if no keyframe is known.
                            let seek_start = cache
                                .nearest_keyframe_before(target)
                                .unwrap_or(Timestamp(target.0.saturating_sub(1_000_000)));

                            // Request slightly past the target so get_range
                            // includes the target frame itself.
                            let seek_end = Timestamp(target.0 + 1);

                            request_seek_batch(
                                &mut state.commands,
                                entity,
                                msg_id,
                                seek_start..seek_end,
                                generation,
                            );
                            stream.state = StreamState::Seeking { generation, target };
                        }
                    }
                }

                // Update last displayed timestamp for next-frame comparison.
                if let Some(ts) = stream.frame_timestamp {
                    cache.last_displayed_ts = Some(ts);
                }

                // Check for stream timeout (no frames received for a while)
                if stream.last_update.elapsed().as_secs_f32() > STREAM_TIMEOUT_SECS {
                    stream.state = StreamState::Disconnected {
                        retry_after: Instant::now() + RECONNECT_DELAY,
                    };
                }
            }
            StreamState::Seeking { generation, target } => {
                let generation = *generation;
                let target = *target;

                // Keep draining the decoder — the seek batch frames flow
                // through the same output channel.
                let seek_done = decoder.render_frame(&mut stream, &mut cache);

                // Also check: has the user scrubbed *again* while we wait?
                // If a new frame arrives via FixedRateMsgStream with a very
                // different timestamp, start a new seek.
                if let Some(new_ts) = stream.frame_timestamp {
                    let delta = (new_ts.0 - target.0).abs();
                    if delta > SCRUB_THRESHOLD_US {
                        // Another scrub happened — start a fresh seek.
                        if let Some((_cached_ts, cached_frame)) = cache.get_nearest(new_ts) {
                            stream.current_frame = Some(cached_frame.clone());
                            cache.last_displayed_ts = Some(new_ts);
                            stream.state = StreamState::Streaming;
                        } else {
                            cache.seek_generation += 1;
                            let new_gen = cache.seek_generation;
                            let seek_start = cache
                                .nearest_keyframe_before(new_ts)
                                .unwrap_or(Timestamp(new_ts.0.saturating_sub(1_000_000)));
                            let seek_end = Timestamp(new_ts.0 + 1);
                            request_seek_batch(
                                &mut state.commands,
                                entity,
                                msg_id,
                                seek_start..seek_end,
                                new_gen,
                            );
                            stream.state = StreamState::Seeking {
                                generation: new_gen,
                                target: new_ts,
                            };
                        }
                        // Skip rest of Seeking logic this tick.
                    } else if seek_done && generation == cache.seek_generation {
                        // Seek batch completed for the current generation.
                        cache.last_displayed_ts = stream.frame_timestamp;
                        stream.last_update = Instant::now();
                        stream.state = StreamState::Streaming;
                    }
                } else if seek_done && generation == cache.seek_generation {
                    cache.last_displayed_ts = stream.frame_timestamp;
                    stream.last_update = Instant::now();
                    stream.state = StreamState::Streaming;
                }
            }
            StreamState::Disconnected { retry_after } => {
                // After a DB restart or TCP reconnection the original req_id
                // handler is orphaned — the DB-side task is gone and the
                // editor-side callback will never be invoked on the new
                // connection. We must re-send the FixedRateMsgStream request
                // so the DB starts a fresh streaming task for us.
                if Instant::now() >= *retry_after {
                    stream.frame_count = 0;
                    stream.state = StreamState::Connecting {
                        since: Instant::now(),
                    };
                    send_stream_request(&mut state.commands, entity, msg_id, stream_id);
                }
            }
            StreamState::Error(_) => {
                // Error state - no automatic recovery
            }
        }

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

        match &stream.state {
            StreamState::None => {
                ui.centered_and_justified(|ui| {
                    ui.label("Initializing...");
                });
            }
            StreamState::WaitingToConnect { frames_waited } => {
                // Set decoder width early so frames are decoded at proper resolution
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);
                ui.centered_and_justified(|ui| {
                    let progress =
                        (*frames_waited as f32 / FRAMES_BEFORE_CONNECT as f32 * 100.0) as u32;
                    ui.label(format!("Initializing video stream... {}%", progress));
                });
            }
            StreamState::Connecting { .. } => {
                // Set decoder width so frames are decoded at proper resolution
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);
                ui.centered_and_justified(|ui| {
                    ui.label("Connecting to video stream...");
                });
            }
            StreamState::Streaming | StreamState::Seeking { .. } => {
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);

                if let Some(frame) = stream.current_frame.take() {
                    upload_frame(frame, &mut image_node, &mut state.images);
                }

                // Show "Loss of Signal" only when the DB has genuinely
                // stopped sending data (wall-clock based).  Brief H.264
                // decode failures (e.g. a missed reference frame before
                // the next keyframe) should NOT trigger the overlay — the
                // last good frame is still displayed and the decoder will
                // recover at the next keyframe.
                if stream.last_update.elapsed().as_secs_f32() > STREAM_TIMEOUT_SECS
                    && matches!(stream.state, StreamState::Streaming)
                {
                    ui.painter()
                        .rect_filled(max_rect, 0, Color32::BLACK.opacity(0.75));
                    ui.put(
                        egui::Rect::from_center_size(
                            max_rect.center_top() + egui::vec2(0., 64.0),
                            egui::vec2(max_rect.width(), 20.0),
                        ),
                        egui::Label::new(
                            egui::RichText::new("Loss of Signal")
                                .size(16.0)
                                .color(get_scheme().highlight),
                        ),
                    );
                }
            }
            StreamState::Disconnected { retry_after } => {
                let secs_until_retry = retry_after
                    .saturating_duration_since(Instant::now())
                    .as_secs();
                ui.centered_and_justified(|ui| {
                    ui.colored_label(
                        get_scheme().highlight,
                        format!(
                            "Stream disconnected. Reconnecting in {}s...",
                            secs_until_retry
                        ),
                    );
                });
            }
            StreamState::Error(error) => {
                ui.centered_and_justified(|ui| {
                    ui.colored_label(get_scheme().error, format!("Error: {}", error));
                });
            }
        }
    }
}

pub fn set_visibility(mut query: Query<(&mut Node, &IsTileVisible)>) {
    for (mut ui_node, is_visible) in &mut query {
        if is_visible.0 {
            ui_node.display = Display::Flex;
        } else {
            ui_node.display = Display::None;
        }
    }
}
