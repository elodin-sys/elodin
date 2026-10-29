//! One [`VideoStream`] component per tile. The decoder is `!Send`, so it lives
//! in a thread-local registry and the component only stores the id.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::image::Image;
use bevy::log::error;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::decoder::Decoder;
use crate::error::Error;
use crate::nal;
use crate::perf_now;
use crate::stats::{Snapshot, Stats};

const MAX_DECODE_QUEUE: u32 = 16;
const MAX_READY_PLAY: usize = 6;
const SEEK_BURST: usize = 64;

pub struct WebcodecsVideoPlugin;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum UploadMode {
    #[default]
    External,
    CopyTo,
}

impl UploadMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::External => "external",
            Self::CopyTo => "copyTo",
        }
    }
}

#[derive(Component)]
pub struct VideoStream {
    id: u64,
    pub upload: UploadMode,
    pub image: Option<Handle<Image>>,
    pub size: Option<(u32, u32)>,
    pub snapshot: Snapshot,
    pub last_presented: Option<i64>,
}

impl VideoStream {
    pub fn new() -> Self {
        let id = NEXT_ID.with(|next| {
            let id = next.get();
            next.set(id.wrapping_add(1).max(1));
            id
        });
        let (decoder, note) = match Decoder::new("") {
            Ok(decoder) => (Some(decoder), None),
            Err(err) => (None, Some(err)),
        };
        REGISTRY.with(|registry| {
            registry.borrow_mut().insert(
                id,
                Slot {
                    decoder,
                    upload: UploadMode::External,
                    target_us: 0,
                    playing: false,
                    pending: None,
                    submitted: None,
                    copy_inflight: false,
                    copy_bytes: None,
                    copy_ms: 0.0,
                    image: None,
                    size: None,
                    seek: None,
                    seen_key: false,
                    last_pts: None,
                    stats: Stats::new(),
                    note,
                    last_upload_ms: 0.0,
                    upload_sum: 0.0,
                    upload_n: 0,
                    last_presented: None,
                },
            );
        });
        Self {
            id,
            upload: UploadMode::External,
            image: None,
            size: None,
            snapshot: Snapshot::default(),
            last_presented: None,
        }
    }

    pub fn with_upload(mut self, mode: UploadMode) -> Self {
        self.set_upload(mode);
        self
    }

    pub fn set_upload(&mut self, mode: UploadMode) {
        self.upload = mode;
        with_slot(self.id, |slot| {
            if mode == UploadMode::CopyTo {
                close_gpu_frames(slot);
            } else {
                slot.copy_bytes = None;
            }
            slot.upload = mode;
        });
    }

    /// Sequential decode. Returns false before the first keyframe, while a
    /// seek is in flight, or when the decoder cannot take another sample yet.
    pub fn push(&mut self, pts_us: i64, annexb: &[u8]) -> bool {
        let key = nal::is_keyframe(annexb);
        with_slot(self.id, |slot| {
            let Some(decoder) = slot.decoder.as_mut() else {
                return false;
            };
            if slot.seek.is_some() || decoder.is_blocked() {
                return false;
            }
            if !slot.seen_key && !key {
                return false;
            }
            if decoder.needs_keyframe() && !key {
                return false;
            }
            if decoder.decode_queue_size() >= MAX_DECODE_QUEUE {
                return false;
            }
            if decoder.open_frames() >= MAX_READY_PLAY {
                return false;
            }
            let duration_us = duration_from(slot.last_pts, pts_us);
            if decoder.submit(pts_us, annexb, key, duration_us).is_err() {
                return false;
            }
            slot.seen_key = true;
            slot.last_pts = Some(pts_us);
            true
        })
        .unwrap_or(false)
    }

    /// Decode `frames` from a keyframe through the frame to show. Only the
    /// last timestamp is displayed. Feeding continues across frames so a long
    /// GOP does not fill the decode queue in one tick.
    pub fn seek(&mut self, frames: Vec<(i64, Vec<u8>)>) {
        let target_pts = frames.last().map(|(pts, _)| *pts).unwrap_or(0);
        self.seek_through(frames, target_pts);
    }

    /// Like [`seek`](Self::seek), but `frames` may include one sample after
    /// `target_pts`. High-profile streams hold a frame until the next sample
    /// arrives, so a one-frame seek would otherwise never paint.
    pub fn seek_through(&mut self, frames: Vec<(i64, Vec<u8>)>, target_pts: i64) {
        if frames.is_empty() {
            return;
        }
        let last_pts = frames.last().map(|(pts, _)| *pts);
        let samples = seek_samples(frames);
        with_slot(self.id, |slot| {
            let Some(decoder) = slot.decoder.as_mut() else {
                return;
            };
            if let Err(err) = decoder.reset() {
                slot.note = Some(err);
                return;
            }
            decoder.set_discard_before(Some(target_pts));
            close_gpu_frames(slot);
            slot.stats.begin_seek(perf_now());
            slot.seek = Some(PendingSeek {
                frames: samples,
                next: 0,
                target_pts,
            });
            slot.seen_key = true;
            slot.last_pts = last_pts;
        });
    }

    pub fn set_time(&mut self, pts_us: i64) {
        with_slot(self.id, |slot| slot.target_us = pts_us);
    }

    /// Safari only flushes a paused decoder. Playback leaves the queue warm.
    pub fn set_playing(&mut self, playing: bool) {
        with_slot(self.id, |slot| slot.playing = playing);
    }

    pub fn is_seeking(&self) -> bool {
        with_slot(self.id, |slot| slot.seek.is_some()).unwrap_or(false)
    }

    /// `flush` ran, so the next sample has to be a keyframe.
    pub fn needs_keyframe(&self) -> bool {
        with_slot(self.id, |slot| {
            slot.decoder
                .as_ref()
                .is_some_and(|decoder| decoder.needs_keyframe())
        })
        .unwrap_or(false)
    }

    pub fn fail(&mut self, err: Error) {
        self.snapshot.status = err.to_string();
        with_slot(self.id, |slot| slot.note = Some(err));
    }
}

impl Default for VideoStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VideoStream {
    fn drop(&mut self) {
        REGISTRY.with(|registry| {
            if let Some(mut slot) = registry.borrow_mut().remove(&self.id) {
                close_gpu_frames(&mut slot);
                slot.seek = None;
            }
        });
    }
}

struct SeekSample {
    pts_us: i64,
    data: Vec<u8>,
    key: bool,
    duration_us: f64,
}

struct PendingSeek {
    frames: Vec<SeekSample>,
    next: usize,
    target_pts: i64,
}

struct Slot {
    decoder: Option<Decoder>,
    upload: UploadMode,
    target_us: i64,
    playing: bool,
    pending: Option<web_sys::VideoFrame>,
    submitted: Option<web_sys::VideoFrame>,
    copy_inflight: bool,
    copy_bytes: Option<Vec<u8>>,
    copy_ms: f64,
    image: Option<Handle<Image>>,
    size: Option<(u32, u32)>,
    seek: Option<PendingSeek>,
    seen_key: bool,
    last_pts: Option<i64>,
    stats: Stats,
    note: Option<Error>,
    last_upload_ms: f64,
    upload_sum: f64,
    upload_n: u64,
    last_presented: Option<i64>,
}

thread_local! {
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static REGISTRY: RefCell<HashMap<u64, Slot>> = RefCell::new(HashMap::new());
}

impl Plugin for WebcodecsVideoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, drive);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            error!("WebcodecsVideoPlugin needs the RenderApp; add DefaultPlugins first");
            return;
        };
        render_app.add_systems(
            Render,
            upload_external.in_set(RenderSystems::PrepareResources),
        );
    }
}

fn drive(mut images: ResMut<Assets<Image>>, mut streams: Query<&mut VideoStream>) {
    for mut stream in &mut streams {
        let id = stream.id;
        let published = REGISTRY.with(|registry| {
            let mut registry = registry.borrow_mut();
            let Some(slot) = registry.get_mut(&id) else {
                return None;
            };
            feed_seek(slot);
            if !slot.playing && seek_submitted(slot) {
                if let Some(decoder) = slot.decoder.as_ref() {
                    let _ = decoder.flush();
                }
            }
            apply_copy(slot, &mut images);
            present(id, slot, &mut images);
            Some((publish(slot), slot.last_presented))
        });
        if let Some(((image, size, snapshot), last_presented)) = published {
            stream.last_presented = last_presented;
            if let Some(image) = image {
                stream.image = Some(image);
            }
            if let Some(size) = size {
                stream.size = Some(size);
            }
            stream.snapshot = snapshot;
            stream.upload = stream.snapshot.renderer.as_str().into_upload();
        }
    }
}

trait UploadLabel {
    fn into_upload(self) -> UploadMode;
}

impl UploadLabel for &str {
    fn into_upload(self) -> UploadMode {
        if self == UploadMode::CopyTo.label() {
            UploadMode::CopyTo
        } else {
            UploadMode::External
        }
    }
}

fn feed_seek(slot: &mut Slot) {
    let mut submitted = 0;
    while submitted < SEEK_BURST {
        let blocked = slot.decoder.as_ref().is_none_or(|decoder| {
            decoder.is_blocked() || decoder.decode_queue_size() >= MAX_DECODE_QUEUE
        });
        if blocked {
            return;
        }
        let sample = {
            let Some(seek) = slot.seek.as_mut() else {
                return;
            };
            if seek.next >= seek.frames.len() {
                return;
            }
            let sample = &mut seek.frames[seek.next];
            let pts_us = sample.pts_us;
            let key = sample.key;
            let duration_us = sample.duration_us;
            let data = std::mem::take(&mut sample.data);
            seek.next += 1;
            (pts_us, key, duration_us, data)
        };
        let Some(decoder) = slot.decoder.as_mut() else {
            return;
        };
        let (pts_us, key, duration_us, data) = sample;
        if let Err(err) = decoder.submit(pts_us, &data, key, duration_us) {
            slot.note = Some(err);
            return;
        }
        submitted += 1;
    }
}

fn seek_submitted(slot: &Slot) -> bool {
    slot.seek
        .as_ref()
        .is_some_and(|seek| seek.next >= seek.frames.len())
}

fn apply_copy(slot: &mut Slot, images: &mut Assets<Image>) {
    if slot.upload != UploadMode::CopyTo {
        slot.copy_bytes = None;
        return;
    }
    let Some(bytes) = slot.copy_bytes.take() else {
        return;
    };
    let ms = slot.copy_ms;
    let Some(handle) = slot.image.as_ref() else {
        slot.copy_bytes = Some(bytes);
        slot.copy_ms = ms;
        return;
    };
    if let Some(mut image) = images.get_mut(handle) {
        image.data = Some(bytes);
        note_upload(slot, ms);
    } else {
        slot.copy_bytes = Some(bytes);
        slot.copy_ms = ms;
    }
}

fn present(id: u64, slot: &mut Slot, images: &mut Assets<Image>) {
    let copy_busy = slot.upload == UploadMode::CopyTo && slot.copy_inflight;
    if copy_busy {
        return;
    }
    let Some((pts_us, frame)) = take_frame(slot) else {
        return;
    };
    slot.last_presented = Some(pts_us);
    ensure_image(slot, &frame, images);
    match slot.upload {
        UploadMode::External => {
            if let Some(previous) = slot.pending.replace(frame) {
                previous.close();
            }
        }
        UploadMode::CopyTo => {
            slot.copy_inflight = true;
            queue_copy(id, frame);
        }
    }
    slot.stats.note_render(slot.last_upload_ms);
}

fn take_frame(slot: &mut Slot) -> Option<(i64, web_sys::VideoFrame)> {
    if let Some(target) = slot.seek.as_ref().map(|seek| seek.target_pts) {
        let frame = slot.decoder.as_mut()?.pop_pts(target)?;
        slot.seek = None;
        slot.decoder.as_mut()?.set_discard_before(None);
        slot.stats.finish_seek(perf_now());
        return Some((target, frame));
    }
    slot.decoder
        .as_mut()?
        .pop_upto(slot.target_us)
        .map(|popped| (popped.pts_us, popped.frame))
}

fn ensure_image(slot: &mut Slot, frame: &web_sys::VideoFrame, images: &mut Assets<Image>) {
    if slot.image.is_some() {
        return;
    }
    let width = frame.display_width();
    let height = frame.display_height();
    if width == 0 || height == 0 {
        return;
    }
    slot.size = Some((width, height));
    slot.image = Some(images.add(video_image(width, height)));
}

fn publish(slot: &mut Slot) -> (Option<Handle<Image>>, Option<(u32, u32)>, Snapshot) {
    let decoded = slot
        .decoder
        .as_ref()
        .map(|decoder| decoder.decoded())
        .unwrap_or(0);
    slot.stats.observe(perf_now(), decoded);
    let (queue, open, dropped) = slot
        .decoder
        .as_ref()
        .map(|decoder| {
            (
                decoder.decode_queue_size(),
                decoder.open_frames(),
                decoder.dropped(),
            )
        })
        .unwrap_or((0, 0, 0));
    let (width, height) = slot.size.unwrap_or((0, 0));
    let mut snapshot = slot.stats.snapshot(
        "",
        width,
        height,
        0,
        None,
        slot.target_us,
        slot.playing,
        slot.upload.label(),
        queue,
        open,
        dropped,
    );
    snapshot.status = slot
        .decoder
        .as_ref()
        .and_then(|decoder| decoder.error())
        .or_else(|| slot.note.clone())
        .map(|err| err.to_string())
        .unwrap_or_else(|| {
            if slot.decoder.is_none() {
                Error::DecoderUnavailable.to_string()
            } else if slot.image.is_some() {
                "ready".to_string()
            } else {
                "loading".to_string()
            }
        });
    snapshot.last_render_ms = slot.last_upload_ms;
    snapshot.avg_render_ms = if slot.upload_n == 0 {
        0.0
    } else {
        slot.upload_sum / slot.upload_n as f64
    };
    (slot.image.clone(), slot.size, snapshot)
}

fn upload_external(queue: Res<RenderQueue>, gpu_images: Res<RenderAssets<GpuImage>>) {
    let pending: Vec<(u64, web_sys::VideoFrame)> = REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .iter_mut()
            .filter_map(|(id, slot)| slot.pending.take().map(|frame| (*id, frame)))
            .collect()
    });
    for (id, frame) in pending {
        let handle = REGISTRY.with(|registry| {
            registry
                .borrow()
                .get(&id)
                .and_then(|slot| slot.image.clone())
        });
        let Some(handle) = handle else {
            restore(id, frame, Error::NoVideoTexture);
            continue;
        };
        let Some(gpu) = gpu_images.get(&handle) else {
            restore(id, frame, Error::NoGpuImage);
            continue;
        };
        let started = perf_now();
        let source = wgpu::CopyExternalImageSourceInfo {
            source: wgpu::ExternalImageSource::VideoFrame(frame),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        };
        queue.copy_external_image_to_texture(
            &source,
            wgpu::CopyExternalImageDestInfo {
                texture: &*gpu.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            gpu.texture_descriptor.size,
        );
        let wgpu::ExternalImageSource::VideoFrame(frame) = source.source else {
            continue;
        };
        let ms = perf_now() - started;
        REGISTRY.with(|registry| {
            let mut registry = registry.borrow_mut();
            let Some(slot) = registry.get_mut(&id) else {
                frame.close();
                return;
            };
            if let Some(previous) = slot.submitted.replace(frame) {
                previous.close();
            }
            if matches!(
                slot.note,
                Some(Error::NoVideoTexture | Error::NoGpuImage)
            ) {
                slot.note = None;
            }
            note_upload(slot, ms);
        });
    }
}

fn restore(id: u64, frame: web_sys::VideoFrame, note: Error) {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let Some(slot) = registry.get_mut(&id) else {
            frame.close();
            return;
        };
        slot.pending = Some(frame);
        slot.note = Some(note);
    });
}

fn queue_copy(id: u64, frame: web_sys::VideoFrame) {
    wasm_bindgen_futures::spawn_local(async move {
        let started = perf_now();
        let copied = copy_rgba(&frame).await;
        let ms = perf_now() - started;
        frame.close();
        REGISTRY.with(|registry| {
            let mut registry = registry.borrow_mut();
            let Some(slot) = registry.get_mut(&id) else {
                return;
            };
            slot.copy_inflight = false;
            match copied {
                Ok(bytes) => {
                    slot.copy_bytes = Some(bytes);
                    slot.copy_ms = ms;
                }
                Err(err) => slot.note = Some(Error::from(err)),
            }
        });
    });
}

async fn copy_rgba(frame: &web_sys::VideoFrame) -> Result<Vec<u8>, JsValue> {
    let options = web_sys::VideoFrameCopyToOptions::new();
    options.set_format(web_sys::VideoPixelFormat::Rgba);
    let size = frame.allocation_size_with_options(&options)?;
    let pixels = js_sys::Uint8Array::new_with_length(size);
    JsFuture::from(frame.copy_to_with_u8_array_and_options(&pixels, &options)).await?;
    let mut bytes = vec![0; size as usize];
    pixels.copy_to(&mut bytes);
    Ok(bytes)
}

fn video_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    image
}

fn note_upload(slot: &mut Slot, ms: f64) {
    slot.last_upload_ms = ms;
    slot.upload_sum += ms;
    slot.upload_n += 1;
}

fn close_gpu_frames(slot: &mut Slot) {
    if let Some(frame) = slot.pending.take() {
        frame.close();
    }
    if let Some(frame) = slot.submitted.take() {
        frame.close();
    }
}

fn with_slot<T>(id: u64, f: impl FnOnce(&mut Slot) -> T) -> Option<T> {
    REGISTRY.with(|registry| registry.borrow_mut().get_mut(&id).map(f))
}

fn duration_from(previous: Option<i64>, pts_us: i64) -> f64 {
    previous
        .map(|prev| (pts_us - prev).max(1) as f64)
        .unwrap_or(1_000_000.0 / 60.0)
}

fn seek_samples(frames: Vec<(i64, Vec<u8>)>) -> Vec<SeekSample> {
    let pts: Vec<i64> = frames.iter().map(|(pts, _)| *pts).collect();
    frames
        .into_iter()
        .enumerate()
        .map(|(index, (pts_us, data))| {
            let duration_us = pts
                .get(index + 1)
                .map(|next| (*next - pts_us).max(1) as f64)
                .unwrap_or_else(|| duration_from(pts.get(index.wrapping_sub(1)).copied(), pts_us));
            let key = nal::is_keyframe(&data);
            SeekSample {
                pts_us,
                data,
                key,
                duration_us,
            }
        })
        .collect()
}

#[cfg(feature = "egui")]
impl VideoStream {
    /// Aspect-fit this stream inside an egui rect.
    pub fn fit_rect(&self, ui_rect: egui::Rect) -> egui::Rect {
        let (width, height) = self.size.unwrap_or((0, 0));
        fit_rect(width, height, ui_rect)
    }

    /// Draw the stream's texture inside `rect`. The picture is letterboxed.
    pub fn paint(
        &self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        textures: &mut bevy_egui::EguiUserTextures,
    ) {
        ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
        let Some(handle) = &self.image else {
            let status = if self.snapshot.status.is_empty() {
                "loading"
            } else {
                self.snapshot.status.as_str()
            };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                status,
                egui::TextStyle::Body.resolve(ui.style()),
                ui.visuals().text_color(),
            );
            return;
        };
        let id = textures.add_image(bevy_egui::EguiTextureHandle::Weak(handle.id()));
        let fitted = self.fit_rect(rect);
        ui.painter().image(
            id,
            fitted,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }
}

#[cfg(feature = "egui")]
pub fn fit_rect(width: u32, height: u32, viewport: egui::Rect) -> egui::Rect {
    let size = viewport.size();
    if width == 0 || height == 0 || size.x <= 0.0 || size.y <= 0.0 {
        return viewport;
    }
    let aspect = height as f32 / width as f32;
    let fitted_height = size.x * aspect;
    let (width, height) = if fitted_height > size.y {
        (size.y / aspect, size.y)
    } else {
        (size.x, fitted_height)
    };
    egui::Rect::from_center_size(viewport.center(), egui::vec2(width, height))
}
