//! WebCodecs H.264 decoder. `VideoDecoder` is `!Send`, so instances live in a
//! thread-local registry and [`super::VideoDecoderHandle`] only stores the id.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension};
use impeller::types::Timestamp;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::{DecoderOutput, avc1_codec_string, is_keyframe};

thread_local! {
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static DECODERS: RefCell<HashMap<u64, WasmH264Decoder>> = RefCell::new(HashMap::new());
}

struct WasmH264Decoder {
    decoder: web_sys::VideoDecoder,
    _output: Closure<dyn FnMut(web_sys::VideoFrame)>,
    _error: Closure<dyn FnMut(JsValue)>,
    configured: bool,
    image_tx: flume::Sender<DecoderOutput>,
    latest_generation: Arc<AtomicU64>,
    /// RGBA copies still in flight. Seek completion waits so frames land
    /// before `SeekComplete`.
    copies_inflight: Arc<AtomicUsize>,
}

pub(super) fn spawn(
    image_tx: flume::Sender<DecoderOutput>,
    latest_generation: Arc<AtomicU64>,
) -> Option<u64> {
    let logged_error = Arc::new(AtomicBool::new(false));
    let copies_inflight = Arc::new(AtomicUsize::new(0));
    let output_tx = image_tx.clone();
    let output_inflight = copies_inflight.clone();
    let output = Closure::wrap(Box::new(move |frame: web_sys::VideoFrame| {
        let tx = output_tx.clone();
        let inflight = output_inflight.clone();
        inflight.fetch_add(1, Ordering::Relaxed);
        spawn_local(async move {
            deliver_frame(frame, tx, inflight).await;
        });
    }) as Box<dyn FnMut(web_sys::VideoFrame)>);
    let error = Closure::wrap(Box::new(move |err: JsValue| {
        if logged_error.swap(true, Ordering::Relaxed) {
            return;
        }
        bevy::log::warn!("H.264 VideoDecoder error: {}", js_error_message(&err));
    }) as Box<dyn FnMut(JsValue)>);
    let init = web_sys::VideoDecoderInit::new(
        error.as_ref().unchecked_ref(),
        output.as_ref().unchecked_ref(),
    );
    let decoder = match web_sys::VideoDecoder::new(&init) {
        Ok(decoder) => decoder,
        Err(err) => {
            bevy::log::warn!("H.264 VideoDecoder unavailable: {}", js_error_message(&err));
            return None;
        }
    };
    let id = NEXT_ID.with(|next| {
        let id = next.get();
        next.set(id.wrapping_add(1).max(1));
        id
    });
    DECODERS.with(|decoders| {
        decoders.borrow_mut().insert(
            id,
            WasmH264Decoder {
                decoder,
                _output: output,
                _error: error,
                configured: false,
                image_tx,
                latest_generation,
                copies_inflight,
            },
        );
    });
    Some(id)
}

pub(super) fn remove(id: u64) {
    if id == 0 {
        return;
    }
    DECODERS.with(|decoders| {
        decoders.borrow_mut().remove(&id);
    });
}

pub(super) fn process_frame(id: u64, timestamp: Timestamp, packet: &[u8]) -> bool {
    with_decoder(id, |decoder| decoder.decode_packet(timestamp, packet)).unwrap_or(false)
}

pub(super) fn send_seek_batch(id: u64, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) -> bool {
    with_decoder(id, |decoder| decoder.seek(frames, generation)).unwrap_or(false)
}

fn with_decoder<R>(id: u64, f: impl FnOnce(&mut WasmH264Decoder) -> R) -> Option<R> {
    if id == 0 {
        return None;
    }
    DECODERS.with(|decoders| decoders.borrow_mut().get_mut(&id).map(f))
}

impl Drop for WasmH264Decoder {
    fn drop(&mut self) {
        if self.decoder.state() != web_sys::CodecState::Closed {
            let _ = self.decoder.close();
        }
    }
}

impl WasmH264Decoder {
    fn decode_packet(&mut self, timestamp: Timestamp, packet: &[u8]) -> bool {
        if self.decoder.state() == web_sys::CodecState::Closed {
            return false;
        }
        let key = is_keyframe(packet);
        if !self.configured && !key {
            return false;
        }
        if !self.configured && !self.configure(packet) {
            return false;
        }
        self.enqueue(timestamp, packet, key)
    }

    fn configure(&mut self, packet: &[u8]) -> bool {
        let Some(codec) = avc1_codec_string(packet) else {
            return false;
        };
        let config = web_sys::VideoDecoderConfig::new(&codec);
        config.set_optimize_for_latency(true);
        if self.decoder.configure(&config).is_err() {
            return false;
        }
        self.configured = self.decoder.state() == web_sys::CodecState::Configured;
        self.configured
    }

    fn enqueue(&self, timestamp: Timestamp, packet: &[u8], key: bool) -> bool {
        let bytes = js_sys::Uint8Array::from(packet);
        let chunk_type = if key {
            web_sys::EncodedVideoChunkType::Key
        } else {
            web_sys::EncodedVideoChunkType::Delta
        };
        let init = web_sys::EncodedVideoChunkInit::new_with_u8_array(&bytes, 0, chunk_type);
        init.set_timestamp_f64(timestamp.0 as f64);
        let Ok(chunk) = web_sys::EncodedVideoChunk::new(&init) else {
            return false;
        };
        self.decoder.decode(&chunk).is_ok()
    }

    fn seek(&mut self, frames: Vec<(Timestamp, Vec<u8>)>, generation: u64) -> bool {
        if self.latest_generation.load(Ordering::Relaxed) != generation {
            let _ = self
                .image_tx
                .try_send(DecoderOutput::SeekComplete(generation));
            return true;
        }
        if self.decoder.reset().is_err() {
            return false;
        }
        self.configured = false;
        for (timestamp, packet) in &frames {
            if self.latest_generation.load(Ordering::Relaxed) != generation {
                break;
            }
            if !self.decode_packet(*timestamp, packet) && !self.configured {
                return false;
            }
        }
        if !self.configured {
            let _ = self
                .image_tx
                .try_send(DecoderOutput::SeekComplete(generation));
            return true;
        }
        let promise = self.decoder.flush();
        let tx = self.image_tx.clone();
        let latest = self.latest_generation.clone();
        let inflight = self.copies_inflight.clone();
        spawn_local(async move {
            let _ = JsFuture::from(promise).await;
            // `copyTo` finishes after `flush` resolves. Wait until those RGBA
            // buffers are queued so `SeekComplete` cannot overtake them.
            let mut spins = 0;
            while latest.load(Ordering::Relaxed) == generation
                && inflight.load(Ordering::Relaxed) > 0
                && spins < 2000
            {
                yield_to_browser().await;
                spins += 1;
            }
            if latest.load(Ordering::Relaxed) == generation {
                let _ = tx.try_send(DecoderOutput::SeekComplete(generation));
            }
        });
        true
    }
}

async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let scheduled = web_sys::window()
            .and_then(|window| {
                window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                    .ok()
            })
            .is_some();
        if !scheduled {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        }
    });
    let _ = JsFuture::from(promise).await;
}

async fn deliver_frame(
    frame: web_sys::VideoFrame,
    tx: flume::Sender<DecoderOutput>,
    inflight: Arc<AtomicUsize>,
) {
    struct InflightGuard(Arc<AtomicUsize>);
    impl Drop for InflightGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _guard = InflightGuard(inflight);
    let timestamp = Timestamp(frame.timestamp().round() as i64);
    let image = copy_frame_rgba(&frame).await;
    frame.close();
    if let Some(image) = image {
        let _ = tx.try_send(DecoderOutput::Frame(Box::new(image), timestamp));
    }
}

async fn copy_frame_rgba(frame: &web_sys::VideoFrame) -> Option<Image> {
    let width = frame.display_width();
    let height = frame.display_height();
    if width == 0 || height == 0 {
        return None;
    }
    let options = web_sys::VideoFrameCopyToOptions::new();
    options.set_format(web_sys::VideoPixelFormat::Rgba);
    if let Ok(size) = frame.allocation_size_with_options(&options) {
        let dest = js_sys::Uint8Array::new_with_length(size);
        let promise = frame.copy_to_with_u8_array_and_options(&dest, &options);
        if JsFuture::from(promise).await.is_ok() {
            let rgba = dest.to_vec();
            let expected = (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(4);
            if rgba.len() == expected {
                return Some(rgba_image(width, height, rgba));
            }
        } else if COPY_TO_FALLBACK
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            bevy::log::warn!("VideoFrame.copyTo(RGBA) failed; using OffscreenCanvas");
        }
    }
    copy_frame_via_canvas(frame)
}

static COPY_TO_FALLBACK: AtomicBool = AtomicBool::new(false);

fn copy_frame_via_canvas(frame: &web_sys::VideoFrame) -> Option<Image> {
    let width = frame.display_width();
    let height = frame.display_height();
    let canvas = web_sys::OffscreenCanvas::new(width, height).ok()?;
    let ctx = canvas.get_context("2d").ok().flatten()?;
    let ctx = ctx
        .dyn_into::<web_sys::OffscreenCanvasRenderingContext2d>()
        .ok()?;
    ctx.draw_image_with_video_frame(frame, 0.0, 0.0).ok()?;
    let sw = i32::try_from(width).ok()?;
    let sh = i32::try_from(height).ok()?;
    let image = ctx.get_image_data(0, 0, sw, sh).ok()?;
    let rgba = image.data().0;
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba.len() != expected {
        return None;
    }
    Some(rgba_image(width, height, rgba))
}

fn rgba_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        bevy_render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
}

fn js_error_message(err: &JsValue) -> String {
    err.dyn_ref::<web_sys::DomException>()
        .map(|exception| exception.message())
        .or_else(|| err.as_string())
        .unwrap_or_else(|| "unknown".to_string())
}
