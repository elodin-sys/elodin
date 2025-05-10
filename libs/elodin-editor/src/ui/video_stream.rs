use std::sync::Arc;
use std::sync::atomic::{self, AtomicU32};
use std::time::{Duration, Instant};

use bevy::ecs::system::InRef;
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, Query, World},
};
use egui::{self, Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use ffmpeg_next::frame::Video;
use ffmpeg_next::{Packet, codec, decoder};
use impeller2::types::OwnedPacket;
use impeller2_bevy::{CommandsExt, PacketGrantR};
use impeller2_wkt::MsgStream;

#[derive(Clone)]
pub struct VideoStreamPane {
    pub entity: Entity,
    pub label: String,
}

#[derive(Component)]
pub struct VideoStream {
    pub msg_id: [u8; 2],
    pub current_frame: Option<ColorImage>,
    pub texture_handle: Option<TextureHandle>,
    pub size: Vec2,
    pub frame_count: usize,
    pub state: StreamState,
    pub last_update: Instant,
}

impl Default for VideoStream {
    fn default() -> Self {
        Self {
            msg_id: [0, 0],
            current_frame: None,
            texture_handle: None,
            size: Vec2::ZERO,
            frame_count: 0,
            state: StreamState::None,
            last_update: Instant::now(),
        }
    }
}

#[derive(Default, Clone)]
pub enum StreamState {
    #[default]
    None,
    Streaming,
    Error(String),
}

#[derive(Component)]
pub struct VideoDecoderHandle {
    tx: flume::Sender<Packet>,
    rx: flume::Receiver<ColorImage>,
    width: Arc<AtomicU32>,
    _handle: std::thread::JoinHandle<()>,
}

impl Default for VideoDecoderHandle {
    fn default() -> Self {
        let (packet_tx, packet_rx) = flume::unbounded();
        let (image_tx, image_rx) = flume::bounded(8);
        let width = Arc::new(AtomicU32::new(0));
        let frame_width = width.clone();
        let _handle = std::thread::spawn(move || {
            let codec = decoder::find(codec::Id::H264).unwrap();
            let mut decoder = codec::context::Context::new_with_codec(codec)
                .decoder()
                .video()
                .unwrap();
            while let Ok(packet) = packet_rx.recv() {
                let _ = decoder.send_packet(&packet);
                while let Some(image) =
                    get_frame(&mut decoder, frame_width.load(atomic::Ordering::Relaxed))
                {
                    let _ = image_tx.send(image);
                }
            }
        });
        VideoDecoderHandle {
            tx: packet_tx,
            rx: image_rx,
            _handle,
            width,
        }
    }
}

impl VideoDecoderHandle {
    pub fn process_frame(&mut self, frame_data: &[u8]) {
        let _ = self.tx.send(Packet::copy(frame_data));
    }

    pub fn render_frame(&mut self, stream: &mut VideoStream) {
        while let Ok(frame) = self.rx.try_recv() {
            stream.current_frame = Some(frame);
            stream.frame_count += 1;
            if stream.size == Vec2::ZERO && stream.current_frame.is_some() {
                if let Some(ref img) = stream.current_frame {
                    stream.size = Vec2::new(img.width() as f32, img.height() as f32);
                }
            }
        }
    }
}

fn get_frame(decoder: &mut decoder::Video, desired_width: u32) -> Option<ColorImage> {
    use ffmpeg_next::format::Pixel;
    use ffmpeg_next::software::scaling::{context::Context as ScalingContext, flag::Flags};
    let mut frame = Video::empty();
    decoder.receive_frame(&mut frame).ok()?;
    let (og_width, og_height) = (frame.width() as u32, frame.height() as u32);
    let aspect_ratio = frame.height() as f64 / frame.width() as f64;
    let width = (frame.width() as f64).min(desired_width as f64);
    let height = width * aspect_ratio;

    let rgb_frame = if frame.format() != Pixel::RGB24 {
        let mut rgb = Video::new(Pixel::RGB24, width as u32, height as u32);

        let mut scaler = ScalingContext::get(
            frame.format(),
            og_width,
            og_height,
            Pixel::RGB24,
            width as u32,
            height as u32,
            Flags::FAST_BILINEAR,
        )
        .ok()?;

        scaler.run(&frame, &mut rgb).ok()?;

        rgb
    } else {
        frame
    };
    Some(video_frame_to_image(rgb_frame))
}

#[derive(SystemParam)]
pub struct VideoStreamWidget<'w, 's> {
    streams: Query<'w, 's, &'static mut VideoStream>,
    decoders: Query<'w, 's, &'static mut VideoDecoderHandle>,
    commands: Commands<'w, 's>,
}

impl super::widgets::WidgetSystem for VideoStreamWidget<'_, '_> {
    type Args = VideoStreamPane;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        VideoStreamPane { entity, .. }: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut stream) = state.streams.get_mut(entity) else {
            return;
        };

        let Ok(mut decoder) = state.decoders.get_mut(entity) else {
            return;
        };

        // Check if it's time to process a new frame
        if stream.last_update.elapsed() > Duration::from_millis(16) {
            if let StreamState::Streaming = stream.state {
                decoder.render_frame(&mut stream);
                stream.last_update = Instant::now();
            }
        }

        if let StreamState::None = stream.state {
            stream.state = StreamState::Streaming;
            let entity = entity;

            state.commands.send_req_reply_raw(
                MsgStream {
                    msg_id: stream.msg_id,
                },
                move |InRef(res): InRef<OwnedPacket<PacketGrantR>>,
                      mut decoders: Query<&mut VideoDecoderHandle>| {
                    match res {
                        OwnedPacket::Msg(msg_buf) => {
                            if let Ok(mut decoder) = decoders.get_mut(entity) {
                                decoder.process_frame(&msg_buf.buf);
                            }
                        }
                        _ => {}
                    };
                    false
                },
            );
        }

        let available_size = ui.available_size();

        match &stream.state {
            StreamState::None => {
                ui.centered_and_justified(|ui| {
                    ui.label("No video stream. Please send a valid H264 stream");
                });
            }
            StreamState::Streaming => {
                let aspect_ratio = if stream.size.y > 0.0 {
                    stream.size.x / stream.size.y
                } else {
                    16.0 / 9.0
                };
                let width = (available_size.x).min(available_size.y * aspect_ratio);
                let height = width / aspect_ratio;
                let display_size = Vec2::new(width, height);
                decoder.width.store(width as u32, atomic::Ordering::Relaxed);

                if let Some(frame) = stream.current_frame.take() {
                    if stream.texture_handle.is_none() {
                        let texture_handle = ui.ctx().load_texture(
                            format!("video_stream_{}", entity.index()),
                            frame,
                            TextureOptions::default(),
                        );
                        stream.texture_handle = Some(texture_handle);
                    } else if let Some(texture) = &mut stream.texture_handle {
                        texture.set(frame, TextureOptions::default());
                    }
                }

                if let Some(texture) = &stream.texture_handle {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            texture.id(),
                            display_size,
                        )));
                    });
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                    });
                }
            }
            StreamState::Error(error) => {
                ui.centered_and_justified(|ui| {
                    ui.colored_label(super::colors::REDDISH_DEFAULT, format!("Error: {}", error));
                });
            }
        }
    }
}

fn video_frame_to_image(frame: Video) -> ColorImage {
    let size = [frame.width() as usize, frame.height() as usize];
    let data = frame.data(0);
    let stride = frame.stride(0);
    let pixel_size_bytes = 3;
    let byte_width: usize = pixel_size_bytes * frame.width() as usize;
    let height: usize = frame.height() as usize;
    let mut pixels = Vec::with_capacity(size[0] * size[1]);
    for line in 0..height {
        let begin = line * stride;
        let end = begin + byte_width;
        let data_line = &data[begin..end];
        pixels.extend(
            data_line
                .chunks_exact(pixel_size_bytes)
                .map(|p| Color32::from_rgb(p[0], p[1], p[2])),
        )
    }
    ColorImage { size, pixels }
}
