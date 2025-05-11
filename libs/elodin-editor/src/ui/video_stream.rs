use std::sync::Arc;
use std::sync::atomic::{self, AtomicU32};
use std::time::{Duration, Instant};

use bevy::ecs::system::InRef;
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, Query, Res, World},
};
use egui::{self, Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use impeller2::types::OwnedPacket;
use impeller2_bevy::{CommandsExt, CurrentStreamId, PacketGrantR};
use impeller2_wkt::{FixedRateMsgStream, FixedRateOp};
use pic_scale::{
    ImageStore, ImageStoreMut, LinearScaler, ResamplingFunction, Scaling, ThreadingPolicy,
};

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
    tx: flume::Sender<Vec<u8>>,
    rx: flume::Receiver<ColorImage>,
    width: Arc<AtomicU32>,
    _handle: std::thread::JoinHandle<()>,
}

impl Default for VideoDecoderHandle {
    fn default() -> Self {
        let (packet_tx, packet_rx) = flume::unbounded::<Vec<u8>>();
        let (image_tx, image_rx) = flume::bounded(8);
        let width = Arc::new(AtomicU32::new(0));
        let frame_width = width.clone();
        let _handle = std::thread::spawn(move || {
            let mut decoder = openh264::decoder::Decoder::new().unwrap();
            let mut scaler = LinearScaler::new(ResamplingFunction::Bilinear);
            scaler.set_threading_policy(ThreadingPolicy::Adaptive);

            let mut rgba = vec![];
            while let Ok(packet) = packet_rx.recv() {
                use openh264::formats::YUVSource;
                if let Ok(Some(yuv)) = decoder.decode(&packet) {
                    let (width, height) = yuv.dimensions();
                    rgba.clear();
                    rgba.resize(width * height * 4, 0);
                    yuv.write_rgba8(&mut rgba);
                    let input = ImageStore::<'_, u8, 4>::borrow(&rgba, width, height).unwrap();

                    let (width, height) = (width as f64, height as f64);
                    let desired_width = frame_width.load(atomic::Ordering::Relaxed);
                    let aspect_ratio = height / width;
                    let new_width = width.min(desired_width as f64);
                    let new_height = (new_width * aspect_ratio) as usize;
                    let new_width = new_width as usize;

                    let mut image = ColorImage::new([new_width, new_height], Color32::TRANSPARENT);
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(
                            image.pixels.as_mut_ptr() as *mut u8,
                            image.pixels.len() * size_of::<Color32>(),
                        )
                    };
                    let mut out =
                        ImageStoreMut::<'_, u8, 4>::borrow(out, new_width, new_height).unwrap();

                    scaler.resize_rgba(&input, &mut out, false).unwrap();
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
        let _ = self.tx.send(frame_data.to_vec());
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

#[derive(SystemParam)]
pub struct VideoStreamWidget<'w, 's> {
    streams: Query<'w, 's, &'static mut VideoStream>,
    decoders: Query<'w, 's, &'static mut VideoDecoderHandle>,
    commands: Commands<'w, 's>,
    stream_id: Res<'w, CurrentStreamId>,
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
                FixedRateMsgStream {
                    msg_id: stream.msg_id,
                    fixed_rate: FixedRateOp {
                        stream_id: state.stream_id.0,
                        behavior: Default::default(),
                    },
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
