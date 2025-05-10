use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::ecs::system::{InRef, NonSendMut};
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, In, NonSend, Query, Resource, World},
};
use egui::{self, Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use ffmpeg_next::frame::Video;
use ffmpeg_next::{codec, decoder};
use impeller2::buf::Slice;
use impeller2::types::{MsgBuf, OwnedPacket};
use impeller2_bevy::{CommandsExt, PacketGrantR};
use impeller2_wkt::{ErrorResponse, MsgStream};

#[derive(Clone)]
pub struct VideoStreamPane {
    pub entity: Entity,
    pub label: String,
}

// This component doesn't contain any FFmpeg references, ensuring thread safety
#[derive(Component)]
pub struct VideoStream {
    pub message_id: u16,
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
            message_id: 0,
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
    Requested(Instant),
    Streaming,
    Error(String),
}

// Simple non-thread-safe video decoder system
#[derive(Default)]
pub struct VideoDecoderManager {
    decoders: HashMap<Entity, VideoDecoder>,
    pending_frames: HashMap<Entity, Vec<Vec<u8>>>,
}

pub struct VideoDecoder {
    decoder: decoder::Video,
}

impl VideoDecoder {
    pub fn new() -> Self {
        let codec = decoder::find(codec::Id::AV1).unwrap();

        let decoder = codec::context::Context::new_with_codec(codec)
            .decoder()
            .video()
            .unwrap();
        Self { decoder }
    }
}

impl VideoDecoderManager {
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
            pending_frames: HashMap::new(),
        }
    }

    // Queue a new frame for processing
    pub fn queue_frame(&mut self, entity: Entity, data: Vec<u8>) {
        let queue = self.pending_frames.entry(entity).or_default();
        queue.push(data);
    }

    // Process entity and update its frame
    pub fn process_frame(&mut self, entity: Entity, stream: &mut VideoStream, frame_data: &[u8]) {
        // Initialize the decoder if needed
        if !self.decoders.contains_key(&entity) {
            let decoder = VideoDecoder::new();
            self.decoders.insert(entity, decoder);
        }

        // Get a frame to process
        println!("encoding frame data");
        // Get the decoder for this entity
        if let Some(decoder) = self.decoders.get_mut(&entity) {
            if let Err(err) = Self::decode_av1_frame(&mut decoder.decoder, &frame_data) {
                if stream.current_frame.is_none() {
                    stream.state = StreamState::Error(format!("Failed to decode frame: {}", err));
                }
            }
        }
    }
    pub fn render_frame(&mut self, entity: Entity, stream: &mut VideoStream) {
        if let Some(decoder) = self.decoders.get_mut(&entity) {
            while let Some(frame) = Self::get_av1_frame(&mut decoder.decoder) {
                println!("got frame");
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

    fn decode_av1_frame(decoder: &mut decoder::Video, frame_data: &[u8]) -> Result<(), String> {
        use ffmpeg_next::util::frame::video::Video;
        use ffmpeg_next::{Error, Packet};

        let packet = Packet::copy(frame_data);

        // Send the packet to the decoder
        if let Err(e) = decoder.send_packet(&packet) {
            return Err(format!("Error sending packet to decoder: {}", e));
        }

        // Allocate a frame to receive decoded data
        //let mut frame = Video::empty();

        // Receive the decoded frame
        // match decoder.receive_frame(&mut frame) {
        //     Ok(()) => {}
        //     Err(Error::Eof) =>{
        //         return Err("End of stream reached".to_string());
        //     }
        //     Err(e) => {
        //         // let codec = decoder::find(codec::Id::AV1).unwrap();
        //         // *decoder = codec::context::Context::new_with_codec(codec)
        //         //     .decoder()
        //         //     .video()
        //         //     .unwrap();

        //         return Err(format!("Error receiving frame: {}", e));
        //     }
        // }
        Ok(())
    }

    fn get_av1_frame(decoder: &mut decoder::Video) -> Option<ColorImage> {
        use ffmpeg_next::format::Pixel;
        use ffmpeg_next::software::scaling::{context::Context as ScalingContext, flag::Flags};
        let mut frame = Video::empty();
        decoder
            .receive_frame(&mut frame)
            .inspect_err(|err| {
                dbg!(err);
            })
            .ok()?;

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        let rgb_frame = if frame.format() != Pixel::RGB24 {
            let mut rgb = Video::new(Pixel::RGB24, width as u32, height as u32);

            let mut scaler = ScalingContext::get(
                frame.format(),
                width as u32,
                height as u32,
                Pixel::RGB24,
                width as u32,
                height as u32,
                Flags::BILINEAR,
            )
            .ok()?;

            scaler.run(&frame, &mut rgb).ok()?;

            rgb
        } else {
            frame
        };
        Some(video_frame_to_image(rgb_frame))
    }
}

#[derive(SystemParam)]
pub struct VideoStreamWidget<'w, 's> {
    streams: Query<'w, 's, &'static mut VideoStream>,
    decoder_manager: NonSendMut<'w, VideoDecoderManager>,
    commands: Commands<'w, 's>,
}

impl super::widgets::WidgetSystem for VideoStreamWidget<'_, '_> {
    type Args = VideoStreamPane;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        VideoStreamPane { entity, label }: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut stream) = state.streams.get_mut(entity) else {
            return;
        };

        // Check if it's time to process a new frame
        if stream.last_update.elapsed() > Duration::from_millis(16) {
            if let StreamState::Streaming = stream.state {
                state.decoder_manager.render_frame(entity, &mut stream);
                stream.last_update = Instant::now();
            }
        }

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading(&label);

                ui.add_space(ui.available_width() - 120.0);

                if ui
                    .add_sized(
                        [100.0, 32.0],
                        super::widgets::button::EButton::green("REFRESH"),
                    )
                    .clicked()
                {
                    if let StreamState::None = stream.state {
                        stream.state = StreamState::Requested(Instant::now());

                        let msg_id = stream.message_id.to_le_bytes();
                        let entity = entity;

                        state.commands.send_req_reply_raw(
                            MsgStream { msg_id },
                            move |InRef(res): InRef<OwnedPacket<PacketGrantR>>,
                                  mut streams: Query<&mut VideoStream>,
                            mut decoder_manager: NonSendMut<VideoDecoderManager>| {
                                let Ok(mut stream) = streams.get_mut(entity) else {
                                    return false;
                                };
                                match res {
                                    OwnedPacket::Msg(msg_buf) => {
                                        if let StreamState::Requested(_) = stream.state {
                                            stream.state = StreamState::Streaming;
                                        }
                                        decoder_manager.process_frame(entity, &mut stream, &msg_buf.buf);
                                    }
                                    _ => {}
                                };
                                false
                            },
                        );
                    }
                }
            });

            let available_size = ui.available_size();

            match &stream.state {
                StreamState::None => {
                    ui.centered_and_justified(|ui| {
                        ui.label("No video stream loaded. Click REFRESH to start streaming.");
                    });
                }
                StreamState::Requested(time) => {
                    let _elapsed = time.elapsed().as_secs_f32();
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                        ui.label(format!(
                            "Requesting video stream for message ID: {}...",
                            stream.message_id
                        ));
                    });
                }
                StreamState::Streaming => {
                    // Display the current frame if available
                    if let Some(frame) = stream.current_frame.clone() {
                        // Initialize texture if needed
                        if stream.texture_handle.is_none() {
                            let texture_handle = ui.ctx().load_texture(
                                format!("video_stream_{}", entity.index()),
                                frame,
                                TextureOptions::default(),
                            );
                            stream.texture_handle = Some(texture_handle);
                        } else if let Some(texture) = &mut stream.texture_handle {
                            texture.set(frame.clone(), TextureOptions::default());
                        }

                        if let Some(texture) = &stream.texture_handle {
                            // Calculate aspect ratio for proper display
                            let aspect_ratio = if stream.size.y > 0.0 {
                                stream.size.x / stream.size.y
                            } else {
                                16.0 / 9.0
                            };
                            let width = (available_size.x).min(available_size.y * aspect_ratio);
                            let height = width / aspect_ratio;
                            let display_size = Vec2::new(width, height);

                            ui.centered_and_justified(|ui| {
                                ui.add(egui::Image::new(egui::load::SizedTexture::new(
                                    texture.id(),
                                    display_size,
                                )));
                            });

                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "Frame: {} | Size: {}x{}",
                                    stream.frame_count, stream.size.x as u32, stream.size.y as u32,
                                ));
                            });
                        } else {
                            ui.centered_and_justified(|ui| {
                                ui.spinner();
                                ui.label("Processing frames...");
                            });
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.spinner();
                            ui.label("Waiting for video frames...");
                        });
                    }
                }
                StreamState::Error(error) => {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(
                            super::colors::REDDISH_DEFAULT,
                            format!("Error: {}", error),
                        );
                    });
                }
            }
        });
    }
}

// Register the necessary resource
pub fn setup_video_system(world: &mut World) {
    world.insert_non_send_resource(VideoDecoderManager::new());
}

fn video_frame_to_image(frame: Video) -> ColorImage {
    let size = [frame.width() as usize, frame.height() as usize];
    let data = frame.data(0);
    let stride = frame.stride(0);
    let pixel_size_bytes = 3;
    let byte_width: usize = pixel_size_bytes * frame.width() as usize;
    let height: usize = frame.height() as usize;
    let mut pixels = vec![];
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
