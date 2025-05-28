use crate::ui::PrimaryWindow;
use bevy::asset::Assets;
use bevy::asset::RenderAssetUsages;
use bevy::ecs::query::QueryData;
use bevy::ecs::system::InRef;
use bevy::image::Image;
use bevy::prelude::With;
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
use impeller2_wkt::{CurrentTimestamp, FixedRateMsgStream, FixedRateOp};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{self};
use std::time::Instant;

use super::colors::{ColorExt, get_scheme};

#[derive(Clone)]
pub struct VideoStreamPane {
    pub entity: Entity,
    pub label: String,
}

#[derive(Component)]
pub struct VideoStream {
    pub msg_id: [u8; 2],
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

#[derive(Default, Clone)]
pub enum StreamState {
    #[default]
    None,
    Streaming,
    Error(String),
}

#[derive(Component)]
pub struct VideoDecoderHandle {
    tx: flume::Sender<(Vec<u8>, Timestamp)>,
    rx: flume::Receiver<(Image, Timestamp)>,
    width: Arc<AtomicUsize>,
    _handle: std::thread::JoinHandle<()>,
}

#[cfg(not(target_os = "macos"))]
fn decode_video(
    frame_width: Arc<AtomicUsize>,
    packet_rx: flume::Receiver<(Vec<u8>, Timestamp)>,
    image_tx: flume::Sender<(Image, Timestamp)>,
) {
    use pic_scale::{
        ImageStore, ImageStoreMut, LinearScaler, ResamplingFunction, Scaling, ThreadingPolicy,
    };
    let mut decoder = openh264::decoder::Decoder::new().unwrap();
    let mut scaler = LinearScaler::new(ResamplingFunction::Bilinear);
    scaler.set_threading_policy(ThreadingPolicy::Adaptive);

    let mut rgba = vec![];
    while let Ok((packet, timestamp)) = packet_rx.recv() {
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

            let mut pixels = vec![0; new_width * new_height * 4];

            {
                let mut out =
                    ImageStoreMut::<'_, u8, 4>::borrow(&mut pixels, new_width, new_height).unwrap();
                scaler.resize_rgba(&input, &mut out, false).unwrap();
            }

            let image = Image::new(
                Extent3d {
                    width: new_width as u32,
                    height: new_height as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                pixels,
                bevy_render::render_resource::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );

            let _ = image_tx.send((image, timestamp));
        }
    }
}

#[cfg(target_os = "macos")]
fn decode_video(
    frame_width: Arc<AtomicUsize>,
    packet_rx: flume::Receiver<(Vec<u8>, Timestamp)>,
    image_tx: flume::Sender<(Image, Timestamp)>,
) {
    let mut video_toolbox = video_toolbox::VideoToolboxDecoder::new(frame_width).unwrap();

    while let Ok((packet, timestamp)) = packet_rx.recv() {
        if let Ok(Some(frame)) = video_toolbox.decode(&packet, 0) {
            let image = Image::new(
                Extent3d {
                    width: frame.width as u32,
                    height: frame.height as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                frame.rgba,
                bevy_render::render_resource::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            let _ = image_tx.send((image, timestamp));
        }
    }
}

impl Default for VideoDecoderHandle {
    fn default() -> Self {
        let (packet_tx, packet_rx) = flume::bounded::<(Vec<u8>, Timestamp)>(8);
        let (image_tx, image_rx) = flume::bounded(8);
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
    pub fn process_frame(&mut self, timestamp: Timestamp, frame_data: &[u8]) {
        let _ = self.tx.try_send((frame_data.to_vec(), timestamp));
    }

    pub fn render_frame(&mut self, stream: &mut VideoStream) {
        while let Ok((frame, timestamp)) = self.rx.try_recv() {
            stream.current_frame = Some(frame);
            stream.frame_timestamp = Some(timestamp);
            stream.frame_count += 1;
            if stream.size == Vec2::ZERO && stream.current_frame.is_some() {
                if let Some(ref img) = stream.current_frame {
                    stream.size = Vec2::new(img.width() as f32, img.height() as f32);
                }
            }
        }
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
pub struct WidgetQuery {
    stream: &'static mut VideoStream,
    decoder: &'static mut VideoDecoderHandle,
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
    window: Query<'w, 's, &'static bevy_egui::EguiContextSettings, With<PrimaryWindow>>,
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

        let Ok(WidgetQueryItem {
            mut stream,
            mut decoder,
            mut image_node,
            mut ui_node,
        }) = state.query.get_mut(entity)
        else {
            return;
        };

        if let StreamState::Streaming = stream.state {
            decoder.render_frame(&mut stream);
            stream.last_update = Instant::now();
        }

        if let StreamState::None = stream.state {
            stream.state = StreamState::Streaming;
            state.commands.send_req_reply_raw(
                FixedRateMsgStream {
                    msg_id: stream.msg_id,
                    fixed_rate: FixedRateOp {
                        stream_id: state.stream_id.0,
                        behavior: Default::default(),
                    },
                },
                move |InRef(pkt): InRef<OwnedPacket<PacketGrantR>>,
                      mut decoders: Query<&mut VideoDecoderHandle>| {
                    if let OwnedPacket::Msg(msg_buf) = pkt {
                        if let Ok(mut decoder) = decoders.get_mut(entity) {
                            if let Some(timestamp) = msg_buf.timestamp {
                                decoder.process_frame(timestamp, &msg_buf.buf);
                            }
                        }
                    }
                    false
                },
            );
        }

        let max_rect = ui.max_rect();

        let Some(egui_settings) = state.window.iter().next() else {
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
                    ui.label("No video stream. Please send a valid H264 stream");
                });
            }
            StreamState::Streaming => {
                decoder
                    .width
                    .store(width as usize, atomic::Ordering::Relaxed);

                if let Some(frame) = stream.current_frame.take() {
                    image_node.image = state.images.add(frame);
                }

                if let Some(frame_timestamp) = stream.frame_timestamp {
                    if (frame_timestamp.0 - state.current_time.0.0).abs() > 500000 {
                        ui.painter()
                            .rect_filled(max_rect, 0, Color32::BLACK.opacity(0.75));
                        ui.put(
                            egui::Rect::from_center_size(
                                max_rect.center_top() + egui::vec2(0., 64.0),
                                egui::vec2(max_rect.width(), 20.0),
                            ),
                            egui::Label::new(
                                egui::RichText::new(
                                    "Loss of Signal - Frame out of date. Waiting for new keyframe",
                                )
                                .size(16.0)
                                .color(get_scheme().highlight),
                            ),
                        );
                    }
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

pub fn set_visibility(mut query: Query<(&mut Node, &IsTileVisible)>) {
    for (mut ui_node, is_visible) in &mut query {
        if is_visible.0 {
            ui_node.display = Display::Block;
        } else {
            ui_node.display = Display::None;
        }
    }
}
