use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next::{self as ffmpeg, codec, encoder, format::Pixel, frame::Video, picture};
use impeller2::types::{LenPacket, Msg, OwnedPacket};
use impeller2_stellar::Client;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

/// Video streamer that encodes video files to AV1 and sends OBUs to elodin-db
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    input: PathBuf,

    msg_id: u16,

    /// Elodin DB address in the format IP:PORT
    #[clap(short, long, default_value = "127.0.0.1:2240")]
    db_addr: SocketAddr,

    /// Output bitrate in kbps
    #[clap(short, long, default_value = "1000")]
    bitrate: usize,

    /// Keyframe interval in frames
    #[clap(short, long, default_value = "60")]
    keyframe_interval: i32,

    /// Speed preset (1-8, where 1 is highest quality, 8 is fastest)
    #[clap(short, long, default_value = "6")]
    speed: i32,
}

struct VideoStreamer {
    args: Args,
    client: Client,
}

impl VideoStreamer {
    async fn new(args: Args) -> Result<Self> {
        let client = Client::connect(args.db_addr)
            .await
            .context("Failed to connect to elodin-db")?;

        Ok(Self { args, client })
    }

    async fn stream_video(&mut self) -> Result<()> {
        ffmpeg::init()
            .context("Failed to initialize FFmpeg")
            .unwrap();

        let mut ictx = ffmpeg::format::input(&self.args.input).unwrap();

        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .context("No video stream found")
            .unwrap();

        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters()).unwrap();
        let mut decoder = context.decoder().video().unwrap();

        let encoder_codec = encoder::find(codec::Id::H264).unwrap();

        let mut encoder = codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .video()
            .unwrap();
        encoder.set_width(decoder.width());
        encoder.set_height(decoder.height());
        encoder.set_time_base((1, 60));
        encoder.set_bit_rate(self.args.bitrate * 1000);
        encoder.set_aspect_ratio(decoder.aspect_ratio());
        encoder.set_format(decoder.format());
        encoder.set_frame_rate(decoder.frame_rate());
        //encoder.set_time_base(Some(decoder.frame_rate().unwrap().invert()));

        //encoder.set_pix_fmt(ffmpeg::format::Pixel::YUV420P);

        // Set AV1 specific options
        let mut opts = ffmpeg::Dictionary::new();
        // opts.set("cpu-used", &self.args.speed.to_string());
        // opts.set("keyint_min", &self.args.keyframe_interval.to_string());
        // opts.set("g", &self.args.keyframe_interval.to_string());
        opts.set("g", "12");
        opts.set("preset", "medium");
        opts.set("profile", "baseline");
        opts.set("header_at_keyframes", &"true");
        let mut encoder = encoder.open_with(opts)?;

        let mut frame = ffmpeg::util::frame::Video::empty();

        let mut frame_count = 0;
        let mut obu_count = 0;

        info!("Sending OBUs with message ID: {}", self.args.msg_id);
        let msg_id = self.args.msg_id.to_le_bytes();
        let video_stream_index = input.index();
        for res in ictx.packets() {
            let (stream, packet) = res;
            if stream.index() != video_stream_index {
                continue;
            }

            decoder.send_packet(&packet).unwrap();

            while decoder.receive_frame(&mut frame).is_ok() {
                frame_count += 1;
                info!("Processing frame {}", frame_count);

                let timestamp = frame.timestamp();
                frame.set_pts(timestamp);
                frame.set_kind(picture::Type::None);
                encoder.send_frame(&frame).unwrap();

                let mut packet = ffmpeg::Packet::empty();
                while encoder.receive_packet(&mut packet).is_ok() {
                    obu_count += 1;
                    if let Some(data) = packet.data() {
                        println!("sending pkt {:?}", data.len());
                        let mut pkt = LenPacket::msg(msg_id, data.len());
                        pkt.extend_from_slice(data);
                        if let Err(err) = self.client.send(pkt).await.0 {
                            self.client = Client::connect(self.args.db_addr)
                                .await
                                .context("Failed to connect to elodin-db")?;
                        }
                    }
                    stellarator::sleep(Duration::from_secs_f64(1.0 / 30.0)).await;
                }
            }
        }

        // Flush the encoder
        encoder.send_eof().unwrap();
        let mut packet = ffmpeg::Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            obu_count += 1;
            if let Some(data) = packet.data() {
                println!("sending pkt {:?}", data.len());
                let mut pkt = LenPacket::msg(msg_id, data.len());
                pkt.extend_from_slice(data);
                if let Err(err) = self.client.send(pkt).await.0 {}
            }
        }

        info!("Video processing complete.");
        info!(
            "Processed {} frames and sent {} OBUs",
            frame_count, obu_count
        );

        Ok(())
    }
}

#[stellarator::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let mut streamer = VideoStreamer::new(args).await.unwrap();
    streamer.stream_video().await.unwrap();

    Ok(())
}
