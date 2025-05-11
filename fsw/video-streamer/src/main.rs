use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next::{self as ffmpeg, codec, encoder, picture};
use impeller2::types::{msg_id, LenPacket, Timestamp};
use impeller2_stellar::Client;
use kdam::term::Colorizer;
use kdam::BarExt;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Video streamer that encodes video files to AV1 and sends OBUs to elodin-db
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    input: PathBuf,

    msg_name: String,

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

        let total_count = input.frames();
        let time_base: f64 = input.time_base().into();
        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters()).unwrap();
        let mut decoder = context.decoder().video().unwrap();

        //let encoder_codec = encoder::find_by_name("h264_videotoolbox").unwrap();
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
        //encoder.set_time_base(rational_time_base);

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

        let msg_id = msg_id(&self.args.msg_name);
        let video_stream_index = input.index();
        let start_time = Timestamp::now();

        kdam::term::init(std::io::stderr().is_terminal());
        let mut bar = kdam::tqdm!(
            total = total_count as usize,
            bar_format = format!(
                "{{animation}} {} ",
                "{percentage:3.1}% {rate:.4}{unit}/s|{elapsed human=true}|{remaining human=true}"
                    .colorize("#EE6FF8")
            ),
            colour = kdam::Colour::gradient(&["#5A56E0", "#EE6FF8"]),
            dynamic_ncols = true,
            unit = "F",
            unit_scale = true,
            force_refresh = true
        );
        let mut frame_timestamp;
        for res in ictx.packets() {
            let (stream, packet) = res;
            if stream.index() != video_stream_index {
                continue;
            }

            decoder.send_packet(&packet).unwrap();

            while decoder.receive_frame(&mut frame).is_ok() {
                frame_count += 1;

                frame_timestamp = frame.timestamp();
                let time = time_base * frame_timestamp.unwrap_or(0) as f64;
                frame.set_pts(frame_timestamp);
                frame.set_kind(picture::Type::None);
                encoder.send_frame(&frame).unwrap();

                let mut packet = ffmpeg::Packet::empty();
                let mut pkt_timestamp = start_time + Duration::from_secs_f64(time);
                while encoder.receive_packet(&mut packet).is_ok() {
                    obu_count += 1;
                    if let Some(data) = packet.data() {
                        let mut pkt =
                            LenPacket::msg_with_timestamp(msg_id, pkt_timestamp, data.len());
                        pkt.extend_from_slice(data);
                        if let Err(_) = self.client.send(pkt).await.0 {
                            self.client = Client::connect(self.args.db_addr)
                                .await
                                .context("Failed to connect to elodin-db")?;
                        }
                    }
                    pkt_timestamp.0 += 1;
                }
                bar.update(1);
            }
        }

        encoder.send_eof().unwrap();
        let mut packet = ffmpeg::Packet::empty();
        frame_timestamp = frame.timestamp();
        let time = time_base * frame_timestamp.unwrap_or(0) as f64;
        let mut pkt_timestamp = start_time + Duration::from_secs_f64(time);
        while encoder.receive_packet(&mut packet).is_ok() {
            obu_count += 1;
            if let Some(data) = packet.data() {
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, pkt_timestamp, data.len());
                pkt.extend_from_slice(data);
                if let Err(_) = self.client.send(pkt).await.0 {}
            }
            pkt_timestamp.0 += 1;
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
