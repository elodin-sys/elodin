use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::{self as ffmpeg, codec, encoder, picture};
use impeller2::types::{LenPacket, Timestamp, msg_id};
use impeller2_stellar::Client;
use kdam::BarExt;
use kdam::term::Colorizer;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

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

    /// The max distance between two key frames
    #[clap(short, long, default_value = "12")]
    keyframe_interval: i32,

    /// Which h264 encoder to use (i.e h264_videotoolbox or libx264)
    #[clap(short, long, default_value = "libopenh264")]
    encoder: String,

    #[clap(short, long)]
    live: bool,
}

impl Args {
    pub async fn run(&mut self) -> anyhow::Result<()> {
        let mut client = Client::connect(self.db_addr)
            .await
            .context("Failed to connect to elodin-db")?;
        ffmpeg::init().context("Failed to initialize FFmpeg")?;

        let mut ictx = ffmpeg::format::input(&self.input).context("ffmpeg input not found")?;

        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .context("No video stream found")?;

        let total_count = input.frames();
        let time_base: f64 = input.time_base().into();

        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context.decoder().video()?;

        let encoder_codec = encoder::find_by_name(&self.encoder).context("encoder not found")?;

        let mut encoder = codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .video()?;
        encoder.set_width(decoder.width());
        encoder.set_height(decoder.height());
        encoder.set_time_base((1, 60));
        encoder.set_bit_rate(self.bitrate * 1000);
        encoder.set_aspect_ratio(decoder.aspect_ratio());
        encoder.set_format(Pixel::YUV420P);
        encoder.set_frame_rate(decoder.frame_rate());

        let mut opts = ffmpeg::Dictionary::new();
        opts.set("g", &self.keyframe_interval.to_string());
        opts.set("preset", "medium");
        opts.set("profile:v", "baseline");
        opts.set("header_at_keyframes", "true");
        let mut encoder = encoder.open_with(opts)?;

        let mut frame = ffmpeg::util::frame::Video::empty();

        let msg_id = msg_id(&self.msg_name);
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

            decoder
                .send_packet(&packet)
                .context("decoder didn't accept packet")?;

            let input_format = decoder.format();
            let mut scaler = ffmpeg::software::scaling::context::Context::get(
                input_format,
                decoder.width(),
                decoder.height(),
                Pixel::YUV420P,
                decoder.width(),
                decoder.height(),
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )
            .context("init scaler failed ")?;
            while decoder.receive_frame(&mut frame).is_ok() {
                frame_timestamp = frame.timestamp();
                let time = time_base * frame_timestamp.unwrap_or(0) as f64;

                if input_format != Pixel::YUV420P {
                    let mut converted_frame = ffmpeg::util::frame::Video::empty();
                    scaler
                        .run(&frame, &mut converted_frame)
                        .context("scaler failed")?;

                    converted_frame.set_pts(frame_timestamp);
                    converted_frame.set_kind(picture::Type::None);
                    encoder
                        .send_frame(&converted_frame)
                        .context("encoder didn't accept frame")?;
                } else {
                    frame.set_pts(frame_timestamp);
                    frame.set_kind(picture::Type::None);
                    encoder
                        .send_frame(&frame)
                        .context("encoder didn't accept frame")?;
                }

                let mut packet = ffmpeg::Packet::empty();
                let mut pkt_timestamp = start_time + Duration::from_secs_f64(time);
                while encoder.receive_packet(&mut packet).is_ok() {
                    if let Some(data) = packet.data() {
                        let mut pkt = LenPacket::msg_with_timestamp(
                            msg_id,
                            if self.live {
                                Timestamp::now()
                            } else {
                                pkt_timestamp
                            },
                            data.len(),
                        );
                        pkt.extend_from_slice(data);
                        if client.send(pkt).await.0.is_err() {
                            client = Client::connect(self.db_addr)
                                .await
                                .context("Failed to connect to elodin-db")?;
                        }
                    }
                    pkt_timestamp.0 += 1;
                }
                let _ = bar.update(1);
            }
        }

        encoder.send_eof().context("eof failed")?;
        let mut packet = ffmpeg::Packet::empty();
        frame_timestamp = frame.timestamp();
        let time = time_base * frame_timestamp.unwrap_or(0) as f64;
        let mut pkt_timestamp = start_time + Duration::from_secs_f64(time);
        while encoder.receive_packet(&mut packet).is_ok() {
            if let Some(data) = packet.data() {
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, pkt_timestamp, data.len());
                pkt.extend_from_slice(data);
                let _ = client.send(pkt).await.0;
            }
            pkt_timestamp.0 += 1;
        }

        Ok(())
    }
}

#[stellarator::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    args.run().await?;

    Ok(())
}
