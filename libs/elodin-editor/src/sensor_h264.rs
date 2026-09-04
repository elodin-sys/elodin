//! Live sensor-camera H.264 encode.
//!
//! Prefers a hardware encoder (NVENC / VideoToolbox via FFmpeg) and
//! falls back to OpenH264. Override with `ELODIN_H264_ENCODER=cpu|auto|<name>`.
//! `h264_vaapi` is not wired: ffmpeg-next has no VAAPI device context.

use std::collections::VecDeque;

use impeller2::types::Timestamp;
use openh264::OpenH264API;
use openh264::encoder::{
    BitRate, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, RateControlMode, SpsPpsStrategy,
};
use openh264::formats::{RgbaSliceU8, YUVBuffer};

#[derive(Default)]
struct ParameterSetCache {
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
}

pub(crate) struct SensorH264Encoder {
    inner: EncoderInner,
    parameter_sets: ParameterSetCache,
}

enum EncoderInner {
    Software(Box<SoftwareEncoder>),
    Hardware(Box<HardwareEncoder>),
}

impl SensorH264Encoder {
    pub(crate) fn new(width: u32, height: u32, fps: f32) -> Result<Self, String> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err("h264 width and height must be positive and even".to_string());
        }
        let fps = fps.max(1.0);
        match encoder_choice() {
            EncoderChoice::Software => Self::software(width, height, fps),
            EncoderChoice::Auto => {
                for name in hw_encoder_names() {
                    match HardwareEncoder::open(name, width, height, fps) {
                        Ok(hw) => {
                            tracing::info!("sensor camera H.264: using {name}");
                            return Ok(Self {
                                inner: EncoderInner::Hardware(Box::new(hw)),
                                parameter_sets: ParameterSetCache::default(),
                            });
                        }
                        Err(err) => {
                            tracing::debug!("sensor camera H.264: {name} unavailable ({err})");
                        }
                    }
                }
                tracing::info!("sensor camera H.264: falling back to OpenH264");
                Self::software(width, height, fps)
            }
            EncoderChoice::Named(name) => match HardwareEncoder::open(&name, width, height, fps) {
                Ok(hw) => {
                    tracing::info!("sensor camera H.264: using {name}");
                    Ok(Self {
                        inner: EncoderInner::Hardware(Box::new(hw)),
                        parameter_sets: ParameterSetCache::default(),
                    })
                }
                Err(err) => Err(format!("ELODIN_H264_ENCODER={name} failed: {err}")),
            },
        }
    }

    pub(crate) fn software(width: u32, height: u32, fps: f32) -> Result<Self, String> {
        Ok(Self {
            inner: EncoderInner::Software(Box::new(SoftwareEncoder::new(width, height, fps)?)),
            parameter_sets: ParameterSetCache::default(),
        })
    }

    pub(crate) fn encode(
        &mut self,
        rgba: &[u8],
        timestamp: Timestamp,
    ) -> Result<Vec<(Timestamp, Vec<u8>)>, String> {
        let frames = match &mut self.inner {
            EncoderInner::Software(enc) => {
                let annex_b = enc.encode(rgba)?;
                if annex_b.is_empty() {
                    Vec::new()
                } else {
                    vec![(timestamp, annex_b)]
                }
            }
            EncoderInner::Hardware(enc) => enc.encode(rgba, timestamp)?,
        };
        Ok(self.with_parameter_sets(frames))
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<(Timestamp, Vec<u8>)>, String> {
        let frames = match &mut self.inner {
            EncoderInner::Software(_) => Vec::new(),
            EncoderInner::Hardware(enc) => enc.flush()?,
        };
        Ok(self.with_parameter_sets(frames))
    }

    fn with_parameter_sets(
        &mut self,
        frames: Vec<(Timestamp, Vec<u8>)>,
    ) -> Vec<(Timestamp, Vec<u8>)> {
        frames
            .into_iter()
            .map(|(timestamp, annex_b)| {
                (
                    timestamp,
                    apply_parameter_sets(annex_b, &mut self.parameter_sets),
                )
            })
            .collect()
    }
}

enum EncoderChoice {
    Auto,
    Software,
    Named(String),
}

fn encoder_choice() -> EncoderChoice {
    match std::env::var("ELODIN_H264_ENCODER") {
        Ok(value) => {
            let value = value.trim();
            if value.is_empty() || value.eq_ignore_ascii_case("auto") {
                EncoderChoice::Auto
            } else if value.eq_ignore_ascii_case("cpu")
                || value.eq_ignore_ascii_case("openh264")
                || value.eq_ignore_ascii_case("software")
            {
                EncoderChoice::Software
            } else {
                EncoderChoice::Named(value.to_string())
            }
        }
        Err(_) => EncoderChoice::Auto,
    }
}

fn hw_encoder_names() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    {
        &["h264_videotoolbox"]
    }
    #[cfg(target_os = "linux")]
    {
        &["h264_nvenc"]
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        &[]
    }
}

fn encode_defaults(width: u32, height: u32, fps: f32) -> (u32, u32) {
    let keyframe_interval = (fps.ceil() as u32).saturating_mul(2).max(1);
    let target_bitrate = ((width as f64) * (height as f64) * (fps as f64) * 3.0)
        .clamp(300_000.0, 12_000_000.0) as u32;
    (keyframe_interval, target_bitrate)
}

struct SoftwareEncoder {
    encoder: Encoder,
    yuv: YUVBuffer,
    width: usize,
    height: usize,
}

impl SoftwareEncoder {
    fn new(width: u32, height: u32, fps: f32) -> Result<Self, String> {
        let (keyframe_interval, target_bitrate) = encode_defaults(width, height, fps);
        let config = EncoderConfig::new()
            .bitrate(BitRate::from_bps(target_bitrate))
            .skip_frames(true)
            .max_frame_rate(FrameRate::from_hz(fps))
            .rate_control_mode(RateControlMode::Bitrate)
            .sps_pps_strategy(SpsPpsStrategy::ConstantId)
            .intra_frame_period(IntraFramePeriod::from_num_frames(keyframe_interval));
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|err| format!("openh264 encoder init: {err}"))?;
        Ok(Self {
            encoder,
            yuv: YUVBuffer::new(width as usize, height as usize),
            width: width as usize,
            height: height as usize,
        })
    }

    fn encode(&mut self, rgba: &[u8]) -> Result<Vec<u8>, String> {
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "sensor frame dimensions overflow".to_string())?;
        if rgba.len() != expected {
            return Err(format!(
                "unexpected RGBA frame size {} (expected {expected})",
                rgba.len()
            ));
        }
        let rgba = RgbaSliceU8::new(rgba, (self.width, self.height));
        self.yuv.read_rgb(rgba);
        self.encoder
            .encode(&self.yuv)
            .map(|bitstream| bitstream.to_vec())
            .map_err(|err| format!("openh264 encode: {err}"))
    }
}

struct HardwareEncoder {
    encoder: ffmpeg_next::encoder::Video,
    scaler: ffmpeg_next::software::scaling::Context,
    rgba_frame: ffmpeg_next::util::frame::Video,
    yuv_frame: ffmpeg_next::util::frame::Video,
    width: u32,
    height: u32,
    pts: i64,
    pending: VecDeque<(i64, Timestamp)>,
    held_headers: Vec<u8>,
}

impl HardwareEncoder {
    fn open(name: &str, width: u32, height: u32, fps: f32) -> Result<Self, String> {
        use ffmpeg_next::codec;
        use ffmpeg_next::format::Pixel;
        use ffmpeg_next::software::scaling::{Context as Scaler, flag::Flags};
        use ffmpeg_next::util::frame::Video;
        use ffmpeg_next::{Dictionary, Rational, encoder};

        if name == "h264_vaapi" {
            return Err(
                "h264_vaapi needs an FFmpeg VAAPI device context; ffmpeg-next does not expose one"
                    .to_string(),
            );
        }
        ffmpeg_next::init().map_err(|err| format!("ffmpeg init: {err}"))?;
        let codec = encoder::find_by_name(name).ok_or_else(|| format!("{name} not in ffmpeg"))?;
        let mut encoder = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|err| format!("{name} video encoder: {err}"))?;
        let fps_num = fps.round().clamp(1.0, 240.0) as i32;
        let (keyframe_interval, target_bitrate) = encode_defaults(width, height, fps);
        encoder.set_width(width);
        encoder.set_height(height);
        encoder.set_format(Pixel::YUV420P);
        encoder.set_time_base(Rational(1, fps_num));
        encoder.set_frame_rate(Some(Rational(fps_num, 1)));
        encoder.set_bit_rate(target_bitrate as usize);
        encoder.set_gop(keyframe_interval);
        encoder.set_max_b_frames(0);

        let mut opts = Dictionary::new();
        opts.set("bf", "0");
        opts.set("g", &keyframe_interval.to_string());
        match name {
            "h264_nvenc" => {
                opts.set("preset", "p1");
                opts.set("tune", "ull");
                opts.set("delay", "0");
                opts.set("zerolatency", "1");
                opts.set("rc", "cbr");
                opts.set("profile", "baseline");
                opts.set("repeat-headers", "1");
            }
            "h264_videotoolbox" => {
                opts.set("realtime", "1");
                opts.set("allow_sw", "0");
                opts.set("profile", "baseline");
            }
            _ => {}
        }

        let encoder = encoder
            .open_with(opts)
            .map_err(|err| format!("{name} open: {err}"))?;

        let scaler = Scaler::get(
            Pixel::RGBA,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            Flags::FAST_BILINEAR,
        )
        .map_err(|err| format!("ffmpeg scaler: {err}"))?;

        Ok(Self {
            encoder,
            scaler,
            rgba_frame: Video::new(Pixel::RGBA, width, height),
            yuv_frame: Video::empty(),
            width,
            height,
            pts: 0,
            pending: VecDeque::new(),
            held_headers: Vec::new(),
        })
    }

    fn encode(
        &mut self,
        rgba: &[u8],
        timestamp: Timestamp,
    ) -> Result<Vec<(Timestamp, Vec<u8>)>, String> {
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "sensor frame dimensions overflow".to_string())?;
        if rgba.len() != expected {
            return Err(format!(
                "unexpected RGBA frame size {} (expected {expected})",
                rgba.len()
            ));
        }
        fill_rgba_frame(&mut self.rgba_frame, rgba, self.width, self.height)?;
        self.scaler
            .run(&self.rgba_frame, &mut self.yuv_frame)
            .map_err(|err| format!("ffmpeg scale: {err}"))?;
        let pts = self.pts;
        self.yuv_frame.set_pts(Some(pts));
        self.yuv_frame.set_kind(ffmpeg_next::picture::Type::None);
        self.encoder
            .send_frame(&self.yuv_frame)
            .map_err(|err| format!("ffmpeg send_frame: {err}"))?;
        self.pending.push_back((pts, timestamp));
        self.pts += 1;
        let packets = self.drain_packets();
        Ok(take_ready_frames(
            &mut self.pending,
            &mut self.held_headers,
            packets,
        ))
    }

    fn flush(&mut self) -> Result<Vec<(Timestamp, Vec<u8>)>, String> {
        if let Err(err) = self.encoder.send_eof() {
            tracing::debug!("ffmpeg send_eof: {err}");
        }
        let packets = self.drain_packets();
        Ok(take_ready_frames(
            &mut self.pending,
            &mut self.held_headers,
            packets,
        ))
    }

    fn drain_packets(&mut self) -> Vec<(Option<i64>, Vec<u8>)> {
        let mut packets = Vec::new();
        let mut packet = ffmpeg_next::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            if let Some(data) = packet.data() {
                packets.push((packet.pts(), to_annex_b(data)));
            }
            packet = ffmpeg_next::Packet::empty();
        }
        packets
    }
}

fn is_non_vcl_only(data: &[u8]) -> bool {
    let nals = annex_b_nals(data);
    !nals.is_empty()
        && nals
            .iter()
            .all(|(_, _, nal_type)| !matches!(*nal_type, 1..=5))
}

fn take_ready_frames(
    pending: &mut VecDeque<(i64, Timestamp)>,
    held_headers: &mut Vec<u8>,
    packets: Vec<(Option<i64>, Vec<u8>)>,
) -> Vec<(Timestamp, Vec<u8>)> {
    let mut out: Vec<(Timestamp, Vec<u8>)> = Vec::new();
    let mut last_pts = None;
    for (pts, data) in packets {
        if data.is_empty() {
            continue;
        }
        if is_non_vcl_only(&data) {
            if let (Some(p), Some(prev)) = (pts, last_pts)
                && p == prev
                && let Some(last) = out.last_mut()
            {
                last.1.extend_from_slice(&data);
                continue;
            }
            held_headers.extend_from_slice(&data);
            continue;
        }
        let data = if held_headers.is_empty() {
            data
        } else {
            held_headers.extend_from_slice(&data);
            std::mem::take(held_headers)
        };
        if let (Some(p), Some(prev)) = (pts, last_pts)
            && p == prev
            && let Some(last) = out.last_mut()
        {
            last.1.extend_from_slice(&data);
            continue;
        }
        let timestamp = match pts {
            Some(p) => pending
                .iter()
                .position(|(queued, _)| *queued == p)
                .and_then(|idx| pending.remove(idx))
                .map(|(_, timestamp)| timestamp)
                .or_else(|| pending.pop_front().map(|(_, timestamp)| timestamp)),
            None => pending.pop_front().map(|(_, timestamp)| timestamp),
        };
        let Some(timestamp) = timestamp else {
            continue;
        };
        last_pts = pts;
        out.push((timestamp, data));
    }
    out
}

fn apply_parameter_sets(mut annex_b: Vec<u8>, cache: &mut ParameterSetCache) -> Vec<u8> {
    if annex_b.is_empty() {
        return annex_b;
    }
    let nals = annex_b_nals(&annex_b);
    for &(start, end, nal_type) in &nals {
        match nal_type {
            7 => cache.sps = Some(annex_b[start..end].to_vec()),
            8 => cache.pps = Some(annex_b[start..end].to_vec()),
            _ => {}
        }
    }
    let is_idr = nals.iter().any(|(_, _, nal_type)| *nal_type == 5);
    let has_sps = nals.iter().any(|(_, _, nal_type)| *nal_type == 7);
    let has_pps = nals.iter().any(|(_, _, nal_type)| *nal_type == 8);
    if is_idr && (!has_sps || !has_pps) {
        let (Some(sps), Some(pps)) = (&cache.sps, &cache.pps) else {
            return annex_b;
        };
        let mut prefixed = Vec::with_capacity(sps.len() + pps.len() + annex_b.len());
        if !has_sps {
            prefixed.extend_from_slice(sps);
        }
        if !has_pps {
            prefixed.extend_from_slice(pps);
        }
        prefixed.extend_from_slice(&annex_b);
        annex_b = prefixed;
    }
    annex_b
}

fn fill_rgba_frame(
    frame: &mut ffmpeg_next::util::frame::Video,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let stride = frame.stride(0);
    let row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| "sensor frame dimensions overflow".to_string())?;
    if stride < row_bytes {
        return Err(format!(
            "ffmpeg RGBA stride {stride} is smaller than row {row_bytes}"
        ));
    }
    let data = frame.data_mut(0);
    for y in 0..height as usize {
        let src = y * row_bytes;
        let dst = y * stride;
        data[dst..dst + row_bytes].copy_from_slice(&rgba[src..src + row_bytes]);
    }
    Ok(())
}

pub(crate) fn to_annex_b(data: &[u8]) -> Vec<u8> {
    if data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1]) {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len() + 8);
    let mut i = 0;
    while i + 4 <= data.len() {
        let n = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;
        if n == 0 || i + n > data.len() {
            return data.to_vec();
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[i..i + n]);
        i += n;
    }
    if out.is_empty() { data.to_vec() } else { out }
}

pub(crate) fn annex_b_nals(data: &[u8]) -> Vec<(usize, usize, u8)> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        let start_code_len = if data.get(i..i + 4) == Some(&[0, 0, 0, 1]) {
            Some(4)
        } else if data.get(i..i + 3) == Some(&[0, 0, 1]) {
            Some(3)
        } else {
            None
        };
        let Some(start_code_len) = start_code_len else {
            i += 1;
            continue;
        };
        let payload_start = i + start_code_len;
        if let Some(&header) = data.get(payload_start) {
            starts.push((i, header & 0x1f));
        }
        i = payload_start + 1;
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &(start, nal_type))| {
            let end = starts
                .get(index + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            (start, end, nal_type)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ParameterSetCache, SensorH264Encoder, annex_b_nals, apply_parameter_sets,
        take_ready_frames, to_annex_b,
    };
    use impeller2::types::Timestamp;
    use std::collections::VecDeque;

    #[test]
    fn delayed_hw_packets_keep_input_timestamps() {
        let mut pending = VecDeque::from([
            (0, Timestamp(100)),
            (1, Timestamp(200)),
            (2, Timestamp(300)),
        ]);
        let mut held = Vec::new();
        assert!(take_ready_frames(&mut pending, &mut held, Vec::new()).is_empty());
        assert_eq!(pending.len(), 3);

        let out = take_ready_frames(
            &mut pending,
            &mut held,
            vec![(Some(0), vec![1]), (Some(1), vec![2])],
        );
        assert_eq!(
            out,
            vec![(Timestamp(100), vec![1]), (Timestamp(200), vec![2])]
        );
        assert_eq!(pending.len(), 1);

        let out = take_ready_frames(&mut pending, &mut held, vec![(Some(2), vec![3])]);
        assert_eq!(out, vec![(Timestamp(300), vec![3])]);
        assert!(pending.is_empty());
    }

    #[test]
    fn same_pts_packets_merge() {
        let mut pending = VecDeque::from([(0, Timestamp(10))]);
        let mut held = Vec::new();
        let out = take_ready_frames(
            &mut pending,
            &mut held,
            vec![(Some(0), vec![1, 2]), (Some(0), vec![3])],
        );
        assert_eq!(out, vec![(Timestamp(10), vec![1, 2, 3])]);
        assert!(pending.is_empty());
    }

    #[test]
    fn avcc_converts_to_annex_b() {
        let avcc = {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&3u32.to_be_bytes());
            bytes.extend_from_slice(&[0x67, 0x42, 0xc0]);
            bytes.extend_from_slice(&2u32.to_be_bytes());
            bytes.extend_from_slice(&[0x68, 0xce]);
            bytes
        };
        assert_eq!(
            to_annex_b(&avcc),
            [0, 0, 0, 1, 0x67, 0x42, 0xc0, 0, 0, 0, 1, 0x68, 0xce]
        );
        let annex = [0, 0, 0, 1, 0x65, 0x88];
        assert_eq!(to_annex_b(&annex), annex);
    }

    #[test]
    fn pps_only_packet_keeps_cached_sps() {
        let mut cache = ParameterSetCache::default();
        let sps_pps = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce];
        apply_parameter_sets(sps_pps.to_vec(), &mut cache);
        apply_parameter_sets([0, 0, 0, 1, 0x68, 0xff].to_vec(), &mut cache);
        assert_eq!(cache.sps.as_deref(), Some(&[0, 0, 0, 1, 0x67, 0x42][..]));
        assert_eq!(cache.pps.as_deref(), Some(&[0, 0, 0, 1, 0x68, 0xff][..]));

        let out = apply_parameter_sets([0, 0, 0, 1, 0x65, 0x88].to_vec(), &mut cache);
        let nals = annex_b_nals(&out);
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 7));
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 8));
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 5));
    }

    #[test]
    fn header_packet_does_not_steal_frame_timestamp() {
        let sps_pps = vec![0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce];
        let idr = vec![0, 0, 0, 1, 0x65, 0x88];
        let mut pending = VecDeque::from([(0, Timestamp(100))]);
        let mut held = Vec::new();

        let out = take_ready_frames(&mut pending, &mut held, vec![(None, sps_pps.clone())]);
        assert!(out.is_empty());
        assert_eq!(pending.len(), 1);
        assert!(!held.is_empty());

        let out = take_ready_frames(&mut pending, &mut held, vec![(Some(0), idr)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, Timestamp(100));
        let nals = annex_b_nals(&out[0].1);
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 7));
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 8));
        assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 5));
        assert!(pending.is_empty());
        assert!(held.is_empty());

        let mut pending = VecDeque::from([(0, Timestamp(200))]);
        let mut held = Vec::new();
        let out = take_ready_frames(
            &mut pending,
            &mut held,
            vec![(None, sps_pps), (Some(0), vec![0, 0, 0, 1, 0x65, 0x88])],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, Timestamp(200));
        assert!(
            annex_b_nals(&out[0].1)
                .iter()
                .any(|(_, _, nal_type)| *nal_type == 5)
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn software_idrs_include_parameter_sets() {
        let mut encoder = SensorH264Encoder::software(32, 32, 30.0).unwrap();
        let mut idr_count = 0;
        for frame in 0..130u8 {
            let rgba = [frame, frame.wrapping_mul(3), 128, 255].repeat(32 * 32);
            for (_, encoded) in encoder.encode(&rgba, Timestamp(i64::from(frame))).unwrap() {
                let nals = annex_b_nals(&encoded);
                if nals.iter().any(|(_, _, nal_type)| *nal_type == 5) {
                    idr_count += 1;
                    assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 7));
                    assert!(nals.iter().any(|(_, _, nal_type)| *nal_type == 8));
                }
            }
        }
        assert!(idr_count >= 2);
    }

    #[test]
    fn rejects_odd_dimensions() {
        assert!(SensorH264Encoder::new(31, 32, 30.0).is_err());
        assert!(SensorH264Encoder::new(32, 31, 30.0).is_err());
    }

    #[test]
    fn vaapi_is_not_opened_without_a_device_context() {
        let err = match super::HardwareEncoder::open("h264_vaapi", 64, 64, 30.0) {
            Ok(_) => panic!("h264_vaapi must not open without a device context"),
            Err(err) => err,
        };
        assert!(err.contains("VAAPI device context"), "{err}");
    }

    #[test]
    fn nvenc_encodes_annex_b_when_available() {
        let Ok(mut encoder) = super::HardwareEncoder::open("h264_nvenc", 256, 256, 30.0) else {
            return;
        };
        let mut saw_idr = false;
        for frame in 0..8u8 {
            let rgba = [frame, 64, 128, 255].repeat(256 * 256);
            for (_, encoded) in encoder
                .encode(&rgba, Timestamp(i64::from(frame)))
                .expect("nvenc encode")
            {
                if encoded.is_empty() {
                    continue;
                }
                assert!(
                    encoded.starts_with(&[0, 0, 0, 1]) || encoded.starts_with(&[0, 0, 1]),
                    "NVENC output must be Annex-B"
                );
                let nals = annex_b_nals(&encoded);
                if nals.iter().any(|(_, _, nal_type)| *nal_type == 5) {
                    saw_idr = true;
                }
            }
        }
        for (_, encoded) in encoder.flush().expect("nvenc flush") {
            let nals = annex_b_nals(&encoded);
            if nals.iter().any(|(_, _, nal_type)| *nal_type == 5) {
                saw_idr = true;
            }
        }
        assert!(saw_idr, "NVENC should emit an IDR in the first GOP");
    }

    #[test]
    fn nvenc_640x512_is_faster_than_60fps() {
        let Ok(mut encoder) = super::HardwareEncoder::open("h264_nvenc", 640, 512, 60.0) else {
            return;
        };
        let rgba = vec![80u8; 640 * 512 * 4];
        let start = std::time::Instant::now();
        let mut packets = 0usize;
        for _ in 0..60 {
            packets += encoder
                .encode(&rgba, Timestamp(0))
                .expect("nvenc encode")
                .len();
        }
        packets += encoder.flush().expect("nvenc flush").len();
        let elapsed = start.elapsed();
        eprintln!(
            "NVENC 640x512: 60 frames in {:.1}ms ({:.0} fps, {packets} packets)",
            elapsed.as_secs_f64() * 1000.0,
            60.0 / elapsed.as_secs_f64().max(1e-6)
        );
        assert!(
            elapsed.as_millis() < 500,
            "NVENC should encode 60 Boson frames well under 1/60s each, took {elapsed:?}"
        );
        assert!(packets >= 50, "NVENC dropped too many frames: {packets}");
    }
}
