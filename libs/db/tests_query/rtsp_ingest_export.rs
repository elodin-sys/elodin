//! End-to-end storage-contract test for the RTSP ingest path, without retina,
//! the network, or hardware.
//!
//! Real H.264 is produced by the openh264 encoder, then reshaped into exactly
//! what `retina` hands us — length-prefixed AVC access units with the SPS/PPS
//! held *out-of-band* — and routed through the `rtsp-ingest` converter (which
//! must re-inject SPS/PPS ahead of the IDR) and `ClockMapper` before landing in
//! `MsgLog` via `DB::push_msg`. Finally `export_videos` muxes an MP4 and we
//! confirm the SPS round-trips with the expected resolution.

use std::collections::HashMap;
use std::io::Cursor;

use elodin_db::DB;
use impeller2::types::{Timestamp, msg_id};
use impeller2_wkt::{MsgMetadata, opaque_bytes_msg_schema};
use openh264::OpenH264API;
use openh264::encoder::{
    BitRate, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, RateControlMode, SpsPpsStrategy,
};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use rtsp_ingest::annexb::{
    AnnexBConverter, ParameterSets, annexb_contains_idr, nal_type, nal_unit_type, split_annexb_nals,
};
use rtsp_ingest::clock::ClockMapper;
use scuffle_h264::Sps;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FPS: u32 = 30;
const FRAMES: usize = 12;
const MSG_NAME: &str = "rtsp-camera";

/// A moving RGBA test pattern so successive frames actually differ.
fn rgba_frame(step: u8) -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let mut frame = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            frame.push((x as u8).wrapping_mul(4).wrapping_add(step.wrapping_mul(7)));
            frame.push((y as u8).wrapping_mul(4).wrapping_add(step.wrapping_mul(3)));
            frame.push(64u8.wrapping_add(step.wrapping_mul(11)));
            frame.push(255);
        }
    }
    frame
}

/// Pulls the first SPS NAL out of the avcC box of an MP4 (big-endian length-prefixed).
fn extract_avcc_sps(mp4: &[u8]) -> Option<&[u8]> {
    let avcc_pos = mp4.windows(4).position(|window| window == b"avcC")?;
    let avcc = mp4.get(avcc_pos + 4..)?;
    if avcc.len() < 8 {
        return None;
    }
    let sps_count = avcc[5] & 0x1f;
    if sps_count == 0 {
        return None;
    }
    let sps_len = u16::from_be_bytes([avcc[6], avcc[7]]) as usize;
    avcc.get(8..8 + sps_len)
}

/// Encodes `FRAMES` synthetic frames into real per-frame H.264 Annex-B bitstreams.
fn encode_h264_frames() -> Vec<Vec<u8>> {
    let target_bitrate = ((WIDTH as f64) * (HEIGHT as f64) * (FPS as f64) * 3.0)
        .clamp(300_000.0, 12_000_000.0) as u32;
    let config = EncoderConfig::new()
        .bitrate(BitRate::from_bps(target_bitrate))
        .skip_frames(false)
        .max_frame_rate(FrameRate::from_hz(FPS as f32))
        .rate_control_mode(RateControlMode::Off)
        .sps_pps_strategy(SpsPpsStrategy::ConstantId)
        // One keyframe well beyond our frame count: a single IDR at the start.
        .intra_frame_period(IntraFramePeriod::from_num_frames(FPS * 4));
    let mut encoder =
        Encoder::with_api_config(OpenH264API::from_source(), config).expect("openh264 init");

    (0..FRAMES)
        .map(|step| {
            let frame = rgba_frame(step as u8);
            let rgba = RgbaSliceU8::new(&frame, (WIDTH as usize, HEIGHT as usize));
            let yuv = YUVBuffer::from_rgb_source(rgba);
            encoder.encode(&yuv).expect("openh264 encode").to_vec()
        })
        .collect()
}

/// Re-frames an Annex-B picture into a `retina`-style AVC access unit: SPS/PPS
/// are stripped to out-of-band parameter sets, the remaining NALs are emitted
/// length-prefixed (4-byte big-endian). Captured parameter sets are merged into
/// `params`.
fn annexb_to_avc_au(annexb: &[u8], params: &mut ParameterSets) -> Vec<u8> {
    let mut au = Vec::new();
    for nal in split_annexb_nals(annexb) {
        match nal_unit_type(nal) {
            Some(nal_type::SPS) => params.sps = nal.to_vec(),
            Some(nal_type::PPS) => params.pps = nal.to_vec(),
            _ => {
                au.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                au.extend_from_slice(nal);
            }
        }
    }
    au
}

#[test]
fn rtsp_avc_access_units_export_to_mp4() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");

    let id = msg_id(MSG_NAME);
    // Name the message log so `export-videos` names the file `rtsp-camera.mp4`
    // (mirrors the SetMsgMetadata the ingest task sends on connect).
    db.with_state_mut(|state| {
        state.set_msg_metadata(
            id,
            MsgMetadata {
                name: MSG_NAME.to_string(),
                schema: opaque_bytes_msg_schema(),
                metadata: HashMap::new(),
            },
            &db_path,
        )
    })
    .expect("set_msg_metadata");

    let frames = encode_h264_frames();

    // The encoder's first picture carries SPS/PPS; capture them out-of-band and
    // strip them from the AUs, exactly as a retina pull would deliver them.
    let mut params = ParameterSets::default();
    let mut avc_aus = Vec::new();
    for annexb in &frames {
        let au = annexb_to_avc_au(annexb, &mut params);
        if !au.is_empty() {
            avc_aus.push(au);
        }
    }
    assert!(
        params.is_complete(),
        "expected SPS+PPS from the encoder's first IDR"
    );
    assert!(!avc_aus.is_empty(), "expected at least one access unit");

    let converter = AnnexBConverter::new(params);
    let mut clock = ClockMapper::new(1_000_000);
    let pts_step = 1_000_000 / FPS as i64;
    let mut saw_injected_keyframe = false;

    for (i, au) in avc_aus.iter().enumerate() {
        let annexb = converter.convert(au).expect("convert AVC->AnnexB");
        if annexb_contains_idr(&annexb) {
            // Injection contract: stored keyframes must carry an in-band SPS.
            let has_sps = split_annexb_nals(&annexb)
                .iter()
                .any(|n| nal_unit_type(n) == Some(nal_type::SPS));
            assert!(has_sps, "keyframe must have in-band SPS after conversion");
            saw_injected_keyframe = true;
        }
        let ts = clock.map(1_000_000 + i as i64 * pts_step);
        db.push_msg(Timestamp(ts), id, &annexb).expect("push_msg");
    }
    assert!(saw_injected_keyframe, "expected at least one keyframe");

    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, FPS).expect("export_videos");

    let mp4_path = out_path.join(format!("{}.mp4", MSG_NAME));
    let mp4 = std::fs::read(&mp4_path).expect("read mp4");
    assert!(mp4.len() > 128, "expected non-empty mp4");
    assert_eq!(mp4.get(4..8), Some(&b"ftyp"[..]), "expected MP4 ftyp box");

    let sps = extract_avcc_sps(&mp4).expect("avcC SPS");
    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps)).expect("parse SPS");
    assert_eq!(sps.width(), WIDTH as u64);
    assert_eq!(sps.height(), HEIGHT as u64);
}
