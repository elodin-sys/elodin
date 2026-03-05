mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use std::{fs, path::Path};
use support::{parse_boxes, Mp4Box, SharedBuffer};

fn read_hex_fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("video_samples")
        .join(name);
    let contents = fs::read_to_string(path).expect("fixture must be readable");
    let hex: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(hex.len() % 2 == 0, "hex fixtures must have even length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex");
        out.push(byte);
    }
    out
}

fn find_box(haystack: &[u8], typ: [u8; 4]) -> Mp4Box {
    *parse_boxes(haystack)
        .iter()
        .find(|b| b.typ == typ)
        .unwrap_or_else(|| panic!("missing box {:?}", std::str::from_utf8(&typ).unwrap()))
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
}

#[test]
fn timebase_30fps_has_exact_3000_tick_delta() -> Result<(), Box<dyn std::error::Error>> {
    let key = read_hex_fixture("frame0_key.264");
    let p = read_hex_fixture("frame1_p.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    // Use the CrabCamera convention: pts = frame_number / framerate.
    muxer.write_video(0.0, &key, true)?;
    muxer.write_video(1.0 / 30.0, &p, false)?;
    muxer.write_video(2.0 / 30.0, &p, false)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);
    let moov = top.iter().find(|b| b.typ == *b"moov").unwrap();

    // Navigate to stts.
    let moov_payload = moov.payload(&produced);
    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);
    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    let stts = find_box(stbl_payload, *b"stts");
    let stts_payload = stts.payload(stbl_payload);

    // Single entry: count=3, delta=3000 (90kHz/30fps).
    assert_eq!(be_u32(&stts_payload[4..8]), 1);
    assert_eq!(be_u32(&stts_payload[8..12]), 3);
    assert_eq!(be_u32(&stts_payload[12..16]), 3000);

    Ok(())
}

#[test]
fn timebase_long_run_does_not_drift_at_30fps() -> Result<(), Box<dyn std::error::Error>> {
    let key = read_hex_fixture("frame0_key.264");
    let p = read_hex_fixture("frame1_p.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    let frames = 300u32; // 10 seconds at 30fps
    for i in 0..frames {
        let pts = (i as f64) / 30.0;
        let is_key = i == 0;
        let data = if is_key { &key } else { &p };
        muxer.write_video(pts, data, is_key)?;
    }
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);
    let moov = top.iter().find(|b| b.typ == *b"moov").unwrap();

    let moov_payload = moov.payload(&produced);
    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);
    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    let stts = find_box(stbl_payload, *b"stts");
    let stts_payload = stts.payload(stbl_payload);

    // Should collapse to one entry: count=300, delta=3000.
    assert_eq!(be_u32(&stts_payload[4..8]), 1);
    assert_eq!(be_u32(&stts_payload[8..12]), frames);
    assert_eq!(be_u32(&stts_payload[12..16]), 3000);

    Ok(())
}
