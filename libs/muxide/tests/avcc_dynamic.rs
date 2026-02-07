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

fn extract_avcc_payload(produced: &[u8]) -> Vec<u8> {
    let top = parse_boxes(produced);
    let moov = top.iter().find(|b| b.typ == *b"moov").expect("moov");
    let moov_payload = moov.payload(produced);

    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);

    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);

    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);

    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    let stsd = find_box(stbl_payload, *b"stsd");
    let stsd_payload = stsd.payload(stbl_payload);

    // Skip full box header + entry count.
    let entries_payload = &stsd_payload[8..];
    let avc1_boxes = parse_boxes(entries_payload);
    let avc1 = avc1_boxes.iter().find(|b| b.typ == *b"avc1").expect("avc1");
    let avc1_payload = avc1.payload(entries_payload);

    let avc_c_index = avc1_payload
        .windows(4)
        .position(|window| window == b"avcC")
        .expect("avcC box must exist in avc1");
    let size_start = avc_c_index - 4;
    let avc_c_size =
        u32::from_be_bytes(avc1_payload[size_start..size_start + 4].try_into().unwrap()) as usize;
    avc1_payload[size_start + 8..size_start + avc_c_size].to_vec()
}

#[test]
fn avcc_uses_sps_pps_from_first_keyframe() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("frame0_key_alt.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let avcc_payload = extract_avcc_payload(&produced);

    // Our alt SPS begins with: 67 4d 00 28 ...
    let expected_profile = 0x4d;
    let expected_compat = 0x00;
    let expected_level = 0x28;

    assert!(avcc_payload
        .windows(6)
        .any(|w| w == [0x67, 0x4d, 0x00, 0x28, 0xaa, 0xbb]));
    assert!(avcc_payload
        .windows(4)
        .any(|w| w == [0x68, 0xee, 0x06, 0xf2]));

    // avcC header bytes must match SPS profile/compat/level.
    assert!(avcc_payload.len() >= 4);
    assert_eq!(avcc_payload[0], 1);
    assert_eq!(avcc_payload[1], expected_profile);
    assert_eq!(avcc_payload[2], expected_compat);
    assert_eq!(avcc_payload[3], expected_level);

    Ok(())
}
