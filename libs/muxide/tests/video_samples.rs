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
fn video_samples_writes_mdat_and_tables() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("frame0_key.264");
    let frame1 = read_hex_fixture("frame1_p.264");
    let frame2 = read_hex_fixture("frame2_p.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.write_video(1.0 / 30.0, &frame1, false)?;
    muxer.write_video(2.0 / 30.0, &frame2, false)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);
    assert_eq!(top[0].typ, *b"ftyp");
    // Fast-start is enabled by default: moov comes before mdat
    assert_eq!(top[1].typ, *b"moov");
    assert_eq!(top[2].typ, *b"mdat");

    let ftyp = top[0];
    let moov = top[1];
    let mdat = top[2];

    // stco should point to the first byte of mdat payload (after moov).
    let expected_chunk_offset = (ftyp.size + moov.size + 8) as u32;

    // Verify mdat begins with a 4-byte NAL length (AVCC format).
    let mdat_payload = mdat.payload(&produced);
    assert!(mdat_payload.len() >= 4);
    let first_nal_len = be_u32(&mdat_payload[0..4]) as usize;
    assert!(first_nal_len > 0);
    assert!(mdat_payload.len() >= 4 + first_nal_len);

    // Navigate to stbl.
    let moov_payload = moov.payload(&produced);
    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);
    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    // stts: single entry with count=3, delta=3000 (90kHz / 30fps).
    let stts = find_box(stbl_payload, *b"stts");
    let stts_payload = stts.payload(stbl_payload);
    assert_eq!(be_u32(&stts_payload[4..8]), 1);
    assert_eq!(be_u32(&stts_payload[8..12]), 3);
    assert_eq!(be_u32(&stts_payload[12..16]), 3000);

    // stsc: one chunk containing all 3 samples.
    let stsc = find_box(stbl_payload, *b"stsc");
    let stsc_payload = stsc.payload(stbl_payload);
    assert_eq!(be_u32(&stsc_payload[4..8]), 1);
    assert_eq!(be_u32(&stsc_payload[8..12]), 1);
    assert_eq!(be_u32(&stsc_payload[12..16]), 3);
    assert_eq!(be_u32(&stsc_payload[16..20]), 1);

    // stsz: sample sizes match AVCC conversion (length prefixes included).
    let stsz = find_box(stbl_payload, *b"stsz");
    let stsz_payload = stsz.payload(stbl_payload);
    assert_eq!(be_u32(&stsz_payload[4..8]), 0);
    assert_eq!(be_u32(&stsz_payload[8..12]), 3);

    // Frame0 contains 3 NALs (SPS, PPS, IDR), each length-prefixed.
    let expected_size0 = (4 + 10) + (4 + 4) + (4 + 5); // based on fixture bytes
    let expected_size1 = 4 + (frame1.len() - 4); // start code removed, 4-byte length added
    let expected_size2 = 4 + (frame2.len() - 4);

    assert_eq!(be_u32(&stsz_payload[12..16]) as usize, expected_size0);
    assert_eq!(be_u32(&stsz_payload[16..20]) as usize, expected_size1);
    assert_eq!(be_u32(&stsz_payload[20..24]) as usize, expected_size2);

    // stco: one chunk offset.
    let stco = find_box(stbl_payload, *b"stco");
    let stco_payload = stco.payload(stbl_payload);
    assert_eq!(be_u32(&stco_payload[4..8]), 1);
    assert_eq!(be_u32(&stco_payload[8..12]), expected_chunk_offset);

    // stss: only the first sample is a sync sample.
    let stss = find_box(stbl_payload, *b"stss");
    let stss_payload = stss.payload(stbl_payload);
    assert_eq!(be_u32(&stss_payload[4..8]), 1);
    assert_eq!(be_u32(&stss_payload[8..12]), 1);

    Ok(())
}
