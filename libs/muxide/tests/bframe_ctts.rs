mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use support::{parse_boxes, Mp4Box, SharedBuffer};

fn find_box(haystack: &[u8], typ: [u8; 4]) -> Mp4Box {
    *parse_boxes(haystack)
        .iter()
        .find(|b| b.typ == typ)
        .unwrap_or_else(|| panic!("missing box {:?}", std::str::from_utf8(&typ).unwrap()))
}

fn try_find_box(haystack: &[u8], typ: [u8; 4]) -> Option<Mp4Box> {
    parse_boxes(haystack).into_iter().find(|b| b.typ == typ)
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
}

fn be_i32(bytes: &[u8]) -> i32 {
    i32::from_be_bytes(bytes.try_into().unwrap())
}

/// Test that B-frame video produces a ctts box with correct composition time offsets.
#[test]
fn bframe_video_produces_ctts_box() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    // Simulated GOP with B-frames: I P B B (decode order)
    // Display order would be: I B B P
    //
    // Frame   DTS     PTS     CTS (pts-dts)
    // I       0       0       0
    // P       3000    9000    6000    (P displayed after 2 B-frames)
    // B       6000    3000    -3000   (B at display pos 1)
    // B       9000    6000    -3000   (B at display pos 2)

    // SPS+PPS+IDR keyframe
    let frame_i = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc,
        0xdd,
    ];
    // P-frame (non-IDR)
    let frame_p = vec![0x00, 0x00, 0x00, 0x01, 0x41, 0xaa, 0xbb, 0xcc];
    // B-frames
    let frame_b1 = vec![0x00, 0x00, 0x00, 0x01, 0x01, 0x11, 0x22, 0x33];
    let frame_b2 = vec![0x00, 0x00, 0x00, 0x01, 0x01, 0x44, 0x55, 0x66];

    // At 30fps, frame duration is 1/30 sec = 3000 timescale units (90kHz)
    let frame_dur = 1.0 / 30.0;

    // Write frames in decode order with explicit DTS
    // I-frame: pts=0, dts=0
    muxer.write_video_with_dts(0.0, 0.0, &frame_i, true)?;
    // P-frame: pts=3*frame_dur (displayed 4th), dts=frame_dur (decoded 2nd)
    muxer.write_video_with_dts(3.0 * frame_dur, 1.0 * frame_dur, &frame_p, false)?;
    // B-frame 1: pts=1*frame_dur (displayed 2nd), dts=2*frame_dur (decoded 3rd)
    muxer.write_video_with_dts(1.0 * frame_dur, 2.0 * frame_dur, &frame_b1, false)?;
    // B-frame 2: pts=2*frame_dur (displayed 3rd), dts=3*frame_dur (decoded 4th)
    muxer.write_video_with_dts(2.0 * frame_dur, 3.0 * frame_dur, &frame_b2, false)?;

    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let _top = parse_boxes(&produced);

    // Navigate to stbl
    let moov = find_box(&produced, *b"moov");
    let moov_payload = moov.payload(&produced);
    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);
    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    // Verify ctts box exists (only when B-frames present)
    let ctts = try_find_box(stbl_payload, *b"ctts");
    assert!(
        ctts.is_some(),
        "ctts box should be present for B-frame video"
    );

    let ctts = ctts.unwrap();
    let ctts_payload = ctts.payload(stbl_payload);

    // ctts header: version(1)+flags(3) = 4 bytes, entry_count = 4 bytes
    let version = ctts_payload[0];
    assert_eq!(version, 1, "ctts should use version 1 for signed offsets");

    let entry_count = be_u32(&ctts_payload[4..8]);
    // We have 4 samples, but run-length encoding may compress them
    assert!(entry_count >= 1, "ctts should have at least one entry");

    // Verify the CTS offsets are correct
    // For this test we wrote:
    // I: pts=0, dts=0 -> cts=0
    // P: pts=9000, dts=3000 -> cts=6000
    // B1: pts=3000, dts=6000 -> cts=-3000
    // B2: pts=6000, dts=9000 -> cts=-3000

    // Read all entries
    let mut offset = 8;
    let mut all_cts: Vec<i32> = Vec::new();
    for _ in 0..entry_count {
        let count = be_u32(&ctts_payload[offset..offset + 4]) as usize;
        let cts = be_i32(&ctts_payload[offset + 4..offset + 8]);
        for _ in 0..count {
            all_cts.push(cts);
        }
        offset += 8;
    }

    assert_eq!(all_cts.len(), 4, "Should have 4 CTS values for 4 samples");
    assert_eq!(all_cts[0], 0, "I-frame: cts should be 0");
    assert_eq!(
        all_cts[1], 6000,
        "P-frame: cts should be 6000 (pts=9000, dts=3000)"
    );
    assert_eq!(
        all_cts[2], -3000,
        "B1: cts should be -3000 (pts=3000, dts=6000)"
    );
    assert_eq!(
        all_cts[3], -3000,
        "B2: cts should be -3000 (pts=6000, dts=9000)"
    );

    Ok(())
}

/// Test that video without B-frames does NOT produce a ctts box.
#[test]
fn non_bframe_video_has_no_ctts_box() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    // Regular I-P-P video (no B-frames)
    let frame_i = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc,
        0xdd,
    ];
    let frame_p1 = vec![0x00, 0x00, 0x00, 0x01, 0x41, 0xaa, 0xbb, 0xcc];
    let frame_p2 = vec![0x00, 0x00, 0x00, 0x01, 0x41, 0xdd, 0xee, 0xff];

    let frame_dur = 1.0 / 30.0;
    muxer.write_video(0.0, &frame_i, true)?;
    muxer.write_video(frame_dur, &frame_p1, false)?;
    muxer.write_video(2.0 * frame_dur, &frame_p2, false)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();

    // Navigate to stbl
    let moov = find_box(&produced, *b"moov");
    let moov_payload = moov.payload(&produced);
    let trak = find_box(moov_payload, *b"trak");
    let trak_payload = trak.payload(moov_payload);
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let minf = find_box(mdia_payload, *b"minf");
    let minf_payload = minf.payload(mdia_payload);
    let stbl = find_box(minf_payload, *b"stbl");
    let stbl_payload = stbl.payload(minf_payload);

    // ctts should NOT be present
    let ctts = try_find_box(stbl_payload, *b"ctts");
    assert!(
        ctts.is_none(),
        "ctts box should NOT be present for non-B-frame video"
    );

    Ok(())
}
