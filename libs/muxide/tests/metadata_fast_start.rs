mod support;

use muxide::api::{Metadata, MuxerBuilder, VideoCodec};
use support::{parse_boxes, Mp4Box, SharedBuffer};

fn find_box(haystack: &[u8], typ: [u8; 4]) -> Mp4Box {
    *parse_boxes(haystack)
        .iter()
        .find(|b| b.typ == typ)
        .unwrap_or_else(|| panic!("missing box {:?}", std::str::from_utf8(&typ).unwrap()))
}

#[test]
fn metadata_title_appears_in_udta_box() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();

    let metadata = Metadata {
        title: Some("Test Video Title".to_string()),
        creation_time: Some(3600), // 1 hour since 1904
        language: None,
    };

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .with_metadata(metadata)
        .build()?;

    // Write a single keyframe
    let frame0 = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc,
        0xdd,
    ];
    muxer.write_video(0.0, &frame0, true)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);

    // Fast-start: ftyp, moov, mdat
    assert_eq!(top[0].typ, *b"ftyp");
    assert_eq!(top[1].typ, *b"moov");
    assert_eq!(top[2].typ, *b"mdat");

    let moov_payload = top[1].payload(&produced);

    // Look for udta box in moov
    let udta = find_box(moov_payload, *b"udta");
    assert!(udta.size > 8, "udta box should contain metadata");

    let udta_payload = udta.payload(moov_payload);

    // udta should contain a meta box
    let meta = find_box(udta_payload, *b"meta");
    assert!(meta.size > 8, "meta box should contain data");

    // Verify the title string appears somewhere in the metadata
    let title_bytes = b"Test Video Title";
    let produced_slice = &produced[..];
    let contains_title = produced_slice
        .windows(title_bytes.len())
        .any(|w| w == title_bytes);
    assert!(contains_title, "Title string should appear in the output");

    Ok(())
}

#[test]
fn fast_start_puts_moov_before_mdat() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();

    // fast_start is true by default
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    let frame0 = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc,
        0xdd,
    ];
    muxer.write_video(0.0, &frame0, true)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);

    assert_eq!(top[0].typ, *b"ftyp", "First box should be ftyp");
    assert_eq!(
        top[1].typ, *b"moov",
        "Second box should be moov (fast-start)"
    );
    assert_eq!(top[2].typ, *b"mdat", "Third box should be mdat");

    // Verify moov comes BEFORE mdat in the byte stream
    let moov_offset = top[1].offset;
    let mdat_offset = top[2].offset;
    assert!(
        moov_offset < mdat_offset,
        "moov should come before mdat for fast start"
    );

    Ok(())
}

#[test]
fn fast_start_false_puts_mdat_before_moov() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .with_fast_start(false) // Disable fast-start
        .build()?;

    let frame0 = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc,
        0xdd,
    ];
    muxer.write_video(0.0, &frame0, true)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);

    assert_eq!(top[0].typ, *b"ftyp", "First box should be ftyp");
    assert_eq!(
        top[1].typ, *b"mdat",
        "Second box should be mdat (standard mode)"
    );
    assert_eq!(top[2].typ, *b"moov", "Third box should be moov");

    Ok(())
}
