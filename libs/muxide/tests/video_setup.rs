mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use support::{parse_boxes, SharedBuffer};

#[test]
fn video_track_structure_contains_expected_boxes() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();

    let muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .build()?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top_boxes = parse_boxes(&produced);
    let moov = top_boxes
        .iter()
        .find(|b| b.typ == *b"moov")
        .expect("moov box must exist");

    let moov_payload = moov.payload(&produced);
    let moov_children = parse_boxes(moov_payload);
    assert!(
        moov_children.iter().any(|b| b.typ == *b"trak"),
        "trak missing"
    );
    let trak = moov_children.iter().find(|b| b.typ == *b"trak").unwrap();

    let trak_payload = trak.payload(moov_payload);
    let mdia_boxes = parse_boxes(trak_payload);
    let mdia = mdia_boxes.iter().find(|b| b.typ == *b"mdia").unwrap();

    let mdia_payload = mdia.payload(trak_payload);
    let minf_boxes = parse_boxes(mdia_payload);
    let minf = minf_boxes.iter().find(|b| b.typ == *b"minf").unwrap();

    let minf_payload = minf.payload(mdia_payload);
    let stbl_boxes = parse_boxes(minf_payload);
    let stbl = stbl_boxes.iter().find(|b| b.typ == *b"stbl").unwrap();

    let stbl_payload = stbl.payload(minf_payload);
    let stsd_boxes = parse_boxes(stbl_payload);
    let stsd = stsd_boxes.iter().find(|b| b.typ == *b"stsd").unwrap();

    let stsd_payload = stsd.payload(stbl_payload);
    let entries_payload = &stsd_payload[8..];
    let avc1_boxes = parse_boxes(entries_payload);
    let avc1 = avc1_boxes.iter().find(|b| b.typ == *b"avc1").unwrap();

    let avc1_payload = avc1.payload(entries_payload);
    let avc_c_index = avc1_payload
        .windows(4)
        .position(|window| window == b"avcC")
        .expect("avcC box must exist in avc1");
    let size_start = avc_c_index - 4;
    let avc_c_size =
        u32::from_be_bytes(avc1_payload[size_start..size_start + 4].try_into().unwrap()) as usize;
    let avc_c_payload = &avc1_payload[size_start + 8..size_start + avc_c_size];
    assert!(
        avc_c_payload.windows(1).any(|w| w[0] == 0x67),
        "SPS missing in avcC"
    );
    assert!(
        avc_c_payload.windows(1).any(|w| w[0] == 0x68),
        "PPS missing in avcC"
    );

    Ok(())
}
