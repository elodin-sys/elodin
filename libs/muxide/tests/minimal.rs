mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use std::{fs, path::Path};
use support::SharedBuffer;

#[test]
fn minimal_writer_matches_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (writer, buffer) = SharedBuffer::new();

    let muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    muxer.finish()?;

    let produced = buffer.lock().unwrap().clone();
    let fixture = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/minimal.mp4"))?;
    assert_eq!(
        produced, fixture,
        "build output must match the golden minimal file"
    );
    Ok(())
}
