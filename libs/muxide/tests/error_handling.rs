use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, MuxerError, VideoCodec};
use std::{fs, path::Path};

mod support;
use support::SharedBuffer;

fn read_hex_fixture(dir: &str, name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(dir)
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

#[test]
fn errors_are_specific_and_descriptive() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("video_samples", "frame0_key.264");
    let non_sps_pps = read_hex_fixture("video_samples", "frame1_p.264");

    // Video pts must be non-negative.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()?;
        let err = muxer.write_video(-0.001, &frame0, true).unwrap_err();
        assert!(matches!(err, MuxerError::NegativeVideoPts { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("negative"),
            "error should mention negative: {}",
            msg
        );
        assert!(
            msg.contains("frame 0"),
            "error should include frame index: {}",
            msg
        );
    }

    // First frame must be a keyframe.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()?;
        let err = muxer.write_video(0.0, &frame0, false).unwrap_err();
        assert!(matches!(err, MuxerError::FirstVideoFrameMustBeKeyframe));
        assert!(err.to_string().contains("keyframe"));
    }

    // First keyframe must contain SPS/PPS.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()?;
        let err = muxer.write_video(0.0, &non_sps_pps, true).unwrap_err();
        assert!(matches!(err, MuxerError::FirstVideoFrameMissingSpsPps));
        assert!(err.to_string().contains("SPS"));
    }

    // Video pts must be strictly increasing.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()?;
        muxer.write_video(0.0, &frame0, true)?;
        let err = muxer.write_video(0.0, &frame0, false).unwrap_err();
        assert!(matches!(err, MuxerError::NonIncreasingVideoPts { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("frame 1"),
            "error should include frame index: {}",
            msg
        );
        assert!(
            msg.contains("increase"),
            "error should mention increasing: {}",
            msg
        );
    }

    // Audio must not arrive before first video frame.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
            .build()?;
        let err = muxer
            .write_audio(0.0, &[0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc])
            .unwrap_err();
        assert!(matches!(err, MuxerError::AudioBeforeFirstVideo { .. }));
        assert!(err.to_string().contains("video"));
    }

    // Invalid ADTS should surface as InvalidAdts.
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
            .build()?;
        muxer.write_video(0.0, &frame0, true)?;
        let err = muxer.write_audio(0.0, &[0, 1, 2, 3]).unwrap_err();
        assert!(matches!(err, MuxerError::InvalidAdtsDetailed { .. }));
        assert!(err.to_string().contains("ADTS"));
    }

    Ok(())
}
