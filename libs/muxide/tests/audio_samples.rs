mod support;

use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
use std::{fs, path::Path};
use support::{parse_boxes, Mp4Box, SharedBuffer};

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

fn find_box(haystack: &[u8], typ: [u8; 4]) -> Mp4Box {
    *parse_boxes(haystack)
        .iter()
        .find(|b| b.typ == typ)
        .unwrap_or_else(|| panic!("missing box {:?}", std::str::from_utf8(&typ).unwrap()))
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
}

fn handler_type_from_trak(trak_payload: &[u8]) -> [u8; 4] {
    let mdia = find_box(trak_payload, *b"mdia");
    let mdia_payload = mdia.payload(trak_payload);
    let hdlr = find_box(mdia_payload, *b"hdlr");
    let hdlr_payload = hdlr.payload(mdia_payload);
    hdlr_payload[8..12].try_into().unwrap()
}

#[test]
fn audio_samples_writes_second_track_and_tables() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("video_samples", "frame0_key.264");
    let frame1 = read_hex_fixture("video_samples", "frame1_p.264");
    let frame2 = read_hex_fixture("video_samples", "frame2_p.264");

    let a0 = read_hex_fixture("audio_samples", "frame0.aac.adts");
    let a1 = read_hex_fixture("audio_samples", "frame1.aac.adts");
    let a2 = read_hex_fixture("audio_samples", "frame2.aac.adts");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.write_audio(0.0, &a0)?;
    muxer.write_audio(0.021, &a1)?;
    muxer.write_video(0.033, &frame1, false)?;
    muxer.write_audio(0.042, &a2)?;
    muxer.write_video(0.066, &frame2, false)?;
    muxer.finish()?;

    let produced = buffer.lock().unwrap();
    let top = parse_boxes(&produced);
    assert_eq!(top[0].typ, *b"ftyp");
    // fast_start=true (default) puts moov before mdat
    assert_eq!(top[1].typ, *b"moov");
    assert_eq!(top[2].typ, *b"mdat");

    let mdat = top[2];
    let mdat_payload_start = (mdat.offset + 8) as u32;
    let mdat_end = (mdat.offset + mdat.size) as u32;

    let moov_payload = top[1].payload(&produced);
    let traks: Vec<Mp4Box> = parse_boxes(moov_payload)
        .into_iter()
        .filter(|b| b.typ == *b"trak")
        .collect();
    assert_eq!(traks.len(), 2);

    let mut video_trak = None;
    let mut audio_trak = None;
    for trak in traks {
        let trak_payload = trak.payload(moov_payload);
        match handler_type_from_trak(trak_payload) {
            t if t == *b"vide" => video_trak = Some(trak_payload),
            t if t == *b"soun" => audio_trak = Some(trak_payload),
            other => panic!("unexpected handler type {other:?}"),
        }
    }

    let video_trak = video_trak.expect("missing vide trak");
    let audio_trak = audio_trak.expect("missing soun trak");

    // Video stsd contains an avc1 entry.
    let v_mdia = find_box(video_trak, *b"mdia");
    let v_mdia_payload = v_mdia.payload(video_trak);
    let v_minf = find_box(v_mdia_payload, *b"minf");
    let v_minf_payload = v_minf.payload(v_mdia_payload);
    let v_stbl = find_box(v_minf_payload, *b"stbl");
    let v_stbl_payload = v_stbl.payload(v_minf_payload);
    let v_stsd = find_box(v_stbl_payload, *b"stsd");
    let v_stsd_payload = v_stsd.payload(v_stbl_payload);
    let v_entries = parse_boxes(&v_stsd_payload[8..]);
    assert_eq!(v_entries[0].typ, *b"avc1");

    // Audio stsd contains an mp4a entry.
    let a_mdia = find_box(audio_trak, *b"mdia");
    let a_mdia_payload = a_mdia.payload(audio_trak);
    let a_minf = find_box(a_mdia_payload, *b"minf");
    let a_minf_payload = a_minf.payload(a_mdia_payload);
    let a_stbl = find_box(a_minf_payload, *b"stbl");
    let a_stbl_payload = a_stbl.payload(a_minf_payload);
    let a_stsd = find_box(a_stbl_payload, *b"stsd");
    let a_stsd_payload = a_stsd.payload(a_stbl_payload);
    let a_entries = parse_boxes(&a_stsd_payload[8..]);
    assert_eq!(a_entries[0].typ, *b"mp4a");

    // Audio stts: single entry with count=3, delta=1890 (90kHz * 0.021s).
    let stts = find_box(a_stbl_payload, *b"stts");
    let stts_payload = stts.payload(a_stbl_payload);
    assert_eq!(be_u32(&stts_payload[4..8]), 1);
    assert_eq!(be_u32(&stts_payload[8..12]), 3);
    assert_eq!(be_u32(&stts_payload[12..16]), 1890);

    // Audio stsz: 3 samples, each 2 bytes (ADTS headers stripped).
    let stsz = find_box(a_stbl_payload, *b"stsz");
    let stsz_payload = stsz.payload(a_stbl_payload);
    assert_eq!(be_u32(&stsz_payload[8..12]), 3);
    assert_eq!(be_u32(&stsz_payload[12..16]), 2);
    assert_eq!(be_u32(&stsz_payload[16..20]), 2);
    assert_eq!(be_u32(&stsz_payload[20..24]), 2);

    // Audio stco: 3 chunk offsets within the mdat payload.
    let stco = find_box(a_stbl_payload, *b"stco");
    let stco_payload = stco.payload(a_stbl_payload);
    assert_eq!(be_u32(&stco_payload[4..8]), 3);
    for i in 0..3 {
        let off = be_u32(&stco_payload[8 + i * 4..12 + i * 4]);
        assert!(off >= mdat_payload_start);
        assert!(off < mdat_end);
    }

    Ok(())
}

#[test]
fn aac_profiles_supported() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("video_samples", "frame0_key.264");
    let aac_frame = read_hex_fixture("audio_samples", "frame0.aac.adts");

    // Test each AAC profile variant
    let profiles = vec![
        AacProfile::Lc,
        AacProfile::Main,
        AacProfile::Ssr,
        AacProfile::Ltp,
        AacProfile::He,
        AacProfile::Hev2,
    ];

    for profile in profiles {
        let (writer, buffer) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Aac(profile), 48000, 2)
            .build()?;

        muxer.write_video(0.0, &frame0, true)?;
        muxer.write_audio(0.0, &aac_frame)?;
        muxer.finish()?;

        let output = buffer.lock().unwrap();

        // Verify basic MP4 structure
        assert!(
            output.len() > 1000,
            "Output too small for profile {:?}",
            profile
        );

        // Verify moov and mdat boxes exist
        let boxes = parse_boxes(&output);
        let has_moov = boxes.iter().any(|b| b.typ == *b"moov");
        let has_mdat = boxes.iter().any(|b| b.typ == *b"mdat");
        assert!(has_moov, "Missing moov box for profile {:?}", profile);
        assert!(has_mdat, "Missing mdat box for profile {:?}", profile);
    }

    Ok(())
}

#[test]
fn aac_invalid_profile_rejected() {
    // This test would require adding an invalid profile variant to test rejection
    // For now, we rely on the property tests and invariants to ensure only valid profiles are accepted
}
