use std::io::Cursor;

use elodin_db::DB;
use impeller2::types::{Timestamp, msg_id};
use impeller2_wkt::{Color, SensorCameraConfig};
use openh264::OpenH264API;
use openh264::encoder::{Encoder, EncoderConfig, FrameRate, IntraFramePeriod, SpsPpsStrategy};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use scuffle_h264::Sps;

fn color(r: f32, g: f32, b: f32, a: f32) -> Color {
    Color { r, g, b, a }
}

fn sensor_camera_config() -> SensorCameraConfig {
    SensorCameraConfig {
        entity_name: "drone".to_string(),
        camera_name: "drone.fpv".to_string(),
        width: 32,
        height: 32,
        fov_degrees: 90.0,
        near: 0.02,
        far: 100.0,
        pos_offset: [0.0; 3],
        rot_offset: [0.0; 3],
        format: "rgba".to_string(),
        effect: "normal".to_string(),
        effect_params: serde_json::json!({}),
        create_frustum: false,
        show_ellipsoids: false,
        frustums_color: color(0.0, 1.0, 0.4, 0.4),
        projection_color: color(0.0, 1.0, 0.4, 0.1),
        frustums_thickness: 0.008,
        fps: 60.0,
        ..Default::default()
    }
}

fn rgba_frame(width: u32, height: u32, step: u8) -> Vec<u8> {
    let mut frame = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            frame.push((x as u8).wrapping_mul(8).wrapping_add(step));
            frame.push((y as u8).wrapping_mul(8).wrapping_add(step.wrapping_mul(3)));
            frame.push(128u8.wrapping_add(step.wrapping_mul(9)));
            frame.push(255);
        }
    }
    frame
}

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

fn encode_h264_frames(camera: &SensorCameraConfig, count: usize) -> Vec<Vec<u8>> {
    let config = EncoderConfig::new()
        .skip_frames(false)
        .max_frame_rate(FrameRate::from_hz(camera.fps))
        .sps_pps_strategy(SpsPpsStrategy::ConstantId)
        .intra_frame_period(IntraFramePeriod::from_num_frames(120));
    let mut encoder =
        Encoder::with_api_config(OpenH264API::from_source(), config).expect("openh264 init");
    (0..count)
        .map(|step| {
            let frame = rgba_frame(camera.width, camera.height, step as u8);
            let rgba = RgbaSliceU8::new(&frame, (camera.width as usize, camera.height as usize));
            encoder
                .encode(&YUVBuffer::from_rgb_source(rgba))
                .expect("openh264 encode")
                .to_vec()
        })
        .collect()
}

#[test]
fn sensor_camera_rgba_msg_log_exports_to_mp4() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");
    let camera = sensor_camera_config();
    let sensor_json = serde_json::to_string(&vec![camera.clone()]).expect("sensor json");

    db.with_state_mut(|state| {
        state
            .db_config
            .metadata
            .insert("sensor_cameras".to_string(), sensor_json);
    });

    let id = msg_id(&camera.camera_name);
    for step in 0..5 {
        let frame = rgba_frame(camera.width, camera.height, step as u8);
        db.push_msg(Timestamp(1_000_000 + step * 33_333), id, &frame)
            .expect("push_msg");
    }
    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, 30).expect("export_videos");

    let mp4_path = out_path.join("drone.fpv.mp4");
    let mp4 = std::fs::read(&mp4_path).expect("read mp4");
    assert!(
        mp4.len() > 128,
        "expected non-empty mp4 at {}",
        mp4_path.display()
    );
    assert_eq!(mp4.get(4..8), Some(&b"ftyp"[..]));

    let sps = extract_avcc_sps(&mp4).expect("avcC SPS");
    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps)).expect("parse SPS");
    assert_eq!(sps.width(), camera.width as u64);
    assert_eq!(sps.height(), camera.height as u64);
}

#[test]
fn sensor_camera_gray8_msg_log_exports_to_mp4() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");
    let mut camera = sensor_camera_config();
    camera.format = "gray8".to_string();
    let sensor_json = serde_json::to_string(&vec![camera.clone()]).expect("sensor json");
    db.with_state_mut(|state| {
        state
            .db_config
            .metadata
            .insert("sensor_cameras".to_string(), sensor_json);
    });

    let id = msg_id(&camera.camera_name);
    for step in 0..5 {
        let frame = (0..camera.width * camera.height)
            .map(|pixel| (pixel as u8).wrapping_add(step as u8))
            .collect::<Vec<_>>();
        db.push_msg(Timestamp(1_000_000 + step * 16_667), id, &frame)
            .expect("push_msg");
    }
    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, 30).expect("export_videos");

    let mp4 = std::fs::read(out_path.join("drone.fpv.mp4")).expect("read mp4");
    let sps = extract_avcc_sps(&mp4).expect("avcC SPS");
    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps)).expect("parse SPS");
    assert_eq!(sps.width(), camera.width as u64);
    assert_eq!(sps.height(), camera.height as u64);
}

#[test]
fn malformed_sensor_camera_frames_do_not_leave_mp4() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");
    let camera = sensor_camera_config();
    let sensor_json = serde_json::to_string(&vec![camera.clone()]).expect("sensor json");

    db.with_state_mut(|state| {
        state
            .db_config
            .metadata
            .insert("sensor_cameras".to_string(), sensor_json);
    });

    let id = msg_id(&camera.camera_name);
    db.push_msg(Timestamp(1_000_000), id, &[0, 1, 2, 3])
        .expect("push malformed msg");
    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, 30).expect("export_videos");

    let mp4_path = out_path.join("drone.fpv.mp4");
    assert!(
        !mp4_path.exists(),
        "malformed sensor_camera frames should not leave {}",
        mp4_path.display()
    );
}

#[test]
fn sensor_camera_h264_msg_log_muxes_without_reencoding() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");
    let mut camera = sensor_camera_config();
    camera.format = "h264".to_string();
    let sensor_json = serde_json::to_string(&vec![camera.clone()]).expect("sensor json");

    db.with_state_mut(|state| {
        state
            .db_config
            .metadata
            .insert("sensor_cameras".to_string(), sensor_json);
    });
    let id = msg_id(&camera.camera_name);
    for (step, frame) in encode_h264_frames(&camera, 5).iter().enumerate() {
        db.push_msg(Timestamp(1_000_000 + step as i64 * 16_667), id, frame)
            .expect("push_msg");
    }
    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, 30).expect("export_videos");

    let mp4 = std::fs::read(out_path.join("drone.fpv.mp4")).expect("read mp4");
    let sps = extract_avcc_sps(&mp4).expect("avcC SPS");
    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps)).expect("parse SPS");
    assert_eq!(sps.width(), camera.width as u64);
    assert_eq!(sps.height(), camera.height as u64);
}

#[test]
fn sensor_camera_h264_skips_leading_non_idr_after_trim() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("db");
    let out_path = tempdir.path().join("out");
    let db = DB::create(db_path.clone()).expect("DB::create");
    let mut camera = sensor_camera_config();
    camera.format = "h264".to_string();
    let sensor_json = serde_json::to_string(&vec![camera.clone()]).expect("sensor json");

    db.with_state_mut(|state| {
        state
            .db_config
            .metadata
            .insert("sensor_cameras".to_string(), sensor_json);
    });
    let id = msg_id(&camera.camera_name);
    let frames = encode_h264_frames(&camera, 5);
    db.push_msg(Timestamp(1_000_000), id, &[0, 0, 1, 0x41, 0, 0, 0])
        .expect("push non-idr");
    for (step, frame) in frames.iter().enumerate() {
        db.push_msg(Timestamp(1_016_667 + step as i64 * 16_667), id, frame)
            .expect("push_msg");
    }
    db.save_db_state().expect("save_db_state");
    db.flush_all().expect("flush_all");
    drop(db);

    elodin_db::export_videos::run(db_path, out_path.clone(), None, 30).expect("export_videos");

    let mp4_path = out_path.join("drone.fpv.mp4");
    let mp4 = std::fs::read(&mp4_path).expect("read mp4");
    assert_eq!(mp4.get(4..8), Some(&b"ftyp"[..]));
    let sps = extract_avcc_sps(&mp4).expect("avcC SPS");
    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps)).expect("parse SPS");
    assert_eq!(sps.width(), camera.width as u64);
    assert_eq!(sps.height(), camera.height as u64);
}
