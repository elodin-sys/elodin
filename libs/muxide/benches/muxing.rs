use criterion::{black_box, criterion_group, criterion_main, Criterion};
use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
use std::io::Cursor;

fn bench_h264_muxing(c: &mut Criterion) {
    c.bench_function("mux_1000_h264_frames", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let writer = Cursor::new(&mut buffer);
            let mut muxer = MuxerBuilder::new(writer)
                .video(VideoCodec::H264, 1920, 1080, 30.0)
                .build()
                .expect("build muxer");

            // Simulate 1000 frames (using dummy data for benchmark)
            let dummy_frame = vec![0u8; 10000]; // ~10KB per frame
            for i in 0..1000 {
                let pts = i as f64 / 30.0;
                let _ = muxer.write_video(pts, &dummy_frame, i % 30 == 0);
            }
            let _ = muxer.finish();
            black_box(buffer);
        });
    });
}

fn bench_h264_with_audio(c: &mut Criterion) {
    c.bench_function("mux_1000_h264_audio_frames", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let writer = Cursor::new(&mut buffer);
            let mut muxer = MuxerBuilder::new(writer)
                .video(VideoCodec::H264, 1920, 1080, 30.0)
                .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
                .build()
                .expect("build muxer");

            let dummy_video = vec![0u8; 10000];
            let dummy_audio = vec![0u8; 1000];
            for i in 0..1000 {
                let pts = i as f64 / 30.0;
                let _ = muxer.write_video(pts, &dummy_video, i % 30 == 0);
                let _ = muxer.write_audio(pts, &dummy_audio);
            }
            let _ = muxer.finish();
            black_box(buffer);
        });
    });
}

criterion_group!(benches, bench_h264_muxing, bench_h264_with_audio);
criterion_main!(benches);
