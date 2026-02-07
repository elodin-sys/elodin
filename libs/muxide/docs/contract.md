# Muxide API Contract (v0.2.x)

This document defines the public API contract and invariants for the v0.2.x series of Muxide. The contract is intended to be a stable reference for users of the crate and for implementers working on the internals. All public items in the `muxide::api` module are covered by this contract.

## High‑Level API

Muxide exposes a builder pattern for creating a `Muxer` instance that writes an MP4 container to an arbitrary writer (implementing `std::io::Write`).  The API is intentionally minimal; configuration options beyond those described here are not available in v0.1.0.

### Types

* `VideoCodec`: Enumeration of supported video codecs.
  * `H264` — H.264/AVC video codec. Bitstreams must be in Annex B format.
  * `H265` — H.265/HEVC video codec. Bitstreams must be in Annex B format.
  * `Av1` — AV1 video codec. Bitstreams must be supplied as an OBU stream.
  * `Vp9` — VP9 video codec. Bitstreams must be supplied as compressed frames.

* `AudioCodec`: Enumeration of supported audio codecs.
  * `Aac` — AAC audio codec, encoded as ADTS frames. Only AAC LC is expected to play back correctly.
  * `Opus` — Opus audio codec, supplied as raw Opus packets. (In MP4, Opus is always signaled at 48 kHz.)
  * `None` — Indicates that no audio track will be created.

* `MuxerBuilder<Writer>` — Type parameterised by an output writer.  Provides methods to configure the container and tracks and to build a `Muxer`.  The builder consumes itself on `build`.

* `Muxer<Writer>` — Type parameterised by an output writer.  Provides methods to write video and audio frames and to finalise the file.  The generic parameter is preserved to allow any type implementing `Write` as the underlying sink.

* `MuxerConfig` — Simple configuration struct for integrations that prefer a config-driven constructor.

* `MuxerStats` — Statistics returned when finishing a mux.

* `MuxerError` — Enumeration of error conditions that may be returned by builder or runtime operations.  This enum may grow as the implementation matures.

### Timebase

Muxide converts incoming timestamps in seconds (`pts: f64`) into a fixed internal media timebase.

- The v0.1.0 implementation uses a **90 kHz** media timescale for track timing (a common convention for MP4/H.264).
- The media timebase is shared between video and audio when both tracks are present.

### MuxerBuilder Methods

* `new(writer: Writer) -> Self` — Constructs a builder for the given writer.  The writer is consumed by the builder and later moved into the `Muxer`.

* `video(self, codec: VideoCodec, width: u32, height: u32, framerate: f64) -> Self` — Configures the video track.  Exactly one call to `video` is required for v0.1.0.  The frame rate must be positive and reasonable (e.g. between 1 and 120).  Non‑integer frame rates (e.g. 29.97) are permitted.

* `audio(self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self` — Configures an optional audio track.  At most one call to `audio` may be made.  Audio is optional; if omitted, the file will contain only video.  If `codec` is `None`, the sample rate and channels are ignored.

* `build(self) -> Result<Muxer<Writer>, MuxerError>` — Validates the configuration and returns a `Muxer` instance on success.  In v0.1.0 the following validation rules apply:
  1. A video track must have been configured.  Otherwise a `MuxerError::MissingVideoConfig` is returned.
  2. If `AudioCodec::None` is selected, the muxer behaves as video-only.

### Muxer Methods

* `new(writer: Writer, config: MuxerConfig) -> Result<Muxer<Writer>, MuxerError>` — Convenience constructor that builds a muxer from a `MuxerConfig`.

* `write_video(&mut self, pts: f64, data: &[u8], is_keyframe: bool) -> Result<(), MuxerError>` — Writes a video frame to the container.

  **Invariants:**
  - `pts` **must be non‑negative and strictly greater than the `pts` of the previous video frame**. Violations produce `MuxerError::NegativeVideoPts` or `MuxerError::NonIncreasingVideoPts`.
  - `data` must contain a complete encoded frame in Annex B format.  The first video frame of a file must be a keyframe and must contain SPS and PPS NAL units; otherwise `MuxerError::FirstVideoFrameMustBeKeyframe` or `MuxerError::FirstVideoFrameMissingSpsPps` is returned.
  - `is_keyframe` must accurately reflect whether the frame is a keyframe (IDR picture).  Incorrect keyframe flags may result in unseekable files.

  **B-frames:**
  - `write_video()` is intended for streams where **PTS == DTS** (no reordering).
  - For streams with B-frames (PTS != DTS), use `write_video_with_dts()` and feed frames in decode order.

* `write_audio(&mut self, pts: f64, data: &[u8]) -> Result<(), MuxerError>` — Writes an audio frame to the container.

  **Invariants:**
  - `pts` **must be non‑negative and strictly greater than or equal to the `pts` of the previous audio frame**.
  - Audio must not arrive before the first video frame (i.e. audio `pts` must be >= video `pts`).
  - `data` must contain a complete encoded audio frame:
    - AAC must be ADTS; invalid ADTS is rejected.
    - Opus must be a structurally valid Opus packet; invalid packets are rejected.

* `finish(self) -> Result<(), MuxerError>` — Finalises the container.  After calling this method, no further `write_*` calls may be made.  This method writes any pending metadata (e.g. `moov` box) to the output writer.

* `finish_with_stats(self) -> Result<MuxerStats, MuxerError>` — Finalises the container and returns muxing statistics.

* `finish_in_place(&mut self) -> Result<(), MuxerError>` — Finalises the container without consuming the muxer.  This is a convenience for applications that want an explicit “finalised” error on double-finish and on writes after finishing.

* `finish_in_place_with_stats(&mut self) -> Result<MuxerStats, MuxerError>` — In-place finalisation that returns muxing statistics.

### Error Semantics

All functions that can fail return a `MuxerError`. New error variants may be added in minor versions.

### Concurrency & Thread Safety

`Muxer<W>` is `Send` when `W: Send` and `Sync` when `W: Sync`.

Muxide itself is implemented as a single-threaded writer; thread-safety here refers to moving/sharing the muxer value when the underlying writer type supports it.

## Invariants & Correctness Rules

1. **Monotonic Timestamps:** For each track, presentation timestamps (`pts`) must be non‑negative and strictly increasing (video) or non‑decreasing (audio). If this invariant is violated, the operation must fail.
2. **Keyframes:** The first video frame must be a keyframe containing SPS and PPS.  Subsequent keyframes must be marked via the `is_keyframe` flag.  Files produced without proper keyframe signalling will not play back correctly and are considered incorrect.
3. **Single Video Track:** Exactly one video track is supported.  Multiple video tracks or the absence of a video track is an error.
4. **Single Audio Track:** At most one audio track is supported.  Adding multiple audio tracks is not allowed.
5. **B‑frames:** Streams with reordering (B-frames) are supported when callers use `write_video_with_dts()`:
  - Frames must be supplied in **decode order**.
  - DTS must be strictly increasing.
  - PTS may differ from DTS.
  For streams without reordering, callers may use `write_video()` which assumes PTS == DTS.
6. **Bitstream formats:**
  - H.264/H.265 video must be provided in Annex B format (start-code-prefixed NAL units).
  - AV1 video must be provided as an OBU stream.
7. **Audio formats:**
  - AAC audio must be provided as ADTS frames.
  - Opus audio must be provided as raw Opus packets.

## B-frame Support

Muxide supports B-frames via the `write_video_with_dts()` method:

- When B-frames are present, callers must provide both PTS (presentation timestamp) and DTS (decode timestamp)
- DTS must be monotonically increasing (decode order)
- PTS may differ from DTS (display order ≠ decode order)
- The `ctts` (composition time offset) box is automatically generated

For streams without B-frames, use `write_video()` which assumes PTS == DTS.

## Examples (Pseudo‑Code)

```
use muxide::api::{MuxerBuilder, VideoCodec, AudioCodec};
use std::fs::File;

// Create an output file
let file = File::create("out.mp4")?;

// Build a muxer for 1920x1080 30 fps video and 48 kHz stereo audio
let mut mux = MuxerBuilder::new(file)
    .video(VideoCodec::H264, 1920, 1080, 30.0)
    .audio(AudioCodec::Aac, 48_000, 2)
    .build()?;

// Write frames (encoded elsewhere)
for (i, frame) in video_frames.iter().enumerate() {
    let pts = (i as f64) / 30.0;
    let is_key = i == 0 || i % 30 == 0;
    mux.write_video(pts, &frame.data, is_key)?;
    // Optionally interleave audio
}

// Finish the file
mux.finish()?;
```

## Stability

The API described here must not change in any breaking way during the v0.1.x series.  Additional methods may be added, but existing signatures and invariants must remain stable.  Breaking changes require a new major version or a new charter.
