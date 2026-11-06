# webcodecs-video

WebCodecs H.264 decoder and a Bevy plugin that uploads each decoded frame into one persistent `Image`. Elodin keeps its frame cache, tiles, and playhead. This crate replaces the wasm decode-and-upload step.

The HTTP loader, canvas renderers, and shared playhead used by the players in this repo live in `player-support`. They are not part of the library.

## What Elodin calls

Elodin's wasm editor (`libs/elodin-editor/src/ui/video_stream.rs`) keeps the `ImageNode` overlay, the raw Annex-B `VideoFrameCache`, and the playhead. Two functions in that file talk to this crate:

- `attach_webcodecs` inserts a `VideoStream` on each video-tile entity after spawn. `EditorPlugin` registers it on `Update`, and registers `WebcodecsVideoPlugin` after `DefaultPlugins`.
- `drive_webcodecs` runs from the widget while the stream is active. It calls `set_time` and `set_playing` from the playhead, `push` for sequential Annex-B packets, and `seek_through` from the nearest keyframe. A still playhead counts as paused so a scrub flushes and paints. When `image` is ready, the widget assigns that handle to the tile's `ImageNode`.

```toml
[target.'cfg(target_family = "wasm")'.dependencies]
webcodecs-video = { path = "../webcodecs-video", features = ["bevy"] }
```

The wasm build needs `--cfg=web_sys_unstable_apis` so `VideoFrame` and `copyExternalImageToTexture` are exposed. Elodin already passes that flag. This repo sets it in `.cargo/config.toml`.

The image is created on the first decoded frame (`Rgba8UnormSrgb`, render attachment). The default upload copies the `VideoFrame` into that texture with `copy_external_image_to_texture`. `UploadMode::CopyTo` is the slower RGBA readback, kept so the examples can compare.

The `bevy-egui-tiles` example reads Elodin's stored video through the same `GetMsgs` pages the editor uses (`ws://127.0.0.1:2241/impeller/msg`). In Elodin the packets already sitting in `VideoFrameCache` are the input to `VideoStream::push` and `VideoStream::seek`.

`push` returns `false` until the first keyframe, while a seek is in flight, or when the decoder queue is full. Call it again on a later frame. `seek` resets the decoder and shows only the last timestamp in the batch.

Native builds of this crate are the Annex-B helpers in `nal`. The decoder and the plugin are `wasm32` only, so Elodin's native editor still links.

## Features

- `bevy` — `WebcodecsVideoPlugin` and the `VideoStream` component
- `egui` — `VideoStream::paint` and `fit_rect` for drawing the texture inside an egui tile (`bevy_egui`). The Elodin overlay does not need this; it uses the `Image` handle directly.
