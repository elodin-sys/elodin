//! WebCodecs H.264 playback for Bevy.
//!
//! Elodin keeps its frame cache and playhead. This crate decodes Annex-B
//! packets and uploads each frame into one persistent [`bevy::image::Image`]
//! (feature `bevy`). The HTTP loader and the canvas players live in the
//! example crate `player-support`, not here.
//!
//! A wasm build of a consumer needs `--cfg=web_sys_unstable_apis` so
//! `VideoFrame` and `copyExternalImageToTexture` are available.

pub mod nal;

#[cfg(target_arch = "wasm32")]
pub mod decoder;
#[cfg(target_arch = "wasm32")]
pub mod error;
#[cfg(target_arch = "wasm32")]
pub mod stats;

#[cfg(all(feature = "bevy", target_arch = "wasm32"))]
pub mod bevy;

#[cfg(target_arch = "wasm32")]
pub use error::{DomException, Error};

#[cfg(target_arch = "wasm32")]
mod platform {
    #[allow(dead_code)]
    pub fn perf_now() -> f64 {
        web_sys::window()
            .and_then(|window| window.performance())
            .map(|performance| performance.now())
            .unwrap_or(0.0)
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub(crate) use platform::perf_now;
