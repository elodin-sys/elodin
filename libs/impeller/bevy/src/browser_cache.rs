//! Optional session-scoped browser storage. Playback never depends on persistence.

use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

use bevy::prelude::*;

#[derive(Resource, Clone, Debug)]
pub struct PlaybackCacheLimits {
    pub video_bytes: usize,
    pub decoded_bytes: usize,
    pub telemetry_bytes: usize,
    pub terrain_bytes: usize,
    pub disk_bytes: usize,
    pub entry_bytes: usize,
}

impl Default for PlaybackCacheLimits {
    fn default() -> Self {
        Self {
            video_bytes: 192 * 1024 * 1024,
            decoded_bytes: 32 * 1024 * 1024,
            telemetry_bytes: 64 * 1024 * 1024,
            terrain_bytes: 64 * 1024 * 1024,
            disk_bytes: 512 * 1024 * 1024,
            entry_bytes: 8 * 1024 * 1024,
        }
    }
}

#[derive(Resource, Clone, Default, Debug)]
pub struct BrowserCacheScope {
    pub key: Option<String>,
    pub assets: Option<String>,
}

static ASSET_SCOPE: Mutex<Option<(BrowserCacheScope, usize, usize)>> = Mutex::new(None);

pub fn asset_scope(url: &str) -> Option<(String, String, usize, usize)> {
    let state = ASSET_SCOPE.lock().ok()?;
    let (scope, max_asset, chunk_bytes) = state.as_ref()?;
    let base = url::Url::parse(scope.assets.as_deref()?).ok()?;
    let target = url::Url::parse(url).ok()?;
    if base.origin() != target.origin() || base.query() != target.query() {
        return None;
    }
    let path = target.path().strip_prefix(base.path())?;
    Some((
        scope.key.clone()?,
        format!("asset/{path}"),
        *max_asset,
        *chunk_bytes,
    ))
}

#[derive(Default)]
struct ReadState {
    result: Option<Option<Vec<u8>>>,
    waker: Option<Waker>,
}

pub struct CacheRead(Arc<Mutex<ReadState>>);

impl CacheRead {
    pub fn poll(&mut self) -> Option<Option<Vec<u8>>> {
        self.0.lock().ok()?.result.take()
    }
}

impl Future for CacheRead {
    type Output = Option<Vec<u8>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let Ok(mut state) = self.0.lock() else {
            return Poll::Ready(None);
        };
        if let Some(result) = state.result.take() {
            return Poll::Ready(result);
        }
        state.waker = Some(context.waker().clone());
        Poll::Pending
    }
}

pub fn lookup(scope: &str, key: &str) -> CacheRead {
    let result = Arc::new(Mutex::new(ReadState::default()));
    #[cfg(target_family = "wasm")]
    {
        use wasm_bindgen::{JsCast, JsValue, closure::Closure};
        let result = result.clone();
        let done = Closure::once_into_js(move |value: JsValue| {
            let waker = result.lock().ok().and_then(|mut state| {
                state.result = Some(bytes(value));
                state.waker.take()
            });
            if let Some(waker) = waker {
                waker.wake();
            }
        });
        web::request(scope, key, done.unchecked_ref());
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let _ = (scope, key);
        result.lock().expect("new cache read").result = Some(None);
    }
    CacheRead(result)
}

pub fn report_resident(kind: &str, bytes: usize) {
    #[cfg(target_family = "wasm")]
    web::resident(kind, bytes as f64);
    #[cfg(not(target_family = "wasm"))]
    let _ = (kind, bytes);
}

pub fn put(scope: &str, key: &str, data: &[u8]) {
    #[cfg(target_family = "wasm")]
    web::put(scope, key, data);
    #[cfg(not(target_family = "wasm"))]
    let _ = (scope, key, data);
}

pub async fn commit(scope: &str, key: &str, data: &[u8]) -> bool {
    let state = Arc::new(Mutex::new(ReadState::default()));
    #[cfg(target_family = "wasm")]
    {
        use wasm_bindgen::{JsCast, JsValue, closure::Closure};
        let result = state.clone();
        let done = Closure::once_into_js(move |value: JsValue| {
            let waker = result.lock().ok().and_then(|mut state| {
                state.result = Some(bytes(value));
                state.waker.take()
            });
            if let Some(waker) = waker {
                waker.wake();
            }
        });
        web::write(scope, key, data, done.unchecked_ref());
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let _ = (scope, key, data);
        state.lock().expect("new cache write").result = Some(None);
    }
    CacheRead(state).await.is_some()
}

pub fn store<T: serde::Serialize>(scope: &str, key: &str, value: &T) {
    if let Ok(bytes) = postcard::to_allocvec(value) {
        put(scope, key, &bytes);
    }
}

pub fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    postcard::from_bytes(bytes).ok()
}

#[cfg(target_family = "wasm")]
fn bytes(value: wasm_bindgen::JsValue) -> Option<Vec<u8>> {
    (!value.is_null() && !value.is_undefined()).then(|| js_sys::Uint8Array::new(&value).to_vec())
}

pub fn sync_scope(
    connection: Option<Res<crate::ConnectionAddr>>,
    status: Option<Res<crate::ThreadConnectionStatus>>,
    settings: Option<Res<impeller_wkt::DbConfig>>,
    limits: Res<PlaybackCacheLimits>,
    mut scope: ResMut<BrowserCacheScope>,
) {
    let assets = connection
        .as_ref()
        .filter(|c| matches!(c.endpoint(), crate::DbEndpoint::Browser(_)))
        .filter(|_| {
            status
                .as_ref()
                .is_some_and(|s| s.status() == crate::ConnectionStatus::Success)
        })
        .map(|c| c.endpoint().http_base());
    let key = assets.as_ref().map(|base| {
        format!(
            "{base}#{}",
            settings.as_ref().map_or(0, |s| s.assets_revision())
        )
    });
    if scope.key != key || limits.is_changed() {
        #[cfg(target_family = "wasm")]
        web::activate(
            key.as_deref().unwrap_or(""),
            limits.disk_bytes as f64,
            limits.entry_bytes as u32,
        );
        scope.key = key;
        scope.assets = assets;
        if let Ok(mut current) = ASSET_SCOPE.lock() {
            *current = scope
                .key
                .as_ref()
                .map(|_| (scope.clone(), limits.terrain_bytes, limits.entry_bytes));
        }
    }
}

#[cfg(target_family = "wasm")]
mod web {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(module = "/src/browser_cache.js")]
    extern "C" {
        pub fn resident(kind: &str, bytes: f64);
        pub fn activate(scope: &str, disk_bytes: f64, entry_bytes: u32);
        pub fn request(scope: &str, key: &str, done: &js_sys::Function);
        pub fn put(scope: &str, key: &str, data: &[u8]);
        pub fn write(scope: &str, key: &str, data: &[u8], done: &js_sys::Function);
    }
}

pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub fn is_seek(previous: i64, current: i64, elapsed: Duration, minimum: u64) -> bool {
    let allowance = elapsed.as_micros().saturating_mul(2).max(minimum as u128);
    previous.abs_diff(current) as u128 > allowance
}

/// Fixed ranges make disk entries reusable after forward and backward seeks.
pub fn window_start(timestamp: i64, width: i64) -> i64 {
    timestamp.div_euclid(width).saturating_mul(width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_stable_across_seeks_and_negative_timestamps() {
        assert_eq!(window_start(1_250_001, 250_000), 1_250_000);
        assert_eq!(window_start(-1, 250_000), -250_000);
        assert_eq!(window_start(1_499_999, 250_000), 1_250_000);
    }

    #[test]
    fn unavailable_storage_returns_a_miss_without_blocking() {
        let mut request = lookup("session", "frame");
        assert_eq!(request.poll(), Some(None));
        assert_eq!(request.poll(), None);
    }
    #[test]
    fn slow_frames_do_not_cancel_normal_playback() {
        assert!(!is_seek(0, 2_000_000, Duration::from_secs(2), 1_000_000));
        assert!(is_seek(0, 10_000_000, Duration::from_millis(16), 1_000_000));
        assert!(is_seek(10_000_000, 0, Duration::from_millis(16), 1_000_000));
        assert!(!is_seek(i64::MIN, i64::MIN, Duration::ZERO, 1_000_000));
        assert!(is_seek(i64::MIN, i64::MAX, Duration::ZERO, 1_000_000));
    }
}
