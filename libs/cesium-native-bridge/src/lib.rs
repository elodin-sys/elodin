//! Narrow, owned-data boundary to Cesium Native. No native pointers escape.
//! The session stays on its creating thread; Cesium owns its worker pool.

pub type SessionHandle = cxx::UniquePtr<ffi::Session>;

#[cxx::bridge(namespace = "elodin_cesium")]
pub mod ffi {
    #[derive(Clone, Copy)]
    struct View {
        position: [f64; 3],
        direction: [f64; 3],
        up: [f64; 3],
        width: f64,
        height: f64,
        horizontal_fov: f64,
        vertical_fov: f64,
    }

    struct Primitive {
        transform: [f64; 16],
        positions: Vec<f32>,
        normals: Vec<f32>,
        uvs: Vec<f32>,
        indices: Vec<u32>,
        base_color: [f32; 4],
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        double_sided: bool,
    }

    struct Tile {
        id: u64,
        primitives: Vec<Primitive>,
    }

    struct Frame {
        loaded: Vec<Tile>,
        visible: Vec<u64>,
        removed: Vec<u64>,
        credits: Vec<String>,
        pending: u32,
        load_failed: bool,
    }

    // SAFETY: CXX checks shared layouts and ownership at build time. Session
    // owns all native state; methods return copies, never borrowed native data.
    // No Send/Sync implementation: creation, update and destruction stay on
    // the main thread. C++ exceptions become Rust errors through Result.
    unsafe extern "C++" {
        include!("elodin_cesium/bridge.h");
        type Session;
        fn open_url(url: &str) -> Result<UniquePtr<Session>>;
        fn open_ion(asset_id: i64, token: &str) -> Result<UniquePtr<Session>>;
        fn update(self: Pin<&mut Session>, views: &[View], seconds: f32) -> Result<Frame>;
    }
}
