#![cfg(feature = "native")]

use cesium_native_bridge::ffi;
use std::time::{Duration, Instant};

#[test]
fn streams_a_local_georeferenced_gltf_without_a_token() {
    let url = format!(
        "file://{}/tests/data/tileset.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut session = ffi::open_url(&url).unwrap();
    let view = ffi::View {
        position: [6_378_200.0, 5.0, -5.0],
        direction: [-1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        width: 1280.0,
        height: 720.0,
        horizontal_fov: 1.4,
        vertical_fov: 0.9,
    };
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut received = false;
    while Instant::now() < deadline {
        let frame = session
            .pin_mut()
            .update(std::slice::from_ref(&view), 0.02)
            .unwrap();
        for tile in frame.loaded {
            assert_eq!(tile.primitives.len(), 1);
            let primitive = &tile.primitives[0];
            assert_eq!(primitive.positions.len(), 9);
            assert_eq!(primitive.normals.len(), 9);
            assert_eq!(primitive.indices, [0, 1, 2]);
            assert!((primitive.transform[12] - 6_378_137.0).abs() < 1e-6);
            assert_eq!(primitive.base_color, [0.2, 0.5, 0.9, 1.0]);
            assert_eq!((primitive.width, primitive.height), (1, 1));
            assert_eq!(primitive.pixels, [240, 80, 20, 255]);
            assert_eq!(primitive.uvs, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
            received = true;
        }
        if received && !frame.visible.is_empty() {
            // Same tileset accepts multiple cameras without reloading the mesh.
            let next = session.pin_mut().update(&[view, view], 0.02).unwrap();
            assert!(next.loaded.is_empty());
            assert_eq!(next.visible.len(), 1);
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("Cesium did not stream the local fixture");
}
