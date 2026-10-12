#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cesium_native_bridge::ffi;
    let url = std::env::args().nth(1).unwrap_or_else(|| {
        "https://3d.geo.admin.ch/ch.swisstopo.swissbuildings3d.3d/v1/tileset.json".into()
    });
    let mut session = ffi::open_url(&url)?;
    // Bern, ECEF coordinates. Look down from above the old town.
    let lat = 46.948_f64.to_radians();
    let lon = 7.4474_f64.to_radians();
    let normal = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let n = 6_378_137.0 / (1.0 - 0.00669437999014 * lat.sin().powi(2)).sqrt();
    let view = ffi::View {
        position: [
            (n + 2500.) * normal[0],
            (n + 2500.) * normal[1],
            (n * (1.0 - 0.00669437999014) + 2500.) * normal[2],
        ],
        direction: normal.map(|n| -n),
        up: [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()],
        width: 1280.,
        height: 720.,
        horizontal_fov: 1.4,
        vertical_fov: 0.9,
    };
    let mut total = 0;
    for _ in 0..3000 {
        let frame = session
            .pin_mut()
            .update(std::slice::from_ref(&view), 0.02)?;
        for tile in frame.loaded {
            for p in tile.primitives {
                assert_eq!(p.positions.len(), p.normals.len());
                assert_eq!(p.uvs.len() / 2, p.positions.len() / 3);
                assert!(
                    p.indices
                        .iter()
                        .all(|i| (*i as usize) < p.positions.len() / 3)
                );
                total += 1;
            }
        }
        if total > 0 && !frame.visible.is_empty() {
            println!(
                "Cesium Native streamed {total} mesh primitives; {} visible tiles",
                frame.visible.len()
            );
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Err("No visible mesh within 60 seconds".into())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("Enable --features native and set CESIUM_NATIVE_DIR");
    std::process::exit(1);
}
