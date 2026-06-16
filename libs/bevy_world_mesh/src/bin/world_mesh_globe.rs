// Top-level binary entry for the spherical Earth renderer. All the
// interesting wiring (WGS84 ellipsoid, big_space, orbital camera, sun
// mesh, EnvScreenshotPlugin) lives in `GlobeScenePlugin`.

use bevy::prelude::*;
use bevy_world_mesh::prelude::GlobeScenePlugin;

fn main() {
    App::new().add_plugins(GlobeScenePlugin).run();
}
