use bevy::prelude::*;
use bevy_world_mesh::prelude::PlanarScenePlugin;

fn main() {
    App::new().add_plugins(PlanarScenePlugin).run();
}
