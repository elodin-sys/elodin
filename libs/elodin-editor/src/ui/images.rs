use bevy::{
    asset::{AssetServer, Handle},
    ecs::world::{FromWorld, World},
    render::texture::Image,
};

pub struct Images {
    pub icon_play: Handle<Image>,
    pub icon_pause: Handle<Image>,
    pub icon_scrub: Handle<Image>,
    pub icon_skip_next: Handle<Image>,
    pub icon_skip_prev: Handle<Image>,
}

impl FromWorld for Images {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.get_resource_mut::<AssetServer>().unwrap();
        Self {
            icon_play: asset_server.load("embedded://elodin_editor/assets/icons/icon_play.png"),
            icon_pause: asset_server.load("embedded://elodin_editor/assets/icons/icon_pause.png"),
            icon_scrub: asset_server.load("embedded://elodin_editor/assets/icons/icon_scrub.png"),
            icon_skip_next: asset_server
                .load("embedded://elodin_editor/assets/icons/icon_skip_next.png"),
            icon_skip_prev: asset_server
                .load("embedded://elodin_editor/assets/icons/icon_skip_prev.png"),
        }
    }
}
