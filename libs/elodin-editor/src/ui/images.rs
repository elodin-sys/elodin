use bevy::{
    asset::{AssetServer, Handle},
    ecs::world::{FromWorld, World},
    render::texture::Image,
};

pub struct Images {
    pub logo: Handle<Image>,
    pub icon_play: Handle<Image>,
    pub icon_pause: Handle<Image>,
    pub icon_scrub: Handle<Image>,
    pub icon_jump_to_end: Handle<Image>,
    pub icon_jump_to_start: Handle<Image>,
    pub icon_frame_forward: Handle<Image>,
    pub icon_frame_back: Handle<Image>,
    pub icon_search: Handle<Image>,
    pub icon_add: Handle<Image>,
    pub icon_close: Handle<Image>,
    pub icon_chart: Handle<Image>,
}

impl FromWorld for Images {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.get_resource_mut::<AssetServer>().unwrap();
        Self {
            logo: asset_server.load("embedded://elodin_editor/assets/logo.png"),
            icon_play: asset_server.load("embedded://elodin_editor/assets/icons/play.png"),
            icon_pause: asset_server.load("embedded://elodin_editor/assets/icons/pause.png"),
            icon_scrub: asset_server.load("embedded://elodin_editor/assets/icons/scrub.png"),
            icon_jump_to_end: asset_server
                .load("embedded://elodin_editor/assets/icons/jump_to_end.png"),
            icon_jump_to_start: asset_server
                .load("embedded://elodin_editor/assets/icons/jump_to_start.png"),
            icon_frame_forward: asset_server
                .load("embedded://elodin_editor/assets/icons/frame_forward.png"),
            icon_frame_back: asset_server
                .load("embedded://elodin_editor/assets/icons/frame_back.png"),
            icon_search: asset_server.load("embedded://elodin_editor/assets/icons/search.png"),
            icon_add: asset_server.load("embedded://elodin_editor/assets/icons/add.png"),
            icon_close: asset_server.load("embedded://elodin_editor/assets/icons/close.png"),
            icon_chart: asset_server.load("embedded://elodin_editor/assets/icons/chart.png"),
        }
    }
}
