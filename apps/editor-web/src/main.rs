use anyhow::anyhow;
use bevy::prelude::{App, In, IntoSystem, PostStartup};
use elo_conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
use paracosm::sync::SyncPlugin;
use paracosm_editor::EditorPlugin;
use tracing::error;

mod web_sock;

fn main() -> anyhow::Result<()> {
    //tracing_wasm::set_as_global_default();
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let url = get_url()?;
    let (sub, bevy_tx) = ConduitSubscribePlugin::pair();
    web_sock::spawn_wasm(url, bevy_tx)?;
    let mut app = App::new();
    app.add_plugins(EditorPlugin)
        .add_plugins(SyncPlugin {
            plugin: sub,
            subscriptions: Subscriptions::default(),
        })
        .add_systems(PostStartup, hide_loader.pipe(handle_error))
        .run();
    Ok(())
}

fn hide_loader() -> anyhow::Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let spinner = document
        .get_element_by_id("editor-spinner")
        .ok_or_else(|| anyhow!("missing editor spinner div"))?;
    let canvas = document
        .get_element_by_id("editor")
        .ok_or_else(|| anyhow!("missing editor canvas div"))?;
    spinner
        .set_attribute("style", "display: none;")
        .map_err(|e| anyhow!("set attr err {:?}", e))?;
    canvas
        .set_attribute("style", "display: block;")
        .map_err(|e| anyhow!("set attr err {:?}", e))?;
    Ok(())
}

fn handle_error(In(result): In<anyhow::Result<()>>) {
    if let Err(err) = result {
        error!(?err, "anyhow error")
    }
}

#[cfg(not(target_family = "wasm"))]
fn get_url() -> anyhow::Result<String> {
    use std::env;
    let args: Vec<String> = env::args().collect();
    Ok(args.into_iter().next().unwrap())
}

#[cfg(target_family = "wasm")]

fn get_url() -> anyhow::Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let container = document
        .get_element_by_id("editor-container")
        .ok_or_else(|| anyhow!("missing editor container div"))?;
    let url = container
        .get_attribute("data-ws-url")
        .ok_or_else(|| anyhow!("data-ws-url required"))?
        .to_string();
    Ok(url)
}
