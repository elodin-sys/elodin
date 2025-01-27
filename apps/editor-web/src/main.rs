use anyhow::anyhow;
use bevy::app::Update;
use bevy::ecs::change_detection::DetectChanges;
use bevy::ecs::{schedule::IntoSystemConfigs, system::Res};
use bevy::prelude::{App, In, IntoSystem, PostStartup};
use elodin_editor::ui::FullscreenState;
use elodin_editor::EditorPlugin;
use impeller2::types::{FilledRecycle, LenPacket};
use impeller2_bevy::CurrentStreamId;
use impeller2_bevy::{PacketRx, PacketTx};
use thingbuf::mpsc;
use tracing::error;

mod web_sock;

fn main() {
    //tracing_wasm::set_as_global_default();
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let url = get_url().unwrap();
    let (incoming_packet_tx, incoming_packet_rx) = mpsc::with_recycle(512, FilledRecycle);
    let (outgoing_packet_tx, outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(512);
    let stream_id = fastrand::u64(..);
    web_sock::spawn_wasm(url, outgoing_packet_rx, incoming_packet_tx, stream_id).unwrap();
    App::new()
        .add_plugins(EditorPlugin::default())
        .add_systems(PostStartup, hide_loader.pipe(handle_error))
        .add_systems(
            PostStartup,
            show_canvas.pipe(handle_error).after(hide_loader),
        )
        .add_systems(Update, fullscreen.pipe(handle_error))
        .insert_resource(PacketTx(outgoing_packet_tx))
        .insert_resource(PacketRx(incoming_packet_rx))
        .insert_resource(CurrentStreamId(stream_id))
        .add_systems(Update, impeller2_bevy::sink)
        .run();
}

fn hide_loader() -> anyhow::Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let spinner = document
        .get_element_by_id("editor-spinner")
        .ok_or_else(|| anyhow!("missing editor spinner div"))?;
    spinner
        .set_attribute("style", "display: none;")
        .map_err(|e| anyhow!("set attr err {:?}", e))?;
    Ok(())
}

fn show_canvas() -> anyhow::Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let canvas = document
        .get_element_by_id("editor")
        .ok_or_else(|| anyhow!("missing editor canvas div"))?;
    canvas
        .set_attribute("style", "display: block; width: 100%; height: 100%;")
        .map_err(|e| anyhow!("set attr err {:?}", e))
}

fn fullscreen(fullscreen: Res<FullscreenState>) -> anyhow::Result<()> {
    if !fullscreen.is_changed() {
        return Ok(());
    }
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let canvas = document
        .get_element_by_id("editor")
        .ok_or_else(|| anyhow!("missing editor canvas div"))?;
    let container = document
        .get_element_by_id("editor-container")
        .ok_or_else(|| anyhow!("missing editor container div"))?;

    if fullscreen.0 {
        canvas
            .set_attribute(
                "style",
                "position: fixed; display: block; width: 100%; height: 100%; top: 0; left: 0; z-index: 1000;",
            )
            .map_err(|e| anyhow!("set attr err {:?}", e))
    } else {
        canvas
            .set_attribute(
                "style",
                "display: block; width: 100%; height: 100%; position: relative; top: 0; left: 0;",
            )
            .map_err(|e| anyhow!("set attr err {:?}", e))?;
        canvas
            .set_attribute("width", &container.scroll_width().to_string())
            .map_err(|e| anyhow!("set attr err {:?}", e))?;
        canvas
            .set_attribute("height", &container.scroll_height().to_string())
            .map_err(|e| anyhow!("set attr err {:?}", e))
    }
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
