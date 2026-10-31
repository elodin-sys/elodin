mod launch;

use bevy::prelude::*;
use elodin_editor::{EditorPlugin, ViewerEvent};
use web_sys::UrlSearchParams;

fn main() {
    console_error_panic_hook::set_once();
    let launch = match browser_launch() {
        Ok(launch) => launch,
        Err(error) => {
            dispatch("elodin-error", Some(&error));
            web_sys::console::error_1(&error.into());
            return;
        }
    };
    let plugin = EditorPlugin::default().with_viewer_capabilities(launch.capabilities);
    #[cfg(target_family = "wasm")]
    let plugin = match launch.endpoint {
        Some(endpoint) => plugin.with_db_endpoint(endpoint),
        None => plugin,
    };
    #[cfg(not(target_family = "wasm"))]
    let _ = launch.endpoint;
    App::new()
        .add_plugins(plugin)
        .add_systems(Last, publish_events)
        .run();
}

fn browser_launch() -> Result<launch::Launch, String> {
    let window = web_sys::window().ok_or("No browser window")?;
    let search = window
        .location()
        .search()
        .map_err(|_| "Cannot read browser location")?;
    let params = UrlSearchParams::new_with_str(&search).map_err(|_| "Invalid launch parameters")?;
    let mut values = Vec::new();
    for name in ["impeller", "messages", "assets", "db", "read_only"] {
        let all = params.get_all(name);
        for index in 0..all.length() {
            values.push((
                name.to_owned(),
                all.get(index)
                    .as_string()
                    .ok_or("Invalid launch parameter")?,
            ));
        }
    }
    launch::parse(&values)
}

fn publish_events(mut events: MessageReader<ViewerEvent>) {
    for event in events.read() {
        match event {
            ViewerEvent::Connecting => dispatch("elodin-connecting", None),
            ViewerEvent::Ready => dispatch("elodin-ready", None),
            ViewerEvent::Failed(error) => dispatch("elodin-error", Some(error)),
        }
    }
}

fn dispatch(name: &str, error: Option<&str>) {
    let init = web_sys::CustomEventInit::new();
    if let Some(error) = error {
        init.set_detail(&error.into());
    }
    if let (Some(window), Ok(event)) = (
        web_sys::window(),
        web_sys::CustomEvent::new_with_event_init_dict(name, &init),
    ) {
        let _ = window.dispatch_event(&event);
    }
}
