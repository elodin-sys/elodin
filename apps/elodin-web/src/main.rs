mod launch;

use bevy::prelude::*;
use elodin_editor::{EditorPlugin, ViewerEvent};

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    let capabilities = match page_capabilities() {
        Ok(capabilities) => capabilities,
        Err(error) => {
            dispatch("elodin-error", Some(&error));
            return;
        }
    };
    let plugin = EditorPlugin::default().with_viewer_capabilities(capabilities);
    #[cfg(target_family = "wasm")]
    let plugin = apply_page_db(plugin);
    App::new()
        .add_plugins(plugin)
        .add_systems(Startup, log_startup)
        .add_systems(Last, publish_events)
        .run();
}

/// `<meta name="elodin-web-db">` when it has a value. Empty content leaves the
/// editor on the connect screen.
#[cfg(target_family = "wasm")]
fn page_db_url() -> Option<String> {
    let document = web_sys::window()?.document()?;
    document
        .query_selector("meta[name='elodin-web-db']")
        .ok()
        .flatten()
        .and_then(|el| el.get_attribute("content"))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_family = "wasm")]
fn apply_page_db(plugin: EditorPlugin) -> EditorPlugin {
    let Some(raw) = page_db_url() else {
        web_sys::console::log_1(&"elodin-web: no page database".into());
        return plugin;
    };
    match raw.parse::<impeller_bevy::HostOrUrl>() {
        Ok(url) => {
            web_sys::console::log_1(&format!("elodin-web: db={url}").into());
            plugin.with_connection_url(url)
        }
        Err(err) => {
            web_sys::console::warn_1(&format!("elodin-web: ignoring db={raw:?}: {err}").into());
            plugin
        }
    }
}

fn log_startup() {
    web_sys::console::log_1(&"elodin-web: Startup".into());
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

fn page_capabilities() -> Result<elodin_editor::ViewerCapabilities, String> {
    let window = web_sys::window().ok_or("No browser window")?;
    let search = window
        .location()
        .search()
        .map_err(|_| "Cannot read browser location")?;
    let params =
        web_sys::UrlSearchParams::new_with_str(&search).map_err(|_| "Invalid launch parameters")?;
    let all = params.get_all("read_only");
    let values = (0..all.length())
        .map(|index| {
            all.get(index)
                .as_string()
                .ok_or_else(|| "Invalid launch parameter".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    launch::capabilities(&values)
}
