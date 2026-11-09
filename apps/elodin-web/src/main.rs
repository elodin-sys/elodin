use bevy::prelude::*;
use elodin_editor::EditorPlugin;

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    // The graphics probe is async. Bevy must not build until it finishes,
    // because wgpu locks in WebGPU as soon as `navigator.gpu` exists.
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async {
            elodin_editor::prepare_wasm_graphics().await;
            start_app();
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    start_app();
}

fn start_app() {
    let plugin = EditorPlugin::default();
    #[cfg(target_family = "wasm")]
    let plugin = apply_page_db(plugin);
    App::new()
        .add_plugins(plugin)
        .add_systems(Startup, log_startup)
        .add_systems(Update, log_first_update)
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

fn log_first_update(mut done: Local<bool>) {
    if *done {
        return;
    }
    *done = true;
    web_sys::console::log_1(&"elodin-web: first Update".into());
    if let (Some(window), Ok(event)) = (web_sys::window(), web_sys::Event::new("elodin-ready")) {
        let _ = window.dispatch_event(&event);
    }
}
