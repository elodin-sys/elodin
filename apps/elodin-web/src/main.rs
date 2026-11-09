use bevy::prelude::*;
use elodin_editor::EditorPlugin;
use web_sys::UrlSearchParams;

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
    let mut plugin = EditorPlugin::default();
    if let Some(db) = db_query_value() {
        plugin = apply_db_query(plugin, &db);
    } else {
        web_sys::console::log_1(&"elodin-web: no db query param".into());
    }
    App::new()
        .add_plugins(plugin)
        .add_systems(Startup, log_startup)
        .add_systems(Update, log_first_update)
        .run();
}

#[cfg(target_family = "wasm")]
fn apply_db_query(plugin: EditorPlugin, db: &str) -> EditorPlugin {
    match plugin.with_connection_str(db) {
        Ok(next) => {
            web_sys::console::log_1(&format!("elodin-web: db={db}").into());
            next
        }
        Err((plugin, err)) => {
            web_sys::console::warn_1(&format!("elodin-web: ignoring db={db:?}: {err}").into());
            plugin
        }
    }
}

/// Host `cargo clippy` typechecks this bin. The connection API is wasm-only.
#[cfg(not(target_family = "wasm"))]
fn apply_db_query(plugin: EditorPlugin, db: &str) -> EditorPlugin {
    let _ = db;
    plugin
}

/// `?db=host:port` connects immediately. `host` may be an IP or a DNS name.
fn db_query_value() -> Option<String> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    let params = UrlSearchParams::new_with_str(&search).ok()?;
    params.get("db")
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
