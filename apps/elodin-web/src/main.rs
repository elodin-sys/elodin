use bevy::prelude::*;
use elodin_editor::EditorPlugin;
use web_sys::UrlSearchParams;

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    let mut plugin = EditorPlugin::default();
    if let Some(db) = query_value("db") {
        plugin = apply_db_query(plugin, &db);
    } else {
        web_sys::console::log_1(&"elodin-web: no db query param".into());
    }
    if let Some(assets) = query_value("assets") {
        plugin = apply_assets_query(plugin, &assets);
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
/// `&assets=host:port` or `&assets=http(s)://…` overrides where `db:` assets are fetched.
fn query_value(key: &str) -> Option<String> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    let params = UrlSearchParams::new_with_str(&search).ok()?;
    params.get(key)
}

#[cfg(target_family = "wasm")]
fn apply_assets_query(plugin: EditorPlugin, assets: &str) -> EditorPlugin {
    match plugin.with_assets(assets) {
        Ok(next) => {
            web_sys::console::log_1(&format!("elodin-web: assets={assets}").into());
            next
        }
        Err((plugin, err)) => {
            web_sys::console::warn_1(
                &format!("elodin-web: ignoring assets={assets:?}: {err}").into(),
            );
            plugin
        }
    }
}

/// Host `cargo clippy` typechecks this bin. The assets API is wasm-only.
#[cfg(not(target_family = "wasm"))]
fn apply_assets_query(plugin: EditorPlugin, assets: &str) -> EditorPlugin {
    let _ = assets;
    plugin
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
