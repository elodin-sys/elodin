use bevy::prelude::*;
use elodin_editor::EditorPlugin;
use std::net::SocketAddr;
use web_sys::UrlSearchParams;

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    let mut plugin = EditorPlugin::default();
    if let Some(db_addr) = db_addr_from_query() {
        web_sys::console::log_1(&format!("elodin-web: db={db_addr}").into());
        plugin = plugin.with_connection_addr(db_addr);
    } else {
        web_sys::console::log_1(&"elodin-web: no db query param".into());
    }
    App::new()
        .add_plugins(plugin)
        .add_systems(Startup, log_startup)
        .add_systems(Update, log_first_update)
        .run();
}

/// `?db=host:port` connects immediately. Anything else shows the startup screen.
fn db_addr_from_query() -> Option<SocketAddr> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    let params = UrlSearchParams::new_with_str(&search).ok()?;
    let value = params.get("db")?;
    match value.parse() {
        Ok(addr) => Some(addr),
        Err(err) => {
            web_sys::console::warn_1(&format!("elodin-web: ignoring db={value:?}: {err}").into());
            None
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
