use bevy::prelude::*;
use elodin_editor::EditorPlugin;
use std::net::SocketAddr;
use web_sys::UrlSearchParams;

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    let db_addr = db_addr_from_query();
    web_sys::console::log_1(&format!("elodin-web: db={db_addr}").into());
    App::new()
        .add_plugins(EditorPlugin::default().with_connection_addr(db_addr))
        .add_systems(Startup, log_startup)
        .add_systems(Update, log_first_update)
        .run();
}

fn db_addr_from_query() -> SocketAddr {
    let default: SocketAddr = "127.0.0.1:2240".parse().expect("default db addr");
    let Some(window) = web_sys::window() else {
        return default;
    };
    let Ok(search) = window.location().search() else {
        return default;
    };
    let Ok(params) = UrlSearchParams::new_with_str(&search) else {
        return default;
    };
    params
        .get("db")
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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
