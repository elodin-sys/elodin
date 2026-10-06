use bevy::prelude::*;
use elodin_editor::EditorPlugin;

fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"elodin-web: main()".into());
    App::new()
        .add_plugins(EditorPlugin::default())
        .add_systems(Startup, log_startup)
        .add_systems(Update, log_first_update)
        .run();
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
