use bevy::prelude::*;
use elodin_editor::EditorPlugin;

fn main() {
    console_error_panic_hook::set_once();
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&"elodin-web: main()".into());
    App::new().add_plugins(EditorPlugin::default()).run();
}
