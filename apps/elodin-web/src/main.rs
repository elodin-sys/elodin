use bevy::prelude::*;
use elodin_editor::EditorPlugin;

fn main() {
    console_error_panic_hook::set_once();
    App::new().add_plugins(EditorPlugin::default()).run();
}
