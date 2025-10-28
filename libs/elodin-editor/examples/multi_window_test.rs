use bevy::prelude::*;
use elodin_editor::{multi_window::*, EditorPlugin};

fn main() {
    App::new()
        .add_plugins(EditorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, test_popout)
        .run();
}

fn setup(mut commands: Commands) {
    // The editor plugin sets up the main window and UI
    println!("Multi-window test application started");
    println!("Press 'P' to test creating a secondary window");
}

fn test_popout(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut requests: ResMut<SecondaryWindowRequests>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        println!("Creating a test secondary window...");
        
        // Create a simple viewport pane for testing
        let test_pane = elodin_editor::ui::tiles::Pane::Viewport(
            elodin_editor::ui::tiles::ViewportPane {
                camera: None,
                nav_gizmo: None,
                nav_gizmo_camera: None,
                rect: None,
                label: "Test Viewport".to_string(),
            }
        );
        
        request_secondary_window(&mut requests, test_pane, None);
        println!("Secondary window request added");
    }
}
