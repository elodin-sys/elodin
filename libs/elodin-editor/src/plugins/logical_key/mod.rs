use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput};
use bevy::prelude::*;
use std::collections::HashSet;

pub struct LogicalKeyPlugin;

#[derive(Resource, Default)]
pub struct LogicalKeyState {
    pressed: HashSet<Key>,
    just_pressed: HashSet<Key>,
    just_released: HashSet<Key>,
}

impl LogicalKeyState {
    pub fn pressed(&self, key: &Key) -> bool {
        self.pressed.contains(key)
    }

    pub fn just_pressed(&self, key: &Key) -> bool {
        self.just_pressed.contains(key)
    }

    pub fn just_released(&self, key: &Key) -> bool {
        self.just_released.contains(key)
    }

    pub fn any_pressed(&self, keys: &[Key]) -> bool {
        keys.iter().any(|key| self.pressed(key))
    }

    pub fn all_pressed(&self, keys: &[Key]) -> bool {
        keys.iter().all(|key| self.pressed(key))
    }
}

fn update_logical_key_state(
    mut key_state: ResMut<LogicalKeyState>,
    mut keyboard_events: MessageReader<KeyboardInput>,
    mut focus_lost: MessageReader<KeyboardFocusLost>,
) {
    key_state.just_pressed.clear();
    key_state.just_released.clear();

    for event in keyboard_events.read() {
        if event.state.is_pressed() {
            if key_state.pressed.insert(event.logical_key.clone()) {
                key_state.just_pressed.insert(event.logical_key.clone());
            }
        } else if key_state.pressed.remove(&event.logical_key) {
            key_state.just_released.insert(event.logical_key.clone());
        }
    }

    // Leaving the window (Alt-Tab, Cmd-Tab) never delivers the key release, which
    // would keep modifiers latched as held for the rest of the session.
    if !focus_lost.is_empty() {
        focus_lost.clear();
        let released = std::mem::take(&mut key_state.pressed);
        key_state.just_released.extend(released);
    }
}

impl Plugin for LogicalKeyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LogicalKeyState>()
            .add_systems(PreUpdate, update_logical_key_state);
    }
}
