use bevy::prelude::Entity;
use bevy::camera::RenderTarget;
use bevy::window::WindowRef;

pub fn window_entity_from_target(target: &RenderTarget, primary_window: Entity) -> Option<Entity> {
    match target {
        RenderTarget::Window(WindowRef::Primary) => Some(primary_window),
        RenderTarget::Window(WindowRef::Entity(entity)) => Some(*entity),
        _ => None,
    }
}
