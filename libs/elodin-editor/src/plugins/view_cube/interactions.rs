//! Interaction handlers for the ViewCube widget

use bevy::ecs::hierarchy::ChildOf;
use bevy::picking::prelude::*;
use bevy::prelude::*;

use super::components::*;
use super::events::ViewCubeEvent;
use super::theme::ViewCubeColors;

// ============================================================================
// Setup System - Called each frame to set up new cube elements from GLB
// ============================================================================

/// Set up cube elements after the GLB is loaded
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn setup_cube_elements(
    mut commands: Commands,
    query: Query<(Entity, &Name), (With<Name>, Without<ViewCubeSetup>)>,
    parents: Query<&ChildOf>,
    children_query: Query<&Children>,
    mesh_roots: Query<Entity, With<ViewCubeMeshRoot>>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    // Support multiple ViewCubes (split viewports)
    if mesh_roots.is_empty() {
        return;
    }

    let colors = ViewCubeColors::default();

    for (entity, name) in query.iter() {
        // Check if this entity is in ANY cube subtree
        let mut in_subtree = false;
        let mut current = entity;
        loop {
            if mesh_roots.get(current).is_ok() {
                in_subtree = true;
                break;
            }
            if let Ok(parent) = parents.get(current) {
                current = parent.0;
            } else {
                break;
            }
        }

        if !in_subtree {
            continue;
        }

        let name_str = name.as_str();

        // Parse element type from name
        let element = if name_str.starts_with("Face_") {
            parse_face(name_str)
        } else if name_str.starts_with("Border_") {
            parse_edge(name_str)
        } else if name_str.starts_with("Corner_") {
            parse_corner(name_str)
        } else {
            None
        };

        if let Some(elem) = element {
            let element_color = colors.get_element_color(&elem);

            commands
                .entity(entity)
                .insert((elem.clone(), ViewCubeSetup));

            // Clone materials for children so they can be highlighted independently
            if let Ok(children) = children_query.get(entity) {
                for child in children.iter() {
                    if let Ok(mat_handle) = material_query.get(child)
                        && materials.get(&mat_handle.0).is_some()
                    {
                        original_materials.colors.insert(child, element_color);

                        let new_mat = StandardMaterial {
                            base_color: element_color,
                            alpha_mode: AlphaMode::Blend,
                            unlit: false,
                            double_sided: true,
                            cull_mode: None,
                            ..default()
                        };
                        let new_handle = materials.add(new_mat);
                        commands.entity(child).insert(MeshMaterial3d(new_handle));
                    }
                }
            }
        }
    }
}

fn parse_face(name: &str) -> Option<CubeElement> {
    let dir = match name {
        "Face_Front" => FaceDirection::North,
        "Face_Back" => FaceDirection::South,
        "Face_Left" => FaceDirection::West,
        "Face_Right" => FaceDirection::East,
        "Face_Top" => FaceDirection::Up,
        "Face_Bottom" => FaceDirection::Down,
        _ => return None,
    };
    Some(CubeElement::Face(dir))
}

fn parse_edge(name: &str) -> Option<CubeElement> {
    let dir = match name {
        "Border_X_y1_z1" => EdgeDirection::XTopFront,
        "Border_X_y1_z-1" => EdgeDirection::XTopBack,
        "Border_X_y-1_z1" => EdgeDirection::XBottomFront,
        "Border_X_y-1_z-1" => EdgeDirection::XBottomBack,
        "Border_Y_x-1_z1" => EdgeDirection::YFrontLeft,
        "Border_Y_x1_z1" => EdgeDirection::YFrontRight,
        "Border_Y_x-1_z-1" => EdgeDirection::YBackLeft,
        "Border_Y_x1_z-1" => EdgeDirection::YBackRight,
        "Border_Z_x-1_y1" => EdgeDirection::ZTopLeft,
        "Border_Z_x1_y1" => EdgeDirection::ZTopRight,
        "Border_Z_x-1_y-1" => EdgeDirection::ZBottomLeft,
        "Border_Z_x1_y-1" => EdgeDirection::ZBottomRight,
        _ => return None,
    };
    Some(CubeElement::Edge(dir))
}

fn parse_corner(name: &str) -> Option<CubeElement> {
    let pos = match name {
        "Corner_x-1_y1_z1" => CornerPosition::TopFrontLeft,
        "Corner_x1_y1_z1" => CornerPosition::TopFrontRight,
        "Corner_x-1_y1_z-1" => CornerPosition::TopBackLeft,
        "Corner_x1_y1_z-1" => CornerPosition::TopBackRight,
        "Corner_x-1_y-1_z1" => CornerPosition::BottomFrontLeft,
        "Corner_x1_y-1_z1" => CornerPosition::BottomFrontRight,
        "Corner_x-1_y-1_z-1" => CornerPosition::BottomBackLeft,
        "Corner_x1_y-1_z-1" => CornerPosition::BottomBackRight,
        _ => return None,
    };
    Some(CubeElement::Corner(pos))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Find the nearest ancestor with a CubeElement component
fn find_cube_element_ancestor(
    entity: Entity,
    cube_elements: &Query<&CubeElement>,
    parents_query: &Query<&ChildOf>,
) -> Option<Entity> {
    find_ancestor(entity, parents_query, |current| {
        cube_elements.get(current).is_ok()
    })
}

// ============================================================================
// Cube Element Hover Handlers
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn on_cube_hover_start(
    trigger: On<Pointer<Over>>,
    mut commands: Commands,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hovered: ResMut<HoveredElement>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    let entity = trigger.entity;

    let Some(target) = find_cube_element_ancestor(entity, &cube_elements, &parents_query) else {
        return;
    };

    if hovered.entity == Some(target) {
        return;
    }

    // Reset previous
    if let Some(prev) = hovered.entity {
        reset_highlight(
            prev,
            &children_query,
            &material_query,
            &mut materials,
            &original_materials,
        );
    }

    // Apply highlight
    apply_highlight(
        target,
        &children_query,
        &material_query,
        &mut materials,
        &mut original_materials,
        &mut commands,
    );

    hovered.entity = Some(target);
}

#[allow(clippy::too_many_arguments)]
pub fn on_cube_hover_end(
    trigger: On<Pointer<Out>>,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hovered: ResMut<HoveredElement>,
    original_materials: Res<OriginalMaterials>,
) {
    let entity = trigger.entity;

    let Some(target) = find_cube_element_ancestor(entity, &cube_elements, &parents_query) else {
        return;
    };

    if hovered.entity != Some(target) {
        return;
    }

    reset_highlight(
        target,
        &children_query,
        &material_query,
        &mut materials,
        &original_materials,
    );
    hovered.entity = None;
}

pub fn on_cube_click(
    trigger: On<Pointer<Click>>,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    let entity = trigger.entity;

    let Some(target_entity) = find_cube_element_ancestor(entity, &cube_elements, &parents_query)
    else {
        return;
    };

    let Ok(element) = cube_elements.get(target_entity) else {
        return;
    };

    // Find the ViewCubeRoot ancestor to identify which ViewCube was clicked
    let source =
        find_root_ancestor(entity, &parents_query, &root_query).unwrap_or(Entity::PLACEHOLDER);

    match element {
        CubeElement::Face(dir) => {
            events.write(ViewCubeEvent::FaceClicked {
                direction: *dir,
                source,
            });
        }
        CubeElement::Edge(dir) => {
            events.write(ViewCubeEvent::EdgeClicked {
                direction: *dir,
                source,
            });
        }
        CubeElement::Corner(pos) => {
            events.write(ViewCubeEvent::CornerClicked {
                position: *pos,
                source,
            });
        }
    }
}

// ============================================================================
// Arrow Hover Handlers
// ============================================================================

pub fn on_arrow_hover_start(
    trigger: On<Pointer<Over>>,
    arrows: Query<&RotationArrow>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.entity;

    if arrows.get(entity).is_err() {
        return;
    }

    let colors = ViewCubeColors::default();
    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = colors.arrow_hover;
    }
}

pub fn on_arrow_hover_end(
    trigger: On<Pointer<Out>>,
    arrows: Query<&RotationArrow>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.entity;

    if arrows.get(entity).is_err() {
        return;
    }

    let colors = ViewCubeColors::default();
    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = colors.arrow_normal;
    }
}

pub fn on_arrow_click(
    trigger: On<Pointer<Click>>,
    arrows: Query<&RotationArrow>,
    parents_query: Query<&ChildOf>,
    camera_link_query: Query<&ViewCubeLink, With<ViewCubeCamera>>,
    root_query: Query<(Entity, &ViewCubeLink), With<ViewCubeRoot>>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    let entity = trigger.entity;

    let Ok(arrow) = arrows.get(entity) else {
        return;
    };

    if trigger.event().button != PointerButton::Primary {
        return;
    }

    // Arrows are children of the ViewCube camera (not ViewCubeRoot).
    // Walk up to find the camera with ViewCubeLink, then find the root with matching main_camera.
    let source =
        find_root_for_camera_child(entity, &parents_query, &camera_link_query, &root_query)
            .unwrap_or(Entity::PLACEHOLDER);

    events.write(ViewCubeEvent::ArrowClicked {
        arrow: *arrow,
        source,
    });
}

// ============================================================================
// Highlight Helpers
// ============================================================================

#[allow(clippy::collapsible_if)]
fn apply_highlight(
    entity: Entity,
    children_query: &Query<&Children>,
    material_query: &Query<&MeshMaterial3d<StandardMaterial>>,
    materials: &mut Assets<StandardMaterial>,
    original_materials: &mut OriginalMaterials,
    _commands: &mut Commands,
) {
    let colors = ViewCubeColors::default();

    if let Ok(mat_handle) = material_query.get(entity) {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            original_materials
                .colors
                .entry(entity)
                .or_insert(mat.base_color);
            mat.base_color = colors.face_hover;
            mat.emissive = colors.highlight_emissive;
        }
    }

    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            if let Ok(mat_handle) = material_query.get(child) {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    original_materials
                        .colors
                        .entry(child)
                        .or_insert(mat.base_color);
                    mat.base_color = colors.face_hover;
                    mat.emissive = colors.highlight_emissive;
                }
            }
        }
    }
}

#[allow(clippy::collapsible_if)]
fn reset_highlight(
    entity: Entity,
    children_query: &Query<&Children>,
    material_query: &Query<&MeshMaterial3d<StandardMaterial>>,
    materials: &mut Assets<StandardMaterial>,
    original_materials: &OriginalMaterials,
) {
    if let Ok(mat_handle) = material_query.get(entity) {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            if let Some(&original_color) = original_materials.colors.get(&entity) {
                mat.base_color = original_color;
            }
            mat.emissive = LinearRgba::BLACK;
        }
    }

    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            if let Ok(mat_handle) = material_query.get(child) {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    if let Some(&original_color) = original_materials.colors.get(&child) {
                        mat.base_color = original_color;
                    }
                    mat.emissive = LinearRgba::BLACK;
                }
            }
        }
    }
}

/// Walk up the entity hierarchy to find the ViewCubeRoot ancestor.
/// This identifies which ViewCube instance an element belongs to.
/// Works for cube faces/edges/corners that are children of ViewCubeRoot.
fn find_root_ancestor(
    entity: Entity,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
) -> Option<Entity> {
    find_ancestor(entity, parents_query, |current| {
        root_query.get(current).is_ok()
    })
}

/// Find the ViewCubeRoot for an entity that is a child of the ViewCube camera
/// (e.g., rotation arrows). Walks up to find the camera with ViewCubeLink,
/// then finds the root that shares the same main_camera.
fn find_root_for_camera_child(
    entity: Entity,
    parents_query: &Query<&ChildOf>,
    camera_link_query: &Query<&ViewCubeLink, With<ViewCubeCamera>>,
    root_query: &Query<(Entity, &ViewCubeLink), With<ViewCubeRoot>>,
) -> Option<Entity> {
    let camera_entity = find_ancestor(entity, parents_query, |current| {
        camera_link_query.get(current).is_ok()
    })?;
    let cam_link = camera_link_query.get(camera_entity).ok()?;

    // Found the camera, now find the root with the same main_camera.
    for (root_entity, root_link) in root_query.iter() {
        if root_link.main_camera == cam_link.main_camera {
            return Some(root_entity);
        }
    }

    None
}

fn find_ancestor(
    entity: Entity,
    parents_query: &Query<&ChildOf>,
    mut predicate: impl FnMut(Entity) -> bool,
) -> Option<Entity> {
    let mut current = entity;
    loop {
        if predicate(current) {
            return Some(current);
        }
        if let Ok(parent) = parents_query.get(current) {
            current = parent.0;
        } else {
            return None;
        }
    }
}
