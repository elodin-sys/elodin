//! Interaction handlers for the ViewCube widget

use bevy::ecs::hierarchy::ChildOf;
use bevy::log::info;
use bevy::picking::prelude::*;
use bevy::prelude::*;

use super::camera::ViewCubeTargetCamera;
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
    mut transforms: Query<&mut Transform>,
    parents: Query<&ChildOf>,
    children_query: Query<&Children>,
    mesh_roots: Query<Entity, With<ViewCubeMeshRoot>>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    const EDGE_HOVER_SCALE: f32 = 1.2;

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
            if matches!(elem, CubeElement::Edge(_))
                && let Ok(mut transform) = transforms.get_mut(entity)
            {
                // Slightly enlarge border meshes so edge/frame hover is easier to trigger.
                transform.scale *= Vec3::splat(EDGE_HOVER_SCALE);
            }

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
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
) -> Option<Entity> {
    find_ancestor(entity, parents_query, |current| {
        cube_elements.get(current).is_ok()
    })
}

fn same_entity_set(lhs: &[Entity], rhs: &[Entity]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().all(|entity| rhs.contains(entity))
}

#[derive(Debug)]
struct EdgeHoverCandidate {
    frame_face: FaceDirection,
    frame_face_dot: f32,
    secondary_face: FaceDirection,
    secondary_face_dot: f32,
    target_face: FaceDirection,
    chosen_up_source: &'static str,
    rotation_angle: f32,
}

fn build_edge_hover_candidate(
    camera_rotation: Quat,
    frame_face: FaceDirection,
    frame_face_dot: f32,
    secondary_face: FaceDirection,
    secondary_face_dot: f32,
) -> Option<EdgeHoverCandidate> {
    let target_face = frame_face.opposite();
    let facing_world = -target_face.to_look_direction();
    let (_up_world, chosen_up_source, rotation_angle) =
        choose_min_rotation_up_world(camera_rotation, facing_world)?;
    Some(EdgeHoverCandidate {
        frame_face,
        frame_face_dot,
        secondary_face,
        secondary_face_dot,
        target_face,
        chosen_up_source,
        rotation_angle,
    })
}

fn choose_min_rotation_up_world(
    camera_rotation: Quat,
    facing_world: Vec3,
) -> Option<(Vec3, &'static str, f32)> {
    let mut best: Option<(Vec3, &'static str, f32)> = None;

    for (label, up_world) in [
        ("world_pos_x", Vec3::X),
        ("world_neg_x", Vec3::NEG_X),
        ("world_pos_y", Vec3::Y),
        ("world_neg_y", Vec3::NEG_Y),
        ("world_pos_z", Vec3::Z),
        ("world_neg_z", Vec3::NEG_Z),
    ] {
        if facing_world.dot(up_world).abs() > 0.99 {
            continue;
        }
        let target_rotation = Transform::default()
            .looking_to(facing_world, up_world)
            .rotation;
        let angle = camera_rotation.angle_between(target_rotation).abs();
        let replace = match best {
            Some((_, _, best_angle)) => angle + 1.0e-6 < best_angle,
            None => true,
        };
        if replace {
            best = Some((up_world, label, angle));
        }
    }

    best
}

#[allow(clippy::too_many_arguments)]
fn compute_hover_targets(
    target: Entity,
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
    root_links: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: &Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
) -> Vec<Entity> {
    let Ok((_, element)) = cube_elements.get(target) else {
        return vec![target];
    };

    let CubeElement::Edge(edge_under_cursor) = *element else {
        return vec![target];
    };

    let Some(root) = find_root_ancestor(target, parents_query, root_query) else {
        return vec![target];
    };
    let Ok(link) = root_links.get(root) else {
        return vec![target];
    };
    let Ok(camera_global) = camera_globals.get(link.main_camera) else {
        return vec![target];
    };

    let (_, cam_rotation, _) = camera_global.to_scale_rotation_translation();
    let camera_dir_world = cam_rotation * Vec3::Z;
    let (face_a, face_b) = edge_under_cursor.adjacent_faces();
    let dot_a = face_a.to_look_direction().dot(camera_dir_world);
    let dot_b = face_b.to_look_direction().dot(camera_dir_world);
    let candidate_a = build_edge_hover_candidate(cam_rotation, face_a, dot_a, face_b, dot_b);
    let candidate_b = build_edge_hover_candidate(cam_rotation, face_b, dot_b, face_a, dot_a);

    let chosen = match (&candidate_a, &candidate_b) {
        (Some(a), Some(b)) => {
            if a.rotation_angle <= b.rotation_angle + 1.0e-6 {
                Some(a)
            } else {
                Some(b)
            }
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let Some(chosen) = chosen else {
        return vec![target];
    };

    let frame_face = chosen.frame_face;
    let edge_group = edges_for_face(frame_face);

    let mut targets = Vec::new();
    for (entity, element) in cube_elements.iter() {
        let CubeElement::Edge(candidate_edge) = *element else {
            continue;
        };
        if !edge_group.contains(&candidate_edge) {
            continue;
        }
        let candidate_root = find_root_ancestor(entity, parents_query, root_query);
        if candidate_root == Some(root) {
            targets.push(entity);
        }
    }

    if targets.is_empty() {
        targets.push(target);
    }

    info!(
        hover_edge = ?edge_under_cursor,
        camera_dir_world = ?camera_dir_world,
        candidate_a_frame_face = ?candidate_a.as_ref().map(|c| c.frame_face),
        candidate_a_target_face = ?candidate_a.as_ref().map(|c| c.target_face),
        candidate_a_rotation_angle = ?candidate_a.as_ref().map(|c| c.rotation_angle),
        candidate_b_frame_face = ?candidate_b.as_ref().map(|c| c.frame_face),
        candidate_b_target_face = ?candidate_b.as_ref().map(|c| c.target_face),
        candidate_b_rotation_angle = ?candidate_b.as_ref().map(|c| c.rotation_angle),
        frame_face = ?chosen.frame_face,
        frame_face_dot = chosen.frame_face_dot,
        secondary_face = ?chosen.secondary_face,
        secondary_face_dot = chosen.secondary_face_dot,
        target_face = ?chosen.target_face,
        chosen_up_source = chosen.chosen_up_source,
        rotation_angle = chosen.rotation_angle,
        edge_group = ?edge_group,
        highlighted_edges = targets.len(),
        "view cube: edge hover group"
    );

    targets
}

fn edges_for_face(face: FaceDirection) -> [EdgeDirection; 4] {
    match face {
        FaceDirection::North => [
            EdgeDirection::XTopFront,
            EdgeDirection::XBottomFront,
            EdgeDirection::YFrontLeft,
            EdgeDirection::YFrontRight,
        ],
        FaceDirection::South => [
            EdgeDirection::XTopBack,
            EdgeDirection::XBottomBack,
            EdgeDirection::YBackLeft,
            EdgeDirection::YBackRight,
        ],
        FaceDirection::East => [
            EdgeDirection::YFrontRight,
            EdgeDirection::YBackRight,
            EdgeDirection::ZTopRight,
            EdgeDirection::ZBottomRight,
        ],
        FaceDirection::West => [
            EdgeDirection::YFrontLeft,
            EdgeDirection::YBackLeft,
            EdgeDirection::ZTopLeft,
            EdgeDirection::ZBottomLeft,
        ],
        FaceDirection::Up => [
            EdgeDirection::XTopFront,
            EdgeDirection::XTopBack,
            EdgeDirection::ZTopLeft,
            EdgeDirection::ZTopRight,
        ],
        FaceDirection::Down => [
            EdgeDirection::XBottomFront,
            EdgeDirection::XBottomBack,
            EdgeDirection::ZBottomLeft,
            EdgeDirection::ZBottomRight,
        ],
    }
}

// ============================================================================
// Cube Element Hover Handlers
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn on_cube_hover_start(
    trigger: On<Pointer<Over>>,
    mut commands: Commands,
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    root_links: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
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

    let target_entities = compute_hover_targets(
        target,
        &cube_elements,
        &parents_query,
        &root_query,
        &root_links,
        &camera_globals,
    );

    if same_entity_set(&hovered.entities, &target_entities) {
        return;
    }

    // Reset previous
    for prev in hovered.entities.iter().copied() {
        reset_highlight(
            prev,
            &children_query,
            &material_query,
            &mut materials,
            &original_materials,
        );
    }

    // Apply highlight
    for hover_target in target_entities.iter().copied() {
        apply_highlight(
            hover_target,
            &children_query,
            &material_query,
            &mut materials,
            &mut original_materials,
            &mut commands,
        );
    }

    hovered.entity = Some(target);
    hovered.entities = target_entities;
}

#[allow(clippy::too_many_arguments)]
pub fn on_cube_hover_end(
    trigger: On<Pointer<Out>>,
    cube_elements: Query<(Entity, &CubeElement)>,
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

    if !hovered.entities.contains(&target) {
        return;
    }

    for hover_target in hovered.entities.iter().copied() {
        reset_highlight(
            hover_target,
            &children_query,
            &material_query,
            &mut materials,
            &original_materials,
        );
    }
    hovered.entity = None;
    hovered.entities.clear();
}

pub fn on_cube_click(
    trigger: On<Pointer<Click>>,
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    names: Query<&Name>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    let entity = trigger.entity;

    let Some(target_entity) = find_cube_element_ancestor(entity, &cube_elements, &parents_query)
    else {
        return;
    };

    // Pointer click events can bubble from mesh children to the CubeElement parent.
    // Handle only the canonical callback on the CubeElement entity to avoid
    // emitting duplicated ViewCubeEvent for a single click.
    if entity != target_entity {
        info!(
            trigger_entity = %entity,
            canonical_entity = %target_entity,
            pointer_event = ?trigger.event(),
            "view cube: skipping bubbled click"
        );
        return;
    }

    let Ok((_, element)) = cube_elements.get(target_entity) else {
        return;
    };

    // Find the ViewCubeRoot ancestor to identify which ViewCube was clicked
    let source =
        find_root_ancestor(entity, &parents_query, &root_query).unwrap_or(Entity::PLACEHOLDER);

    let target_name = names
        .get(target_entity)
        .map(|name| name.as_str())
        .unwrap_or("<unnamed>");
    let source_name = names
        .get(source)
        .map(|name| name.as_str())
        .unwrap_or("<unnamed>");
    info!(
        trigger_entity = %entity,
        target_entity = %target_entity,
        target_name = target_name,
        source = %source,
        source_name = source_name,
        element = ?element,
        pointer_event = ?trigger.event(),
        "view cube: on_cube_click resolved"
    );

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
    names: Query<&Name>,
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

    let arrow_name = names
        .get(entity)
        .map(|name| name.as_str())
        .unwrap_or("<unnamed>");
    let source_name = names
        .get(source)
        .map(|name| name.as_str())
        .unwrap_or("<unnamed>");
    info!(
        trigger_entity = %entity,
        trigger_name = arrow_name,
        source = %source,
        source_name = source_name,
        arrow = ?arrow,
        pointer_event = ?trigger.event(),
        "view cube: on_arrow_click resolved"
    );

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
