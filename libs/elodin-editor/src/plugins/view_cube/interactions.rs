//! Interaction handlers for the ViewCube widget

use bevy::ecs::hierarchy::ChildOf;
use bevy::log::debug;
use bevy::picking::prelude::*;
use bevy::prelude::*;

use super::camera::ViewCubeTargetCamera;
use super::components::*;
use super::config::ViewCubeConfig;
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
    const CORNER_PICK_SCALE: f32 = 1.15;

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
            if matches!(elem, CubeElement::Corner(_))
                && let Ok(mut transform) = transforms.get_mut(entity)
            {
                // Slightly enlarge corners to improve click tolerance.
                transform.scale *= Vec3::splat(CORNER_PICK_SCALE);
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
        // In axes-cube.glb, Face_Back is translated to +Z and Face_Front to -Z.
        // Map names to world directions, not to lexical front/back wording.
        "Face_Front" => FaceDirection::South,
        "Face_Back" => FaceDirection::North,
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

const FACE_FRONT_DOT_THRESHOLD: f32 = 0.95;
const FACE_VISIBLE_DOT_THRESHOLD: f32 = 0.0;

#[derive(Clone, Copy, Debug)]
struct EdgeInteractionContext {
    camera_dir_cube: Vec3,
    front_face: FaceDirection,
    front_dot: f32,
}

fn face_dot(face: FaceDirection, camera_dir_cube: Vec3) -> f32 {
    face.to_look_direction().dot(camera_dir_cube)
}

fn front_face(camera_dir_cube: Vec3) -> (FaceDirection, f32) {
    [
        FaceDirection::East,
        FaceDirection::West,
        FaceDirection::North,
        FaceDirection::South,
        FaceDirection::Up,
        FaceDirection::Down,
    ]
    .into_iter()
    .map(|face| (face, face_dot(face, camera_dir_cube)))
    .max_by(|(_, dot_a), (_, dot_b)| dot_a.total_cmp(dot_b))
    .unwrap_or((FaceDirection::North, 0.0))
}

#[allow(clippy::too_many_arguments)]
fn edge_interaction_context(
    target: Entity,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
    root_links: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: &Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: &Query<&GlobalTransform, With<ViewCubeRoot>>,
    config: &ViewCubeConfig,
) -> Option<(Entity, EdgeInteractionContext)> {
    let root = find_root_ancestor(target, parents_query, root_query)?;
    let link = root_links.get(root).ok()?;
    let camera_global = camera_globals.get(link.main_camera).ok()?;
    let cube_global = root_globals
        .get(root)
        .ok()
        .map(|transform| (transform.translation(), transform.rotation()));

    let (_, cam_rotation, _) = camera_global.to_scale_rotation_translation();
    let camera_dir_world = cam_rotation * Vec3::Z;
    let cube_rotation = if config.sync_with_camera {
        cam_rotation.conjugate() * config.effective_axis_correction()
    } else {
        cube_global
            .map(|(_, rotation)| rotation)
            .unwrap_or(Quat::IDENTITY)
    };
    let view_dir_world = if config.use_overlay {
        Vec3::Z
    } else if let Some((cube_translation, _)) = cube_global {
        (camera_global.translation() - cube_translation).normalize_or_zero()
    } else {
        camera_dir_world
    };
    let camera_dir_cube = cube_rotation.inverse() * view_dir_world;
    let (front_face, front_dot) = front_face(camera_dir_cube);

    Some((
        root,
        EdgeInteractionContext {
            camera_dir_cube,
            front_face,
            front_dot,
        },
    ))
}

fn resolve_edge_target_face(
    edge_under_cursor: EdgeDirection,
    context: EdgeInteractionContext,
) -> Option<FaceDirection> {
    if context.front_dot >= FACE_FRONT_DOT_THRESHOLD {
        if edges_for_face(context.front_face).contains(&edge_under_cursor) {
            return Some(context.front_face.opposite());
        }
        return None;
    }

    let (face_a, face_b) = edge_under_cursor.adjacent_faces();
    let dot_a = face_dot(face_a, context.camera_dir_cube);
    let dot_b = face_dot(face_b, context.camera_dir_cube);
    let a_visible = dot_a >= FACE_VISIBLE_DOT_THRESHOLD;
    let b_visible = dot_b >= FACE_VISIBLE_DOT_THRESHOLD;

    match (a_visible, b_visible) {
        (true, false) => Some(face_b),
        (false, true) => Some(face_a),
        (false, false) => {
            // Degenerate case: both faces hidden. Prefer the more hidden face.
            if dot_a <= dot_b {
                Some(face_a)
            } else {
                Some(face_b)
            }
        }
        (true, true) => None,
    }
}

fn edge_is_visible_for_target_face(
    edge: EdgeDirection,
    target_face: FaceDirection,
    camera_dir_cube: Vec3,
) -> bool {
    let (face_a, face_b) = edge.adjacent_faces();
    let other_face = if face_a == target_face {
        face_b
    } else if face_b == target_face {
        face_a
    } else {
        return false;
    };

    face_dot(other_face, camera_dir_cube) >= FACE_VISIBLE_DOT_THRESHOLD
}

fn edge_group_for_target_face(
    edge_under_cursor: EdgeDirection,
    target_face: FaceDirection,
    context: EdgeInteractionContext,
) -> Vec<EdgeDirection> {
    let mut group: Vec<EdgeDirection> = if context.front_dot >= FACE_FRONT_DOT_THRESHOLD
        && target_face == context.front_face.opposite()
    {
        edges_for_face(context.front_face).into_iter().collect()
    } else {
        edges_for_face(target_face)
            .into_iter()
            .filter(|edge| {
                edge_is_visible_for_target_face(*edge, target_face, context.camera_dir_cube)
            })
            .collect()
    };

    if !group.contains(&edge_under_cursor) {
        group.push(edge_under_cursor);
    }

    group
}

fn collect_edge_entities_for_root(
    root: Entity,
    edge_group: &[EdgeDirection],
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
) -> Vec<Entity> {
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
    targets
}

#[allow(clippy::too_many_arguments)]
fn compute_hover_targets(
    target: Entity,
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
    root_links: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: &Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: &Query<&GlobalTransform, With<ViewCubeRoot>>,
    config: &ViewCubeConfig,
) -> Vec<Entity> {
    let Ok((_, element)) = cube_elements.get(target) else {
        return vec![];
    };

    let CubeElement::Edge(edge_under_cursor) = *element else {
        return vec![target];
    };

    let Some((root, context)) = edge_interaction_context(
        target,
        parents_query,
        root_query,
        root_links,
        camera_globals,
        root_globals,
        config,
    ) else {
        return vec![];
    };

    let Some(target_face) = resolve_edge_target_face(edge_under_cursor, context) else {
        return vec![];
    };
    let edge_group = edge_group_for_target_face(edge_under_cursor, target_face, context);
    let targets =
        collect_edge_entities_for_root(root, &edge_group, cube_elements, parents_query, root_query);

    debug!(
        hover_edge = ?edge_under_cursor,
        front_face = ?context.front_face,
        front_dot = context.front_dot,
        target_face = ?target_face,
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
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    root_links: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: Query<&GlobalTransform, With<ViewCubeRoot>>,
    config: Res<ViewCubeConfig>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hovered: ResMut<HoveredElement>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    let entity = trigger.entity;
    let colors = ViewCubeColors::default();

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
        &root_globals,
        &config,
    );

    if target_entities.is_empty() {
        for prev in hovered.entities.iter().copied() {
            reset_highlight(
                prev,
                &children_query,
                &material_query,
                &mut materials,
                &original_materials,
            );
        }
        hovered.entity = None;
        hovered.entities.clear();
        return;
    }

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
        let (hover_color, hover_emissive) =
            if let Ok((_, element)) = cube_elements.get(hover_target) {
                (colors.get_element_hover(element), colors.highlight_emissive)
            } else {
                (colors.face_hover, colors.highlight_emissive)
            };
        apply_highlight(
            hover_target,
            &children_query,
            &material_query,
            &mut materials,
            &mut original_materials,
            hover_color,
            hover_emissive,
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
    root_links: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: Query<&GlobalTransform, With<ViewCubeRoot>>,
    config: Res<ViewCubeConfig>,
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
        debug!(
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
    debug!(
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
            let Some((_root, context)) = edge_interaction_context(
                target_entity,
                &parents_query,
                &root_query,
                &root_links,
                &camera_globals,
                &root_globals,
                &config,
            ) else {
                return;
            };
            let Some(target_face) = resolve_edge_target_face(*dir, context) else {
                return;
            };

            events.write(ViewCubeEvent::EdgeClicked {
                direction: *dir,
                target_face,
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
    debug!(
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
    hover_color: Color,
    hover_emissive: LinearRgba,
) {
    if let Ok(mat_handle) = material_query.get(entity) {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            original_materials
                .colors
                .entry(entity)
                .or_insert(mat.base_color);
            mat.base_color = hover_color;
            mat.emissive = hover_emissive;
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
                    mat.base_color = hover_color;
                    mat.emissive = hover_emissive;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn context(camera_dir_cube: Vec3) -> EdgeInteractionContext {
        let (front_face, front_dot) = front_face(camera_dir_cube);
        EdgeInteractionContext {
            camera_dir_cube,
            front_face,
            front_dot,
        }
    }

    #[test]
    fn edge_target_face_uses_opposite_when_face_on() {
        let ctx = context(Vec3::Z);
        let edge = EdgeDirection::YFrontLeft;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, Some(FaceDirection::South));
    }

    #[test]
    fn edge_target_face_selects_hidden_face_in_oblique_view() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let edge = EdgeDirection::YFrontLeft;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, Some(FaceDirection::West));

        let group = edge_group_for_target_face(edge, FaceDirection::West, ctx);
        assert_eq!(group.len(), 2);
        assert!(group.contains(&EdgeDirection::YFrontLeft));
        assert!(group.contains(&EdgeDirection::ZTopLeft));
    }

    #[test]
    fn edge_between_two_visible_faces_is_not_clickable() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let edge = EdgeDirection::XTopFront; // Up + North (both visible in this view)
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, None);
    }
}
