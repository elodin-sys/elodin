//! Interaction handlers for the ViewCube widget.

use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::system::SystemParam;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy_editor_cam::controller::component::EditorCam;

use super::camera::ViewCubeTargetCamera;
use super::components::*;
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;
use super::theme::ViewCubeColors;
use crate::plugins::camera_anchor::camera_anchor_from_transform;

#[derive(Clone)]
struct ArrowHold {
    arrow: RotationArrow,
    source: Entity,
    timer: Timer,
}

#[derive(Resource, Default)]
pub struct ActiveArrowHold {
    active: Option<ArrowHold>,
}

const ARROW_HOLD_REPEAT_INTERVAL_SECS: f32 = 0.1;

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
    const CORNER_PICK_SCALE: f32 = 1.45;
    const CORNER_OUTWARD_BIAS: f32 = 0.035;

    if mesh_roots.is_empty() {
        return;
    }

    let colors = ViewCubeColors::default();

    for (entity, name) in query.iter() {
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

        let element = if name_str.starts_with("Face_") {
            transforms
                .get(entity)
                .ok()
                .and_then(|transform| classify_face_from_translation(transform.translation))
                .map(CubeElement::Face)
                .or_else(|| parse_face(name_str))
        } else if name_str.starts_with("Border_") {
            transforms
                .get(entity)
                .ok()
                .and_then(|transform| classify_edge_from_translation(transform.translation))
                .map(CubeElement::Edge)
                .or_else(|| parse_edge(name_str))
        } else if name_str.starts_with("Corner_") {
            transforms
                .get(entity)
                .ok()
                .and_then(|transform| classify_corner_from_translation(transform.translation))
                .map(CubeElement::Corner)
                .or_else(|| parse_corner(name_str))
        } else {
            None
        };

        if let Some(elem) = element {
            let element_color = colors.get_element_color(&elem);
            let make_theme_material = |base_color: Color| StandardMaterial {
                base_color,
                alpha_mode: AlphaMode::Opaque,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            };
            if matches!(elem, CubeElement::Edge(_))
                && let Ok(mut transform) = transforms.get_mut(entity)
            {
                transform.scale *= Vec3::splat(EDGE_HOVER_SCALE);
            }
            if matches!(elem, CubeElement::Corner(_))
                && let Ok(mut transform) = transforms.get_mut(entity)
            {
                let outward = transform.translation.normalize_or_zero();
                if outward.length_squared() > 1.0e-6 {
                    transform.translation += outward * CORNER_OUTWARD_BIAS;
                }
                transform.scale *= Vec3::splat(CORNER_PICK_SCALE);
            }

            commands
                .entity(entity)
                .insert((elem.clone(), ViewCubeSetup));

            if let Ok(mat_handle) = material_query.get(entity)
                && materials.get(&mat_handle.0).is_some()
            {
                original_materials.colors.insert(entity, element_color);
                let new_mat = make_theme_material(element_color);
                let new_handle = materials.add(new_mat);
                commands.entity(entity).insert(MeshMaterial3d(new_handle));
            }

            if let Ok(children) = children_query.get(entity) {
                for child in children.iter() {
                    if let Ok(mat_handle) = material_query.get(child)
                        && materials.get(&mat_handle.0).is_some()
                    {
                        original_materials.colors.insert(child, element_color);
                        let new_mat = make_theme_material(element_color);
                        let new_handle = materials.add(new_mat);
                        commands.entity(child).insert(MeshMaterial3d(new_handle));
                    }
                }
            }
        }
    }
}

fn classify_face_from_translation(translation: Vec3) -> Option<FaceDirection> {
    let abs = translation.abs();
    let max = abs.max_element();
    if max <= 1.0e-3 {
        return None;
    }

    if abs.x >= abs.y && abs.x >= abs.z {
        Some(if translation.x >= 0.0 {
            FaceDirection::East
        } else {
            FaceDirection::West
        })
    } else if abs.y >= abs.x && abs.y >= abs.z {
        Some(if translation.y >= 0.0 {
            FaceDirection::Up
        } else {
            FaceDirection::Down
        })
    } else {
        Some(if translation.z >= 0.0 {
            FaceDirection::North
        } else {
            FaceDirection::South
        })
    }
}

fn classify_edge_from_translation(translation: Vec3) -> Option<EdgeDirection> {
    const AXIS_NEAR_ZERO_EPS: f32 = 0.18;
    let abs = translation.abs();

    if abs.x <= AXIS_NEAR_ZERO_EPS {
        let is_top = translation.y >= 0.0;
        let is_front = translation.z >= 0.0;
        return Some(match (is_top, is_front) {
            (true, true) => EdgeDirection::XTopFront,
            (true, false) => EdgeDirection::XTopBack,
            (false, true) => EdgeDirection::XBottomFront,
            (false, false) => EdgeDirection::XBottomBack,
        });
    }

    if abs.y <= AXIS_NEAR_ZERO_EPS {
        let is_front = translation.z >= 0.0;
        let is_left = translation.x < 0.0;
        return Some(match (is_front, is_left) {
            (true, true) => EdgeDirection::YFrontLeft,
            (true, false) => EdgeDirection::YFrontRight,
            (false, true) => EdgeDirection::YBackLeft,
            (false, false) => EdgeDirection::YBackRight,
        });
    }

    if abs.z <= AXIS_NEAR_ZERO_EPS {
        let is_top = translation.y >= 0.0;
        let is_left = translation.x < 0.0;
        return Some(match (is_top, is_left) {
            (true, true) => EdgeDirection::ZTopLeft,
            (true, false) => EdgeDirection::ZTopRight,
            (false, true) => EdgeDirection::ZBottomLeft,
            (false, false) => EdgeDirection::ZBottomRight,
        });
    }

    None
}

fn classify_corner_from_translation(translation: Vec3) -> Option<CornerPosition> {
    const CORNER_MIN_ABS: f32 = 0.18;
    let abs = translation.abs();
    if abs.min_element() <= CORNER_MIN_ABS {
        return None;
    }

    let is_top = translation.y >= 0.0;
    let is_front = translation.z >= 0.0;
    let is_left = translation.x < 0.0;

    Some(match (is_top, is_front, is_left) {
        (true, true, true) => CornerPosition::TopFrontLeft,
        (true, true, false) => CornerPosition::TopFrontRight,
        (true, false, true) => CornerPosition::TopBackLeft,
        (true, false, false) => CornerPosition::TopBackRight,
        (false, true, true) => CornerPosition::BottomFrontLeft,
        (false, true, false) => CornerPosition::BottomFrontRight,
        (false, false, true) => CornerPosition::BottomBackLeft,
        (false, false, false) => CornerPosition::BottomBackRight,
    })
}

fn parse_face(name: &str) -> Option<CubeElement> {
    let dir = match name {
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
const FACE_HIDDEN_CLASS_DOT_THRESHOLD: f32 = 0.05;
const FACE_ON_SECOND_FACE_MAX_DOT: f32 = 0.08;
const EDGE_GROUP_NEIGHBOR_MIN_DOT: f32 = -0.02;

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
    let mut face_dots: Vec<(FaceDirection, f32)> = [
        FaceDirection::East,
        FaceDirection::West,
        FaceDirection::North,
        FaceDirection::South,
        FaceDirection::Up,
        FaceDirection::Down,
    ]
    .into_iter()
    .map(|face| (face, face_dot(face, camera_dir_cube)))
    .collect();
    face_dots.sort_by(|(_, dot_a), (_, dot_b)| dot_b.total_cmp(dot_a));

    let (front_face, front_dot) = face_dots
        .first()
        .copied()
        .unwrap_or((FaceDirection::North, 0.0));
    (front_face, front_dot)
}

fn second_most_visible_face_dot(camera_dir_cube: Vec3, front_face: FaceDirection) -> f32 {
    [
        FaceDirection::East,
        FaceDirection::West,
        FaceDirection::North,
        FaceDirection::South,
        FaceDirection::Up,
        FaceDirection::Down,
    ]
    .into_iter()
    .filter(|face| *face != front_face)
    .map(|face| face_dot(face, camera_dir_cube))
    .max_by(|dot_a, dot_b| dot_a.total_cmp(dot_b))
    .unwrap_or(-1.0)
}

fn is_face_on(context: EdgeInteractionContext) -> bool {
    context.front_dot >= FACE_FRONT_DOT_THRESHOLD
        && second_most_visible_face_dot(context.camera_dir_cube, context.front_face)
            <= FACE_ON_SECOND_FACE_MAX_DOT
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
    let cube_global = root_globals.get(root).ok().map(GlobalTransform::rotation);

    let (_, cam_rotation, _) = camera_global.to_scale_rotation_translation();
    let cube_rotation = if config.sync_with_camera {
        cam_rotation.conjugate() * config.effective_axis_correction()
    } else {
        cube_global.unwrap_or(Quat::IDENTITY)
    };
    let camera_dir_cube = cube_rotation.inverse() * Vec3::Z;
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
    if is_face_on(context) {
        if edges_for_face(context.front_face).contains(&edge_under_cursor) {
            return Some(context.front_face.opposite());
        }
        return None;
    }

    let (face_a, face_b) = edge_under_cursor.adjacent_faces();
    let dot_a = face_dot(face_a, context.camera_dir_cube);
    let dot_b = face_dot(face_b, context.camera_dir_cube);
    let a_visible = dot_a >= FACE_HIDDEN_CLASS_DOT_THRESHOLD;
    let b_visible = dot_b >= FACE_HIDDEN_CLASS_DOT_THRESHOLD;
    let a_hidden = dot_a <= -FACE_HIDDEN_CLASS_DOT_THRESHOLD;
    let b_hidden = dot_b <= -FACE_HIDDEN_CLASS_DOT_THRESHOLD;

    match (a_visible, b_visible, a_hidden, b_hidden) {
        (true, false, false, true) => Some(face_b),
        (false, true, true, false) => Some(face_a),
        _ => None,
    }
}

fn other_face_dot_for_target_face(
    edge: EdgeDirection,
    target_face: FaceDirection,
    camera_dir_cube: Vec3,
) -> Option<f32> {
    let (face_a, face_b) = edge.adjacent_faces();
    let other_face = if face_a == target_face {
        face_b
    } else if face_b == target_face {
        face_a
    } else {
        return None;
    };

    Some(face_dot(other_face, camera_dir_cube))
}

fn edge_is_visible_for_target_face(
    edge: EdgeDirection,
    target_face: FaceDirection,
    camera_dir_cube: Vec3,
) -> bool {
    other_face_dot_for_target_face(edge, target_face, camera_dir_cube)
        .is_some_and(|dot| dot >= EDGE_GROUP_NEIGHBOR_MIN_DOT)
}

fn edge_group_for_target_face(
    target_face: FaceDirection,
    context: EdgeInteractionContext,
) -> Vec<EdgeDirection> {
    let mut group: Vec<EdgeDirection> =
        if is_face_on(context) && target_face == context.front_face.opposite() {
            edges_for_face(context.front_face).into_iter().collect()
        } else {
            let mut group: Vec<EdgeDirection> = edges_for_face(target_face)
                .into_iter()
                .filter(|edge| {
                    edge_is_visible_for_target_face(*edge, target_face, context.camera_dir_cube)
                })
                .collect();

            // Keep hidden-face groups coherent in oblique views: if filtering gets too strict,
            // include the best in-plane candidate so the group remains at least 2 edges.
            if group.len() < 2 {
                let mut fallback: Vec<(EdgeDirection, f32)> = edges_for_face(target_face)
                    .into_iter()
                    .filter_map(|edge| {
                        other_face_dot_for_target_face(edge, target_face, context.camera_dir_cube)
                            .map(|dot| (edge, dot))
                    })
                    .filter(|(_, dot)| *dot > -FACE_HIDDEN_CLASS_DOT_THRESHOLD)
                    .collect();
                fallback.sort_by(|(_, dot_a), (_, dot_b)| dot_b.total_cmp(dot_a));
                for (edge, _) in fallback {
                    if !group.contains(&edge) {
                        group.push(edge);
                    }
                    if group.len() >= 2 {
                        break;
                    }
                }
            }

            group
        };

    if group.len() < 2 {
        for edge in edges_for_face(target_face) {
            if !group.contains(&edge) {
                group.push(edge);
            }
            if group.len() >= 2 {
                break;
            }
        }
    }

    group
}

fn faces_for_corner(corner: CornerPosition) -> [FaceDirection; 3] {
    match corner {
        CornerPosition::TopFrontLeft => {
            [FaceDirection::Up, FaceDirection::North, FaceDirection::West]
        }
        CornerPosition::TopFrontRight => {
            [FaceDirection::Up, FaceDirection::North, FaceDirection::East]
        }
        CornerPosition::TopBackLeft => {
            [FaceDirection::Up, FaceDirection::South, FaceDirection::West]
        }
        CornerPosition::TopBackRight => {
            [FaceDirection::Up, FaceDirection::South, FaceDirection::East]
        }
        CornerPosition::BottomFrontLeft => [
            FaceDirection::Down,
            FaceDirection::North,
            FaceDirection::West,
        ],
        CornerPosition::BottomFrontRight => [
            FaceDirection::Down,
            FaceDirection::North,
            FaceDirection::East,
        ],
        CornerPosition::BottomBackLeft => [
            FaceDirection::Down,
            FaceDirection::South,
            FaceDirection::West,
        ],
        CornerPosition::BottomBackRight => [
            FaceDirection::Down,
            FaceDirection::South,
            FaceDirection::East,
        ],
    }
}

fn corners_for_edge(edge: EdgeDirection) -> [CornerPosition; 2] {
    match edge {
        EdgeDirection::XTopFront => [CornerPosition::TopFrontLeft, CornerPosition::TopFrontRight],
        EdgeDirection::XTopBack => [CornerPosition::TopBackLeft, CornerPosition::TopBackRight],
        EdgeDirection::XBottomFront => [
            CornerPosition::BottomFrontLeft,
            CornerPosition::BottomFrontRight,
        ],
        EdgeDirection::XBottomBack => [
            CornerPosition::BottomBackLeft,
            CornerPosition::BottomBackRight,
        ],
        EdgeDirection::YFrontLeft => [
            CornerPosition::TopFrontLeft,
            CornerPosition::BottomFrontLeft,
        ],
        EdgeDirection::YFrontRight => [
            CornerPosition::TopFrontRight,
            CornerPosition::BottomFrontRight,
        ],
        EdgeDirection::YBackLeft => [CornerPosition::TopBackLeft, CornerPosition::BottomBackLeft],
        EdgeDirection::YBackRight => [
            CornerPosition::TopBackRight,
            CornerPosition::BottomBackRight,
        ],
        EdgeDirection::ZTopLeft => [CornerPosition::TopFrontLeft, CornerPosition::TopBackLeft],
        EdgeDirection::ZTopRight => [CornerPosition::TopFrontRight, CornerPosition::TopBackRight],
        EdgeDirection::ZBottomLeft => [
            CornerPosition::BottomFrontLeft,
            CornerPosition::BottomBackLeft,
        ],
        EdgeDirection::ZBottomRight => [
            CornerPosition::BottomFrontRight,
            CornerPosition::BottomBackRight,
        ],
    }
}

fn corner_group_for_edge_group(edge_group: &[EdgeDirection]) -> Vec<CornerPosition> {
    let mut corners = Vec::new();
    for edge in edge_group.iter().copied() {
        for corner in corners_for_edge(edge) {
            if !corners.contains(&corner) {
                corners.push(corner);
            }
        }
    }
    corners
}

fn corner_is_visible_for_target_face(
    corner: CornerPosition,
    target_face: FaceDirection,
    camera_dir_cube: Vec3,
) -> bool {
    let corner_faces = faces_for_corner(corner);
    if !corner_faces.contains(&target_face) {
        return false;
    }
    corner_faces
        .into_iter()
        .filter(|face| *face != target_face)
        .any(|face| face_dot(face, camera_dir_cube) >= EDGE_GROUP_NEIGHBOR_MIN_DOT)
}

#[cfg(test)]
fn hidden_target_face_for_corner(
    corner: CornerPosition,
    context: EdgeInteractionContext,
) -> Option<FaceDirection> {
    if is_face_on(context) {
        return None;
    }

    let mut hidden_faces: Vec<(FaceDirection, f32)> = faces_for_corner(corner)
        .into_iter()
        .map(|face| (face, face_dot(face, context.camera_dir_cube)))
        .filter(|(_, dot)| *dot <= -FACE_HIDDEN_CLASS_DOT_THRESHOLD)
        .collect();

    hidden_faces.sort_by(|(_, dot_a), (_, dot_b)| dot_a.total_cmp(dot_b));

    hidden_faces.into_iter().map(|(face, _)| face).find(|face| {
        edges_for_face(*face)
            .into_iter()
            .any(|edge| edge_is_visible_for_target_face(edge, *face, context.camera_dir_cube))
    })
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

fn collect_corner_entities_for_root(
    root: Entity,
    corner_group: &[CornerPosition],
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
) -> Vec<Entity> {
    let mut targets = Vec::new();
    for (entity, element) in cube_elements.iter() {
        let CubeElement::Corner(candidate_corner) = *element else {
            continue;
        };
        if !corner_group.contains(&candidate_corner) {
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

    match *element {
        CubeElement::Edge(edge_under_cursor) => {
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
            let mut edge_group = edge_group_for_target_face(target_face, context);
            if !edge_group.contains(&edge_under_cursor) {
                edge_group.push(edge_under_cursor);
            }
            let mut targets = collect_edge_entities_for_root(
                root,
                &edge_group,
                cube_elements,
                parents_query,
                root_query,
            );

            let corner_group: Vec<CornerPosition> = if is_face_on(context) {
                corners_for_face(context.front_face).to_vec()
            } else {
                corner_group_for_edge_group(&edge_group)
                    .into_iter()
                    .filter(|corner| {
                        corner_is_visible_for_target_face(
                            *corner,
                            target_face,
                            context.camera_dir_cube,
                        )
                    })
                    .collect()
            };

            let corner_targets = collect_corner_entities_for_root(
                root,
                &corner_group,
                cube_elements,
                parents_query,
                root_query,
            );
            targets.extend(corner_targets);

            targets
        }
        CubeElement::Corner(_) => vec![target],
        _ => vec![target],
    }
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

fn corners_for_face(face: FaceDirection) -> [CornerPosition; 4] {
    match face {
        FaceDirection::North => [
            CornerPosition::TopFrontLeft,
            CornerPosition::TopFrontRight,
            CornerPosition::BottomFrontLeft,
            CornerPosition::BottomFrontRight,
        ],
        FaceDirection::South => [
            CornerPosition::TopBackLeft,
            CornerPosition::TopBackRight,
            CornerPosition::BottomBackLeft,
            CornerPosition::BottomBackRight,
        ],
        FaceDirection::East => [
            CornerPosition::TopFrontRight,
            CornerPosition::TopBackRight,
            CornerPosition::BottomFrontRight,
            CornerPosition::BottomBackRight,
        ],
        FaceDirection::West => [
            CornerPosition::TopFrontLeft,
            CornerPosition::TopBackLeft,
            CornerPosition::BottomFrontLeft,
            CornerPosition::BottomBackLeft,
        ],
        FaceDirection::Up => [
            CornerPosition::TopFrontLeft,
            CornerPosition::TopFrontRight,
            CornerPosition::TopBackLeft,
            CornerPosition::TopBackRight,
        ],
        FaceDirection::Down => [
            CornerPosition::BottomFrontLeft,
            CornerPosition::BottomFrontRight,
            CornerPosition::BottomBackLeft,
            CornerPosition::BottomBackRight,
        ],
    }
}

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

    let mixed_edge_corner_group = if target_entities.len() > 1 {
        let mut has_edge = false;
        let mut has_corner = false;
        for hover_target in target_entities.iter().copied() {
            if let Ok((_, element)) = cube_elements.get(hover_target) {
                match element {
                    CubeElement::Edge(_) => has_edge = true,
                    CubeElement::Corner(_) => has_corner = true,
                    CubeElement::Face(_) => {}
                }
            }
        }
        has_edge && has_corner
    } else {
        false
    };

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

    for prev in hovered.entities.iter().copied() {
        reset_highlight(
            prev,
            &children_query,
            &material_query,
            &mut materials,
            &original_materials,
        );
    }

    for hover_target in target_entities.iter().copied() {
        let (hover_color, hover_emissive) = if mixed_edge_corner_group {
            (colors.edge_hover, colors.highlight_emissive)
        } else if let Ok((_, element)) = cube_elements.get(hover_target) {
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

#[derive(SystemParam)]
pub struct OnCubeClickLookup<'w, 's> {
    root_query: Query<'w, 's, Entity, With<ViewCubeRoot>>,
    root_links: Query<'w, 's, &'static ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: Query<'w, 's, &'static GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: Query<'w, 's, &'static GlobalTransform, With<ViewCubeRoot>>,
    globals: Query<'w, 's, &'static GlobalTransform>,
    config: Res<'w, ViewCubeConfig>,
}

#[allow(clippy::too_many_arguments)]
fn frame_hover_swap_target(
    target_entity: Entity,
    hovered: &HoveredElement,
    cube_elements: &Query<(Entity, &CubeElement)>,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
    root_links: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
    camera_globals: &Query<&GlobalTransform, With<ViewCubeTargetCamera>>,
    root_globals: &Query<&GlobalTransform, With<ViewCubeRoot>>,
    config: &ViewCubeConfig,
) -> Option<(FaceDirection, Entity)> {
    if !hovered.entities.contains(&target_entity) {
        return None;
    }

    let mut hovered_edges = Vec::new();
    for entity in hovered.entities.iter().copied() {
        let Ok((_, element)) = cube_elements.get(entity) else {
            return None;
        };
        if let CubeElement::Edge(edge) = *element {
            hovered_edges.push(edge);
        }
    }

    let (root, context) = edge_interaction_context(
        target_entity,
        parents_query,
        root_query,
        root_links,
        camera_globals,
        root_globals,
        config,
    )?;
    if !is_face_on(context) {
        return None;
    }

    let frame_edges = edges_for_face(context.front_face);
    if !frame_edges
        .into_iter()
        .all(|edge| hovered_edges.contains(&edge))
    {
        return None;
    }

    Some((context.front_face.opposite(), root))
}

pub fn on_cube_click(
    trigger: On<Pointer<Click>>,
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    lookup: OnCubeClickLookup,
    hovered: Res<HoveredElement>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    if trigger.event().button != PointerButton::Primary {
        return;
    }

    let entity = trigger.entity;

    let Some(target_entity) = find_cube_element_ancestor(entity, &cube_elements, &parents_query)
    else {
        return;
    };

    let click_root = find_root_ancestor(target_entity, &parents_query, &lookup.root_query);
    let selected_entity = if hovered.entities.len() == 1 {
        let hovered_entity = hovered.entities[0];
        let hovered_root = find_root_ancestor(hovered_entity, &parents_query, &lookup.root_query);
        if hovered_root.is_some()
            && hovered_root == click_root
            && matches!(
                cube_elements.get(hovered_entity),
                Ok((_, CubeElement::Corner(_)))
            )
        {
            hovered_entity
        } else {
            target_entity
        }
    } else {
        target_entity
    };

    let Ok((_, element)) = cube_elements.get(selected_entity) else {
        return;
    };

    let Some(source) = find_root_ancestor(selected_entity, &parents_query, &lookup.root_query)
    else {
        return;
    };

    match element {
        CubeElement::Face(dir) => {
            events.write(ViewCubeEvent::FaceClicked {
                direction: *dir,
                source,
            });
        }
        CubeElement::Edge(dir) => {
            if let Some((direction, source)) = frame_hover_swap_target(
                selected_entity,
                &hovered,
                &cube_elements,
                &parents_query,
                &lookup.root_query,
                &lookup.root_links,
                &lookup.camera_globals,
                &lookup.root_globals,
                &lookup.config,
            ) {
                events.write(ViewCubeEvent::FaceClicked { direction, source });
                return;
            }

            let Some((_root, context)) = edge_interaction_context(
                selected_entity,
                &parents_query,
                &lookup.root_query,
                &lookup.root_links,
                &lookup.camera_globals,
                &lookup.root_globals,
                &lookup.config,
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
            let local_direction = corner_local_direction(
                selected_entity,
                source,
                &lookup.globals,
                &lookup.root_globals,
                pos.to_look_direction(),
            );
            events.write(ViewCubeEvent::CornerClicked {
                position: *pos,
                local_direction,
                source,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn on_cube_drag(
    drag: On<Pointer<Drag>>,
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    root_links: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    mut cameras: Query<(&Transform, &mut EditorCam, &Camera), With<ViewCubeTargetCamera>>,
    dragging_roots: Query<(), With<ViewCubeDragging>>,
    mut commands: Commands,
) {
    if drag.button != PointerButton::Secondary {
        return;
    }

    let Some(target_entity) =
        find_cube_element_ancestor(drag.entity, &cube_elements, &parents_query)
    else {
        return;
    };
    let Some(root) = find_root_ancestor(target_entity, &parents_query, &root_query) else {
        return;
    };
    let Ok(link) = root_links.get(root) else {
        return;
    };
    let Ok((transform, mut editor_cam, camera)) = cameras.get_mut(link.main_camera) else {
        return;
    };

    if dragging_roots.get(root).is_err() {
        commands.entity(root).insert(ViewCubeDragging);
        editor_cam.end_move();
        let anchor = camera_anchor_from_transform(transform);
        editor_cam.start_orbit(anchor);
    }

    let viewport_size = camera
        .physical_viewport_size()
        .unwrap_or_else(|| UVec2::new(256, 256));
    let delta = drag.delta * viewport_size.as_vec2() / 75.0;
    editor_cam.send_screenspace_input(delta);
}

pub fn on_cube_drag_end(
    drag_end: On<Pointer<DragEnd>>,
    cube_elements: Query<(Entity, &CubeElement)>,
    parents_query: Query<&ChildOf>,
    root_query: Query<Entity, With<ViewCubeRoot>>,
    root_links: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    mut cameras: Query<&mut EditorCam, With<ViewCubeTargetCamera>>,
    mut commands: Commands,
) {
    if drag_end.button != PointerButton::Secondary {
        return;
    }

    let Some(target_entity) =
        find_cube_element_ancestor(drag_end.entity, &cube_elements, &parents_query)
    else {
        return;
    };
    let Some(root) = find_root_ancestor(target_entity, &parents_query, &root_query) else {
        return;
    };
    let Ok(link) = root_links.get(root) else {
        return;
    };

    if let Ok(mut editor_cam) = cameras.get_mut(link.main_camera) {
        editor_cam.end_move();
    }
    commands.entity(root).remove::<ViewCubeDragging>();
}

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

pub fn on_action_button_hover_start(
    trigger: On<Pointer<Over>>,
    action_buttons: Query<&ViewportActionButton>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.entity;
    if action_buttons.get(entity).is_err() {
        return;
    }

    let colors = ViewCubeColors::default();
    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = colors.arrow_hover;
    }
}

pub fn on_action_button_hover_end(
    trigger: On<Pointer<Out>>,
    action_buttons: Query<&ViewportActionButton>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.entity;
    if action_buttons.get(entity).is_err() {
        return;
    }

    let colors = ViewCubeColors::default();
    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = colors.arrow_normal;
    }
}

pub fn on_arrow_pressed(
    trigger: On<Pointer<Press>>,
    arrows: Query<&RotationArrow>,
    parents_query: Query<&ChildOf>,
    camera_link_query: Query<&ViewCubeLink, With<ViewCubeCamera>>,
    root_query: Query<(Entity, &ViewCubeLink), With<ViewCubeRoot>>,
    mut hold: ResMut<ActiveArrowHold>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    if trigger.event().button != PointerButton::Primary {
        return;
    }

    let entity = trigger.entity;
    let Ok(arrow) = arrows.get(entity) else {
        return;
    };

    let Some(source) =
        find_root_for_camera_child(entity, &parents_query, &camera_link_query, &root_query)
    else {
        return;
    };

    // First step happens immediately on press.
    events.write(ViewCubeEvent::ArrowClicked {
        arrow: *arrow,
        source,
    });

    hold.active = Some(ArrowHold {
        arrow: *arrow,
        source,
        timer: Timer::from_seconds(ARROW_HOLD_REPEAT_INTERVAL_SECS, TimerMode::Repeating),
    });
}

pub fn on_action_button_click(
    trigger: On<Pointer<Click>>,
    action_buttons: Query<&ViewportActionButton>,
    parents_query: Query<&ChildOf>,
    camera_link_query: Query<&ViewCubeLink, With<ViewCubeCamera>>,
    root_query: Query<(Entity, &ViewCubeLink), With<ViewCubeRoot>>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    if trigger.event().button != PointerButton::Primary {
        return;
    }

    let entity = trigger.entity;
    let Ok(action) = action_buttons.get(entity) else {
        return;
    };

    let Some(source) =
        find_root_for_camera_child(entity, &parents_query, &camera_link_query, &root_query)
    else {
        return;
    };

    events.write(ViewCubeEvent::ViewportActionClicked {
        action: *action,
        source,
    });
}

pub fn on_arrow_released(
    trigger: On<Pointer<Release>>,
    arrows: Query<&RotationArrow>,
    mut hold: ResMut<ActiveArrowHold>,
) {
    if trigger.event().button != PointerButton::Primary {
        return;
    }

    let entity = trigger.entity;
    let Ok(released_arrow) = arrows.get(entity) else {
        return;
    };

    let Some(active) = hold.active.as_ref() else {
        return;
    };

    if active.arrow == *released_arrow {
        hold.active = None;
    }
}

pub fn repeat_held_arrow(
    time: Res<Time>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut hold: ResMut<ActiveArrowHold>,
    mut events: MessageWriter<ViewCubeEvent>,
) {
    // If primary is no longer held (release happened off-target), stop repeating.
    if !mouse_buttons.pressed(MouseButton::Left) {
        hold.active = None;
        return;
    }

    let Some(active) = hold.active.as_mut() else {
        return;
    };

    active.timer.tick(time.delta());
    let repeats = active.timer.times_finished_this_tick();
    for _ in 0..repeats {
        events.write(ViewCubeEvent::ArrowClicked {
            arrow: active.arrow,
            source: active.source,
        });
    }
}

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

fn find_root_ancestor(
    entity: Entity,
    parents_query: &Query<&ChildOf>,
    root_query: &Query<Entity, With<ViewCubeRoot>>,
) -> Option<Entity> {
    find_ancestor(entity, parents_query, |current| {
        root_query.get(current).is_ok()
    })
}

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

fn corner_local_direction(
    corner_entity: Entity,
    root_entity: Entity,
    globals: &Query<&GlobalTransform>,
    root_globals: &Query<&GlobalTransform, With<ViewCubeRoot>>,
    fallback: Vec3,
) -> Vec3 {
    let Ok(corner_global) = globals.get(corner_entity) else {
        return fallback;
    };
    let Ok(root_global) = root_globals.get(root_entity) else {
        return fallback;
    };

    let corner_local = root_global
        .to_matrix()
        .inverse()
        .transform_point3(corner_global.translation());
    let dir = corner_local.normalize_or_zero();
    if dir.length_squared() > 1.0e-6 {
        dir
    } else {
        fallback
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

        let group = edge_group_for_target_face(FaceDirection::West, ctx);
        assert_eq!(group.len(), 2);
        assert!(group.contains(&EdgeDirection::YFrontLeft));
        assert!(group.contains(&EdgeDirection::ZTopLeft));
    }

    #[test]
    fn edge_between_two_visible_faces_is_not_clickable() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let edge = EdgeDirection::XTopFront;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, None);
    }

    #[test]
    fn edge_between_two_hidden_faces_is_not_clickable() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let edge = EdgeDirection::YBackLeft;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, None);
    }

    #[test]
    fn edge_is_inactive_when_adjacent_face_is_only_in_screen_plane() {
        let ctx = context(Vec3::new(0.0, 0.2, 1.0).normalize());
        let edge = EdgeDirection::YFrontLeft;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, None);
    }

    #[test]
    fn face_on_detection_accepts_small_tilt_but_rejects_oblique() {
        let axis_aligned = context(Vec3::new(0.03, 0.0, 1.0).normalize());
        assert!(is_face_on(axis_aligned));

        let near_face = context(Vec3::new(0.08, 0.0, 1.0).normalize());
        assert!(is_face_on(near_face));

        let oblique = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        assert!(!is_face_on(oblique));
    }

    #[test]
    fn hidden_face_edge_group_has_at_least_two_edges_in_oblique_view() {
        let ctx = context(Vec3::new(0.15, 0.2, 1.0).normalize());
        let edge = EdgeDirection::YFrontLeft;
        let target = resolve_edge_target_face(edge, ctx);
        assert_eq!(target, Some(FaceDirection::West));

        let group = edge_group_for_target_face(FaceDirection::West, ctx);
        assert!(group.len() >= 2);
    }

    #[test]
    fn corner_in_oblique_view_resolves_to_hidden_face() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let target = hidden_target_face_for_corner(CornerPosition::TopFrontLeft, ctx);
        assert_eq!(target, Some(FaceDirection::West));
    }

    #[test]
    fn hidden_face_corner_group_has_at_least_two_corners() {
        let ctx = context(Vec3::new(1.0, 1.0, 1.0).normalize());
        let edge_group = edge_group_for_target_face(FaceDirection::West, ctx);
        let corners: Vec<CornerPosition> = corner_group_for_edge_group(&edge_group)
            .into_iter()
            .filter(|corner| {
                corner_is_visible_for_target_face(*corner, FaceDirection::West, ctx.camera_dir_cube)
            })
            .collect();
        assert!(corners.len() >= 2);
    }
}
