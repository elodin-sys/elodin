use bevy::ecs::{hierarchy::ChildOf, relationship::Relationship};
use bevy::log::warn_once;
use bevy::prelude::Mesh;
use bevy::prelude::*;
use bevy::scene::{SceneInstance, SceneRoot, SceneSpawner};
use bevy_render::alpha::AlphaMode;
use big_space::GridCell;
use bitvec::prelude::*;
use eql::Expr;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, Object3D, Object3DIconSource};
use nox::Array;
use smallvec::smallvec;

use crate::icon_rasterizer::IconTextureCache;
use crate::iter::JoinDisplayExt;
use crate::{BevyExt, EqlContext, MainCamera, plugins::navigation_gizmo::NavGizmoCamera};
use std::collections::HashSet;
use std::fmt;

type ImportedCameraFilter = (Added<Camera>, Without<NavGizmoCamera>, Without<MainCamera>);

type ImportedCameraQuery<'w, 's> = Query<'w, 's, (Entity, &'static ChildOf), ImportedCameraFilter>;

/// ExprObject3D component that holds an EQL expression for dynamic positioning
#[derive(Component)]
pub struct Object3DState {
    pub compiled_expr: Option<CompiledExpr>,
    pub scale_expr: Option<CompiledExpr>,
    pub scale_error: Option<String>,
    pub joint_animations: Vec<(String, String)>, // (joint_name, eql_expr) - compiled in attach_joint_animations
    pub data: Object3D,
}

#[derive(Component)]
pub struct EllipsoidVisual {
    pub child: Entity,
    pub color: impeller2_wkt::Color,
    pub oversized: bool,
    pub max_extent: f32,
}

/// Component attached to joint entities for animation
/// The compiled_expr is filled in lazily on first access when the EQL context is available
#[derive(Component)]
pub struct JointAnimationComponent {
    // pub eql_expr: String,
    pub compiled_expr: CompiledExpr,
    pub original_transform: Transform,
}

#[derive(Default)]
pub struct EditableEQL {
    pub eql: String,
    pub compiled_expr: Option<CompiledExpr>,
}

impl EditableEQL {
    pub fn new(eql: String, compiled_expr: CompiledExpr) -> Self {
        Self {
            eql,
            compiled_expr: Some(compiled_expr),
        }
    }
}

#[derive(Component)]
pub struct Object3DIconState {
    pub billboard_entity: Entity,
    pub billboard_material: Handle<StandardMaterial>,
    pub swap_distance: f32,
    pub screen_size_px: f32,
    pub base_color: Color,
}

const BILLBOARD_FADE_BAND: f32 = 0.2;

#[derive(Component)]
pub struct BillboardIcon;

#[derive(Component)]
pub struct Object3DMeshChild;

type ExprFn = dyn for<'a, 'b> Fn(
        &'a EntityMap,
        &'a Query<'b, 'b, &'static ComponentValue>,
    ) -> Result<ComponentValue, String>
    + Send
    + Sync;

pub enum CompiledExpr {
    Closure(Box<ExprFn>),
    Value(ComponentValue),
}

impl CompiledExpr {
    pub fn closure<F>(closure: F) -> Self
    where
        F: for<'a, 'b> Fn(
                &'a EntityMap,
                &'a Query<'b, 'b, &'static ComponentValue>,
            ) -> Result<ComponentValue, String>
            + Send
            + Sync
            + 'static,
    {
        Self::Closure(Box::new(closure))
    }

    /// Executes the compiled expression
    pub fn execute<'a, 'b>(
        &'a self,
        entity_map: &'a EntityMap,
        values: &'a Query<'b, 'b, &'static ComponentValue>,
    ) -> Result<ComponentValue, String> {
        match self {
            Self::Closure(c) => (c)(entity_map, values),
            Self::Value(value) => Ok(value.clone()),
        }
    }
}

// ============================================================================
// Quaternion and transform helper functions
// ============================================================================

/// Reference frame for rotation/translation operations
#[derive(Clone, Copy)]
enum Frame {
    Body,
    World,
}

/// Hamilton product for quaternion multiplication: q1 * q2
/// Quaternions are stored as (x, y, z, w) where w is the scalar component
fn quat_multiply(q1: (f64, f64, f64, f64), q2: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let (x1, y1, z1, w1) = q1;
    let (x2, y2, z2, w2) = q2;
    (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, // x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, // y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, // z
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, // w
    )
}

/// Create rotation quaternion from axis index (0=X, 1=Y, 2=Z) and angle in radians
fn axis_angle_to_quat(axis: usize, angle_rad: f64) -> (f64, f64, f64, f64) {
    let half = angle_rad / 2.0;
    let (s, c) = (half.sin(), half.cos());
    match axis {
        0 => (s, 0.0, 0.0, c),
        1 => (0.0, s, 0.0, c),
        2 => (0.0, 0.0, s, c),
        _ => (0.0, 0.0, 0.0, 1.0),
    }
}

/// Rotate vector by quaternion: q * v * q^-1
fn rotate_vector_by_quat(q: (f64, f64, f64, f64), v: (f64, f64, f64)) -> (f64, f64, f64) {
    let (qx, qy, qz, qw) = q;
    let (vx, vy, vz) = v;
    // t = q * v (treating v as quaternion with w=0)
    let tw = -qx * vx - qy * vy - qz * vz;
    let tx = qw * vx + qy * vz - qz * vy;
    let ty = qw * vy + qz * vx - qx * vz;
    let tz = qw * vz + qx * vy - qy * vx;
    // result = t * q_conjugate
    (
        tw * (-qx) + tx * qw + ty * (-qz) - tz * (-qy),
        tw * (-qy) - tx * (-qz) + ty * qw + tz * (-qx),
        tw * (-qz) + tx * (-qy) - ty * (-qx) + tz * qw,
    )
}

/// Extract spatial transform data (qx, qy, qz, qw, px, py, pz), validating it has 7 elements
fn extract_spatial(val: ComponentValue) -> Result<[f64; 7], String> {
    use nox::ArrayBuf;
    let ComponentValue::F64(array) = val else {
        return Err("requires a spatial transform".to_string());
    };
    let data = array.buf.as_buf();
    if data.len() < 7 {
        return Err(format!("requires 7-element array, got {}", data.len()));
    }
    Ok([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6],
    ])
}

/// Extract a scalar f64 from component value
fn extract_scalar(val: ComponentValue) -> Result<f64, String> {
    use nox::ArrayBuf;
    let ComponentValue::F64(arr) = val else {
        return Err("must be a number".to_string());
    };
    let d = arr.buf.as_buf();
    if d.is_empty() {
        return Err("cannot be empty".to_string());
    }
    Ok(d[0])
}

/// Build result array from quaternion and position
fn build_spatial_result(q: (f64, f64, f64, f64), pos: (f64, f64, f64)) -> ComponentValue {
    let result = vec![q.0, q.1, q.2, q.3, pos.0, pos.1, pos.2];
    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
    ComponentValue::F64(result_array)
}

/// Build result array from a 3-vector (e.g. direction in world frame)
fn build_vec3_result(v: (f64, f64, f64)) -> ComponentValue {
    let result = vec![v.0, v.1, v.2];
    let result_array = Array::from_shape_vec(smallvec![3], result).unwrap();
    ComponentValue::F64(result_array)
}

/// Compiles a formula expression into a runtime closure
fn compile_formula(formula_name: &str, inner_expr: eql::Expr) -> CompiledExpr {
    match formula_name {
        // Single-axis rotation formulas (body and world frame)
        "rotate_x" | "rotate_y" | "rotate_z" | "rotate_world_x" | "rotate_world_y"
        | "rotate_world_z" => {
            let (axis, frame) = match formula_name {
                "rotate_x" => (0, Frame::Body),
                "rotate_y" => (1, Frame::Body),
                "rotate_z" => (2, Frame::Body),
                "rotate_world_x" => (0, Frame::World),
                "rotate_world_y" => (1, Frame::World),
                "rotate_world_z" => (2, Frame::World),
                _ => unreachable!(),
            };

            let eql::Expr::Tuple(elements) = inner_expr else {
                let error = format!("{} requires tuple expression", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            };
            if elements.len() != 2 {
                let error = format!("{} requires receiver and angle", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            }

            let receiver_compiled = compile_eql_expr(elements[0].clone());
            let angle_compiled = compile_eql_expr(elements[1].clone());

            CompiledExpr::closure(move |entity_map, component_values| {
                let spatial = receiver_compiled.execute(entity_map, component_values)?;
                let data = extract_spatial(spatial)?;
                let angle_val = angle_compiled.execute(entity_map, component_values)?;
                let angle_deg = extract_scalar(angle_val)?;

                let q_input = (data[0], data[1], data[2], data[3]);
                let q_rot = axis_angle_to_quat(axis, angle_deg.to_radians());

                // Body-frame: q_input * q_rot, World-frame: q_rot * q_input
                let q_result = match frame {
                    Frame::Body => quat_multiply(q_input, q_rot),
                    Frame::World => quat_multiply(q_rot, q_input),
                };

                Ok(build_spatial_result(q_result, (data[4], data[5], data[6])))
            })
        }

        // Multi-axis rotation (body and world frame)
        "rotate" | "rotate_world" => {
            let frame = if formula_name == "rotate" {
                Frame::Body
            } else {
                Frame::World
            };

            let eql::Expr::Tuple(elements) = inner_expr else {
                let error = format!("{} requires tuple expression", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            };
            if elements.len() != 4 {
                let error = format!(
                    "{} requires receiver and three angles (x, y, z)",
                    formula_name
                );
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            }

            let receiver_compiled = compile_eql_expr(elements[0].clone());
            let x_angle_compiled = compile_eql_expr(elements[1].clone());
            let y_angle_compiled = compile_eql_expr(elements[2].clone());
            let z_angle_compiled = compile_eql_expr(elements[3].clone());

            CompiledExpr::closure(move |entity_map, component_values| {
                let spatial = receiver_compiled.execute(entity_map, component_values)?;
                let data = extract_spatial(spatial)?;

                let x_deg =
                    extract_scalar(x_angle_compiled.execute(entity_map, component_values)?)?;
                let y_deg =
                    extract_scalar(y_angle_compiled.execute(entity_map, component_values)?)?;
                let z_deg =
                    extract_scalar(z_angle_compiled.execute(entity_map, component_values)?)?;

                let mut q = (data[0], data[1], data[2], data[3]);

                // Apply rotations in order: X, then Y, then Z
                for (axis, deg) in [(0, x_deg), (1, y_deg), (2, z_deg)] {
                    if deg.abs() > 1e-10 {
                        let q_rot = axis_angle_to_quat(axis, deg.to_radians());
                        q = match frame {
                            Frame::Body => quat_multiply(q, q_rot),
                            Frame::World => quat_multiply(q_rot, q),
                        };
                    }
                }

                Ok(build_spatial_result(q, (data[4], data[5], data[6])))
            })
        }

        // Single-axis translation (body and world frame)
        "translate_x" | "translate_y" | "translate_z" | "translate_world_x"
        | "translate_world_y" | "translate_world_z" => {
            let (axis, frame) = match formula_name {
                "translate_x" => (0, Frame::Body),
                "translate_y" => (1, Frame::Body),
                "translate_z" => (2, Frame::Body),
                "translate_world_x" => (0, Frame::World),
                "translate_world_y" => (1, Frame::World),
                "translate_world_z" => (2, Frame::World),
                _ => unreachable!(),
            };

            let eql::Expr::Tuple(elements) = inner_expr else {
                let error = format!("{} requires tuple expression", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            };
            if elements.len() != 2 {
                let error = format!("{} requires receiver and distance", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            }

            let receiver_compiled = compile_eql_expr(elements[0].clone());
            let distance_compiled = compile_eql_expr(elements[1].clone());

            CompiledExpr::closure(move |entity_map, component_values| {
                let spatial = receiver_compiled.execute(entity_map, component_values)?;
                let data = extract_spatial(spatial)?;
                let dist_val = distance_compiled.execute(entity_map, component_values)?;
                let dist = extract_scalar(dist_val)?;

                let q = (data[0], data[1], data[2], data[3]);
                let offset_body = match axis {
                    0 => (dist, 0.0, 0.0),
                    1 => (0.0, dist, 0.0),
                    2 => (0.0, 0.0, dist),
                    _ => unreachable!(),
                };

                // Body-frame: rotate offset to world frame; World-frame: use directly
                let (dx, dy, dz) = match frame {
                    Frame::Body => rotate_vector_by_quat(q, offset_body),
                    Frame::World => offset_body,
                };

                Ok(build_spatial_result(
                    q,
                    (data[4] + dx, data[5] + dy, data[6] + dz),
                ))
            })
        }

        // Multi-axis translation (body and world frame)
        "translate" | "translate_world" => {
            let frame = if formula_name == "translate" {
                Frame::Body
            } else {
                Frame::World
            };

            let eql::Expr::Tuple(elements) = inner_expr else {
                let error = format!("{} requires tuple expression", formula_name);
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            };
            if elements.len() != 4 {
                let error = format!(
                    "{} requires receiver and three distances (x, y, z)",
                    formula_name
                );
                return CompiledExpr::closure(move |_, _| Err(error.clone()));
            }

            let receiver_compiled = compile_eql_expr(elements[0].clone());
            let x_dist_compiled = compile_eql_expr(elements[1].clone());
            let y_dist_compiled = compile_eql_expr(elements[2].clone());
            let z_dist_compiled = compile_eql_expr(elements[3].clone());

            CompiledExpr::closure(move |entity_map, component_values| {
                let spatial = receiver_compiled.execute(entity_map, component_values)?;
                let data = extract_spatial(spatial)?;

                let dx = extract_scalar(x_dist_compiled.execute(entity_map, component_values)?)?;
                let dy = extract_scalar(y_dist_compiled.execute(entity_map, component_values)?)?;
                let dz = extract_scalar(z_dist_compiled.execute(entity_map, component_values)?)?;

                let q = (data[0], data[1], data[2], data[3]);

                // Body-frame: rotate offset to world frame; World-frame: use directly
                let (rx, ry, rz) = match frame {
                    Frame::Body => rotate_vector_by_quat(q, (dx, dy, dz)),
                    Frame::World => (dx, dy, dz),
                };

                Ok(build_spatial_result(
                    q,
                    (data[4] + rx, data[5] + ry, data[6] + rz),
                ))
            })
        }

        // direction(x, y, z): body-frame direction transformed to world frame (returns 3-vector)
        "direction" => {
            let eql::Expr::Tuple(elements) = inner_expr else {
                return CompiledExpr::closure(move |_, _| {
                    Err("direction requires tuple (receiver, x, y, z)".to_string())
                });
            };
            if elements.len() != 4 {
                return CompiledExpr::closure(move |_, _| {
                    Err("direction requires receiver and three components (x, y, z)".to_string())
                });
            }

            let receiver_compiled = compile_eql_expr(elements[0].clone());
            let x_compiled = compile_eql_expr(elements[1].clone());
            let y_compiled = compile_eql_expr(elements[2].clone());
            let z_compiled = compile_eql_expr(elements[3].clone());

            CompiledExpr::closure(move |entity_map, component_values| {
                let spatial = receiver_compiled.execute(entity_map, component_values)?;
                let data = extract_spatial(spatial)?;
                let dx = extract_scalar(x_compiled.execute(entity_map, component_values)?)?;
                let dy = extract_scalar(y_compiled.execute(entity_map, component_values)?)?;
                let dz = extract_scalar(z_compiled.execute(entity_map, component_values)?)?;
                let q = (data[0], data[1], data[2], data[3]);
                let world = rotate_vector_by_quat(q, (dx, dy, dz));
                Ok(build_vec3_result(world))
            })
        }

        _ => {
            let error = format!(
                "formula '{}' is not supported in editor runtime",
                formula_name
            );
            CompiledExpr::closure(move |_, _| Err(error.clone()))
        }
    }
}

/// Compiles an EQL expression into a closure-based form
pub fn compile_eql_expr(expression: eql::Expr) -> CompiledExpr {
    match expression {
        Expr::ComponentPart(component) => {
            let component_id = component.id;
            CompiledExpr::closure(move |entity_map, component_value| {
                let entity_id = entity_map.get(&component_id).ok_or_else(|| {
                    format!("component '{}' not found in entity map", component.name)
                })?;

                component_value
                    .get(*entity_id)
                    .map_err(|_| {
                        format!(
                            "no component value map found for component '{}'",
                            component.name
                        )
                    })
                    .cloned()
            })
        }
        Expr::ArrayAccess(expr, index) => {
            let compiled_expr = compile_eql_expr(*expr);
            CompiledExpr::closure(move |entity_map, component_value_maps| {
                let resolved_expr = compiled_expr.execute(entity_map, component_value_maps)?;
                match resolved_expr {
                    ComponentValue::F64(array) => {
                        use nox::ArrayBuf;
                        let data = array.buf.as_buf();
                        if index < data.len() {
                            let value = data[index];
                            let value = nox::Array::<_, ()> { buf: value }.to_dyn();
                            Ok(ComponentValue::F64(value))
                        } else {
                            Err(format!(
                                "array index {} out of bounds (length: {})",
                                index,
                                data.len()
                            ))
                        }
                    }
                    ComponentValue::F32(array) => {
                        use nox::ArrayBuf;
                        let data = array.buf.as_buf();
                        if index < data.len() {
                            let value = data[index];
                            let value = nox::Array::<_, ()> { buf: value }.to_dyn();
                            Ok(ComponentValue::F32(value))
                        } else {
                            Err(format!(
                                "array index {} out of bounds (length: {})",
                                index,
                                data.len()
                            ))
                        }
                    }
                    _ => Err("array access can only be applied to numeric arrays".to_string()),
                }
            })
        }
        Expr::Tuple(exprs) => {
            let compiled_exprs: Vec<CompiledExpr> =
                exprs.into_iter().map(compile_eql_expr).collect();
            CompiledExpr::closure(move |entity_map, component_value_maps| {
                use nox::ArrayBuf;
                let mut values = Vec::new();
                for compiled_expr in &compiled_exprs {
                    let resolved = compiled_expr.execute(entity_map, component_value_maps)?;
                    match resolved {
                        ComponentValue::F64(array) => {
                            values.extend_from_slice(array.buf.as_buf());
                        }
                        ComponentValue::F32(array) => {
                            let f32_data = array.buf.as_buf();
                            values.extend(f32_data.iter().map(|&x| x as f64));
                        }
                        _ => return Err("tuple elements must be numeric".to_string()),
                    }
                }
                let tuple_array = Array::from_shape_vec(smallvec![values.len()], values).unwrap();
                Ok(ComponentValue::F64(tuple_array))
            })
        }
        Expr::BinaryOp(left, right, op) => {
            let left_compiled = compile_eql_expr(*left);
            let right_compiled = compile_eql_expr(*right);
            CompiledExpr::closure(move |entity_map, component_value_maps| {
                let left_val = left_compiled.execute(entity_map, component_value_maps)?;
                let right_val = right_compiled.execute(entity_map, component_value_maps)?;

                match (left_val, right_val) {
                    (ComponentValue::F64(left), ComponentValue::F64(right)) => {
                        if !nox::array::can_broadcast(left.shape(), right.shape()) {
                            return Err(
                                "binary operation requires arrays be broadcastable".to_string()
                            );
                        }
                        let result = match op {
                            eql::BinaryOp::Add => left.add(&right),
                            eql::BinaryOp::Sub => left.sub(&right),
                            eql::BinaryOp::Mul => left.mul(&right),
                            eql::BinaryOp::Div => left.div(&right),
                        };

                        Ok(ComponentValue::F64(result))
                    }
                    _ => Err("binary operations only supported for F64 arrays".to_string()),
                }
            })
        }
        Expr::FloatLiteral(f) => CompiledExpr::Value(ComponentValue::F64(nox::array!(f).to_dyn())),
        Expr::Formula(formula, inner_expr) => compile_formula(formula.name(), *inner_expr),
        expr => {
            let error = format!("{:?} can't be converted to a component value", expr);
            CompiledExpr::closure(move |_, _| Err(error.clone()))
        }
    }
}

pub fn compile_scale_eql(scale: &str, ctx: &eql::Context) -> Result<CompiledExpr, String> {
    let trimmed = scale.trim();
    if trimmed.is_empty() {
        return Err("scale expression cannot be empty".to_string());
    }

    ctx.parse_str(trimmed)
        .map(compile_eql_expr)
        .map_err(|err| err.to_string())
}

#[derive(Debug)]
pub enum ScaleEvalError {
    Expr(String),
    NotEnoughComponents { len: usize },
    InvalidComponentType,
}

impl fmt::Display for ScaleEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(err) => write!(f, "{}", err),
            Self::NotEnoughComponents { len } => write!(
                f,
                "scale expression must yield at least 3 values, got {}",
                len
            ),
            Self::InvalidComponentType => {
                f.write_str("scale expression must yield an F32 or F64 array")
            }
        }
    }
}

impl From<String> for ScaleEvalError {
    fn from(value: String) -> Self {
        Self::Expr(value)
    }
}

impl std::error::Error for ScaleEvalError {}

const ELLIPSOID_OVERSIZED_THRESHOLD: f32 = 10_000.0;

fn find_entities<'a, T>(
    entity: Entity,
    children: &'a Query<&Children>,
    mut predicate: impl FnMut(Entity) -> Option<T> + 'a,
) -> impl Iterator<Item = (Entity, T)> + 'a {
    children
        .iter_descendants(entity)
        .filter_map(move |id| predicate(id).map(|x| (id, x)))
}

/// Conditional system that checks if scenes are ready. Returns true when at
/// least one scene is ready, false otherwise.
pub fn on_scene_ready(
    mut scene_queue: Local<HashSet<Entity>>,
    added_scenes: Query<Entity, Added<SceneRoot>>,
    scene_instances: Query<&SceneInstance>,
    scene_roots: Query<&SceneRoot>,
    names: Query<&Name>,
    scene_spawner: Res<SceneSpawner>,
) -> Option<Entity> {
    // Add newly added scenes to the queue.
    for entity in added_scenes.iter() {
        scene_queue.insert(entity);
    }

    // Collect which queued scenes became ready this frame.
    let mut ready_entity = None;
    scene_queue.retain(|&entity| {
        if ready_entity.is_some() {
            // Keep in queue to evaluate later. We do this one frame at a
            // time rather than allocating a Vec.
            true
        } else if let Ok(instance) = scene_instances.get(entity) {
            if scene_spawner.instance_is_ready(**instance) {
                ready_entity = Some(entity);
                false // Remove from queue since it's ready.
            } else {
                true // Keep in queue since it's not ready yet.
            }
        } else {
            // SceneInstance not found yet; keep in queue.
            true
        }
    });

    if let Some(entity) = ready_entity.as_ref() {
        let name = names
            .get(*entity)
            .map(|n| n.as_str())
            .unwrap_or("<no name>");
        let scene_info = scene_roots
            .get(*entity)
            .map(|r| format!("handle id={:?}", r.0.id()))
            .unwrap_or_else(|_| "<no SceneRoot>".to_string());
        info!(
            entity = ?entity,
            name = %name,
            scene = %scene_info,
            "A scene is ready."
        );
    }
    ready_entity
}

/// System that updates 3D object entities based on their EQL expressions
pub fn update_object_3d_system(
    mut objects_query: Query<(
        Entity,
        &mut Object3DState,
        &mut impeller2_wkt::WorldPos,
        Option<&mut EllipsoidVisual>,
    )>,
    mut transforms: Query<&mut Transform>,
    entity_map: Res<EntityMap>,
    component_value_maps: Query<&'static ComponentValue>,
) {
    // return;
    for (_entity, mut object_3d, mut pos, ellipse) in objects_query.iter_mut() {
        if let Some(compiled_expr) = &object_3d.compiled_expr
            && let Ok(component_value) = compiled_expr.execute(&entity_map, &component_value_maps)
            && let Some(world_pos) = component_value.as_world_pos()
        {
            *pos = world_pos;
        }

        let Some(mut ellipse) = ellipse else {
            continue;
        };

        if !matches!(
            object_3d.data.mesh,
            impeller2_wkt::Object3DMesh::Ellipsoid { .. }
        ) {
            continue;
        }

        match evaluate_scale(&object_3d, &entity_map, &component_value_maps) {
            Ok(scale) => {
                let scale = scale.max(Vec3::splat(f32::EPSILON));
                if let Ok(mut child_transform) = transforms.get_mut(ellipse.child) {
                    child_transform.scale = scale;
                    child_transform.translation = Vec3::ZERO;
                }
                ellipse.max_extent = scale.max_element();
                ellipse.oversized = ellipse.max_extent > ELLIPSOID_OVERSIZED_THRESHOLD;
                if object_3d.scale_expr.is_some() {
                    object_3d.scale_error = None;
                }
            }
            Err(err) => {
                object_3d.scale_error = Some(err.to_string());
                ellipse.oversized = false;
                ellipse.max_extent = 0.0;
            }
        }
    }
}

/// Converts a ComponentValue to an angle-axis rotation (axis Vec3, angle f32 in radians).
///
/// Input: 3-element vector where direction is axis and magnitude is angle in DEGREES
/// Output: (normalized axis, angle in radians)
fn component_value_to_axis_angle(value: &ComponentValue) -> Result<(Vec3, f32), String> {
    use nox::ArrayBuf;
    match value {
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            if data.len() == 3 {
                // 3-element format: [x, y, z] where direction is axis and magnitude is angle in degrees
                let vec = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32);
                let angle_deg = vec.length();
                let normalized_axis = if angle_deg > f32::EPSILON {
                    vec / angle_deg
                } else {
                    Vec3::Y // Default axis if vector is zero
                };
                // Convert degrees to radians
                let angle_rad = angle_deg.to_radians();
                Ok((normalized_axis, angle_rad))
            } else {
                Err(format!(
                    "Expected 3 elements for rotation_vector (axis direction with magnitude as angle in degrees), got {}",
                    data.len()
                ))
            }
        }
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            if data.len() == 3 {
                // 3-element format: [x, y, z] where direction is axis and magnitude is angle in degrees
                let vec = Vec3::new(data[0], data[1], data[2]);
                let angle_deg = vec.length();
                let normalized_axis = if angle_deg > f32::EPSILON {
                    vec / angle_deg
                } else {
                    Vec3::Y // Default axis if vector is zero
                };
                // Convert degrees to radians
                let angle_rad = angle_deg.to_radians();
                Ok((normalized_axis, angle_rad))
            } else {
                Err(format!(
                    "Expected 3 elements for rotation_vector (axis direction with magnitude as angle in degrees), got {}",
                    data.len()
                ))
            }
        }
        _ => Err("Invalid component type for rotation_vector rotation".to_string()),
    }
}

/// System that attaches JointAnimationComponent to joint entities when scenes load
/// This runs when Object3DState changes (e.g., when a scene finishes loading)
#[allow(clippy::too_many_arguments)]
pub fn attach_joint_animations(
    In(scene_entity): In<Option<Entity>>,
    objects_query: Query<&Object3DState>,
    children: Query<&Children>,
    parent: Query<&ChildOf>,
    names: Query<&Name>,
    transforms: Query<&Transform>,
    existing_components: Query<Entity, With<JointAnimationComponent>>,
    mut commands: Commands,
    ctx: Res<EqlContext>,
) {
    let Some(scene_entity) = scene_entity else {
        return;
    };
    debug!("Run attach joint animations for scene {scene_entity}.");
    let Ok(object_3d_entity) = parent.get(scene_entity).map(|p| p.get()) else {
        // This can be ok. So I'm leaving debug instead of warn because the axes
        // cube triggers it.
        debug!("Could not get parent for scene {scene_entity}.");
        return;
    };

    if let Ok(object_3d) = objects_query.get(object_3d_entity) {
        // Only process GLB meshes with animations.
        if !matches!(object_3d.data.mesh, impeller2_wkt::Object3DMesh::Glb { .. }) {
            debug!("Not a mesh for object 3d {object_3d_entity}.");
            return;
        }

        if object_3d.joint_animations.is_empty() {
            debug!("No joint animations for object 3d {object_3d_entity}.");
            return;
        }
        const MAX_ANIMATIONS: usize = 32 * 4;
        let mut found_animations_store = bitarr![u32, Lsb0; 0; MAX_ANIMATIONS];
        let found_animations = found_animations_store.as_mut_bitslice();

        if object_3d.joint_animations.len() > MAX_ANIMATIONS {
            warn!(
                "The object_3d with path '{:?}' has {} animations; cannot account for missing animations past {}.",
                object_3d.data.mesh,
                object_3d.joint_animations.len(),
                MAX_ANIMATIONS
            );
        }

        // Find entities that match joint names and compile their expressions.
        let entity_compiled_expr = find_entities(object_3d_entity, &children, |id| {
            names.get(id).ok().and_then(|name| {
                object_3d.joint_animations.iter().enumerate().find_map(
                    |(i, (joint_name, eql_expr))| {
                        if name.as_str() == joint_name {
                            found_animations.set(i, true);
                            // Compile the EQL expression here.
                            ctx.0
                                .parse_str(eql_expr)
                                .map(compile_eql_expr)
                                .map_err(|err| {
                                    warn!(
                                        joint = %joint_name,
                                        eql = %eql_expr,
                                        error = %err,
                                        "Failed to compile EQL expression for joint animation."
                                    );
                                    err
                                })
                                .ok()
                        } else {
                            None
                        }
                    },
                )
            })
        });

        let mut entity_count = 0;
        for (joint_entity, compiled_expr) in entity_compiled_expr {
            entity_count += 1;
            if existing_components.contains(joint_entity) {
                continue;
            }

            // Get the original transform to preserve the bone's initial rotation.
            let original_transform = transforms
                .get(joint_entity)
                .cloned()
                .unwrap_or(Transform::IDENTITY);

            // Attach the component with the compiled expression and original transform.
            commands
                .entity(joint_entity)
                .insert(JointAnimationComponent {
                    compiled_expr,
                    original_transform,
                });
        }

        info!(
            "For {} joint animations, found {} matching entities.",
            object_3d.joint_animations.len(),
            entity_count
        );

        let found_animations = found_animations_store.as_bitslice();

        if found_animations[..object_3d.joint_animations.len()].not_all() {
            let items = found_animations[..object_3d.joint_animations.len()]
                .iter_zeros()
                .map(|i| &object_3d.joint_animations[i].0);
            warn!(
                "The object_3d {} did not have any animation joints named: {}",
                &object_3d.data.mesh,
                items.join_display(", ")
            );
        }
    } else {
        warn!(
            "Could not get `Object3dState` for entity {object_3d_entity} for scene {scene_entity}."
        );
    }
}

/// System that updates joint animations based on EQL expressions. This queries
/// for entities with both Transform and JointAnimationComponent.
pub fn update_joint_animations(
    mut joint_query: Query<(&mut Transform, &JointAnimationComponent)>,
    entity_map: Res<EntityMap>,
    component_value_maps: Query<&'static ComponentValue>,
) {
    for (mut joint_transform, joint_anim) in joint_query.iter_mut() {
        // Evaluate the EQL expression
        let component_value = match joint_anim
            .compiled_expr
            .execute(&entity_map, &component_value_maps)
        {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    error = %err,
                    "Failed to evaluate EQL expression for joint animation"
                );
                continue;
            }
        };

        // Convert to axis-angle rotation
        let (axis, angle) = match component_value_to_axis_angle(&component_value) {
            Ok(result) => result,
            Err(err) => {
                warn!(
                    error = %err,
                    "Failed to convert EQL result to axis-angle rotation for joint animation"
                );
                continue;
            }
        };

        // Create the animation rotation
        let animation_rotation = Quat::from_axis_angle(axis, angle);

        // Multiply by the original transform's rotation to preserve the bone's initial orientation
        joint_transform.rotation = joint_anim.original_transform.rotation * animation_rotation;

        // Preserve the original translation and scale
        joint_transform.translation = joint_anim.original_transform.translation;
        joint_transform.scale = joint_anim.original_transform.scale;
    }
}

/// Emits a warning when a GLB spawns a camera so the user can decide how to handle it.
pub fn warn_imported_cameras(
    object_states: Query<&Object3DState>,
    parents: Query<&ChildOf>,
    imported_cameras: ImportedCameraQuery,
) {
    for (camera_entity, parent) in imported_cameras.iter() {
        let mut current = parent.0;
        let mut owning_object = None;

        // Walk up the hierarchy to see if this camera belongs to an object_3d root.
        loop {
            if object_states.get(current).is_ok() {
                owning_object = Some(current);
                break;
            }

            let Ok(next_parent) = parents.get(current) else {
                break;
            };
            current = next_parent.0;
        }

        let Some(object_root) = owning_object else {
            continue;
        };

        if let Ok(state) = object_states.get(object_root) {
            let source = match &state.data.mesh {
                impeller2_wkt::Object3DMesh::Glb { path, .. } => format!("GLB '{path}'"),
                _ => "object_3d".to_string(),
            };
            warn_once!(
                "Imported {source} contains camera entity {camera_entity:?}; \
                 embedded cameras stay active. Remove the camera from the asset if this is unintended."
            );
        } else {
            warn_once!(
                "Imported object_3d contains camera entity {camera_entity:?}; \
                 embedded cameras stay active. Remove the camera from the asset if this is unintended."
            );
        }
    }
}

fn evaluate_scale(
    state: &Object3DState,
    entity_map: &EntityMap,
    component_value_maps: &Query<&'static ComponentValue>,
) -> Result<Vec3, ScaleEvalError> {
    if let Some(expr) = &state.scale_expr {
        let value = expr
            .execute(entity_map, component_value_maps)
            .map_err(ScaleEvalError::from)?;
        component_value_to_vec3(&value)
    } else {
        Ok(Vec3::ONE)
    }
}

fn component_value_to_vec3(value: &ComponentValue) -> Result<Vec3, ScaleEvalError> {
    use nox::ArrayBuf;
    match value {
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            if data.len() >= 3 {
                Ok(Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32).abs())
            } else {
                Err(ScaleEvalError::NotEnoughComponents { len: data.len() })
            }
        }
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            if data.len() >= 3 {
                Ok(Vec3::new(data[0], data[1], data[2]).abs())
            } else {
                Err(ScaleEvalError::NotEnoughComponents { len: data.len() })
            }
        }
        _ => Err(ScaleEvalError::InvalidComponentType),
    }
}

pub trait ComponentArrayExt {
    fn as_world_pos(&self) -> Option<impeller2_wkt::WorldPos>;
}

impl ComponentArrayExt for ComponentValue {
    fn as_world_pos(&self) -> Option<impeller2_wkt::WorldPos> {
        if let ComponentValue::F64(array) = self {
            use nox::ArrayBuf;
            let data = array.buf.as_buf();
            if data.len() >= 7 {
                return Some(impeller2_wkt::WorldPos {
                    att: nox::Quaternion::new(data[3], data[0], data[1], data[2]),
                    pos: nox::Vector3::new(data[4], data[5], data[6]),
                });
            }
        }
        None
    }
}

pub fn create_object_3d_entity(
    commands: &mut Commands,
    data: impeller2_wkt::Object3D,
    expr: eql::Expr,
    ctx: &eql::Context,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    assets: &Res<AssetServer>,
) -> Entity {
    let (scale_expr, scale_error) = match &data.mesh {
        impeller2_wkt::Object3DMesh::Ellipsoid { scale, .. } => {
            match compile_scale_eql(scale, ctx) {
                Ok(expr) => (Some(expr), None),
                Err(err) => (None, Some(err)),
            }
        }
        _ => (None, None),
    };

    let joint_animations = match &data.mesh {
        impeller2_wkt::Object3DMesh::Glb {
            animations, path, ..
        } => {
            info!(
                count = animations.len(),
                path = %path,
                "Found {} animation(s) in GLB mesh '{}'",
                animations.len(),
                path
            );
            if animations.is_empty() {
                debug!("GLB mesh '{}' has no animations.", path);
            }
            animations
                .iter()
                .map(|anim| (anim.joint_name.clone(), anim.eql_expr.clone()))
                .collect()
        }
        _ => Vec::new(),
    };

    let entity_id = commands
        .spawn((
            Object3DState {
                compiled_expr: Some(compile_eql_expr(expr)),
                scale_expr,
                scale_error,
                joint_animations,
                data: data.clone(),
            },
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
            GridCell::<i128>::default(),
            impeller2_wkt::WorldPos::default(),
            Name::new("object_3d"),
        ))
        .id();

    if let Some(ellipse) = spawn_mesh(
        commands,
        entity_id,
        &data.mesh,
        material_assets,
        mesh_assets,
        assets,
    ) {
        commands.entity(entity_id).insert(ellipse);
    }

    entity_id
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_billboard_icon(
    commands: &mut Commands,
    parent: Entity,
    icon: &impeller2_wkt::Object3DIcon,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    image_assets: &mut ResMut<Assets<Image>>,
    asset_server: &Res<AssetServer>,
    icon_cache: &mut ResMut<IconTextureCache>,
) {
    let texture_handle: Handle<Image> = match &icon.source {
        Object3DIconSource::Path(path) => asset_server.load(path.clone()),
        Object3DIconSource::Builtin(name) => {
            let raster_size = (icon.size * 2.0).max(64.0) as u32;
            icon_cache.get_or_insert(name, raster_size, image_assets)
        }
    };

    let icon_color = Color::srgba(icon.color.r, icon.color.g, icon.color.b, 0.0);

    let material = material_assets.add(StandardMaterial {
        base_color: icon_color,
        base_color_texture: Some(texture_handle),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..Default::default()
    });

    let quad = mesh_assets.add(Mesh::from(bevy::math::primitives::Rectangle::new(1.0, 1.0)));

    let material_handle = material.clone();

    let billboard_entity = commands
        .spawn((
            Mesh3d(quad),
            MeshMaterial3d(material),
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            Visibility::Inherited,
            InheritedVisibility::default(),
            ViewVisibility::default(),
            BillboardIcon,
            ChildOf(parent),
            Name::new("billboard_icon"),
        ))
        .id();

    let base_color = Color::srgba(icon.color.r, icon.color.g, icon.color.b, icon.color.a);

    commands.entity(parent).insert(Object3DIconState {
        billboard_entity,
        billboard_material: material_handle,
        swap_distance: icon.swap_distance,
        screen_size_px: icon.size,
        base_color,
    });
}

pub fn spawn_mesh(
    commands: &mut Commands,
    entity: Entity,
    mesh: &impeller2_wkt::Object3DMesh,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    assets: &Res<AssetServer>,
) -> Option<EllipsoidVisual> {
    match mesh {
        impeller2_wkt::Object3DMesh::Glb {
            path,
            scale,
            translate,
            rotate,
            animations: _,
        } => {
            let url = format!("{path}#Scene0");
            let scene = assets.load(&url);

            // Create transform for offset (translate, rotate, scale)
            let translation = Vec3::new(translate.0, translate.1, translate.2);
            let rotation = Quat::from_euler(
                EulerRot::XYZ,
                rotate.0.to_radians(),
                rotate.1.to_radians(),
                rotate.2.to_radians(),
            );
            let offset_transform = Transform {
                translation,
                rotation,
                scale: Vec3::splat(*scale),
            };

            commands.spawn((
                SceneRoot(scene),
                offset_transform,
                GlobalTransform::default(),
                Visibility::default(),
                InheritedVisibility::default(),
                ViewVisibility::default(),
                Object3DMeshChild,
                ChildOf(entity),
                Name::new(format!("object_3d_scene {}", path)),
            ));

            commands
                .entity(entity)
                .insert(Name::new(format!("object_3d {}", path)));
            None
        }
        impeller2_wkt::Object3DMesh::Mesh { mesh, material } => {
            let mut material = material.clone().into_bevy();
            if matches!(mesh, impeller2_wkt::Mesh::Plane { .. }) {
                material.double_sided = true;
                material.cull_mode = None;
            }
            let material = material_assets.add(material);
            let mesh = mesh.clone().into_bevy();
            let mesh = mesh_assets.add(mesh);
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                Visibility::default(),
                InheritedVisibility::default(),
                ViewVisibility::default(),
                Object3DMeshChild,
                ChildOf(entity),
                Name::new("object_3d_mesh"),
            ));
            None
        }
        impeller2_wkt::Object3DMesh::Ellipsoid { color, .. } => {
            let bevy_color = Color::srgba(color.r, color.g, color.b, color.a);
            let alpha_mode = if color.a < 1.0 {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            };

            let material_handle = material_assets.add(StandardMaterial {
                base_color: bevy_color,
                alpha_mode,
                unlit: false,
                double_sided: true,
                cull_mode: None,
                perceptual_roughness: 0.6,
                ..Default::default()
            });

            let mesh_handle =
                mesh_assets.add(Mesh::from(bevy::math::primitives::Sphere { radius: 1.0 }));

            let child = commands
                .spawn((
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(material_handle),
                    Transform::IDENTITY,
                    GlobalTransform::IDENTITY,
                    Visibility::default(),
                    InheritedVisibility::default(),
                    ViewVisibility::default(),
                    Object3DMeshChild,
                    ChildOf(entity),
                ))
                .id();

            Some(EllipsoidVisual {
                child,
                color: *color,
                oversized: false,
                max_extent: 0.0,
            })
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn update_object_3d_billboard_system(
    objects: Query<(
        Entity,
        &Object3DIconState,
        &GlobalTransform,
        &impeller2_wkt::WorldPos,
    )>,
    cameras: Query<(&Camera, &GlobalTransform, &Projection), With<MainCamera>>,
    mut transforms_and_vis: Query<
        (&mut Transform, &mut Visibility),
        (Without<Object3DIconState>, Without<MainCamera>),
    >,
    children_query: Query<&Children>,
    mesh_child_markers: Query<(), With<Object3DMeshChild>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (parent_entity, icon_state, obj_gt, world_pos) in objects.iter() {
        if *world_pos == impeller2_wkt::WorldPos::default() {
            continue;
        }

        let obj_pos = obj_gt.translation();
        let mut best_distance = f32::MAX;
        let mut best_cam_rotation = Quat::IDENTITY;
        let mut best_viewport_height = 1080.0f32;
        let mut best_fov = std::f32::consts::FRAC_PI_4;
        let mut best_in_view = false;

        for (camera, cam_gt, projection) in cameras.iter() {
            let viewport_size = camera.logical_viewport_size();
            let viewport_h = viewport_size.map(|s| s.y).unwrap_or(0.0);
            if viewport_h < 1.0 {
                continue;
            }

            let distance = (obj_pos - cam_gt.translation()).length();

            let in_view = camera
                .world_to_viewport(cam_gt, obj_pos)
                .is_ok_and(|screen_pos| {
                    if let Some(vp) = viewport_size {
                        let inset = vp.y * 0.01;
                        screen_pos.x >= inset
                            && screen_pos.x <= vp.x - inset
                            && screen_pos.y >= inset
                            && screen_pos.y <= vp.y - inset
                    } else {
                        true
                    }
                });

            if distance < best_distance {
                best_distance = distance;
                best_cam_rotation = cam_gt.to_scale_rotation_translation().1;
                best_viewport_height = viewport_h;
                best_fov = match projection {
                    Projection::Perspective(persp) => persp.fov,
                    _ => std::f32::consts::FRAC_PI_4,
                };
                best_in_view = in_view;
            }
        }

        let swap = icon_state.swap_distance;
        let fade_half = swap * BILLBOARD_FADE_BAND * 0.5;
        let fade_start = swap - fade_half;
        let fade_end = swap + fade_half;

        let distance_alpha = if best_distance <= fade_start {
            0.0f32
        } else if best_distance >= fade_end {
            1.0f32
        } else {
            (best_distance - fade_start) / (fade_end - fade_start)
        };

        let billboard_alpha = if best_in_view { distance_alpha } else { 0.0 };

        let parent_rotation = obj_gt.to_scale_rotation_translation().1;

        if let Ok((mut bb_transform, _bb_vis)) =
            transforms_and_vis.get_mut(icon_state.billboard_entity)
        {
            bb_transform.rotation = parent_rotation.inverse() * best_cam_rotation;

            if billboard_alpha > 0.0 {
                let world_size =
                    best_distance * icon_state.screen_size_px * 2.0 * (best_fov / 2.0).tan()
                        / best_viewport_height;
                bb_transform.scale = Vec3::splat(world_size);
            } else {
                bb_transform.scale = Vec3::ZERO;
            }
        }

        if let Some(mat) = materials.get_mut(&icon_state.billboard_material) {
            let mut c = icon_state.base_color;
            c.set_alpha(c.alpha() * billboard_alpha);
            mat.base_color = c;
        }

        let hide_mesh = billboard_alpha > 0.99;

        if let Ok(children) = children_query.get(parent_entity) {
            for child in children.iter() {
                if child == icon_state.billboard_entity {
                    continue;
                }
                if mesh_child_markers.get(child).is_ok()
                    && let Ok((_, mut vis)) = transforms_and_vis.get_mut(child)
                {
                    *vis = if hide_mesh {
                        Visibility::Hidden
                    } else {
                        Visibility::Inherited
                    };
                }
            }
        }
    }
}

pub struct Object3DPlugin;

impl Plugin for Object3DPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IconTextureCache>().add_systems(
            Update,
            (
                update_object_3d_system,
                on_scene_ready.pipe(attach_joint_animations),
                update_joint_animations,
                warn_imported_cameras,
                update_object_3d_billboard_system,
            ),
        );
    }
}
