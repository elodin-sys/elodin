use bevy::ecs::hierarchy::ChildOf;
use bevy::prelude::Mesh;
use bevy::prelude::*;
use bevy_render::alpha::AlphaMode;
use big_space::GridCell;
use eql::Expr;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, Object3D};
use nox::Array;
use smallvec::smallvec;

use crate::BevyExt;
use std::fmt;

/// ExprObject3D component that holds an EQL expression for dynamic positioning
#[derive(Component)]
pub struct Object3DState {
    pub compiled_expr: Option<CompiledExpr>,
    pub scale_expr: Option<CompiledExpr>,
    pub scale_error: Option<String>,
    pub data: Object3D,
}

#[derive(Component)]
pub struct EllipsoidVisual {
    pub child: Entity,
    pub color: impeller2_wkt::Color,
    pub oversized: bool,
    pub max_extent: f32,
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

    let entity_id = commands
        .spawn((
            Object3DState {
                compiled_expr: Some(compile_eql_expr(expr)),
                scale_expr,
                scale_error,
                data: data.clone(),
            },
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
            GridCell::<i128>::default(),
            impeller2_wkt::WorldPos::default(),
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

pub fn spawn_mesh(
    commands: &mut Commands,
    entity: Entity,
    mesh: &impeller2_wkt::Object3DMesh,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    assets: &Res<AssetServer>,
) -> Option<EllipsoidVisual> {
    match mesh {
        impeller2_wkt::Object3DMesh::Glb(path) => {
            let url = format!("{path}#Scene0");
            let scene = assets.load(&url);
            commands.entity(entity).insert(SceneRoot(scene));
            None
        }
        impeller2_wkt::Object3DMesh::Mesh { mesh, material } => {
            let material = material.clone().into_bevy();
            let material = material_assets.add(material);
            commands.entity(entity).insert(MeshMaterial3d(material));
            let mesh = mesh.clone().into_bevy();
            let mesh = mesh_assets.add(mesh);
            commands.entity(entity).insert(Mesh3d(mesh));
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

pub struct Object3DPlugin;

impl Plugin for Object3DPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_object_3d_system);
    }
}
