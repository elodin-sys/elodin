use bevy::prelude::Mesh;
use bevy::prelude::*;
use big_space::GridCell;
use eql::Expr;
use impeller2_bevy::EntityMap;
use impeller2_wkt::ComponentValue;
use nox::Array;
use smallvec::smallvec;

use crate::BevyExt;

/// ExprObject3D component that holds an EQL expression for dynamic positioning
#[derive(Component, Deref, DerefMut)]
pub struct Object3D(EditableEQL);

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
        &'a Query<'b, 'b, &'static ComponentValue, Without<Object3D>>,
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
                &'a Query<'b, 'b, &'static ComponentValue, Without<Object3D>>,
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
        values: &'a Query<'b, 'b, &'static ComponentValue, Without<Object3D>>,
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
                            let value = nox::array!(value).to_dyn();
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
                            let value = nox::array!(value).to_dyn();
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

/// System that updates 3D object entities based on their EQL expressions
pub fn update_object_3d_system(
    mut objects_query: Query<(Entity, &Object3D, &mut impeller2_wkt::WorldPos)>,
    entity_map: Res<EntityMap>,
    component_value_maps: Query<&'static ComponentValue, Without<Object3D>>,
) {
    for (entity, object_3d, mut pos) in objects_query.iter_mut() {
        let Some(compiled_expr) = &object_3d.compiled_expr else {
            continue;
        };
        match compiled_expr.execute(&entity_map, &component_value_maps) {
            Ok(component_value) => {
                if let Some(world_pos) = component_value.as_world_pos() {
                    *pos = world_pos;
                }
            }
            Err(e) => {
                println!(
                    "failed to resolve 3D object expression for entity {:?}: {}",
                    entity, e
                );
            }
        }
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
    eql: String,
    expr: eql::Expr,
    mesh_source: Option<impeller2_wkt::Object3D>,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    assets: &Res<AssetServer>,
) -> Entity {
    let mut entity = commands.spawn((
        Object3D(EditableEQL::new(eql, compile_eql_expr(expr))),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        GridCell::<i128>::default(),
        impeller2_wkt::WorldPos::default(),
    ));

    if let Some(source) = mesh_source {
        match source {
            impeller2_wkt::Object3D::Glb(path) => {
                let url = format!("{path}#Scene0");
                let scene = assets.load(&url);
                entity.insert(SceneRoot(scene));
            }
            impeller2_wkt::Object3D::Mesh { mesh, material } => {
                let material = material.clone().into_bevy();
                let material = material_assets.add(material);
                entity.insert(MeshMaterial3d(material));
                let mesh = mesh.clone().into_bevy();
                let mesh = mesh_assets.add(mesh);
                entity.insert(Mesh3d(mesh));
            }
        }
    }

    entity.id()
}

pub struct Object3DPlugin;

impl Plugin for Object3DPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_object_3d_system);
    }
}
