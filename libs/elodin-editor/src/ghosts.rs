use bevy::prelude::*;
use big_space::GridCell;
use eql::Expr;
use impeller2::{component::Component, types::EntityId};
use impeller2_bevy::{ComponentValueMap, EntityMap};
use impeller2_wkt::{ComponentValue, EntityMetadata, Glb, WorldPos};
use nox::Array;
use smallvec::smallvec;

/// Ghost component that holds an EQL expression for dynamic positioning
#[derive(Component)]
pub struct Ghost {
    pub eql: String,
    pub compiled_expr: CompiledExpr,
}

impl Ghost {
    pub fn new(eql: String, expr: eql::Expr) -> Self {
        Self {
            eql,
            compiled_expr: compile_eql_expr(expr),
        }
    }
}

type ExprFn = dyn for<'a, 'b> Fn(
        &'a EntityMap,
        &'a Query<'b, 'b, &'static ComponentValueMap, Without<Ghost>>,
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
                &'a Query<'b, 'b, &'static ComponentValueMap, Without<Ghost>>,
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
        component_value_maps: &'a Query<'b, 'b, &'static ComponentValueMap, Without<Ghost>>,
    ) -> Result<ComponentValue, String> {
        match self {
            Self::Closure(c) => (c)(entity_map, component_value_maps),
            Self::Value(value) => Ok(value.clone()),
        }
    }
}

/// Compiles an EQL expression into a closure-based form
pub fn compile_eql_expr(expression: eql::Expr) -> CompiledExpr {
    match expression {
        Expr::Component(component) => {
            let entity = component.entity;
            CompiledExpr::closure(move |entity_map, component_value_maps| {
                let entity_id = entity_map.get(&entity).ok_or_else(|| {
                    format!("entity '{}' not found in entity map", component.entity_name)
                })?;

                let component_map = component_value_maps.get(*entity_id).map_err(|_| {
                    format!(
                        "no component value map found for entity '{}'",
                        component.entity_name
                    )
                })?;

                component_map.get(&component.id).cloned().ok_or_else(|| {
                    format!(
                        "component '{}' not found for entity '{}'",
                        component.name, component.entity_name
                    )
                })
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

/// System that updates ghost entities based on their EQL expressions
pub fn update_ghost_system(
    mut ghosts_query: Query<(
        Entity,
        &Ghost,
        &mut impeller2_wkt::WorldPos,
        &mut ComponentValueMap,
    )>,
    entity_map: Res<EntityMap>,
    component_value_maps: Query<&'static ComponentValueMap, Without<Ghost>>,
) {
    for (entity, ghost, mut pos, mut value_map) in ghosts_query.iter_mut() {
        match ghost
            .compiled_expr
            .execute(&entity_map, &component_value_maps)
        {
            Ok(component_value) => {
                if let ComponentValue::F64(array) = component_value {
                    use nox::ArrayBuf;
                    let data = array.buf.as_buf();
                    if data.len() >= 7 {
                        *pos = impeller2_wkt::WorldPos {
                            att: nox::Quaternion::new(data[3], data[0], data[1], data[2]),
                            pos: nox::Vector3::new(data[4], data[5], data[6]),
                        };
                        value_map
                            .insert(WorldPos::COMPONENT_ID, ComponentValue::F64(array.to_dyn()));
                    }
                }
            }
            Err(e) => {
                println!(
                    "failed to resolve ghost expression for entity {:?}: {}",
                    entity, e
                );
            }
        }
    }
}

pub fn create_ghost_entity(
    commands: &mut Commands,
    eql: String,
    expr: eql::Expr,
    gltf_path: Option<String>,
) -> Entity {
    let entity_id = EntityId(fastrand::u64(..));
    let mut entity = commands.spawn((
        Ghost::new(eql, expr),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        GridCell::<i128>::default(),
        impeller2_wkt::WorldPos::default(),
        entity_id,
        ComponentValueMap(Default::default()),
        EntityMetadata {
            entity_id,
            name: "Ghost".to_string(),
            metadata: Default::default(),
        },
    ));

    if let Some(path) = gltf_path {
        entity.insert(impeller2_bevy::AssetHandle::<Glb>::new(fastrand::u64(..)));
        entity.insert(Glb(path));
    }

    entity.id()
}

pub struct GhostsPlugin;

impl Plugin for GhostsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_ghost_system);
    }
}
