use bevy::prelude::*;
use big_space::GridCell;
use eql::{Context as EqlContext, Expr};
use impeller2::{component::Component, types::EntityId};
use impeller2_bevy::{ComponentValueMap, EntityMap};
use impeller2_wkt::{ComponentValue, EntityMetadata, Glb, WorldPos};
use nox::Array;
use smallvec::smallvec;

/// Ghost component that holds an EQL expression for dynamic positioning
#[derive(Component, Clone, Debug)]
pub struct Ghost {
    pub eql: String,
    pub expr: eql::Expr,
}

impl Ghost {
    pub fn new(eql: String, expr: eql::Expr) -> Self {
        Self { eql, expr }
    }
}

pub fn resolve_eql_expression(
    expression: &eql::Expr,
    eql_context: &EqlContext,
    entity_map: &EntityMap,
    component_value_maps: &Query<&ComponentValueMap, Without<Ghost>>,
) -> Result<ComponentValue, String> {
    match expression {
        Expr::Component(component) => {
            let entity_id = entity_map.get(&component.entity).ok_or_else(|| {
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
        }
        Expr::ArrayAccess(expr, index) => {
            let resolved_expr =
                resolve_eql_expression(expr, eql_context, entity_map, component_value_maps)?;
            match resolved_expr {
                ComponentValue::F64(array) => {
                    use nox::ArrayBuf;
                    let data = array.buf.as_buf();
                    if *index < data.len() {
                        let value = data[*index];
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
                    if *index < data.len() {
                        let value = data[*index];
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
        }
        Expr::Tuple(expressions) => {
            let mut values = Vec::new();
            for expr in expressions {
                let resolved =
                    resolve_eql_expression(expr, eql_context, entity_map, component_value_maps)?;
                match resolved {
                    ComponentValue::F64(array) => {
                        use nox::ArrayBuf;
                        values.extend_from_slice(array.buf.as_buf());
                    }
                    ComponentValue::F32(array) => {
                        use nox::ArrayBuf;
                        let f32_data = array.buf.as_buf();
                        values.extend(f32_data.iter().map(|&x| x as f64));
                    }
                    _ => return Err("tuple elements must be numeric".to_string()),
                }
            }
            let tuple_array =
                Array::from_shape_vec(smallvec![values.len()], values.into()).unwrap();
            Ok(ComponentValue::F64(tuple_array))
        }
        Expr::BinaryOp(left, right, op) => {
            let left_val =
                resolve_eql_expression(left, eql_context, entity_map, component_value_maps)?;
            let right_val =
                resolve_eql_expression(right, eql_context, entity_map, component_value_maps)?;

            match (left_val, right_val) {
                (ComponentValue::F64(left), ComponentValue::F64(right)) => {
                    if !nox::array::can_broadcast(left.shape(), right.shape()) {
                        return Err("binary operation requires arrays be broadcastable".to_string());
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
        }

        Expr::FloatLiteral(f) => Ok(ComponentValue::F64(nox::array!(*f).to_dyn())),
        expr => Err(format!(
            "{:?} can't be converted to a component value",
            expr
        )),
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
    eql_context: Res<crate::EqlContext>,
    entity_map: Res<EntityMap>,
    component_value_maps: Query<&ComponentValueMap, Without<Ghost>>,
) {
    for (entity, ghost, mut pos, mut value_map) in ghosts_query.iter_mut() {
        match resolve_eql_expression(
            &ghost.expr,
            &eql_context.0,
            &entity_map,
            &component_value_maps,
        ) {
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
