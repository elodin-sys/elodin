use bevy::ecs::hierarchy::ChildOf;
use bevy::log::warn_once;
use bevy::prelude::Mesh;
use bevy::prelude::*;
use bevy_render::alpha::AlphaMode;
use big_space::GridCell;
use eql::Expr;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, Object3D};
use nox::Array;
use smallvec::smallvec;

use crate::{BevyExt, MainCamera, plugins::navigation_gizmo::NavGizmoCamera};
use std::fmt;

type ImportedCameraFilter = (Added<Camera>, Without<NavGizmoCamera>, Without<MainCamera>);

type ImportedCameraQuery<'w, 's> = Query<'w, 's, (Entity, &'static ChildOf), ImportedCameraFilter>;

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

/// Compiles a formula expression into a runtime closure
fn compile_formula(formula_name: &str, inner_expr: eql::Expr) -> CompiledExpr {
    use nox::ArrayBuf;
    
    match formula_name {
        // Rotation formulas
        "rotate_x" | "rotate_y" | "rotate_z" => {
            // Inner expr is Tuple(receiver, angle)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 2 {
                    let error = format!("{} requires receiver and angle", formula_name);
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let angle_compiled = compile_eql_expr(elements[1].clone());
                let axis_index = match formula_name {
                    "rotate_x" => 0,
                    "rotate_y" => 1,
                    "rotate_z" => 2,
                    _ => unreachable!(),
                };
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    // Get the spatial transform (7-element array: qx,qy,qz,qw,px,py,pz)
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("rotate requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("rotate requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get the angle in degrees
                    let angle_val = angle_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(angle_array) = angle_val else {
                        return Err("angle must be a number".to_string());
                    };
                    let angle_data = angle_array.buf.as_buf();
                    if angle_data.is_empty() {
                        return Err("angle cannot be empty".to_string());
                    }
                    let angle_deg = angle_data[0];
                    let angle_rad = angle_deg.to_radians();
                    
                    // Create rotation quaternion from axis-angle
                    let half_angle = angle_rad / 2.0;
                    let sin_half = half_angle.sin();
                    let cos_half = half_angle.cos();
                    
                    let (rot_qx, rot_qy, rot_qz, rot_qw) = match axis_index {
                        0 => (sin_half, 0.0, 0.0, cos_half),  // X axis
                        1 => (0.0, sin_half, 0.0, cos_half),  // Y axis
                        2 => (0.0, 0.0, sin_half, cos_half),  // Z axis
                        _ => unreachable!(),
                    };
                    
                    // Extract input quaternion (stored as x,y,z,w)
                    let qx = data[0];
                    let qy = data[1];
                    let qz = data[2];
                    let qw = data[3];
                    
                    // Quaternion multiplication: q_result = q_input * q_rotation
                    // Hamilton product: (w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    //                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    //                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    //                    w1*z2 + x1*y2 - y1*x2 + z1*w2)
                    let new_qw = qw * rot_qw - qx * rot_qx - qy * rot_qy - qz * rot_qz;
                    let new_qx = qw * rot_qx + qx * rot_qw + qy * rot_qz - qz * rot_qy;
                    let new_qy = qw * rot_qy - qx * rot_qz + qy * rot_qw + qz * rot_qx;
                    let new_qz = qw * rot_qz + qx * rot_qy - qy * rot_qx + qz * rot_qw;
                    
                    // Keep the same position
                    let result = vec![new_qx, new_qy, new_qz, new_qw, data[4], data[5], data[6]];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = format!("{} requires tuple expression", formula_name);
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        "rotate" => {
            // Inner expr is Tuple(receiver, x_angle, y_angle, z_angle)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 4 {
                    let error = "rotate requires receiver and three angles (x, y, z)".to_string();
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let x_angle_compiled = compile_eql_expr(elements[1].clone());
                let y_angle_compiled = compile_eql_expr(elements[2].clone());
                let z_angle_compiled = compile_eql_expr(elements[3].clone());
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("rotate requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("rotate requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get angles
                    let get_angle = |compiled: &CompiledExpr| -> Result<f64, String> {
                        let val = compiled.execute(entity_map, component_values)?;
                        let ComponentValue::F64(arr) = val else {
                            return Err("angle must be a number".to_string());
                        };
                        let d = arr.buf.as_buf();
                        if d.is_empty() {
                            return Err("angle cannot be empty".to_string());
                        }
                        Ok(d[0])
                    };
                    
                    let x_deg = get_angle(&x_angle_compiled)?;
                    let y_deg = get_angle(&y_angle_compiled)?;
                    let z_deg = get_angle(&z_angle_compiled)?;
                    
                    // Apply rotations in order: X, then Y, then Z
                    let mut qx = data[0];
                    let mut qy = data[1];
                    let mut qz = data[2];
                    let mut qw = data[3];
                    
                    // Helper to apply a rotation
                    let apply_rot = |qx: f64, qy: f64, qz: f64, qw: f64, 
                                    rx: f64, ry: f64, rz: f64, rw: f64| -> (f64, f64, f64, f64) {
                        let new_qw = qw * rw - qx * rx - qy * ry - qz * rz;
                        let new_qx = qw * rx + qx * rw + qy * rz - qz * ry;
                        let new_qy = qw * ry - qx * rz + qy * rw + qz * rx;
                        let new_qz = qw * rz + qx * ry - qy * rx + qz * rw;
                        (new_qx, new_qy, new_qz, new_qw)
                    };
                    
                    // X rotation
                    if x_deg.abs() > 1e-10 {
                        let half = x_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_rot(qx, qy, qz, qw, 
                                                     half.sin(), 0.0, 0.0, half.cos());
                    }
                    
                    // Y rotation
                    if y_deg.abs() > 1e-10 {
                        let half = y_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_rot(qx, qy, qz, qw, 
                                                     0.0, half.sin(), 0.0, half.cos());
                    }
                    
                    // Z rotation
                    if z_deg.abs() > 1e-10 {
                        let half = z_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_rot(qx, qy, qz, qw, 
                                                     0.0, 0.0, half.sin(), half.cos());
                    }
                    
                    let result = vec![qx, qy, qz, qw, data[4], data[5], data[6]];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = "rotate requires tuple expression".to_string();
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        // Translation formulas
        "translate_x" | "translate_y" | "translate_z" => {
            // Inner expr is Tuple(receiver, distance)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 2 {
                    let error = format!("{} requires receiver and distance", formula_name);
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let distance_compiled = compile_eql_expr(elements[1].clone());
                let axis_index = match formula_name {
                    "translate_x" => 0,
                    "translate_y" => 1,
                    "translate_z" => 2,
                    _ => unreachable!(),
                };
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("translate requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("translate requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get the distance
                    let dist_val = distance_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(dist_array) = dist_val else {
                        return Err("distance must be a number".to_string());
                    };
                    let dist_data = dist_array.buf.as_buf();
                    if dist_data.is_empty() {
                        return Err("distance cannot be empty".to_string());
                    }
                    let dist = dist_data[0];
                    
                    // Create offset in body frame
                    let offset_body = match axis_index {
                        0 => (dist, 0.0, 0.0),
                        1 => (0.0, dist, 0.0),
                        2 => (0.0, 0.0, dist),
                        _ => unreachable!(),
                    };
                    
                    // Extract quaternion
                    let qx = data[0];
                    let qy = data[1];
                    let qz = data[2];
                    let qw = data[3];
                    
                    // Rotate offset from body frame to world frame
                    // v' = q * v * q^-1
                    // For unit quaternions, q^-1 = conjugate
                    let (ox, oy, oz) = offset_body;
                    
                    // First: q * v (treating v as quaternion with w=0)
                    let t_w = -qx * ox - qy * oy - qz * oz;
                    let t_x = qw * ox + qy * oz - qz * oy;
                    let t_y = qw * oy + qz * ox - qx * oz;
                    let t_z = qw * oz + qx * oy - qy * ox;
                    
                    // Second: t * q_conjugate
                    let rx = t_w * (-qx) + t_x * qw + t_y * (-qz) - t_z * (-qy);
                    let ry = t_w * (-qy) - t_x * (-qz) + t_y * qw + t_z * (-qx);
                    let rz = t_w * (-qz) + t_x * (-qy) - t_y * (-qx) + t_z * qw;
                    
                    // Add to existing position
                    let new_px = data[4] + rx;
                    let new_py = data[5] + ry;
                    let new_pz = data[6] + rz;
                    
                    let result = vec![qx, qy, qz, qw, new_px, new_py, new_pz];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = format!("{} requires tuple expression", formula_name);
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        // World-frame rotation formulas
        "rotate_world_x" | "rotate_world_y" | "rotate_world_z" => {
            // Inner expr is Tuple(receiver, angle)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 2 {
                    let error = format!("{} requires receiver and angle", formula_name);
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let angle_compiled = compile_eql_expr(elements[1].clone());
                let axis_index = match formula_name {
                    "rotate_world_x" => 0,
                    "rotate_world_y" => 1,
                    "rotate_world_z" => 2,
                    _ => unreachable!(),
                };
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    // Get the spatial transform (7-element array: qx,qy,qz,qw,px,py,pz)
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("rotate_world requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("rotate_world requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get the angle in degrees
                    let angle_val = angle_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(angle_array) = angle_val else {
                        return Err("angle must be a number".to_string());
                    };
                    let angle_data = angle_array.buf.as_buf();
                    if angle_data.is_empty() {
                        return Err("angle cannot be empty".to_string());
                    }
                    let angle_deg = angle_data[0];
                    let angle_rad = angle_deg.to_radians();
                    
                    // Create rotation quaternion from axis-angle
                    let half_angle = angle_rad / 2.0;
                    let sin_half = half_angle.sin();
                    let cos_half = half_angle.cos();
                    
                    let (rot_qx, rot_qy, rot_qz, rot_qw) = match axis_index {
                        0 => (sin_half, 0.0, 0.0, cos_half),  // X axis
                        1 => (0.0, sin_half, 0.0, cos_half),  // Y axis
                        2 => (0.0, 0.0, sin_half, cos_half),  // Z axis
                        _ => unreachable!(),
                    };
                    
                    // Extract input quaternion (stored as x,y,z,w)
                    let qx = data[0];
                    let qy = data[1];
                    let qz = data[2];
                    let qw = data[3];
                    
                    // World-frame rotation: q_result = q_rotation * q_input (reversed order)
                    let new_qw = rot_qw * qw - rot_qx * qx - rot_qy * qy - rot_qz * qz;
                    let new_qx = rot_qw * qx + rot_qx * qw + rot_qy * qz - rot_qz * qy;
                    let new_qy = rot_qw * qy - rot_qx * qz + rot_qy * qw + rot_qz * qx;
                    let new_qz = rot_qw * qz + rot_qx * qy - rot_qy * qx + rot_qz * qw;
                    
                    // Keep the same position
                    let result = vec![new_qx, new_qy, new_qz, new_qw, data[4], data[5], data[6]];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = format!("{} requires tuple expression", formula_name);
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        "rotate_world" => {
            // Inner expr is Tuple(receiver, x_angle, y_angle, z_angle)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 4 {
                    let error = "rotate_world requires receiver and three angles (x, y, z)".to_string();
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let x_angle_compiled = compile_eql_expr(elements[1].clone());
                let y_angle_compiled = compile_eql_expr(elements[2].clone());
                let z_angle_compiled = compile_eql_expr(elements[3].clone());
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("rotate_world requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("rotate_world requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get angles
                    let get_angle = |compiled: &CompiledExpr| -> Result<f64, String> {
                        let val = compiled.execute(entity_map, component_values)?;
                        let ComponentValue::F64(arr) = val else {
                            return Err("angle must be a number".to_string());
                        };
                        let d = arr.buf.as_buf();
                        if d.is_empty() {
                            return Err("angle cannot be empty".to_string());
                        }
                        Ok(d[0])
                    };
                    
                    let x_deg = get_angle(&x_angle_compiled)?;
                    let y_deg = get_angle(&y_angle_compiled)?;
                    let z_deg = get_angle(&z_angle_compiled)?;
                    
                    // Apply rotations in order: X, then Y, then Z (in world frame)
                    let mut qx = data[0];
                    let mut qy = data[1];
                    let mut qz = data[2];
                    let mut qw = data[3];
                    
                    // Helper to apply a world-frame rotation
                    let apply_world_rot = |qx: f64, qy: f64, qz: f64, qw: f64, 
                                           rx: f64, ry: f64, rz: f64, rw: f64| -> (f64, f64, f64, f64) {
                        // World-frame: q_rotation * q_input
                        let new_qw = rw * qw - rx * qx - ry * qy - rz * qz;
                        let new_qx = rw * qx + rx * qw + ry * qz - rz * qy;
                        let new_qy = rw * qy - rx * qz + ry * qw + rz * qx;
                        let new_qz = rw * qz + rx * qy - ry * qx + rz * qw;
                        (new_qx, new_qy, new_qz, new_qw)
                    };
                    
                    // X rotation
                    if x_deg.abs() > 1e-10 {
                        let half = x_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_world_rot(qx, qy, qz, qw, 
                                                           half.sin(), 0.0, 0.0, half.cos());
                    }
                    
                    // Y rotation
                    if y_deg.abs() > 1e-10 {
                        let half = y_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_world_rot(qx, qy, qz, qw, 
                                                           0.0, half.sin(), 0.0, half.cos());
                    }
                    
                    // Z rotation
                    if z_deg.abs() > 1e-10 {
                        let half = z_deg.to_radians() / 2.0;
                        (qx, qy, qz, qw) = apply_world_rot(qx, qy, qz, qw, 
                                                           0.0, 0.0, half.sin(), half.cos());
                    }
                    
                    let result = vec![qx, qy, qz, qw, data[4], data[5], data[6]];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = "rotate_world requires tuple expression".to_string();
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        // World-frame translation formulas
        "translate_world_x" | "translate_world_y" | "translate_world_z" => {
            // Inner expr is Tuple(receiver, distance)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 2 {
                    let error = format!("{} requires receiver and distance", formula_name);
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let distance_compiled = compile_eql_expr(elements[1].clone());
                let axis_index = match formula_name {
                    "translate_world_x" => 0,
                    "translate_world_y" => 1,
                    "translate_world_z" => 2,
                    _ => unreachable!(),
                };
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("translate_world requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("translate_world requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get the distance
                    let dist_val = distance_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(dist_array) = dist_val else {
                        return Err("distance must be a number".to_string());
                    };
                    let dist_data = dist_array.buf.as_buf();
                    if dist_data.is_empty() {
                        return Err("distance cannot be empty".to_string());
                    }
                    let dist = dist_data[0];
                    
                    // Apply offset directly in world frame (no rotation)
                    let (dx, dy, dz) = match axis_index {
                        0 => (dist, 0.0, 0.0),
                        1 => (0.0, dist, 0.0),
                        2 => (0.0, 0.0, dist),
                        _ => unreachable!(),
                    };
                    
                    // Keep quaternion, add offset to position
                    let result = vec![data[0], data[1], data[2], data[3], 
                                     data[4] + dx, data[5] + dy, data[6] + dz];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = format!("{} requires tuple expression", formula_name);
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        "translate_world" => {
            // Inner expr is Tuple(receiver, x, y, z)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 4 {
                    let error = "translate_world requires receiver and three distances (x, y, z)".to_string();
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let x_dist_compiled = compile_eql_expr(elements[1].clone());
                let y_dist_compiled = compile_eql_expr(elements[2].clone());
                let z_dist_compiled = compile_eql_expr(elements[3].clone());
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("translate_world requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("translate_world requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get distances
                    let get_dist = |compiled: &CompiledExpr| -> Result<f64, String> {
                        let val = compiled.execute(entity_map, component_values)?;
                        let ComponentValue::F64(arr) = val else {
                            return Err("distance must be a number".to_string());
                        };
                        let d = arr.buf.as_buf();
                        if d.is_empty() {
                            return Err("distance cannot be empty".to_string());
                        }
                        Ok(d[0])
                    };
                    
                    let dx = get_dist(&x_dist_compiled)?;
                    let dy = get_dist(&y_dist_compiled)?;
                    let dz = get_dist(&z_dist_compiled)?;
                    
                    // Apply offsets directly in world frame (no rotation)
                    let result = vec![data[0], data[1], data[2], data[3],
                                     data[4] + dx, data[5] + dy, data[6] + dz];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = "translate_world requires tuple expression".to_string();
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        "translate" => {
            // Inner expr is Tuple(receiver, x, y, z)
            if let eql::Expr::Tuple(elements) = inner_expr {
                if elements.len() != 4 {
                    let error = "translate requires receiver and three distances (x, y, z)".to_string();
                    return CompiledExpr::closure(move |_, _| Err(error.clone()));
                }
                
                let receiver_compiled = compile_eql_expr(elements[0].clone());
                let x_dist_compiled = compile_eql_expr(elements[1].clone());
                let y_dist_compiled = compile_eql_expr(elements[2].clone());
                let z_dist_compiled = compile_eql_expr(elements[3].clone());
                
                CompiledExpr::closure(move |entity_map, component_values| {
                    let spatial = receiver_compiled.execute(entity_map, component_values)?;
                    let ComponentValue::F64(array) = spatial else {
                        return Err("translate requires a spatial transform".to_string());
                    };
                    
                    let data = array.buf.as_buf();
                    if data.len() < 7 {
                        return Err(format!("translate requires 7-element array, got {}", data.len()));
                    }
                    
                    // Get distances
                    let get_dist = |compiled: &CompiledExpr| -> Result<f64, String> {
                        let val = compiled.execute(entity_map, component_values)?;
                        let ComponentValue::F64(arr) = val else {
                            return Err("distance must be a number".to_string());
                        };
                        let d = arr.buf.as_buf();
                        if d.is_empty() {
                            return Err("distance cannot be empty".to_string());
                        }
                        Ok(d[0])
                    };
                    
                    let dx = get_dist(&x_dist_compiled)?;
                    let dy = get_dist(&y_dist_compiled)?;
                    let dz = get_dist(&z_dist_compiled)?;
                    
                    // Extract quaternion
                    let qx = data[0];
                    let qy = data[1];
                    let qz = data[2];
                    let qw = data[3];
                    
                    // Rotate offset from body frame to world frame
                    // v' = q * v * q^-1
                    let t_w = -qx * dx - qy * dy - qz * dz;
                    let t_x = qw * dx + qy * dz - qz * dy;
                    let t_y = qw * dy + qz * dx - qx * dz;
                    let t_z = qw * dz + qx * dy - qy * dx;
                    
                    let rx = t_w * (-qx) + t_x * qw + t_y * (-qz) - t_z * (-qy);
                    let ry = t_w * (-qy) - t_x * (-qz) + t_y * qw + t_z * (-qx);
                    let rz = t_w * (-qz) + t_x * (-qy) - t_y * (-qx) + t_z * qw;
                    
                    let new_px = data[4] + rx;
                    let new_py = data[5] + ry;
                    let new_pz = data[6] + rz;
                    
                    let result = vec![qx, qy, qz, qw, new_px, new_py, new_pz];
                    let result_array = Array::from_shape_vec(smallvec![7], result).unwrap();
                    Ok(ComponentValue::F64(result_array))
                })
            } else {
                let error = "translate requires tuple expression".to_string();
                CompiledExpr::closure(move |_, _| Err(error.clone()))
            }
        }
        
        _ => {
            let error = format!("formula '{}' is not supported in editor runtime", formula_name);
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
        Expr::Formula(formula, inner_expr) => {
            compile_formula(formula.name(), *inner_expr)
        }
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

    // Get initial transform scale from GLB if applicable
    let initial_transform = match &data.mesh {
        impeller2_wkt::Object3DMesh::Glb { scale, .. } if *scale != 1.0 => {
            Transform::from_scale(Vec3::splat(*scale))
        }
        _ => Transform::default(),
    };

    let entity_id = commands
        .spawn((
            Object3DState {
                compiled_expr: Some(compile_eql_expr(expr)),
                scale_expr,
                scale_error,
                data: data.clone(),
            },
            initial_transform,
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

pub fn spawn_mesh(
    commands: &mut Commands,
    entity: Entity,
    mesh: &impeller2_wkt::Object3DMesh,
    material_assets: &mut ResMut<Assets<StandardMaterial>>,
    mesh_assets: &mut ResMut<Assets<Mesh>>,
    assets: &Res<AssetServer>,
) -> Option<EllipsoidVisual> {
    match mesh {
        impeller2_wkt::Object3DMesh::Glb { path, .. } => {
            let url = format!("{path}#Scene0");
            let scene = assets.load(&url);
            commands.entity(entity).insert(SceneRoot(scene));
            commands
                .entity(entity)
                .insert(Name::new(format!("object_3d {}", &path)));
            None
        }
        impeller2_wkt::Object3DMesh::Mesh { mesh, material } => {
            let mut material = material.clone().into_bevy();
            if matches!(mesh, impeller2_wkt::Mesh::Plane { .. }) {
                material.double_sided = true;
                material.cull_mode = None;
            }
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
        app.add_systems(Update, (update_object_3d_system, warn_imported_cameras));
    }
}
