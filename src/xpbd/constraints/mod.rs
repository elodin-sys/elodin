use nalgebra::{Matrix3, Quaternion, UnitQuaternion, UnitVector3, Vector3};

mod distance;
mod revolute;
pub use distance::*;
pub use revolute::*;

use crate::Pos;

use super::components::EntityQueryItem;

/// Calculate the update of the lagrange multiplier for some
/// generalized coordinate `c`
///
/// Defined in equation 4 of [Detailed Rigid Body Simulation with Extended Position Based Dynamics](https://matthias-research.github.io/pages/publications/PBDBodies.pdf)
pub fn lagrange_multiplier_delta(
    c: f64,
    lagrange_multiplier: f64,
    compliance: f64,
    dt: f64,
    inverse_masses: impl Iterator<Item = f64>,
    gradients: impl Iterator<Item = Vector3<f64>>,
) -> f64 {
    let sum_inverse_masses: f64 = inverse_masses
        .zip(gradients)
        .map(|(m, g)| m * g.norm_squared())
        .sum();
    if sum_inverse_masses <= f64::EPSILON {
        return 0.0;
    }
    let alpha_tilde = compliance / dt.powi(2);
    (-c - alpha_tilde * lagrange_multiplier) / (sum_inverse_masses + alpha_tilde)
}

/// Calculate the positonal impulse from a lagrangian delta
pub fn impulse(delta_lagrange: f64, n: UnitVector3<f64>) -> Vector3<f64> {
    delta_lagrange * n.into_inner()
}

/// Calculate the update to a rigid bodies position from a positional impulse
///
/// Defined in equation 6 of [Detailed Rigid Body Simulation with Extended Position Based Dynamics](https://matthias-research.github.io/pages/publications/PBDBodies.pdf)
pub fn pos_delta_pos_impulse(pos_impulse: Vector3<f64>, mass: f64) -> Vector3<f64> {
    pos_impulse / mass
}
/// Calculate the update to a rigid bodies attitude from a positional impulse
pub fn att_delta_pos_impulse(
    att: Quaternion<f64>,
    pos_impulse: Vector3<f64>,
    r: Vector3<f64>,
    inverse_inertia: Matrix3<f64>,
) -> Quaternion<f64> {
    Quaternion::from_parts(0.0, 0.5 * (inverse_inertia * r.cross(&pos_impulse))) * att
}

pub fn pos_generalized_inverse_mass(
    mass: f64,
    inverse_inertia: Matrix3<f64>,
    r: Vector3<f64>,
    n: UnitVector3<f64>,
) -> f64 {
    let r_cross_n = r.cross(&n);
    1.0 / mass + r_cross_n.dot(&(inverse_inertia * r_cross_n))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_distance_constraint(
    entity_a: &mut EntityQueryItem<'_>,
    entity_b: &mut EntityQueryItem<'_>,
    c: f64,
    n: UnitVector3<f64>,
    inverse_mass_a: f64,
    inverse_mass_b: f64,
    lagrange_multiplier: &mut f64,
    compliance: f64,
    sub_dt: f64,
    world_anchor_a: Pos,
    world_anchor_b: Pos,
) {
    if c <= f64::EPSILON {
        return;
    }
    let delta_lagrange = lagrange_multiplier_delta(
        c,
        *lagrange_multiplier,
        compliance,
        sub_dt,
        [inverse_mass_a, inverse_mass_b].into_iter(),
        [n.into_inner(), -n.into_inner()].into_iter(),
    );
    *lagrange_multiplier += delta_lagrange;
    let pos_impulse = impulse(delta_lagrange, n);
    if !entity_a.fixed.0 {
        entity_a.pos.0 += pos_delta_pos_impulse(pos_impulse, entity_a.mass.0);
        entity_a.att.0 = UnitQuaternion::new_normalize(
            entity_a.att.0.into_inner()
                + att_delta_pos_impulse(
                    *entity_a.att.0,
                    pos_impulse,
                    world_anchor_a.0,
                    entity_a.inverse_inertia.to_world(entity_a).0,
                ),
        );
    }
    if !entity_b.fixed.0 {
        entity_b.pos.0 -= pos_delta_pos_impulse(pos_impulse, entity_b.mass.0);
        let att = entity_b.att.0.into_inner();
        let delta = att_delta_pos_impulse(
            *entity_b.att.0,
            pos_impulse,
            world_anchor_b.0,
            entity_b.inverse_inertia.to_world(entity_b).0,
        );
        entity_b.att.0 = UnitQuaternion::new_normalize(att - delta);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rot_constraint(
    entity_a: &mut EntityQueryItem<'_>,
    entity_b: &mut EntityQueryItem<'_>,
    delta_q: Vector3<f64>,
    inverse_mass_a: f64,
    inverse_mass_b: f64,
    lagrange_multiplier: &mut f64,
    compliance: f64,
    sub_dt: f64,
) {
    let angle = delta_q.norm();
    if angle <= f64::EPSILON {
        return;
    }
    let axis = UnitVector3::new_normalize(delta_q);
    let delta_lagrange = lagrange_multiplier_delta(
        angle,
        *lagrange_multiplier,
        compliance,
        sub_dt,
        [inverse_mass_a, inverse_mass_b].into_iter(),
        [axis.into_inner(), -axis.into_inner()].into_iter(),
    );
    *lagrange_multiplier += delta_lagrange;
    let ang_impulse = -1. * impulse(delta_lagrange, axis);

    if !entity_a.fixed.0 {
        let inverse_inertia = entity_a.inverse_inertia.to_world(entity_a).0;
        entity_a.att.0 = UnitQuaternion::new_normalize(
            *entity_a.att.0 + att_delta_ang_impulse(inverse_inertia, ang_impulse, *entity_a.att.0),
        );
    }
    if !entity_b.fixed.0 {
        let inverse_inertia = entity_b.inverse_inertia.to_world(entity_b).0;
        entity_b.att.0 = UnitQuaternion::new_normalize(
            *entity_b.att.0 + att_delta_ang_impulse(inverse_inertia, ang_impulse, *entity_b.att.0),
        );
    }
}

pub fn rot_generalized_inverse_mass(inverse_inertia: Matrix3<f64>, axis: UnitVector3<f64>) -> f64 {
    axis.dot(&(inverse_inertia * *axis))
}

pub fn att_delta_ang_impulse(
    inverse_mass: Matrix3<f64>,
    ang_impulse: Vector3<f64>,
    q: Quaternion<f64>,
) -> Quaternion<f64> {
    Quaternion::from_parts(0.0, 0.5 * (inverse_mass * ang_impulse)) * q
}
