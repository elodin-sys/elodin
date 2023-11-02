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

/// Calculate the positional impulse from a lagrangian delta
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
    effect_a: &mut Effect,
    effect_b: &mut Effect,
    c: f64,
    n: UnitVector3<f64>,
    inverse_mass_a: f64,
    inverse_mass_b: f64,
    lagrange_multiplier: &mut f64,
    compliance: f64,
    sub_dt: f64,
    world_anchor_a: Vector3<f64>,
    world_anchor_b: Vector3<f64>,
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
    let sub_dt_squared = sub_dt.powi(2);
    if !entity_a.fixed.0 {
        let force = pos_impulse / sub_dt_squared;
        let torque = world_anchor_a.cross(&pos_impulse) / sub_dt_squared;
        effect_a.force.0 += force;
        effect_a.torque.0 += torque;
    }
    if !entity_b.fixed.0 {
        let force = pos_impulse / sub_dt_squared;
        let torque = world_anchor_b.cross(&pos_impulse) / sub_dt_squared;
        effect_b.force.0 -= force;
        effect_b.torque.0 -= torque;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rot_constraint(
    entity_a: &mut EntityQueryItem<'_>,
    entity_b: &mut EntityQueryItem<'_>,
    effect_a: &mut Effect,
    effect_b: &mut Effect,
    delta_q: Vector3<f64>,
    lagrange_multiplier: &mut f64,
    compliance: f64,
    sub_dt: f64,
) {
    let angle = delta_q.norm();
    if angle <= f64::EPSILON {
        return;
    }
    let axis = UnitVector3::new_normalize(delta_q);
    let inverse_inertia_a = entity_a.world_pos.0.transform() * entity_a.inverse_inertia.0;
    let inverse_inertia_b = entity_b.world_pos.0.transform() * entity_b.inverse_inertia.0;
    let inverse_mass_a = rot_generalized_inverse_mass(inverse_inertia_a, axis);
    let inverse_mass_b = rot_generalized_inverse_mass(inverse_inertia_b, axis);

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
    let sub_dt_squared = sub_dt.powi(2);
    if !entity_a.fixed.0 {
        let torque = ang_impulse / sub_dt_squared;
        effect_a.torque.0 += torque;
    }
    if !entity_b.fixed.0 {
        let torque = ang_impulse / sub_dt_squared;
        effect_b.torque.0 -= torque;
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
    0.5 * Quaternion::from_parts(0.0, inverse_mass * ang_impulse) * q
}
