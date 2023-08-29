use nalgebra::{Matrix3, Quaternion, UnitVector3, Vector3};

mod distance;
pub use distance::*;

/// Calculate the update of the lagrange multiplier for some
/// generalized coordinate `c`
///
/// Defined in equation 4 of [Detailed Rigid Body Simulation with Extended Position Based Dynamics](https://matthias-research.github.io/pages/publications/PBDBodies.pdf)
pub fn lagrange_multiplier_delta(
    c: f64,
    lagrange_multiplier: f64,
    alpha: f64,
    dt: f64,
    inverse_masses: impl Iterator<Item = f64>,
) -> f64 {
    let alpha_tilde = alpha / dt.powi(2);
    let sum_inverse_masses: f64 = inverse_masses.sum();
    (-c - alpha_tilde * lagrange_multiplier) / (sum_inverse_masses + alpha_tilde)
}

/// Calculate the positonal impulse from a lagrangian delta
pub fn pos_impulse(delta_lagrange: f64, n: UnitVector3<f64>) -> Vector3<f64> {
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
    0.5 * Quaternion::from_parts(0.0, inverse_inertia * r.cross(&pos_impulse)) * att
}

pub fn pos_generalized_inverse_mass(
    mass: f64,
    inverse_inertia: Matrix3<f64>,
    r: Vector3<f64>,
    n: UnitVector3<f64>,
) -> f64 {
    let r_cross_n = r.cross(&n);
    let mat = 1.0 / mass * r_cross_n.transpose() * inverse_inertia * r_cross_n;
    mat.into_scalar()
}
