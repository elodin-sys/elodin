use nox::{ArrayRepr, OwnedRepr, Quaternion, Repr, Vector};

/// Calculates the LQR control matrices for a yang LQR controller.
///
/// Based on the paper: "Analytic LQR Design for Spacecraft Control System Based on Quaternion Model"
/// by Yang et al.
///
/// # Arguments
/// * `j` - The principal moments of inertia of the spacecraft
/// * `q` - Weights for the state variables (first 3 for angular velocity, last 3 for position)
/// * `r` - Weights for the control inputs
///
/// # Returns
/// * `(d, k)` - Diagonal elements of the D and K gain matrices
pub fn lqr_control_mats<R: Repr + OwnedRepr>(
    j: Vector<f64, 3, R>,
    q_ang_vel: Vector<f64, 3, R>,
    q_pos: Vector<f64, 3, R>,
    r: Vector<f64, 3, R>,
) -> (Vector<f64, 3, R>, Vector<f64, 3, R>) {
    let k_diag = (q_pos / &r).sqrt();
    let d_diag = (q_ang_vel / r + j * &k_diag).sqrt();

    (d_diag, k_diag)
}

/// Computes control torque using the LQR controller.
///
/// # Arguments
/// * `att_est` - Current attitude estimate (quaternion)
/// * `ang_vel` - Current angular velocity estimate
/// * `goal` - Target attitude (quaternion)
/// * `d` - D gain matrix diagonal elements
/// * `k` - K gain matrix diagonal elements
///
/// # Returns
/// * Control torque to apply in body frame
pub fn control(
    att_est: Quaternion<f64, ArrayRepr>,
    ang_vel: Vector<f64, 3, ArrayRepr>,
    goal: Quaternion<f64, ArrayRepr>,
    d: Vector<f64, 3, ArrayRepr>,
    k: Vector<f64, 3, ArrayRepr>,
) -> Vector<f64, 3, ArrayRepr> {
    let error = (att_est.inverse() * goal).0;
    let sign = error.get(3);
    let error_vector: Vector<f64, 3, _> = error.fixed_slice(&[0]);

    let ang_vel_term = -1.0 * (ang_vel * d);
    let error_term = sign * (error_vector * k);

    ang_vel_term + error_term
}

/// Yang LQR spacecraft attitude controller
///
/// Based on the paper: "Analytic LQR Design for Spacecraft Control System Based on Quaternion Model"
/// by Yang et al.
pub struct YangLQR {
    d: Vector<f64, 3, ArrayRepr>,
    k: Vector<f64, 3, ArrayRepr>,
}

impl YangLQR {
    /// Returns a new Yang LQR controller.
    ///
    /// # Arguments
    /// * `j` - The principal moments of inertia of the spacecraft
    /// * `q` - Weights for the state variables (first 3 for angular velocity, last 3 for position)
    /// * `r` - Weights for the control inputs
    pub fn new(
        j: Vector<f64, 3, ArrayRepr>,
        q_ang_vel: Vector<f64, 3, ArrayRepr>,
        q_pos: Vector<f64, 3, ArrayRepr>,
        r: Vector<f64, 3, ArrayRepr>,
    ) -> YangLQR {
        let (d, k) = lqr_control_mats(j, q_ang_vel, q_pos, r);
        YangLQR { d, k }
    }

    /// Computes control torque using the LQR controller.
    ///
    /// # Arguments
    /// * `att_est` - Current attitude estimate (quaternion)
    /// * `ang_vel` - Current angular velocity estimate
    /// * `goal` - Target attitude (quaternion)
    ///
    /// # Returns
    /// * Control torque to apply in body frame
    pub fn control(
        &self,
        att_est: Quaternion<f64, ArrayRepr>,
        ang_vel: Vector<f64, 3, ArrayRepr>,
        target: Quaternion<f64, ArrayRepr>,
    ) -> Vector<f64, 3, ArrayRepr> {
        control(att_est, ang_vel, target, self.d, self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nox::tensor;

    pub fn test_cubesat_gains() -> (Vector<f64, 3, ArrayRepr>, Vector<f64, 3, ArrayRepr>) {
        let j = tensor![15204079.70002, 14621352.61765, 6237758.3131] * 1e-9;
        let q = tensor![5.0, 5.0, 5.0];
        let r = tensor![8.0, 8.0, 8.0];

        lqr_control_mats(j, q, q, r)
    }

    #[test]
    fn test_lqr_control_mat() {
        let j = tensor![15204079.70002, 14621352.61765, 6237758.3131] * 1e-9;
        let q = tensor![5.0, 5.0, 5.0];
        let r = tensor![8.0, 8.0, 8.0];

        let (d, k) = lqr_control_mats(j, q, q, r);

        // values from python impl
        let expected_d = tensor![0.79813525, 0.7978466, 0.79368217];
        let expected_k = tensor![0.79056942, 0.79056942, 0.79056942];

        assert_relative_eq!(d, expected_d, epsilon = 1e-4);
        assert_relative_eq!(k, expected_k, epsilon = 1e-4);
    }

    #[test]
    fn test_control() {
        let att_est = Quaternion::identity();
        let ang_vel = tensor![0.1, -0.2, 0.3];
        let goal = Quaternion::from_axis_angle(tensor![0.0, 0.0, 1.0], 0.1);

        let (d, k) = test_cubesat_gains();

        let torque = control(att_est, ang_vel, goal, d, k);
        assert_relative_eq!(
            torque,
            tensor![
                -0.07981352519433647,
                0.15956931963081875,
                -0.19864202695933386
            ]
        )
    }
}
