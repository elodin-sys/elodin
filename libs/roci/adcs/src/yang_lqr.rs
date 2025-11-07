use nalgebra::{UnitQuaternion, Vector3};

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
pub fn lqr_control_mats(
    j: Vector3<f64>,
    q_ang_vel: Vector3<f64>,
    q_pos: Vector3<f64>,
    r: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let k_diag = q_pos.component_div(&r).map(|x| x.sqrt());
    let d_diag = (q_ang_vel.component_div(&r) + j.component_mul(&k_diag)).map(|x| x.sqrt());

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
    att_est: UnitQuaternion<f64>,
    ang_vel: Vector3<f64>,
    goal: UnitQuaternion<f64>,
    d: Vector3<f64>,
    k: Vector3<f64>,
) -> Vector3<f64> {
    let error = att_est.inverse() * goal;
    let sign = error.w;
    let error_vector = error.vector();

    let ang_vel_term = -ang_vel.component_mul(&d);
    let error_term = sign * error_vector.component_mul(&k);

    ang_vel_term + error_term
}

/// Yang LQR spacecraft attitude controller
///
/// Based on the paper: "Analytic LQR Design for Spacecraft Control System Based on Quaternion Model"
/// by Yang et al.
pub struct YangLQR {
    d: Vector3<f64>,
    k: Vector3<f64>,
}

impl YangLQR {
    /// Returns a new Yang LQR controller.
    ///
    /// # Arguments
    /// * `j` - The principal moments of inertia of the spacecraft
    /// * `q` - Weights for the state variables (first 3 for angular velocity, last 3 for position)
    /// * `r` - Weights for the control inputs
    pub fn new(
        j: Vector3<f64>,
        q_ang_vel: Vector3<f64>,
        q_pos: Vector3<f64>,
        r: Vector3<f64>,
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
        att_est: UnitQuaternion<f64>,
        ang_vel: Vector3<f64>,
        target: UnitQuaternion<f64>,
    ) -> Vector3<f64> {
        control(att_est, ang_vel, target, self.d, self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Vector3, vector};

    pub fn test_cubesat_gains() -> (Vector3<f64>, Vector3<f64>) {
        let j = vector![15204079.70002, 14621352.61765, 6237758.3131] * 1e-9;
        let q = vector![5.0, 5.0, 5.0];
        let r = vector![8.0, 8.0, 8.0];

        lqr_control_mats(j, q, q, r)
    }

    #[test]
    fn test_lqr_control_mat() {
        let j = vector![15204079.70002, 14621352.61765, 6237758.3131] * 1e-9;
        let q = vector![5.0, 5.0, 5.0];
        let r = vector![8.0, 8.0, 8.0];

        let (d, k) = lqr_control_mats(j, q, q, r);

        // values from python impl
        let expected_d = vector![0.79813525, 0.7978466, 0.79368217];
        let expected_k = vector![0.79056942, 0.79056942, 0.79056942];

        assert_relative_eq!(d, expected_d, epsilon = 1e-4);
        assert_relative_eq!(k, expected_k, epsilon = 1e-4);
    }

    #[test]
    fn test_control() {
        let att_est = UnitQuaternion::identity();
        let ang_vel = vector![0.1, -0.2, 0.3];
        let goal = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.1);

        let (d, k) = test_cubesat_gains();

        let torque = control(att_est, ang_vel, goal, d, k);
        assert_relative_eq!(
            torque,
            vector![
                -0.07981352519433647,
                0.15956931963081875,
                -0.19864202695933386
            ],
            epsilon = 1e-10
        )
    }
}
