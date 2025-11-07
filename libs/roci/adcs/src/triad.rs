use nalgebra::{Matrix3, Vector3};

pub fn triad(
    body_1: Vector3<f64>,
    body_2: Vector3<f64>,
    ref_1: Vector3<f64>,
    ref_2: Vector3<f64>,
) -> Matrix3<f64> {
    let r_r = ref_1.cross(&ref_2).normalize();
    let q_r = ref_1;
    let s_r = q_r.cross(&r_r);
    // Create matrix from row vectors
    let m_r_t = Matrix3::from_rows(&[q_r.transpose(), r_r.transpose(), s_r.transpose()]);
    let r_b = body_1.cross(&body_2).normalize();
    let q_b = body_1;
    let s_b = q_b.cross(&r_b);
    let m_b = Matrix3::from_rows(&[q_b.transpose(), r_b.transpose(), s_b.transpose()]).transpose();
    m_b * m_r_t
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, UnitQuaternion, Vector3, matrix, vector};

    use super::*;

    #[test]
    fn test_triad() {
        fn test_triad_inner(q: UnitQuaternion<f64>) -> Matrix3<f64> {
            let ref_a = vector![0.0, 1.0, 0.0].normalize();
            let ref_b = vector![1.0, 0.0, 0.0].normalize();
            // Nalgebra: q * vector rotates by q, so we use q (not q.inverse()) to go from inertial to body
            let body_a = (q * ref_a).normalize();
            let body_b = (q * ref_b).normalize();
            let dcm = triad(body_a, body_b, ref_a, ref_b);
            let a = dcm * vector![1.0, 0.0, 0.0];
            let b = q * vector![1.0, 0.0, 0.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm * vector![0.0, 1.0, 0.0];
            let b = q * vector![0.0, 1.0, 0.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm * vector![0.0, 0.0, 1.0];
            let b = q * vector![0.0, 0.0, 1.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm * vector![1.0, 0.0, 1.0];
            let b = q * vector![1.0, 0.0, 1.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);
            dcm
        }

        let q: UnitQuaternion<f64> =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::PI / 4.0);
        let dcm = test_triad_inner(q);
        // cos(45°) = sin(45°) = √2/2 ≈ 0.7071067811865476
        // Rotation matrix for 45° rotation around z-axis
        let sqrt2_over_2 = core::f64::consts::SQRT_2 / 2.0;
        approx::assert_relative_eq!(
            matrix![
                sqrt2_over_2, -sqrt2_over_2, 0.;
                sqrt2_over_2, sqrt2_over_2, 0.;
                0., 0., 1.;
            ],
            dcm,
            epsilon = 1.0e-10
        );

        for i in -124..124 {
            let ang = i as f64 * 0.05;
            let x = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), ang);
            let y = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), ang);
            let z = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), ang);
            test_triad_inner(x);
            test_triad_inner(y);
            test_triad_inner(z);
            test_triad_inner(x * y);
            test_triad_inner(x * z);
        }
    }
}
