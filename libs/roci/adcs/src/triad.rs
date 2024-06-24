use nox::{ArrayRepr, Matrix, Tensor, Vector};

pub fn triad(
    body_1: Vector<f64, 3, ArrayRepr>,
    body_2: Vector<f64, 3, ArrayRepr>,
    ref_1: Vector<f64, 3, ArrayRepr>,
    ref_2: Vector<f64, 3, ArrayRepr>,
) -> Matrix<f64, 3, 3, ArrayRepr> {
    let r_r = ref_1.cross(&ref_2).normalize();
    let q_r = ref_1;
    let s_r = q_r.cross(&r_r);
    let m_r_t = Matrix::from_buf([q_r, r_r, s_r].map(Tensor::into_buf));
    let r_b = body_1.cross(&body_2).normalize();
    let q_b = body_1;
    let s_b = q_b.cross(&r_b);
    let m_b = Matrix::from_buf([q_b, r_b, s_b].map(Tensor::into_buf)).transpose();
    m_b.dot(&m_r_t)
}

#[cfg(test)]
mod tests {
    use nox::tensor;
    use nox::{ArrayRepr, Matrix, Quaternion, Vector};

    use super::*;

    #[test]
    fn test_triad() {
        fn test_triad_inner(q: Quaternion<f64, ArrayRepr>) -> Matrix<f64, 3, 3, ArrayRepr> {
            let ref_a = tensor![0.0, 1.0, 0.0].normalize();
            let ref_b = tensor![1.0, 0.0, 0.0].normalize();
            let body_a = (q.inverse() * ref_a).normalize();
            let body_b = (q.inverse() * ref_b).normalize();
            let dcm = triad(body_a, body_b, ref_a, ref_b);
            let a = dcm.dot(&tensor![1.0, 0.0, 0.0]);
            let b = q.inverse() * tensor![1.0, 0.0, 0.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm.dot(&tensor![0.0, 1.0, 0.0]);
            let b = q.inverse() * tensor![0.0, 1.0, 0.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm.dot(&tensor![0.0, 0.0, 1.0]);
            let b = q.inverse() * tensor![0.0, 0.0, 1.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);

            let a = dcm.dot(&tensor![1.0, 0.0, 1.0]);
            let b = q.inverse() * tensor![1.0, 0.0, 1.0];
            approx::assert_relative_eq!(&a, &b, epsilon = 1e-5);
            dcm
        }

        let q: Quaternion<f64, ArrayRepr> =
            Quaternion::from_axis_angle(Vector::z_axis(), 3.14 / 4.0);
        let dcm = test_triad_inner(q);
        approx::assert_relative_eq!(
            tensor![
                [0.70738827, 0.70682518, 0.],
                [-0.70682518, 0.70738827, 0.],
                [0., 0., 1.]
            ],
            dcm,
            epsilon = 1.0e-4
        );

        for i in -124..124 {
            let ang = i as f64 * 0.05;
            let x = Quaternion::from_axis_angle(Vector::x_axis(), ang);
            let y = Quaternion::from_axis_angle(Vector::y_axis(), ang);
            let z = Quaternion::from_axis_angle(Vector::z_axis(), ang);
            test_triad_inner(x);
            test_triad_inner(y);
            test_triad_inner(z);
            test_triad_inner(x * y);
            test_triad_inner(x * z);
        }
    }
}
