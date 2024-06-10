use nox::{Array, ArrayRepr, Matrix, Tensor, Vector};

pub fn triad(
    body_1: Vector<f64, 3, ArrayRepr>,
    body_2: Vector<f64, 3, ArrayRepr>,
    ref_1: Vector<f64, 3, ArrayRepr>,
    ref_2: Vector<f64, 3, ArrayRepr>,
) -> Matrix<f64, 3, 3, ArrayRepr> {
    let r_r = ref_1.cross(&ref_2).normalize();
    let q_r = ref_1;
    let s_r = q_r.cross(&r_r);
    let m_r_t: Matrix<f64, 3, 3, ArrayRepr> = Matrix::from_inner(Array {
        buf: [q_r, r_r, s_r].map(Tensor::into_buf),
    });

    let r_b = body_1.cross(&body_2).normalize();
    let q_b = body_1;
    let s_b = q_b.cross(&r_b);

    let m_b = Matrix::from_inner(
        Array {
            buf: [q_b, r_b, s_b].map(Tensor::into_buf),
        }
        .transpose(),
    );
    m_b.dot(&m_r_t)
}

#[cfg(test)]
mod tests {
    use nox::{tensor, Quaternion};

    use super::*;

    #[test]
    fn test_triad() {
        let q: Quaternion<f64, ArrayRepr> =
            Quaternion::from_axis_angle(Vector::z_axis(), 3.14 / 4.0);
        let ref_a = tensor![0.0, 1.0, 0.0].normalize();
        let ref_b = tensor![1.0, 0.0, 0.0].normalize();
        let body_a = (q.inverse() * ref_a).normalize();
        let body_b = (q.inverse() * ref_b).normalize();
        let dcm = triad(body_a, body_b, ref_a, ref_b);
        let a = dcm.dot(&tensor![1.0, 0.0, 0.0]);
        let b = q.inverse() * tensor![1.0, 0.0, 0.0];
        approx::assert_relative_eq!(&a.into_buf()[..], &b.into_buf()[..]);
    }
}
