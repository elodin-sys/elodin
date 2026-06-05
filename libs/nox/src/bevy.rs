//! Conversions between nox tensors and Bevy math types.
use crate::{ArrayRepr, Matrix3, Quaternion, Vec3, Tensor};

/// Row-major `[[T; 3]; 3]` (nox layout) to Bevy column-major `Mat3`.
#[inline]
pub fn mat3_from_buf(buf: [[f32; 3]; 3]) -> bevy_math::Mat3 {
    bevy_math::Mat3::from_cols_array(&[
        buf[0][0], buf[1][0], buf[2][0], buf[0][1], buf[1][1], buf[2][1], buf[0][2], buf[1][2],
        buf[2][2],
    ])
}

/// Row-major `[[T; 3]; 3]` (nox layout) to Bevy column-major `DMat3`.
#[inline]
pub fn dmat3_from_buf(buf: [[f64; 3]; 3]) -> bevy_math::DMat3 {
    bevy_math::DMat3::from_cols_array(&[
        buf[0][0], buf[1][0], buf[2][0], buf[0][1], buf[1][1], buf[2][1], buf[0][2], buf[1][2],
        buf[2][2],
    ])
}

impl From<Matrix3<f32, ArrayRepr>> for bevy_math::Mat3 {
    fn from(mat: Matrix3<f32, ArrayRepr>) -> Self {
        mat3_from_buf(mat.into_buf())
    }
}

impl From<Matrix3<f64, ArrayRepr>> for bevy_math::DMat3 {
    fn from(mat: Matrix3<f64, ArrayRepr>) -> Self {
        dmat3_from_buf(mat.into_buf())
    }
}

impl From<bevy_math::Vec3> for Vec3<f32> {
    fn from(v: bevy_math::Vec3) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

impl From<bevy_math::DVec3> for Vec3<f64> {
    fn from(v: bevy_math::DVec3) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

impl From<Vec3<f32>> for bevy_math::Vec3 {
    fn from(v: Vec3<f32>) -> Self {
        let [x, y, z] = v.parts().map(Tensor::into_buf);
        bevy_math::Vec3::new(x, y, z)
    }
}

impl From<Vec3<f64>> for bevy_math::DVec3 {
    fn from(v: Vec3<f64>) -> Self {
        let [x, y, z] = v.parts().map(Tensor::into_buf);
        bevy_math::DVec3::new(x, y, z)
    }
}

impl From<Quaternion<f32, ArrayRepr>> for bevy_math::Quat {
    fn from(q: Quaternion<f32, ArrayRepr>) -> Self {
        let [x, y, z, w] = q.0.into_buf();
        bevy_math::Quat::from_xyzw(x, y, z, w)
    }
}

impl From<Quaternion<f64, ArrayRepr>> for bevy_math::DQuat {
    fn from(q: Quaternion<f64, ArrayRepr>) -> Self {
        let [x, y, z, w] = q.0.into_buf();
        bevy_math::DQuat::from_xyzw(x, y, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix3, Quaternion, Vector3, tensor};

    #[test]
    fn identity_to_bevy_mat3() {
        let m: Matrix3<f64, ArrayRepr> = Matrix3::eye();
        let bevy: bevy_math::DMat3 = m.into();
        assert!(bevy.abs_diff_eq(bevy_math::DMat3::IDENTITY, 1e-6));
    }

    #[test]
    fn look_at_to_bevy_mat3() {
        let m = Matrix3::look_at_rh(
            Vector3::new(1.0f32, 2.0, 3.0).normalize(),
            Vector3::y_axis(),
        );
        let buf = m.into_buf();
        let bevy: bevy_math::Mat3 = Matrix3::from_buf(buf).into();
        for col in 0..3 {
            let axis = [bevy.x_axis, bevy.y_axis, bevy.z_axis][col];
            for row in 0..3 {
                assert!((axis[row] - buf[row][col]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn tensor_literal_matches_from_cols() {
        let m: Matrix3<f32, ArrayRepr> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let bevy: bevy_math::Mat3 = m.into();
        let expected = bevy_math::Mat3::from_cols(
            bevy_math::Vec3::new(1.0, 4.0, 7.0),
            bevy_math::Vec3::new(2.0, 5.0, 8.0),
            bevy_math::Vec3::new(3.0, 6.0, 9.0),
        );
        assert!(bevy.abs_diff_eq(expected, 1e-6));
    }

    #[test]
    fn vec3_from_bevy() {
        let bevy = bevy_math::Vec3::new(1.0, 2.0, 3.0);
        let nox: Vec3<f32> = bevy.into();
        assert_eq!(nox.into_buf(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn vec3_from_bevy_dvec3() {
        let bevy = bevy_math::DVec3::new(4.0, 5.0, 6.0);
        let nox: Vec3<f64> = bevy.into();
        assert_eq!(nox.into_buf(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn quat_from_nox() {
        let q = Quaternion::new(0.5, 0.5, 0.5, 0.5);
        let bevy: bevy_math::Quat = q.into();
        assert!((bevy.x - 0.5).abs() < 1e-6);
        assert!((bevy.y - 0.5).abs() < 1e-6);
        assert!((bevy.z - 0.5).abs() < 1e-6);
        assert!((bevy.w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dquat_from_nox() {
        let q = Quaternion::from_axis_angle(Vector3::x_axis(), core::f64::consts::FRAC_PI_2);
        let bevy: bevy_math::DQuat = q.into();
        let expected = bevy_math::DQuat::from_axis_angle(bevy_math::DVec3::X, core::f64::consts::FRAC_PI_2);
        assert!(bevy.abs_diff_eq(expected, 1e-6));
    }
}
