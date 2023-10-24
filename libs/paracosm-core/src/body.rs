use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Integrate a bodies position and velocity
///
/// This function integrates position and velocity of a body.
/// It takes into account external forces, but does not solve
/// any constraints.
pub fn integrate_pos(
    pos: &mut Vector3<f64>,
    prev_pos: &mut Vector3<f64>,
    vel: &mut Vector3<f64>,
    joint_accel: Vector3<f64>,
    dt: f64,
) {
    *vel += dt * joint_accel;
    *prev_pos = *pos;
    *pos += *vel * dt;
}

/// Integrate a bodies attitude and angular velocity
///
/// This function integrates the angular position and veloctiy of a body,
/// taking into account any external torque.
pub fn integrate_att(
    att: &mut UnitQuaternion<f64>,
    prev_att: &mut UnitQuaternion<f64>,
    ang_vel: &mut Vector3<f64>,
    joint_ang_accel: Vector3<f64>,
    dt: f64,
) {
    *ang_vel += dt * joint_ang_accel; // FIXME? maybe remove the cross product here -inv_inertia * ang_vel.cross(&(inertia * *ang_vel)));
    *prev_att = *att;
    let non_unit_att = att.into_inner();
    let new_att = non_unit_att
        + dt * 0.5 * Quaternion::new(0., ang_vel.x, ang_vel.y, ang_vel.z) * non_unit_att;
    *att = UnitQuaternion::new_normalize(new_att);
}

/// Calculates a new velocity from the previous position and current position
#[inline]
pub fn calc_vel(pos: Vector3<f64>, prev_pos: Vector3<f64>, dt: f64) -> Vector3<f64> {
    let dx = pos - prev_pos;
    dx / dt
}

/// Calculates a new angular velocity from the previous attitude and current attitude
#[inline]
pub fn calc_ang_vel(
    att: UnitQuaternion<f64>,
    prev_att: UnitQuaternion<f64>,
    dt: f64,
) -> Vector3<f64> {
    let delta_att = att * prev_att.inverse();
    delta_att.w.signum() * 2.0 * Vector3::new(delta_att.i, delta_att.j, delta_att.k) / dt
}
