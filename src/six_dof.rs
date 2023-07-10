use std::ops::{Add, AddAssign, Mul};

use nalgebra::{Quaternion, Vector3};

use crate::{Force, FromState, Mass, Pos, Sim, State, StateEffect, Time, Torque};

#[derive(Default, Debug, Clone, Copy)]
pub struct SixDof {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub mass: f64,
    pub ang: Quaternion<f64>,
    pub ang_vel: Vector3<f64>,
    pub time: f64,
}

impl SixDof {
    pub fn sim(self) -> Sim<SixDof> {
        Sim::new(self)
    }
    pub fn pos(mut self, pos: Vector3<f64>) -> Self {
        self.pos = pos;
        self
    }
    pub fn vel(mut self, vel: Vector3<f64>) -> Self {
        self.vel = vel;
        self
    }
    pub fn mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }
    pub fn ang(mut self, ang: Quaternion<f64>) -> Self {
        self.ang = ang;
        self
    }
    pub fn ang_vel(mut self, ang_vel: Vector3<f64>) -> Self {
        self.ang_vel = ang_vel;
        self
    }
    pub fn time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }
}

impl Add for SixDof {
    type Output = SixDof;

    fn add(self, rhs: Self) -> Self::Output {
        SixDof {
            pos: self.pos + rhs.pos,
            vel: self.vel + rhs.vel,
            mass: self.mass + rhs.mass,
            ang: self.ang + rhs.ang,
            ang_vel: self.ang_vel + rhs.ang_vel,
            time: self.time + rhs.time,
        }
    }
}

impl Mul<SixDof> for f64 {
    type Output = SixDof;

    fn mul(self, rhs: SixDof) -> Self::Output {
        SixDof {
            pos: rhs.pos * self,
            vel: rhs.vel * self,
            mass: rhs.mass * self,
            ang: rhs.ang * self,
            ang_vel: rhs.ang_vel * self,
            time: rhs.time * self,
        }
    }
}

impl FromState<SixDof> for Mass {
    fn from_state(state: &SixDof) -> Self {
        Mass(state.mass)
    }
}

impl FromState<SixDof> for Pos {
    fn from_state(state: &SixDof) -> Self {
        Pos(state.pos)
    }
}

impl FromState<SixDof> for Time {
    fn from_state(state: &SixDof) -> Self {
        Time(state.time)
    }
}

impl StateEffect<SixDof> for Force {
    fn apply(&self, init_state: &SixDof, inc_state: &mut SixDof) {
        let accl = self.0 / init_state.mass;
        inc_state.vel += accl;
    }
}

impl StateEffect<SixDof> for Torque {
    fn apply(&self, init_state: &SixDof, inc_state: &mut SixDof) {
        let ang_accl = self.0 / init_state.mass;
        //let accl = self.0 / init_state.mass;
        inc_state.ang_vel += ang_accl;
    }
}

impl State for SixDof {
    fn step(&self, inc_state: &mut SixDof) {
        inc_state.pos += self.vel;
        inc_state.ang =
            0.5 * Quaternion::new(0., self.ang_vel.x, self.ang_vel.y, self.ang_vel.z) * self.ang;
        // NOTE: This relies on the small angle approx a bunch, I think this is safe for small dt, but for very high angular vel it will fail
        inc_state.time += 1.0;
    }
}

impl AddAssign for SixDof {
    fn add_assign(&mut self, rhs: Self) {
        self.pos += rhs.pos;
        self.vel += rhs.vel;
        self.mass += rhs.mass;
        self.ang += rhs.ang;
        self.ang_vel += rhs.ang_vel;
        self.time += rhs.time;
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, UnitVector3};

    use super::*;

    #[test]
    fn impulse_torque() {
        let mut sim = SixDof::default()
            .mass(1.0)
            .ang(*UnitQuaternion::from_axis_angle(
                &UnitVector3::new_unchecked(Vector3::new(0.0, 1.0, 0.0)),
                0.0,
            ))
            .sim()
            .effector(|Time(t)| {
                if t <= 1.0 {
                    Torque(Vector3::new(1.0, 0.0, 0.0))
                } else {
                    Torque(Vector3::zeros())
                }
            });
        const DT: f64 = 0.01;
        for _ in 0..100 {
            sim.tick(DT);
        }
        assert_relative_eq!(
            sim.state.ang,
            *UnitQuaternion::from_axis_angle(
                &UnitVector3::new_unchecked(Vector3::new(1.0, 0.0, 0.0)),
                0.5,
            ),
            epsilon = 1e-9,
        )
    }
}
