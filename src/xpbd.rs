use std::ops::Add;

use crate::{
    effector::{Effector, UnifiedEffector},
    sensor::{ErasedSensor, Sensor},
    Force, FromState, Mass, Pos, Time, Torque, Vel,
};
use nalgebra::{Matrix3, Quaternion, Vector3};
use stable_vec::ExternStableVec;

#[derive(Default)]
pub struct Xpbd {
    time: f64,
    entities: ExternStableVec<Entity>,
}

impl Xpbd {
    pub fn tick(&mut self, dt: f64) {
        for (_, entity) in &mut self.entities {
            entity.integrate(Time(self.time), dt);
            entity.update_vel(dt);
            entity.sense(Time(self.time));
        }
        self.time += dt;
    }

    pub fn entity(&mut self, entity: impl Into<Entity>) -> EntityHandle {
        EntityHandle(self.entities.push(entity.into()))
    }

    pub fn get_entity(&self, handle: EntityHandle) -> &Entity {
        &self.entities[handle.0]
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct EntityHandle(usize);

pub struct Builder<T>(T);

pub struct Entity {
    // pos
    prev_pos: Vector3<f64>,
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,

    // attitude
    prev_att: Quaternion<f64>,
    att: Quaternion<f64>,
    ang_vel: Vector3<f64>,

    mass: f64,
    inertia: Matrix3<f64>,
    inverse_inertia: Matrix3<f64>,

    effectors: Vec<Box<dyn Effector<(), Entity, Effect = XpbdEffects>>>,
    sensors: Vec<Box<dyn Sensor<(), Entity>>>,
}

impl Default for Entity {
    fn default() -> Self {
        Self {
            prev_pos: Default::default(),
            pos: Default::default(),
            vel: Default::default(),
            prev_att: Default::default(),
            att: Quaternion::new(1., 0., 0., 0.),
            ang_vel: Default::default(),
            mass: Default::default(),
            inertia: Default::default(),
            inverse_inertia: Default::default(),
            effectors: Default::default(),
            sensors: Default::default(),
        }
    }
}

impl Entity {
    pub fn builder() -> Builder<Self> {
        Builder(Default::default())
    }

    fn integrate(&mut self, time: Time, dt: f64) {
        let effects = self
            .effectors
            .iter()
            .fold(XpbdEffects::default(), |s, e| s + e.effect(time, self));

        // integrate position
        self.vel += dt * effects.force / self.mass;
        self.prev_pos = self.pos;
        self.pos += self.vel * dt;

        // integrate rotation
        self.ang_vel += dt
            * self.inverse_inertia
            * (effects.torque - self.ang_vel.cross(&(self.inertia * self.ang_vel)));
        self.prev_att = self.att;
        self.att += dt
            * 0.5
            * Quaternion::new(0., self.ang_vel.x, self.ang_vel.y, self.ang_vel.z)
            * self.att;
        self.att.normalize_mut();
    }

    fn update_vel(&mut self, dt: f64) {
        self.vel = (self.pos - self.prev_pos) / dt;

        let delta_att = self.att * self.prev_att.try_inverse().unwrap();
        self.ang_vel =
            delta_att.w.signum() * 2.0 * Vector3::new(delta_att.i, delta_att.j, delta_att.k) / dt;
    }

    fn sense(&self, time: Time) {
        for sensor in &self.sensors {
            sensor.sense(time, self)
        }
    }
}

impl Builder<Entity> {
    pub fn mass(mut self, mass: f64) -> Self {
        self.0.mass = mass;
        self
    }

    pub fn vel(mut self, vel: Vector3<f64>) -> Self {
        self.0.vel = vel;
        self
    }

    pub fn pos(mut self, pos: Vector3<f64>) -> Self {
        self.0.pos = pos;
        self
    }

    pub fn att(mut self, att: Quaternion<f64>) -> Self {
        self.0.att = att;
        self
    }

    pub fn effector<T, E, EF>(mut self, effector: E) -> Self
    where
        T: 'static,
        E: Effector<T, Entity, Effect = EF> + 'static,
        EF: Into<XpbdEffects>,
    {
        let unified = UnifiedEffector::new(effector);
        self.0.effectors.push(Box::new(unified));
        self
    }

    pub fn sensor<T, E>(mut self, sensor: E) -> Self
    where
        T: 'static,
        E: Sensor<T, Entity> + 'static,
    {
        let erased = ErasedSensor::new(sensor);
        self.0.sensors.push(Box::new(erased));
        self
    }
}

impl From<Builder<Entity>> for Entity {
    fn from(value: Builder<Entity>) -> Self {
        value.0
    }
}

impl<T> Builder<T> {
    pub fn build(self) -> T {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct XpbdEffects {
    force: Vector3<f64>,
    torque: Vector3<f64>,
}

impl Add for XpbdEffects {
    type Output = XpbdEffects;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            force: self.force + rhs.force,
            torque: self.torque + rhs.torque,
        }
    }
}

impl Into<XpbdEffects> for Force {
    fn into(self) -> XpbdEffects {
        XpbdEffects {
            force: self.0,
            ..Default::default()
        }
    }
}

impl Into<XpbdEffects> for Torque {
    fn into(self) -> XpbdEffects {
        XpbdEffects {
            torque: self.0,
            ..Default::default()
        }
    }
}

impl FromState<Entity> for Mass {
    fn from_state(_time: Time, state: &Entity) -> Self {
        Mass(state.mass)
    }
}

impl FromState<Entity> for Pos {
    fn from_state(_time: Time, state: &Entity) -> Self {
        Pos(state.pos)
    }
}

impl FromState<Entity> for Time {
    fn from_state(time: Time, _state: &Entity) -> Self {
        time
    }
}

impl FromState<Entity> for Vel {
    fn from_state(_time: Time, state: &Entity) -> Self {
        Vel(state.vel)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use nalgebra::vector;

    use crate::forces::gravity;

    use super::*;

    #[test]
    fn test_gravity() {
        let mut xpbd = Xpbd::default();
        let handle = xpbd.entity(
            Entity::builder()
                .mass(1.0)
                .pos(vector![1.0, 0.0, 0.0])
                .vel(vector![0.0, 1.0, 0.0])
                .effector(gravity(1.0 / 6.649e-11, Vector3::zeros())),
        );
        let mut time = 0.0;
        let dt = 0.01;
        while time <= 2.0 * 3.14 {
            xpbd.tick(dt);
            time += dt;
        }
        let entity = xpbd.get_entity(handle);
        assert_relative_eq!(entity.pos, Vector3::new(1.0, 0., 0.), epsilon = 0.01)
    }
}
