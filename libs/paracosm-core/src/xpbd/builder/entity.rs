use bevy::prelude::{Mesh, PbrBundle, StandardMaterial};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::{effector::Effector, sensor::Sensor, xpbd::components::*};

use super::{AssetHandle, ConcreteEffector, ConcreteSensor};

pub struct EntityBuilder {
    mass: f64,
    pos: Vector3<f64>,
    vel: Vector3<f64>,
    att: UnitQuaternion<f64>,
    ang_vel: Vector3<f64>,
    inertia: Matrix3<f64>,
    inverse_inertia: Matrix3<f64>,

    effectors: Effectors,
    sensors: Sensors,

    pub(crate) editor_bundle: Option<PbrBundle>,

    fixed: bool,
    pub(crate) trace: Option<Vector3<f64>>,
}

impl Default for EntityBuilder {
    fn default() -> Self {
        Self {
            mass: Default::default(),
            pos: Default::default(),
            vel: Default::default(),
            att: Default::default(),
            ang_vel: Default::default(),
            inertia: Matrix3::identity(),
            inverse_inertia: Matrix3::identity(),
            effectors: Default::default(),
            sensors: Default::default(),
            editor_bundle: Default::default(),
            fixed: false,
            trace: None,
        }
    }
}

impl EntityBuilder {
    pub fn mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    pub fn vel(mut self, vel: Vector3<f64>) -> Self {
        self.vel = vel;
        self
    }

    pub fn pos(mut self, pos: Vector3<f64>) -> Self {
        self.pos = pos;
        self
    }

    pub fn att(mut self, att: UnitQuaternion<f64>) -> Self {
        self.att = att;
        self
    }

    pub fn ang_vel(mut self, ang_vel: Vector3<f64>) -> Self {
        self.ang_vel = ang_vel;
        self
    }

    pub fn inertia(mut self, inertia: Inertia) -> Self {
        self.inertia = inertia.0;
        self.inverse_inertia = inertia.0.try_inverse().unwrap();
        self
    }

    pub fn effector<T, E, EF>(mut self, effector: E) -> Self
    where
        T: 'static + Send + Sync,
        E: for<'a> Effector<T, EntityStateRef<'a>, Effect = EF> + Send + Sync + 'static,
        EF: Into<Effect> + Send + Sync,
    {
        let unified: ConcreteEffector<E, T> = ConcreteEffector::new(effector);
        self.effectors.0.push(Box::new(unified));
        self
    }

    pub fn sensor<T, E>(mut self, sensor: E) -> Self
    where
        T: Send + Sync + 'static,
        E: for<'a> Sensor<T, EntityStateRef<'a>> + Send + Sync + 'static,
    {
        let erased = ConcreteSensor::new(sensor);
        self.sensors.0.push(Box::new(erased));
        self
    }

    pub fn mesh(mut self, mesh: AssetHandle<Mesh>) -> Self {
        if let Some(mesh) = mesh.0 {
            let editor_bundle = self.editor_bundle.get_or_insert_with(Default::default);
            editor_bundle.mesh = mesh;
        }
        self
    }

    pub fn material(mut self, material: AssetHandle<StandardMaterial>) -> Self {
        if let Some(material) = material.0 {
            let editor_bundle = self.editor_bundle.get_or_insert_with(Default::default);
            editor_bundle.material = material;
        }
        self
    }

    pub fn fixed(mut self) -> Self {
        self.fixed = true;
        self
    }

    pub fn trace(mut self, anchor: Vector3<f64>) -> Self {
        self.trace = Some(anchor);
        self
    }

    pub fn bundle(self) -> EntityBundle {
        EntityBundle {
            pos: Pos(self.pos),
            prev_pos: PrevPos(Vector3::zeros()),
            vel: Vel(self.vel),

            att: Att(self.att),
            prev_att: PrevAtt(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.0)),
            ang_vel: AngVel(self.ang_vel),

            mass: Mass(self.mass),
            inertia: Inertia(self.inertia),
            inverse_inertia: InverseInertia(self.inverse_inertia),

            effectors: self.effectors,
            sensors: self.sensors,

            effect: Effect::default(),
            fixed: Fixed(self.fixed),
        }
    }
}
