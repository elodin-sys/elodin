use std::cell::RefMut;
use std::marker::PhantomData;

use bevy::{
    asset::Asset,
    prelude::{Handle, Mesh, PbrBundle, StandardMaterial},
};
use bevy_ecs::system::Insert;
use bevy_ecs::{entity::Entities, system::Spawn};
use bevy_ecs::{prelude::Entity, system::CommandQueue, world::Mut};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::{effector::Effector, sensor::Sensor, Time};

use super::{
    components::*,
    constraints::{DistanceConstraint, RevoluteJoint},
};

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

    editor_bundle: Option<PbrBundle>,

    fixed: bool,
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

    pub fn intertia(mut self, inertia: Inertia) -> Self {
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

struct ConcreteEffector<ER, E> {
    effector: ER,
    _phantom: PhantomData<(E,)>,
}

impl<ER, E> ConcreteEffector<ER, E> {
    fn new(effector: ER) -> Self {
        Self {
            effector,
            _phantom: PhantomData,
        }
    }
}

impl<ER, T, Eff> XpbdEffector for ConcreteEffector<ER, T>
where
    ER: for<'s> Effector<T, EntityStateRef<'s>, Effect = Eff>,
    Eff: Into<Effect>,
{
    fn effect(&self, time: Time, state: EntityStateRef<'_>) -> Effect {
        self.effector.effect(time, &state).into()
    }
}

struct ConcreteSensor<ER, E> {
    sensor: ER,
    _phantom: PhantomData<(E,)>,
}

impl<ER, E> ConcreteSensor<ER, E> {
    fn new(sensor: ER) -> Self {
        Self {
            sensor,
            _phantom: PhantomData,
        }
    }
}

impl<ER, T> XpbdSensor for ConcreteSensor<ER, T>
where
    ER: for<'a> Sensor<T, EntityStateRef<'a>>,
{
    fn sense(&mut self, time: crate::Time, state: EntityStateRef<'_>) {
        self.sensor.sense(time, &state)
    }
}

pub trait XpbdEffector {
    fn effect(&self, time: Time, state: EntityStateRef<'_>) -> Effect;
}

pub trait XpbdSensor {
    fn sense(&mut self, time: Time, state: EntityStateRef<'_>);
}

pub struct XpbdBuilder<'a> {
    pub(crate) queue: RefMut<'a, CommandQueue>,
    pub(crate) entities: &'a Entities,
}

impl<'a> XpbdBuilder<'a> {
    pub fn entity(&mut self, mut entity_builder: EntityBuilder) -> Entity {
        let entity = self.entities.reserve_entity();
        if let Some(pbr) = entity_builder.editor_bundle.take() {
            self.queue.push(Insert {
                entity,
                bundle: (pbr, entity_builder.bundle()),
            });
        } else {
            self.queue.push(Insert {
                entity,
                bundle: entity_builder.bundle(),
            });
        }
        entity
    }

    pub fn distance_constraint(&mut self, distance_constriant: DistanceConstraint) {
        self.queue.push(Spawn {
            bundle: distance_constriant,
        });
    }

    pub fn revolute_join(&mut self, revolute_join: RevoluteJoint) {
        self.queue.push(Spawn {
            bundle: revolute_join,
        });
    }
}

pub struct Assets<'a>(pub(crate) Option<AssetsInner<'a>>);

pub(crate) struct AssetsInner<'a> {
    pub(crate) meshes: Mut<'a, bevy::prelude::Assets<Mesh>>,
    pub(crate) materials: Mut<'a, bevy::prelude::Assets<StandardMaterial>>,
}

impl<'a> Assets<'a> {
    pub fn mesh(&mut self, mesh: Mesh) -> AssetHandle<Mesh> {
        AssetHandle(if let Some(inner) = self.0.as_mut() {
            Some(inner.meshes.add(mesh))
        } else {
            None
        })
    }

    pub fn material(&mut self, material: StandardMaterial) -> AssetHandle<StandardMaterial> {
        AssetHandle(if let Some(inner) = self.0.as_mut() {
            Some(inner.materials.add(material))
        } else {
            None
        })
    }
}

pub struct AssetHandle<T: Asset>(Option<Handle<T>>);
