use bevy::{
    prelude::{Handle, Mesh, PbrBundle, StandardMaterial},
    scene::Scene,
};
use bevy_ecs::entity::Entity;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::{
    effector::Effector,
    sensor::Sensor,
    spatial::{SpatialForce, SpatialInertia, SpatialMotion, SpatialPos},
    tree::Joint,
    types::*,
    WorldPos, WorldVel,
};

use crate::builder::{AssetHandle, ConcreteEffector, ConcreteSensor};

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
    pub(crate) scene: Option<Handle<Scene>>,

    fixed: bool,
    pub(crate) trace: Option<Vector3<f64>>,

    pub(crate) parent: Option<Entity>,
    pub(crate) joint: Joint,
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
            scene: None,
            parent: None,
            joint: Joint {
                pos: Vector3::zeros(),
                joint_type: crate::tree::JointType::Free,
            },
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

    pub fn model(mut self, model: AssetHandle<Scene>) -> Self {
        self.scene = model.0;
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

    pub fn parent(mut self, parent: Entity, joint: Joint) -> Self {
        self.parent = Some(parent);
        self.joint = joint;
        self
    }

    pub fn joint(mut self, joint: Joint) -> Self {
        self.joint = joint;
        self
    }

    pub fn bundle(self) -> EntityBundle {
        EntityBundle {
            pos: BodyPos(SpatialPos {
                pos: self.pos,
                att: self.att,
            }),
            prev_pos: PrevPos(Vector3::zeros()),
            vel: BodyVel(SpatialMotion {
                vel: self.vel,
                ang_vel: self.ang_vel,
            }),

            prev_att: PrevAtt(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.0)),

            mass: Mass(self.mass),
            inertia: Inertia(self.inertia),
            inverse_inertia: InverseInertia(self.inverse_inertia),

            effectors: self.effectors,
            sensors: self.sensors,

            effect: Effect::default(),
            fixed: Fixed(self.fixed),
            picked: Picked(false),
            joint: self.joint,

            world_pos: WorldPos(Default::default()),
            world_vel: WorldVel(Default::default()),
            world_accel: WorldAccel(Default::default()),
            joint_accel: JointAccel(SpatialMotion::default()),
            joint_force: JointForce(SpatialForce::default()),
            tree_index: TreeIndex(0),
            subtree_inertia: SubtreeInertia(SpatialInertia::default()),
            bias_force: BiasForce(SpatialForce::default()),
        }
    }
}
