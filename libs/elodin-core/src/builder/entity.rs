use std::sync::Arc;

use bevy::prelude::{Mesh, StandardMaterial};
use nalgebra::{matrix, vector, Matrix3, UnitVector3, Vector3, Vector6};

use crate::{
    effector::Effector,
    sensor::Sensor,
    spatial::{
        GeneralizedForce, GeneralizedMotion, GeneralizedPos, SpatialForce, SpatialInertia,
        SpatialMotion, SpatialPos, SpatialTransform,
    },
    tree::{Joint, JointType},
    types::*,
    FixedBody, WorldPos, WorldVel,
};

use crate::builder::{ConcreteEffector, ConcreteSensor};

use super::RigidBodyHandle;

#[derive(Clone)]
pub struct EntityBuilder {
    mass: f64,
    inertia: Matrix3<f64>,
    inverse_inertia: Matrix3<f64>,

    effectors: Effectors,
    #[cfg(feature = "nox")]
    pub xla_effectors: XlaEffectors,
    sensors: Sensors,

    fixed: bool,

    pub(crate) trace: Option<Vector3<f64>>,

    pub(crate) material: Option<Box<StandardMaterial>>,
    pub(crate) mesh: Option<Box<Mesh>>,
    pub(crate) scene: Option<String>,

    pub(crate) parent: Option<RigidBodyHandle>,
    pub(crate) joint: Arc<dyn JointBuilder>,

    body_pos: SpatialPos,
    body_vel: SpatialMotion,
}

impl Default for EntityBuilder {
    fn default() -> Self {
        Self {
            mass: Default::default(),
            inertia: Matrix3::identity(),
            inverse_inertia: Matrix3::identity(),
            effectors: Default::default(),
            #[cfg(feature = "nox")]
            xla_effectors: Default::default(),
            sensors: Default::default(),
            fixed: false,
            trace: None,
            parent: None,
            joint: Arc::<Free>::default(),
            body_pos: Default::default(),
            body_vel: Default::default(),
            material: None,
            mesh: None,
            scene: None,
        }
    }
}

impl EntityBuilder {
    pub fn mass(mut self, mass: f64) -> Self {
        self.mass = mass;
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
        self.effectors.0.push(Arc::new(unified));
        self
    }

    pub fn sensor<T, E>(mut self, sensor: E) -> Self
    where
        T: Send + Sync + 'static,
        E: for<'a> Sensor<T, EntityStateRef<'a>> + Send + Sync + 'static,
    {
        let erased = ConcreteSensor::new(sensor);
        self.sensors.0.push(Arc::new(erased));
        self
    }

    pub fn model(mut self, model: String) -> Self {
        self.scene = Some(model);
        self
    }

    pub fn mesh(mut self, mesh: Mesh) -> Self {
        self.mesh = Some(Box::new(mesh));
        self
    }

    pub fn material(mut self, material: StandardMaterial) -> Self {
        self.material = Some(Box::new(material));
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

    pub fn parent(mut self, parent: RigidBodyHandle) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn joint(mut self, joint: impl JointBuilder) -> Self {
        self.joint = Arc::new(joint);
        self
    }

    pub fn body_pos(mut self, pos: SpatialPos) -> Self {
        self.body_pos = pos;
        self
    }

    pub fn body_vel(mut self, vel: SpatialMotion) -> Self {
        self.body_vel = vel;
        self
    }

    pub fn bundle(self) -> EntityBundle {
        EntityBundle {
            mass: Mass(self.mass),
            inertia: Inertia(self.inertia),
            inverse_inertia: InverseInertia(self.inverse_inertia),

            effectors: self.effectors,
            #[cfg(feature = "nox")]
            xla_effectors: self.xla_effectors,
            sensors: self.sensors,

            effect: Effect::default(),
            fixed: FixedBody(self.fixed),
            picked: Picked(false),

            world_pos: WorldPos(Default::default()),
            world_anchor_pos: WorldAnchorPos(SpatialTransform::identity()),
            world_vel: WorldVel(Default::default()),
            world_accel: WorldAccel(Default::default()),
            tree_index: TreeIndex(0),
            subtree_inertia: SubtreeInertia(SpatialInertia::default()),
            bias_force: BiasForce(SpatialForce::default()),

            joint: self.joint.apply(),
            body_pos: BodyPos(self.body_pos),
            subtree_com: SubtreeCoM(Default::default()),
            subtree_com_sum: SubtreeCoMSum(Default::default()),
            subtree_mass: SubtreeMass(Default::default()),
        }
    }
}

pub trait JointBuilder: Send + Sync + 'static {
    fn apply(&self) -> JointBundle;
}

pub struct Revolute {
    pos: f64,
    vel: f64,
    axis: UnitVector3<f64>,
    offset: Vector3<f64>,
}

impl Revolute {
    pub fn new(axis: UnitVector3<f64>) -> Self {
        Revolute {
            pos: 0.0,
            vel: 0.0,
            axis,
            offset: Vector3::zeros(),
        }
    }

    pub fn pos(mut self, pos: f64) -> Self {
        self.pos = pos;
        self
    }

    pub fn vel(mut self, vel: f64) -> Self {
        self.vel = vel;
        self
    }

    pub fn anchor(mut self, offset: Vector3<f64>) -> Self {
        self.offset = offset;
        self
    }
}

impl JointBuilder for Revolute {
    fn apply(&self) -> JointBundle {
        JointBundle {
            joint: Joint {
                pos: self.offset,
                joint_type: JointType::Revolute { axis: self.axis },
            },
            pos: JointPos(GeneralizedPos {
                dof: 1,
                is_quat: false,
                inner: vector![self.pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }),
            vel: JointVel(GeneralizedMotion {
                dof: 1,
                inner: vector![self.vel, 0.0, 0.0, 0.0, 0.0, 0.0],
            }),
            joint_accel: JointAccel(crate::spatial::GeneralizedMotion {
                dof: 1,
                inner: Vector6::default(),
            }),
            joint_force: JointForce(crate::spatial::GeneralizedForce {
                dof: 1,
                inner: Vector6::default(),
            }),
        }
    }
}

#[derive(Default)]
pub struct Free {
    pos: SpatialPos,
    vel: SpatialMotion,
}

impl Free {
    pub fn pos(mut self, pos: SpatialPos) -> Self {
        self.pos = pos;
        self
    }

    pub fn vel(mut self, vel: SpatialMotion) -> Self {
        self.vel = vel;
        self
    }
}

impl JointBuilder for Free {
    fn apply(&self) -> JointBundle {
        JointBundle {
            joint: Joint {
                pos: Vector3::zeros(),
                joint_type: JointType::Free,
            },
            pos: JointPos(GeneralizedPos {
                dof: 7,
                is_quat: true,
                inner: matrix![
                    self.pos.att[0];
                    self.pos.att[1];
                    self.pos.att[2];
                    self.pos.att[3];
                    self.pos.pos[0];
                    self.pos.pos[1];
                    self.pos.pos[2]
                ],
            }),
            vel: JointVel(GeneralizedMotion {
                dof: 6,
                inner: self.vel.vector(),
            }),
            joint_accel: JointAccel::default(),
            joint_force: JointForce::default(),
        }
    }
}

#[derive(Default)]
pub struct FixedJoint;

impl JointBuilder for FixedJoint {
    fn apply(&self) -> JointBundle {
        JointBundle {
            joint: Joint {
                pos: Vector3::zeros(),
                joint_type: JointType::Fixed,
            },
            pos: JointPos(GeneralizedPos {
                dof: 0,
                is_quat: false,
                inner: matrix![ 0.; 0.; 0.; 0.; 0.; 0.; 0.; ],
            }),
            vel: JointVel(GeneralizedMotion {
                dof: 0,
                inner: Vector6::zeros(),
            }),
            joint_accel: JointAccel(GeneralizedMotion {
                dof: 0,
                ..Default::default()
            }),
            joint_force: JointForce(GeneralizedForce {
                dof: 0,
                ..Default::default()
            }),
        }
    }
}
