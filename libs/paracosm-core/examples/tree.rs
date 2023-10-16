use std::sync::{atomic::AtomicU64, Arc};

use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, UnitQuaternion, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    editor::{editor, Input},
    forces::earth_gravity,
    runner::IntoSimRunner,
    tree::{Joint, JointType},
    BodyPos, Force, Torque, WorldPos,
};

fn main() {
    editor(sim.substep_count(10))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    let last_change = Arc::new(AtomicU64::new(0));
    let root = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            .fixed()
            .pos(vector![0.0, 2.0, 0.0])
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into()))
            .joint(Joint::fixed()),
    );
    let rod_a_angle = f64::to_radians(30.0);
    let rod_a = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0., -0.5, 0.0])
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                rod_a_angle,
            ))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            //.effector(move || Torque(*input.0.load() * Vector3::z()))
            .effector(|WorldPos(p)| Force(p.att.inverse() * Vector3::new(0., -9.8, 0.)))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            }))
            .parent(
                root,
                Joint {
                    pos: vector![0., 0.0, 0.0],
                    joint_type: JointType::Revolute {
                        axis: Vector3::z_axis(),
                    },
                },
            ),
    );

    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0., -0.5, 0.0])
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                70.0f64.to_radians(),
            ))
            .trace(Vector3::new(0., -0.5, 0.))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            .effector(|WorldPos(p)| Force(p.att.inverse() * Vector3::new(0., -9.8, 0.)))
            .material(assets.material(Color::hex("FF9838").unwrap().into()))
            .parent(
                rod_a,
                Joint {
                    pos: vector![0., -0.5, 0.0],
                    joint_type: JointType::Revolute {
                        axis: Vector3::z_axis(),
                    },
                },
            ),
    );
}
