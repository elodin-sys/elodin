use std::sync::{atomic::AtomicU64, Arc};

use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, UnitQuaternion, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, FixedJoint, Free, Revolute, XpbdBuilder},
    editor::{editor, Input},
    forces::earth_gravity,
    runner::IntoSimRunner,
    spatial::SpatialPos,
    tree::{Joint, JointType},
    Effect, Force, JointPos, Torque, WorldPos,
};

fn main() {
    editor(sim.substep_count(10))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    let last_change = Arc::new(AtomicU64::new(0));
    let root = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            //.fixed()
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into()))
            .body_pos(SpatialPos::linear(vector![0., -2.0, 0.0]))
            .joint(FixedJoint),
    );
    let rod_a = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .body_pos(SpatialPos::linear(vector![0., -0.5, 0.0]))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            //.effector(move || Torque(*input.0.load() * Vector3::z()))
            //.effector(|WorldPos(p)| Force(-9.8 * (Vector3::y())))
            // .effector(|WorldPos(p)|{
            //     Effect::force_at_point(
            //         p.pos,
            //         Vector3::new(0., -9.8, 0.),
            //         UnitQuaternion::identity(),
            //     )
            // })
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            }))
            .parent(
                root,
                Revolute::new(Vector3::z_axis()).pos(0f64.to_radians()),
            ),
    );

    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .body_pos(SpatialPos::linear(vector![0., -0.5, 0.0]))
            .trace(Vector3::new(0., -0.5, 0.))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            .material(assets.material(Color::hex("FF9838").unwrap().into()))
            .parent(
                rod_a,
                Revolute::new(Vector3::z_axis())
                    .anchor(vector![0., -0.5, 0.0])
                    .pos(10f64.to_radians()),
            ),
    );
}
