use std::{
    f64::consts::PI,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, UnitQuaternion, Vector3};
use paracosm::{
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        constraints::{Angle, RevoluteJoint},
        editor::{editor, ObservableInput},
    },
    Force, Pos, Time,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: ObservableInput) {
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
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
    );
    let rod_a_angle = f64::to_radians(0.0);
    let rod_a_pos = vector![0.5 * rod_a_angle.sin(), 2.0 - 0.5 * rod_a_angle.cos(), 0.0];
    let rod_a = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(rod_a_pos)
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                rod_a_angle,
            ))
            .intertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            .effector(|Time(_)| Force(vector![0.0, -9.8, 0.0]))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })),
    );
    builder.revolute_join(
        RevoluteJoint::new(root, rod_a)
            .join_axis(Vector3::z_axis())
            .anchor_b(Pos(vector![0., 0.5, 0.0]))
            .angle_limits(-PI / 2.0..PI / 2.0)
            .compliance(0.0)
            .ang_damping(0.5)
            .pos_damping(0.5)
            .effector(move |Time(t)| {
                let t = (t * 1000.0) as u64;
                if input.0.has_changed() {
                    last_change.store(t, Ordering::SeqCst);
                }
                let last_change_val = last_change.load(Ordering::SeqCst);
                if t < last_change_val + 250 {
                    Some(Angle(*input.0.load()))
                } else {
                    None
                }
            }),
    );

    let rod_b_angle = f64::to_radians(0.0);
    let rod_b = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(rod_a_pos + vector![1.0 * rod_b_angle.sin(), -1.0 * rod_b_angle.cos(), 0.0])
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                rod_a_angle,
            ))
            .trace(Vector3::new(0., -0.5, 0.))
            .intertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            .effector(|Time(_)| Force(vector![0.0, -9.8, 0.0]))
            .material(assets.material(Color::hex("FF9838").unwrap().into())),
    );

    builder.revolute_join(
        RevoluteJoint::new(rod_a, rod_b)
            .ang_damping(0.5)
            .pos_damping(0.5)
            .join_axis(Vector3::z_axis())
            .anchor_a(Pos(vector![0., -0.5, 0.0]))
            .anchor_b(Pos(vector![0., 0.5, 0.0]))
            .compliance(0.0),
    );
}
