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
    let craft = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .intertia(paracosm::Inertia::solid_box(0.2, 0.2, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 0.2, 0.2))))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })),
    );
    let wheel = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .mesh(assets.mesh(Mesh::from(shape::Cylinder {
                radius: 0.05,
                height: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })),
    );
    builder.revolute_join(
        RevoluteJoint::new(craft, wheel)
            .join_axis(Vector3::z_axis())
            .anchor_a(Pos(vector![0., 0.0, 0.2]))
            .compliance(0.0),
    );
}
