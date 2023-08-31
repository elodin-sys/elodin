use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, UnitQuaternion, Vector3};
use paracosm::{
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        constraints::RevoluteJoint,
        editor::editor,
    },
    Force, Pos, Time,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let root = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            .fixed()
            .pos(vector![0.0, 0.0, 0.0])
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
    );
    let rod_a_angle = f64::to_radians(5.0);
    let rod_a_pos = vector![0.5 * rod_a_angle.sin(), -0.5 * rod_a_angle.cos(), 0.0];
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
            .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
    );
    builder.revolute_join(
        RevoluteJoint::new(root, rod_a)
            .join_axis(Vector3::z_axis())
            .anchor_b(Pos(vector![0., 0.5, 0.0]))
            .compliance(0.0000),
    );

    let rod_b_angle = f64::to_radians(0.0);
    let rod_b = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(rod_a_pos + vector![0.5 * rod_b_angle.sin(), -0.5 * rod_b_angle.cos(), 0.0])
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                rod_a_angle,
            ))
            .intertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            .effector(|Time(_)| Force(vector![0.0, -9.8, 0.0]))
            .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
    );

    builder.revolute_join(
        RevoluteJoint::new(rod_a, rod_b)
            .join_axis(Vector3::z_axis())
            .anchor_a(Pos(vector![0., -0.5, 0.0]))
            .anchor_b(Pos(vector![0., 0.5, 0.0]))
            .compliance(0.0000),
    );
}
