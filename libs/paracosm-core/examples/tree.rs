use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, UnitQuaternion, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    editor::{editor, Input},
    runner::IntoSimRunner,
    tree::{Joint, JointType},
    BodyPos, Torque,
};

fn main() {
    editor(sim.substep_count(32))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    let rod_a_angle = f64::to_radians(45.0);
    let rod_a = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            //.ang_vel(1.0 * Vector3::y())
            .effector(move |BodyPos(p)| Torque(p.att * (*input.0.load() * Vector3::z())))
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                rod_a_angle,
            ))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            //.effector(|Time(_)| Force(vector![0.0, -9.8, 0.0]))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })),
    );
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0., -0.5, 0.0])
            .att(UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                -25.0f64.to_radians(),
            ))
            .trace(Vector3::new(0., -0.5, 0.))
            .inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2))))
            //.effector(|Time(_)| Force(vector![0.0, -9.8, 0.0]))
            .material(assets.material(Color::hex("FF9838").unwrap().into()))
            .parent(
                rod_a,
                Joint {
                    pos: vector![0., -0.5, 0.0],
                    //joint_type: JointType::Sphere,
                    joint_type: JointType::Revolute {
                        axis: Vector3::z_axis(),
                    },
                },
            ),
    );
}
