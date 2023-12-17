use bevy::prelude::{shape, Color, Mesh};
use bevy::ecs::system::Resource;
use nalgebra::{vector, Vector3};
use elodin::{
    forces::earth_gravity,
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        constraints::FixedJoint,
        editor::editor,
        editor::Editable,
        runner::IntoSimRunner,
    },
    Att, Force, Pos, SharedNum,
};
use elodin_macros::Editable;

fn main() {
    editor(sim.substep_count(1).delta_t(1.0 / 600.0))
}

fn sim(
    mut builder: XpbdBuilder<'_>,
    mut assets: Assets,
    motor_a_f: MotorForceA,
    motor_b_f: MotorForceB,
    motor_c_f: MotorForceC,
    motor_c_d: MotorForceD,
) {
    let drone_body = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.5, 0.2, 0.5))))
            .effector(earth_gravity)
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );
    let motor_a = builder.entity(
        EntityBuilder::default()
            .mass(0.25)
            .pos(vector![0.25, 0.0, 0.25])
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .effector(earth_gravity)
            .effector(move |Att(a)| Force(Vector3::y() * 9.8 * 2.0))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38C35A").unwrap(),
                ..Default::default()
            })),
    );

    builder
        .fixed_joint(FixedJoint::new(drone_body, motor_a).anchor_a(Pos(vector![0.25, 0.0, 0.25])));

    let motor_b = builder.entity(
        EntityBuilder::default()
            .mass(0.25)
            .pos(vector![-0.25, 0.0, 0.25])
            .trace(Vector3::zeros())
            //.effector(move |Att(a)| Force(Vector3::y() * 9.8 / 2.0))
            .effector(earth_gravity)
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("FFF").unwrap(),
                ..Default::default()
            })),
    );

    builder
        .fixed_joint(FixedJoint::new(drone_body, motor_b).anchor_a(Pos(vector![-0.25, 0.0, 0.25])));

    let motor_c = builder.entity(
        EntityBuilder::default()
            .mass(0.25)
            .pos(vector![0.25, 0.0, -0.25])
            .trace(Vector3::zeros())
            .effector(earth_gravity)
            //.effector(move |Att(a)| Force(Vector3::y() * 9.8 / 2.0))
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("FFF").unwrap(),
                ..Default::default()
            })),
    );

    builder
        .fixed_joint(FixedJoint::new(drone_body, motor_c).anchor_a(Pos(vector![0.25, 0.0, -0.25])));

    let motor_d = builder.entity(
        EntityBuilder::default()
            .mass(0.25)
            .pos(vector![-0.25, 0.0, -0.25])
            //.effector(move |Att(a)| Force(Vector3::y() * 9.8))
            .effector(earth_gravity)
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::Box::new(0.1, 0.1, 0.1))))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("FFF").unwrap(),
                ..Default::default()
            })),
    );

    builder.fixed_joint(
        FixedJoint::new(drone_body, motor_d).anchor_a(Pos(vector![-0.25, 0.0, -0.25])),
    );
}

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-10.0", range_max = 10.0, name = "motor a")]
pub struct MotorForceA(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-10.0", range_max = 10.0, name = "motor b")]
pub struct MotorForceB(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-10.0", range_max = 10.0, name = "motor c")]
pub struct MotorForceC(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-10.0", range_max = 10.0, name = "motor d")]
pub struct MotorForceD(pub SharedNum<f64>);
