use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    editor::{editor, Input},
    forces::gravity,
    Force, Pos, Time,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0.0, 0.0, 1.0])
            .vel(vector![1.0, 0.0, 0.0])
            .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
            .effector(|Time(t)| {
                if (9.42..10.0).contains(&t) {
                    Force(Vector3::new(0.0, -0.3, 0.5))
                } else {
                    Force(Vector3::zeros())
                }
            })
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(Color::rgba_u8(0x41, 0xBB, 0xFF, 0xFF).into())),
    );
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0.0, 1.0, 0.0])
            .vel(vector![0.0, 0.0, 1.0])
            .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
            .effector(move |Pos(pos)| Force(*input.0.load() * pos.normalize()))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(Color::BLUE.into())),
    );

    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );
}
