use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        editor::{editor, Input},
    },
    Time, Torque,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder, mut assets: Assets, input: Input) {
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0.0, 0.75, 0.0])
            .effector(move |Time(_)| {
                let torque = *input.0.load();
                Torque(Vector3::new(0.0, torque, 0.0))
            })
            .mesh(assets.mesh(Mesh::from(shape::Cube { size: 1.5 })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("#0085FF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })),
    );
}