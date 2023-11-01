use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    builder::{EntityBuilder, FixedJoint, Revolute, SimBuilder},
    runner::IntoSimRunner,
    spatial::SpatialPos,
};
use paracosm_editor::editor;

fn main() {
    editor(sim.substep_count(10))
}

fn sim() -> SimBuilder {
    let mut builder = SimBuilder::default();
    let root = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            //.fixed()
            .mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            }))
            .material(Color::rgb(1.0, 0.0, 0.0).into())
            .body_pos(SpatialPos::linear(vector![0., 0.0, 0.0]))
            .joint(FixedJoint),
    );
    let rod_a = builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .body_pos(SpatialPos::linear(vector![0., -0.5, 0.0]))
            //.inertia(paracosm::Inertia::solid_box(0.2, 1.0, 0.2, 1.0))
            .mesh(Mesh::from(shape::Box::new(0.2, 1.0, 0.2)))
            .material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("38ACFF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.1,
                ..Default::default()
            })
            .parent(root)
            .joint(Revolute::new(Vector3::z_axis()).pos(-90f64.to_radians())),
    );

    builder
}
