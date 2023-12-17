use std::f64::consts::PI;

use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use elodin::{
    builder::{Assets, EntityBuilder, Free, XpbdBuilder},
    constraints::GravityConstraint,
    editor::editor,
    spatial::{SpatialMotion, SpatialPos},
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let a = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .joint(
                Free::default()
                    .pos(SpatialPos::linear(vector![1.0, 0.0, 0.0]))
                    .vel(SpatialMotion::linear(
                        0.55 * vector![(0.5 * PI).cos(), (0.5 * PI).sin(), 0.0],
                    )),
            )
            .trace(Vector3::zeros())
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
    let b = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .joint(
                Free::default()
                    .pos(SpatialPos::linear(vector![
                        (2. / 3. * PI).cos(),
                        (2. / 3. * PI).sin(),
                        0.0
                    ]))
                    .vel(SpatialMotion::linear(
                        0.55 * vector![(1.166 * PI).cos(), (1.166 * PI).sin(), 0.0],
                    )),
            )
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let c = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .joint(
                Free::default()
                    .pos(SpatialPos::linear(vector![
                        (4. / 3. * PI).cos(),
                        (4. / 3. * PI).sin(),
                        0.0
                    ]))
                    .vel(SpatialMotion::linear(
                        0.55 * vector![(1.833 * PI).cos(), (1.833 * PI).sin(), 0.0],
                    )),
            )
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 15.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    builder.gravity_constraint(GravityConstraint::new(a, b));
    builder.gravity_constraint(GravityConstraint::new(a, c));
    builder.gravity_constraint(GravityConstraint::new(b, c));
}
