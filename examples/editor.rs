use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    forces::gravity,
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        editor::{editor, Input},
    },
    Force, Mass, Time, Torque,
};

fn main() {
    editor(sim)
}

fn sim(mut assets: Assets, input: Input) -> XpbdBuilder {
    XpbdBuilder::default().entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(Vector3::zeros())
            .effector(move |Time(_)| {
                let torque = *input.0.load();
                Torque(Vector3::new(0.0, torque, 0.0))
            })
            .mesh(assets.mesh(Mesh::from(shape::Cube { size: 0.5 })))
            .material(assets.material(Color::rgb(1.0, 0.5, 0.5).into())),
    )
}

// fn sim(mut assets: Assets) -> XpbdBuilder {
//     XpbdBuilder::default()
//         .entity(
//             EntityBuilder::default()
//                 .mass(1.0)
//                 .pos(Vector3::zeros())
//                 .effector(|Mass(m)| Force(m * Vector3::new(0.0, -9.8, 0.0)))
//                 .effector(|Mass(m), Time(t)| {
//                     if t < 0.4 {
//                         Force(m * Vector3::new(0.0, 25.0, 0.0))
//                     } else {
//                         Force::default()
//                     }
//                 })
//                 .mesh(assets.mesh(Mesh::from(shape::Cube { size: 0.5 })))
//                 .material(assets.material(Color::rgb(1.0, 0.5, 0.5).into())),
//         )
//         .entity(
//             EntityBuilder::default()
//                 .mass(1.0)
//                 .pos(vector![0.0, 0.0, -0.1])
//                 .mesh(assets.mesh(shape::Plane::from_size(5.0).into()))
//                 .material(assets.material(Color::rgb(0.3, 0.5, 0.3).into())),
//         )
// }

// fn orbit_sim(mut assets: Assets, input: Input) -> XpbdBuilder {
//     XpbdBuilder::default()
//         .entity(
//             EntityBuilder::default()
//                 .mass(1.0)
//                 .pos(vector![0.0, 0.0, 1.0])
//                 .vel(vector![1.0, 0.0, 0.0])
//                 .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
//                 .effector(move |Time(_)| Force(Vector3::new(0.0, 0.0, *input.0.load())))
//                 .mesh(assets.mesh(Mesh::from(shape::UVSphere {
//                     radius: 0.1,
//                     ..Default::default()
//                 })))
//                 .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
//         )
//         .entity(
//             EntityBuilder::default()
//                 .mass(1.0)
//                 .pos(Vector3::zeros())
//                 .mesh(assets.mesh(Mesh::from(shape::UVSphere {
//                     radius: 0.1,
//                     ..Default::default()
//                 })))
//                 .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
//         )
// }
