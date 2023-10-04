use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::xpbd::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    constraints::GravityConstraint,
    editor::editor,
    runner::IntoSimRunner,
};

fn main() {
    editor(sim.substep_count(64).scale(2.).delta_t(100. / 60.))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let sun = builder.entity(
        EntityBuilder::default()
            .fixed()
            .mass(1.99e30)
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
    let earth = builder.entity(
        EntityBuilder::default()
            .mass(5.972e24)
            .pos(helio_to_bevy(vector![
                9.932794033922092e-1,
                -8.115895094412964e-2,
                2.123378844575767e-4,
            ]))
            .vel(helio_to_bevy(vector![
                1.072_468_396_698_36e-3,
                1.708345322037895e-2,
                -3.545157932769615e-7
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let jupiter = builder.entity(
        EntityBuilder::default()
            .mass(1.89e27)
            .pos(helio_to_bevy(vector![
                4.006_423_068_610_66,
                2.921_866_864_588_122,
                -1.017_570_096_026_684e-1
            ]))
            .vel(helio_to_bevy(vector![
                -4.529_782_018_388_927e-3,
                6.452_946_144_697_471e-3,
                7.457_109_084_988_2e-5
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 145.0 / 255.0 * 20., 44.0 / 255.0 * 20.),
                base_color: Color::hex("FF912C").unwrap(),
                ..Default::default()
            })),
    );

    let mars = builder.entity(
        EntityBuilder::default()
            .mass(6.39e23)
            .pos(helio_to_bevy(vector![
                -1.455_093_533_090_44,
                -7.026_325_579_976_354e-1,
                2.101_911_966_287_818e-2,
            ]))
            .vel(helio_to_bevy(vector![
                6.625_189_920_311_338e-3,
                -1.140_793_077_770_015e-2,
                -4.013_568_238_263_378e-4
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 15.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let mercury = builder.entity(
        EntityBuilder::default()
            .mass(4.867e24)
            .pos(helio_to_bevy(vector![
                2.028_847_101_112_924e-1,
                2.308_731_142_232_934e-1,
                -1.276_223_148_323_567e-4
            ]))
            .vel(helio_to_bevy(vector![
                -2.640_829_017_559_176e-2,
                2.006_600_272_464_892e-2,
                4.063_076_271_928_278e-3
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let venus = builder.entity(
        EntityBuilder::default()
            .mass(4.867e24)
            .pos(helio_to_bevy(vector![
                6.798_017_112_259_706e-1,
                2.237_111_260_436_163e-1,
                -3.639_561_594_712_085e-2
            ]))
            .vel(helio_to_bevy(vector![
                -6.371_222_592_894_96e-3,
                1.912_020_851_757_406e-2,
                6.305_071_804_394_223e-4
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let saturn = builder.entity(
        EntityBuilder::default()
            .mass(5.683e26)
            .pos(helio_to_bevy(vector![
                8.783_226_102_061_334,
                -4.247_296_553_985_379,
                -2.758_526_575_085_964e-1
            ]))
            .vel(helio_to_bevy(vector![
                2.116_571_054_761_369e-3,
                5.012_229_319_169_293e-3,
                -1.716_983_622_031_672e-4,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let neptune = builder.entity(
        EntityBuilder::default()
            .mass(1.024e26)
            .pos(helio_to_bevy(vector![
                2.981_579_078_855_878e1,
                -2.121_513_644_468_658,
                -6.434_483_800_948_522e-1,
            ]))
            .vel(helio_to_bevy(vector![
                2.021_264_053_090_83e-4,
                3.149_741_242_740_432e-3,
                -6.967_831_155_254_172e-5,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let uranus = builder.entity(
        EntityBuilder::default()
            .mass(1.024e26)
            .pos(helio_to_bevy(vector![
                1.258_255_102_521_142e1,
                1.505_737_113_095_83e1,
                -1.070_858_413_350_803e-1,
            ]))
            .vel(helio_to_bevy(vector![
                -3.046_918_944_187_991e-3,
                2.338_765_156_267_45e-3,
                4.819_887_062_168_329e-5,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    builder.gravity_constraint(GravityConstraint::new(earth, sun).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(earth, jupiter).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, jupiter).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, mars).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, venus).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, saturn).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, neptune).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, mercury).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, uranus).constant(1.4883e-34));
}

fn helio_to_bevy(vec: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(vec.x, vec.z, vec.y)
}
