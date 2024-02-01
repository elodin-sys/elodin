use bevy::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};

use nalgebra::{vector, Vector3};

use elodin_core::{
    forces::gravity,
    runtime::JobSpec,
    xpbd::{
        builder::{EntityBuilder, XpbdBuilder},
        constraints::DistanceConstraint,
        runner::{IntoSimRunner, RunMode},
    },
    Force, Time,
};

fn criterion_benchmark(c: &mut Criterion) {
    fn grav_sim(mut builder: XpbdBuilder<'_>) {
        builder.entity(
            EntityBuilder::default()
                .mass(1.0)
                .pos(vector![0.0, 0.0, 1.0])
                .vel(vector![1.0, 0.0, 0.0])
                .effector(gravity(1.0 / 6.649e-11, Vector3::zeros())),
        );
    }

    fn sim(mut builder: XpbdBuilder<'_>) {
        builder.entity(
            EntityBuilder::default()
                .mass(1.0)
                .pos(vector![0.0, 0.0, 1.0])
                .vel(vector![1.0, 0.0, 0.0]),
        );
    }

    c.bench_function("one_body_gravity_1000", |b| {
        b.iter(|| {
            JobSpec::default()
                .sim(grav_sim.run_mode(RunMode::FixedTicks(500)))
                .run()
        })
    });

    c.bench_function("one_body_1000", |b| {
        b.iter(|| {
            JobSpec::default()
                .sim(sim.run_mode(RunMode::FixedTicks(500)))
                .run()
        })
    });

    fn chain(mut builder: XpbdBuilder<'_>) {
        let mut previous_link = builder.entity(
            EntityBuilder::default()
                .mass(10.0)
                .fixed()
                .pos(vector![0.0, 0.0, 0.0]),
        );
        for i in 1..=20 {
            let link = builder.entity(
                EntityBuilder::default()
                    .mass(1.0)
                    .pos(vector![(i as f64) / 4.0, 0.0, 0.0])
                    .effector(|Time(_)| Force(vector![0.0, -5.0, 0.0])),
            );
            builder.distance_constraint(
                DistanceConstraint::new(link, previous_link)
                    .distance_target(0.25)
                    .compliance(0.0),
            );
            previous_link = link
        }
    }

    c.bench_function("chain_500", |b| {
        b.iter(|| {
            JobSpec::default()
                .sim(chain.run_mode(RunMode::FixedTicks(500)))
                .run()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
