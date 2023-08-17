use std::sync::{Arc, Mutex};

use nalgebra::{vector, Vector3};
use paracosm::{
    forces::gravity,
    sensor,
    xpbd::{builder::EntityBuilder, Xpbd},
    *,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec_stream = rerun::RecordingStreamBuilder::new("paracosm_orbit").save("./out.rrd")?;
    let points = Arc::new(Mutex::new(vec![]));
    let sensor_points = points.clone();

    {
        let mut xpbd = Xpbd::default();
        xpbd.entity(
            EntityBuilder::default()
                .mass(1.0)
                .pos(vector![1.0, 0.0, 0.0])
                .vel(vector![0.0, 1.0, 0.0])
                .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
                .effector(|Time(t)| {
                    if (9.42..10.0).contains(&t) {
                        Force(Vector3::new(0., 0.4, 0.5))
                    } else if (12.0..13.0).contains(&t) {
                        Force(Vector3::new(0.2, 0.0, 0.5))
                    } else {
                        Force(Vector3::zeros())
                    }
                })
                .sensor(sensor::rerun::time_pos_sensor(rec_stream.clone()))
                .sensor(sensor::rerun::vel_sensor(rec_stream.clone()))
                .sensor(sensor::rerun::pos_sensor(rec_stream.clone()))
                .sensor(move |Pos(p)| {
                    sensor_points
                        .lock()
                        .unwrap()
                        .push(rerun::components::Point3D::new(
                            p.x as f32, p.y as f32, p.z as f32,
                        ));
                }),
        );

        let mut time = 0;
        while time <= 60_000 {
            time += 1;
            xpbd.tick();
        }
    }

    rerun::MsgSender::new("central_body")
        .with_component(&[rerun::components::Point3D::new(0.0, 0.0, 0.0)])?
        .with_component(&[rerun::components::Radius(0.05)])?
        .with_timeless(true)
        .send(&rec_stream)?;
    let points = points.lock().unwrap();
    rerun::MsgSender::new("total_pos")
        .with_component(&points[..])?
        .with_timeless(true)
        .send(&rec_stream)?;

    Ok(())
}
