use nalgebra::{vector, Vector3};
use paracosm::{
    forces::gravity,
    sensor,
    xpbd::{Entity, Xpbd},
    *,
};
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec_stream = rerun::RecordingStreamBuilder::new("proso_orbit").save("./out.rrd")?;
    let mut xpbd = Xpbd::default();
    let points = Rc::new(RefCell::new(vec![]));
    let entity_points = points.clone();
    xpbd.entity(
        Entity::builder()
            .mass(1.0)
            .pos(vector![1.0, 0.0, 0.0])
            .vel(vector![0.0, 1.0, 0.0])
            .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
            .effector(|Time(t)| {
                if (9.42..10.0).contains(&t) {
                    Force(Vector3::new(0., 0.4, 0.5))
                } else {
                    Force(Vector3::zeros())
                }
            })
            .sensor(sensor::rerun::time_pos_sensor(rec_stream.clone()))
            .sensor(sensor::rerun::vel_sensor(rec_stream.clone()))
            .sensor(sensor::rerun::pos_sensor(rec_stream.clone()))
            .sensor(move |Pos(p)| {
                entity_points
                    .borrow_mut()
                    .push(rerun::components::Point3D::new(
                        p.x as f32, p.y as f32, p.z as f32,
                    ));
            }),
    );

    let mut time = 0;
    let dt = 0.001;
    while time <= 60_000 {
        xpbd.tick(dt);
        time += 1;
    }

    rerun::MsgSender::new("central_body")
        .with_component(&[rerun::components::Point3D::new(0.0, 0.0, 0.0)])?
        .with_component(&[rerun::components::Radius(0.05)])?
        .with_timeless(true)
        .send(&rec_stream)?;

    rerun::MsgSender::new("total_pos")
        .with_component(&points.borrow()[..])?
        .with_timeless(true)
        .send(&rec_stream)?;
    Ok(())
}
