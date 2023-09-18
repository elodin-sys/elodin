use nalgebra::Vector3;
use paracosm::{six_dof::SixDof, *};
use rerun::{components::Vec3D, time::Timeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec_stream = rerun::RecordingStreamBuilder::new("proso_orbit").save("./out.rrd")?;

    let mut six_dof = SixDof::default()
        .pos(Vector3::new(1.0, 0., 0.))
        .vel(Vector3::new(0.0, 1.0, 0.0))
        .mass(1.0)
        .sim()
        .effector(forces::gravity(1.0 / 6.649e-11, Vector3::zeros()))
        .effector(|Time(t)| {
            if (9.42..10.0).contains(&t) {
                Force(Vector3::new(0.0, -0.3, 0.5))
            } else {
                Force(Vector3::zeros())
            }
        });
    let mut time = 0;
    let mut points = vec![];
    let dt = 0.001;
    while time <= 60_000 {
        six_dof.tick(dt);
        rerun::MsgSender::new("vel")
            .with_component(&[rerun::components::Arrow3D {
                origin: Vec3D::new(
                    six_dof.state.pos.x as f32,
                    six_dof.state.pos.y as f32,
                    six_dof.state.pos.z as f32,
                ),
                vector: Vec3D::new(
                    six_dof.state.vel.x as f32,
                    six_dof.state.vel.y as f32,
                    six_dof.state.vel.z as f32,
                ),
            }])?
            .with_time(Timeline::default(), time)
            .send(&rec_stream)?;

        rerun::MsgSender::new("pos")
            .with_component(&[rerun::components::Point3D::new(
                six_dof.state.pos.x as f32,
                six_dof.state.pos.y as f32,
                six_dof.state.pos.z as f32,
            )])?
            .with_component(&[rerun::components::ColorRGBA::from_rgb(0, 0, 0)])?
            .with_component(&[rerun::components::Radius(0.05)])?
            .with_time(Timeline::default(), time)
            .send(&rec_stream)?;
        points.push(rerun::components::Point3D::new(
            six_dof.state.pos.x as f32,
            six_dof.state.pos.y as f32,
            six_dof.state.pos.z as f32,
        ));
        time += 1;
    }

    rerun::MsgSender::new("total_pos")
        .with_component(&points)?
        .with_timeless(true)
        .send(&rec_stream)?;
    rerun::MsgSender::new("central_body")
        .with_component(&[rerun::components::Point3D::new(0.0, 0.0, 0.0)])?
        .with_component(&[rerun::components::Radius(0.1)])?
        .with_timeless(true)
        .send(&rec_stream)?;

    Ok(())
}
