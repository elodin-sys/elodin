use clap::Parser;
use db_macros::{AsVTable, Metadatatize};
use impeller::types::{LenPacket, PacketId, Timestamp};
use impeller_stellar::Client;
use impeller_stellar::{SinkExt, StreamExt};
use mlua::LuaSerdeExt;
use nox::{
    array::{Mat3, Quat, SpatialTransform, Vec3},
    tensor,
};
use roci_adcs::mekf::State;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};
use zerocopy::{Immutable, IntoBytes, KnownLayout, TryFromBytes};

#[derive(AsVTable, Default, Debug, Clone, TryFromBytes, Immutable, KnownLayout)]
#[db(parent = "imu")]
pub struct Input {
    pub accel: Vec3<f32>,
    pub gyro: Vec3<f32>,
    pub mag: Vec3<f32>,
}

#[derive(AsVTable, Metadatatize, IntoBytes, Immutable, Debug)]
#[db(parent = "aleph")]
#[repr(C)]
pub struct Output {
    #[db(timestamp)]
    pub time: i64,
    pub q_hat: Quat<f64>,
    pub b_hat: Vec3<f64>,
    pub gyro_est: Vec3<f64>,
    pub world_pos: SpatialTransform<f64>,
    pub mag_cal: Vec3<f64>,
}

async fn connect(config: &Config) -> anyhow::Result<()> {
    let mut client = Client::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let id: PacketId = fastrand::u16(..).to_le_bytes();
    client.init_world::<Output>(id).await?;
    let mut mekf = State::new(
        tensor![1., 1., 1.] * config.mekf.gyro_sigma,
        tensor![1., 1., 1.] * config.mekf.gyro_bias_sigma,
        config.mekf.dt,
    );
    let mut sub = client.subscribe::<Input>().await?;
    loop {
        let input = sub.next().await?;
        let (next, output) = step(mekf, input, config, Timestamp::now().0);
        mekf = next;
        let mut table = LenPacket::table(id, core::mem::size_of::<Output>());
        table.extend_from_slice(output.as_bytes());
        sub.send(table).await.0?;
    }
}

fn step(mut mekf: State, input: Input, config: &Config, time: i64) -> (State, Output) {
    let gyro = Vec3::from_buf(input.gyro.into_buf().map(|x| x.to_radians() as f64));
    // a non-finite gyro sample is replaced with b_hat: a zero corrected rate, so the attitude holds
    let gyro_finite = gyro.into_buf().iter().all(|x: &f64| x.is_finite());
    mekf.omega = if gyro_finite { gyro } else { mekf.b_hat };
    let accel = Vec3::from_buf(input.accel.into_buf().map(|x| x as f64));
    let mag = Vec3::from_buf(input.mag.into_buf().map(|x| x as f64));
    // sensor-fw sends a zero mag when the BMM350 is missing
    let mag_valid = unit(mag).is_some();
    let mag = mag - config.mag_cal.b;
    let mag = config.mag_cal.a.dot(&mag);
    let mag = mag * tensor![-1.0, 1.0, -1.0];
    let mag = unit(mag).filter(|_| mag_valid);
    let accel = unit(accel);
    let accel_ref = Vec3::new(0.0, 0.0, 1.0);
    let mag_ref = config.mekf.mag_ref.normalize();
    let (accel_sigma, mag_sigma) = (config.mekf.accel_sigma, config.mekf.mag_sigma);
    mekf = match (accel, mag) {
        (Some(accel), Some(mag)) => {
            mekf.estimate_attitude([accel, mag], [accel_ref, mag_ref], [accel_sigma, mag_sigma])
        }
        (Some(accel), None) => mekf.estimate_attitude([accel], [accel_ref], [accel_sigma]),
        (None, Some(mag)) => mekf.estimate_attitude([mag], [mag_ref], [mag_sigma]),
        (None, None) => mekf.estimate_attitude([], [], []),
    };
    mekf.reset_if_invalid();
    let world_pos = nox::SpatialTransform::from_angular(mekf.q_hat);
    // estimate_attitude already stores omega - b_hat in omega
    let gyro_est = mekf.omega;
    let output = Output {
        time,
        q_hat: mekf.q_hat,
        world_pos,
        b_hat: mekf.b_hat,
        mag_cal: mag.unwrap_or_default(),
        gyro_est,
    };
    (mekf, output)
}

fn unit(v: Vec3<f64>) -> Option<Vec3<f64>> {
    v.norm().into_buf().is_normal().then(|| v.normalize())
}

async fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut config = Config::default();
    if let Ok(script) = std::fs::read_to_string(args.config_path) {
        let lua = mlua::Lua::new();
        let config_val = lua.to_value(&config)?;
        lua.globals().set(
            "wmm",
            lua.create_function(|lua, (lat, long, alt): (f64, f64, f64)| {
                let mut model = wmm::MagneticModel::default();
                let epoch = wmm::Epoch::now().unwrap();
                let geodetic = wmm::GeodeticCoords::with_geoid_height(lat, long, alt);
                let (elements, _) = model.calculate_field(epoch, geodetic);
                let field = Vec3::new(elements.x, elements.y, elements.z) * 1.0e-3;
                lua.to_value(&field)
            })?,
        )?;
        lua.globals().set("config", config_val)?;
        lua.load(script).exec()?;
        config = lua.from_value(lua.globals().get("config")?)?;
    };
    println!("{config:#?}");
    loop {
        if let Err(err) = connect(&config).await {
            println!("error connecting {:?}", err);
            stellarator::sleep(std::time::Duration::from_millis(250)).await;
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    #[serde(default = "default_glb")]
    glb_url: String,
    #[serde(default)]
    mag_cal: MagCal,
    #[serde(default)]
    mekf: Mekf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            glb_url: default_glb(),
            mag_cal: Default::default(),
            mekf: Default::default(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct MagCal {
    a: Mat3<f64>,
    b: Vec3<f64>,
}

impl Default for MagCal {
    fn default() -> Self {
        Self {
            a: Mat3::eye(),
            b: tensor![0., 0., 0.],
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Mekf {
    accel_sigma: f64,
    mag_sigma: f64,
    gyro_sigma: f64,
    gyro_bias_sigma: f64,
    dt: f64,
    mag_ref: Vec3<f64>,
}

impl Default for Mekf {
    fn default() -> Self {
        Self {
            accel_sigma: 160.0e-6,
            mag_sigma: 3e-4,
            gyro_sigma: 0.008f64.to_radians(),
            gyro_bias_sigma: 0.001f64,
            dt: 1.0 / 800.0,
            mag_ref: Vec3::new(22.382, 5.157, -41.567),
        }
    }
}

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(default_value = "/root/mekf.lua")]
    config_path: PathBuf,
}

fn default_glb() -> String {
    "https://assets.elodin.systems/assets/aleph.glb".to_string()
}

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state(config: &Config) -> State {
        State::new(
            tensor![1., 1., 1.] * config.mekf.gyro_sigma,
            tensor![1., 1., 1.] * config.mekf.gyro_bias_sigma,
            config.mekf.dt,
        )
    }

    // level vehicle with body axes aligned to the reference frame (identity mag_cal)
    fn level_input(config: &Config, gyro_rad: [f64; 3]) -> Input {
        let mag = config.mekf.mag_ref.normalize() * tensor![-1.0, 1.0, -1.0];
        Input {
            accel: Vec3::from_buf([0.0, 0.0, 9.81]),
            gyro: Vec3::from_buf(gyro_rad.map(|x| x.to_degrees() as f32)),
            mag: Vec3::from_buf(mag.into_buf().map(|x| x as f32)),
        }
    }

    fn att_err(q: &Quat<f64>) -> f64 {
        let [_, _, _, w] = q.0.into_buf();
        2.0 * w.abs().min(1.0).acos()
    }

    fn assert_finite(output: &Output) {
        let q = output.q_hat.0.into_buf();
        let gyro_est = output.gyro_est.into_buf();
        assert!(
            q.iter().chain(gyro_est.iter()).all(|x| x.is_finite()),
            "non-finite output: q_hat = {q:?}, gyro_est = {gyro_est:?}"
        );
    }

    #[test]
    fn gyro_est_subtracts_bias_once() {
        let config = Config::default();
        let bias = [0.02, -0.03, 0.05];
        let mut mekf = new_state(&config);
        mekf.b_hat = Vec3::from_buf(bias);
        // stationary vehicle: the raw gyro reads only the bias
        let (_, output) = step(mekf, level_input(&config, bias), &config, 0);
        let gyro_est = output.gyro_est.into_buf();
        assert!(
            gyro_est.iter().all(|x| x.abs() < 1e-6),
            "gyro_est = {gyro_est:?}, expected ~0"
        );
    }

    #[test]
    fn missing_mag_falls_back_to_accel() {
        let config = Config::default();
        let q = Quat::from_axis_angle(Vec3::x_axis(), 0.3);
        let down = Vec3::new(0.0, 0.0, 1.0);
        let accel = q.inverse() * down;
        let input = Input {
            accel: Vec3::from_buf((accel * 9.81).into_buf().map(|x| x as f32)),
            gyro: Vec3::from_buf([0.0; 3]),
            // sensor-fw sends a zero mag when the BMM350 is missing
            mag: Vec3::from_buf([0.0; 3]),
        };
        let mut mekf = new_state(&config);
        for _ in 0..800 {
            let (next, output) = step(mekf, input.clone(), &config, 0);
            mekf = next;
            assert_finite(&output);
        }
        let err = (mekf.q_hat.inverse() * down - accel).norm().into_buf();
        assert!(err < 1e-3, "gravity direction error {err}");
    }

    #[test]
    fn missing_mag_ignored_with_offset_cal() {
        let mut config = Config::default();
        config.mag_cal.b = tensor![0.1, -0.2, 0.3];
        let mut mekf = new_state(&config);
        let mut input = level_input(&config, [0.0; 3]);
        input.mag = Vec3::from_buf([0.0; 3]);
        for _ in 0..800 {
            let (next, output) = step(mekf, input.clone(), &config, 0);
            mekf = next;
            assert_finite(&output);
        }
        // a zero mag calibrates to -A b, which must not be fused as a field reading
        let err = att_err(&mekf.q_hat);
        assert!(err < 1e-3, "attitude error {err}");
    }

    #[test]
    fn non_finite_gyro_sample_is_skipped() {
        let config = Config::default();
        let bias = [0.02, -0.03, 0.05];
        let mut mekf = new_state(&config);
        mekf.b_hat = Vec3::from_buf(bias);
        let mut input = level_input(&config, bias);
        input.gyro = Vec3::from_buf([f32::NAN, f32::INFINITY, 0.0]);
        let (mekf, output) = step(mekf, input, &config, 0);
        assert_finite(&output);
        assert!(att_err(&mekf.q_hat) < 1e-3);
        let b = mekf.b_hat.into_buf();
        // the learned bias survives instead of being reset to zero
        assert!(
            b.iter()
                .zip(bias)
                .all(|(b, want): (&f64, f64)| (b - want).abs() < 1e-3),
            "b_hat = {b:?}"
        );
    }

    #[test]
    fn non_finite_state_recovers() {
        let config = Config::default();
        let mut mekf = new_state(&config);
        mekf.b_hat = Vec3::from_buf([f64::NAN, 0.0, 0.0]);
        for _ in 0..100 {
            let (next, output) = step(mekf, level_input(&config, [0.0; 3]), &config, 0);
            mekf = next;
            assert_finite(&output);
        }
        assert!(att_err(&mekf.q_hat) < 1e-3);
    }
}
