use clap::Parser;
use impeller2::types::{LenPacket, PacketId};
use impeller2_stellar::Client;
use impeller2_wkt::{AssetHandle, Glb, SetAsset};
use mlua::LuaSerdeExt;
use nox::{
    array::{Mat3, Quat, SpatialTransform, Vec3},
    tensor,
};
use roci::{
    AsVTable, Componentize, Decomponentize, Metadatatize,
    tcp::{SinkExt, StreamExt},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};
use zerocopy::{Immutable, IntoBytes, KnownLayout, TryFromBytes};

#[derive(
    Componentize,
    Decomponentize,
    AsVTable,
    Default,
    Debug,
    Clone,
    TryFromBytes,
    Immutable,
    KnownLayout,
)]
#[roci(entity_id = 1)]
pub struct Input {
    pub mag: Vec3<f32>,
    pub accel: Vec3<f32>,
    pub gyro: Vec3<f32>,
}

#[derive(Componentize, AsVTable, Metadatatize, IntoBytes, Immutable, Debug)]
#[roci(entity_id = 1)]
pub struct Output {
    pub q_hat: Quat<f64>,
    pub b_hat: Vec3<f64>,
    pub gyro_est: Vec3<f64>,
    pub world_pos: SpatialTransform<f64>,
    pub mag_cal: Vec3<f64>,
    #[roci(asset = true)]
    pub asset_handle_glb: AssetHandle<Glb>,
}

async fn connect(config: &Config) -> anyhow::Result<()> {
    let mut client = Client::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let id: PacketId = fastrand::u16(..).to_le_bytes();
    client.init_world::<Output>(id).await?;
    let mut mekf = roci_adcs::mekf::State::new(
        tensor![1., 1., 1.] * config.mekf.gyro_sigma,
        tensor![1., 1., 1.] * config.mekf.gyro_bias_sigma,
        config.mekf.dt,
    );
    let glb_id = fastrand::u64(..);
    client
        .send(&SetAsset::new(glb_id, Glb(config.glb_url.clone()))?)
        .await
        .0?;
    let mut sub = client.subscribe::<Input>().await?;
    loop {
        let input = sub.next().await?;
        mekf.omega = Vec3::from_buf(input.gyro.into_buf().map(|x| x.to_radians() as f64))
            * tensor![-1., -1., 1.];
        let accel =
            Vec3::from_buf(input.accel.into_buf().map(|x| x as f64)) * tensor![-1.0, -1.0, 1.0];
        let mag = Vec3::from_buf(input.mag.into_buf().map(|x| x as f64));
        let mag = mag - config.mag_cal.b;
        let mag = config.mag_cal.a.dot(&mag);
        let mag = mag * tensor![-1.0, 1.0, -1.0];
        let mag = mag.normalize();
        let accel = accel.normalize();
        mekf = mekf.estimate_attitude(
            [accel, mag],
            [Vec3::new(0.0, 0.0, 1.0), config.mekf.mag_ref.normalize()],
            [config.mekf.accel_sigma, config.mekf.mag_sigma],
        );
        let world_pos = nox::SpatialTransform::from_angular(mekf.q_hat);
        let gyro_est = mekf.omega - mekf.b_hat;
        let mut table = LenPacket::table(id, 64);
        let output = Output {
            q_hat: mekf.q_hat,
            world_pos,
            asset_handle_glb: AssetHandle::new(glb_id),
            b_hat: mekf.b_hat,
            mag_cal: mag,
            gyro_est,
        };
        table.extend_from_slice(output.as_bytes());
        sub.send(table).await.0?;
    }
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
            dt: 1.0 / 50.0,
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
    "https://storage.googleapis.com/elodin-marketing/models/aleph.glb".to_string()
}

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}
