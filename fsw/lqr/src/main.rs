use clap::Parser;
use impeller2::types::{LenPacket, PacketId};
use impeller2_wkt::{Stream, StreamBehavior};
use mlua::LuaSerdeExt;
use nox::{
    array::{Quat, Vec3},
    tensor,
};
use roci::{
    AsVTable, Componentize, Decomponentize, Metadatatize,
    tcp::{SinkExt, StreamExt},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};
use stellarator::net::TcpStream;
use stellarator::{io::SplitExt, rent};
use zerocopy::{Immutable, IntoBytes};

#[derive(Componentize, Decomponentize, AsVTable, Default, Debug)]
#[roci(entity_id = 1)]
pub struct Input {
    pub gyro_est: Vec3<f64>,
    pub q_hat: Quat<f64>,
    pub target_att: Quat<f64>,
}

#[derive(Componentize, AsVTable, Metadatatize, IntoBytes, Immutable, Debug)]
#[roci(entity_id = 1)]
pub struct Output {
    pub control_torque: Vec3<f64>,
}

async fn connect(config: &Config) -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stellar::PacketSink::new(tx);
    let rx = impeller2_stellar::PacketStream::new(rx);
    let id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.init_world::<Output>(id).await?;
    let mut sub = rx.subscribe::<Input>();
    let mut read = vec![0; 256];
    tx.send(&Stream {
        behavior: StreamBehavior::RealTime,
        filter: Default::default(),
        id: fastrand::u64(..),
    })
    .await
    .0
    .unwrap();
    let lqr = roci_adcs::yang_lqr::YangLQR::new(config.j, config.q_ang_vel, config.q_pos, config.r);
    loop {
        let Some(input) = rent!(sub.next(read).await.unwrap(), read) else {
            continue;
        };
        let mut table = LenPacket::table(id, 64);
        let control_torque = lqr.control(input.q_hat, input.gyro_est, input.target_att);
        let output = Output { control_torque };
        table.extend_from_slice(output.as_bytes());
        tx.send(table).await.0?;
    }
}

async fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut config = Config::default();
    if let Ok(script) = std::fs::read_to_string(args.config_path) {
        let lua = mlua::Lua::new();
        let config_val = lua.to_value(&config)?;
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
    j: Vec3<f64>,
    q_ang_vel: Vec3<f64>,
    q_pos: Vec3<f64>,
    r: Vec3<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            j: tensor![15204079.70002, 14621352.61765, 6237758.3131] * 1e-9,
            q_ang_vel: tensor![5.0, 5.0, 5.0],
            q_pos: tensor![5.0, 5.0, 5.0],
            r: tensor![8.0, 8.0, 8.0],
        }
    }
}

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(default_value = "/root/lqr.lua")]
    config_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}
