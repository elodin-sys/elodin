use impeller2::types::{LenPacket, PacketId};
use impeller2_wkt::{AssetHandle, Glb, SetAsset, Stream, StreamBehavior};
use nox::{
    array::{Quaternion, SpatialTransform, Vector3},
    tensor,
};
use roci::{
    AsVTable, Componentize, Decomponentize, Metadatatize,
    tcp::{SinkExt, StreamExt},
};
use std::net::SocketAddr;
use stellarator::net::TcpStream;
use stellarator::{io::SplitExt, rent};
use zerocopy::{Immutable, IntoBytes};

#[derive(Componentize, Decomponentize, AsVTable, Default, Debug)]
#[roci(entity_id = 1)]
pub struct Input {
    pub mag: Vector3<f32>,
    pub accel: Vector3<f32>,
    pub gyro: Vector3<f32>,
}

#[derive(Componentize, AsVTable, Metadatatize, IntoBytes, Immutable, Debug)]
#[roci(entity_id = 1)]
pub struct Output {
    pub q_hat: Quaternion<f64>,
    pub b_hat: Vector3<f64>,
    pub world_pos: SpatialTransform<f64>,
    pub mag_cal: Vector3<f64>,
    #[roci(asset = true)]
    pub asset_handle_glb: AssetHandle<Glb>,
}

async fn connect() -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);
    let rx = impeller2_stella::PacketStream::new(rx);
    let id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.init_world::<Output>(id).await?;
    let mut sub = rx.subscribe::<Input>();
    let mut read = vec![0; 256];
    let mut mekf = roci_adcs::mekf::State::new(
        tensor![1., 1., 1.] * 0.008f64.to_radians(),
        tensor![1., 1., 1.] * 0.001f64,
        1.0 / 50.0,
    );
    let glb_id = fastrand::u64(..);
    tx.send(&SetAsset::new(
        glb_id,
        Glb("https://storage.googleapis.com/elodin-marketing/models/aleph.glb".to_string()),
    )?)
    .await
    .0?;
    tx.send(&Stream {
        behavior: StreamBehavior::RealTime,
        filter: Default::default(),
        id: fastrand::u64(..),
    })
    .await
    .0
    .unwrap();
    loop {
        let Some(input) = rent!(sub.next(read).await.unwrap(), read) else {
            continue;
        };
        mekf.omega = Vector3::from_buf(input.gyro.into_buf().map(|x| x.to_radians() as f64))
            * tensor![-1., -1., 1.];
        let accel =
            Vector3::from_buf(input.accel.into_buf().map(|x| x as f64)) * tensor![-1.0, -1.0, 1.0];
        let mag = Vector3::from_buf(input.mag.into_buf().map(|x| x as f64));
        let mag = mag - tensor![-10.5115, -14.5147, 4.7037];
        let mag = tensor![
            [1.2099, 0.0912, -0.1484],
            [0.0912, 1.0458, -0.0917],
            [-0.1484, -0.0917, 0.8200],
        ]
        .dot(&mag);
        let mag = mag * tensor![-1.0, 1.0, -1.0];
        mekf = mekf.estimate_attitude(
            [accel, mag],
            [
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(22.382, 5.157, -41.567),
            ],
            [160.0e-6, 350e-3],
        );
        let world_pos = nox::SpatialTransform::from_angular(mekf.q_hat);
        let mut table = LenPacket::table(id, 64);
        let output = Output {
            q_hat: mekf.q_hat,
            world_pos,
            asset_handle_glb: AssetHandle::new(glb_id),
            b_hat: mekf.b_hat,
            mag_cal: mag,
        };
        table.extend_from_slice(output.as_bytes());
        tx.send(table).await.0?;
    }
}

async fn run() -> anyhow::Result<()> {
    loop {
        if let Err(err) = connect().await {
            println!("error connecting {:?}", err);
            stellarator::sleep(std::time::Duration::from_millis(250)).await;
        }
    }
}

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}
