use impeller2::types::{EntityId, LenPacket, PacketId};
use impeller2_wkt::SetEntityMetadata;
use roci::{AsVTable, Metadatatize, tcp::SinkExt};
use std::{net::SocketAddr, time::Duration};
use stellarator::io::SplitExt;
use stellarator::net::TcpStream;
use zerocopy::{FromBytes, Immutable, KnownLayout};

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}

async fn run() -> anyhow::Result<()> {
    loop {
        if let Err(err) = connect().await {
            println!("error connecting {:?}", err);
        }
    }
}

#[derive(AsVTable, Metadatatize, FromBytes, Immutable, KnownLayout, Debug)]
#[roci(entity_id = 1)]
#[repr(C)]
pub struct Record {
    pub ts: u32, // in milliseconds
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
    pub baro: f32,
    pub baro_temp: f32,
}

async fn connect() -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (_, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);

    let id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.send(&SetEntityMetadata::new(EntityId(1), "Aleph"))
        .await
        .0?;
    tx.init_world::<Record>(id).await?;
    let mut port = serialport::new("/dev/ttyTHS0", 115200)
        .timeout(Duration::MAX)
        .open()?;
    let mut read = vec![0; 256];
    while let Ok(n) = port.read(&mut read) {
        let mut i = 0;
        let buf = &read[..n];
        'decode: loop {
            match buf.get(i) {
                Some(0) => {
                    let Ok(decode) = cobs::decode_vec(&buf[(i + 1)..]) else {
                        break 'decode;
                    };
                    i += decode.len();
                    let mut table = LenPacket::table(id, 64);
                    if <Record>::ref_from_bytes(&decode).is_err() {
                        println!("failed to decode record");
                        continue;
                    };
                    table.extend_from_slice(&decode);
                    tx.send(table).await.0?;
                }
                Some(_) => {
                    i += 1;
                }
                None => {
                    break 'decode;
                }
            }
        }
    }
    Ok(())
}
