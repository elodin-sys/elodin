use impeller2::types::{EntityId, LenPacket, Msg, PacketId};
use impeller2_wkt::{MsgStream, SetEntityMetadata};
use roci::{AsVTable, Metadatatize, tcp::SinkExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use stellarator::io::{AsyncRead, AsyncWrite};
use stellarator::net::TcpStream;
use stellarator::rent;
use stellarator::{io::SplitExt, struc_con::Joinable};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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

#[derive(Serialize, Deserialize, postcard_schema::Schema, Debug, IntoBytes, Immutable)]
struct Command {
    gpios: [bool; 8],
}

pub async fn connect() -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);
    let mut rx = impeller2_stella::PacketStream::new(rx);

    let id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.send(&SetEntityMetadata::new(EntityId(1), "Aleph"))
        .await
        .0?;
    tx.init_world::<Record>(id).await?;
    tx.init_msg::<Command>().await?;
    tx.send(&MsgStream {
        msg_id: Command::ID,
    })
    .await
    .0?;
    let mut port = stellarator::serial::SerialPort::open("/dev/ttyTHS0").await?;
    port.set_baud(stellarator::serial::Baud::B115200)?;
    let (port_rx, port_tx) = port.split();

    let write = stellarator::struc_con::stellar::<anyhow::Result<()>, _, _>(move || async move {
        let mut buf = vec![0; 256];
        loop {
            let pkt = rx.next(buf).await?;
            match &pkt {
                impeller2::types::OwnedPacket::Msg(m) if m.id == Command::ID => {
                    let cmd = m.parse::<Command>()?;
                    println!("cmd {cmd:?}");
                    let buf = cobs::encode_vec(cmd.as_bytes());
                    println!("buf {buf:?}");
                    port_tx.write_all(buf).await.0?;
                }
                _ => {}
            }
            buf = pkt.into_buf().into_inner();
        }
    });
    let read = stellarator::struc_con::stellar(move || async move {
        let mut buf = vec![0u8; 512];
        let mut frame = impeller2_frame::FrameDecoder::<Vec<u8>>::default();
        loop {
            let n = rent!(port_rx.read(buf).await, buf)?;
            let data = &buf[..n];
            if let Some(decoded) = frame.push(data)? {
                let mut table = LenPacket::table(id, 64);
                if <Record>::ref_from_bytes(decoded).is_err() {
                    println!("failed to decode record");
                    continue;
                };
                table.extend_from_slice(decoded);
                tx.send(table).await.0?;
            }
        }
    });
    futures_lite::future::race(async { write.join().await.unwrap() }, async {
        read.join().await.unwrap()
    })
    .await
    .unwrap()
}
