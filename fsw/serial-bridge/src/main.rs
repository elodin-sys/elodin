use std::{net::SocketAddr, time::Duration};

use impeller2::types::{LenPacket, MsgExt};
use impeller2::{
    table::VTableBuilder,
    types::{EntityId, PrimType},
};
use impeller2_wkt::VTableMsg;
use stellarator::io::SplitExt;
use stellarator::net::TcpStream;

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

async fn connect() -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (_, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);

    let mut vtable: VTableBuilder<Vec<_>, Vec<_>> = VTableBuilder::default();
    vtable.column("ts", PrimType::U32, [1], [EntityId(0)])?;
    vtable.column("mag", PrimType::F32, [3], [EntityId(0)])?;
    vtable.column("gyro", PrimType::F32, [3], [EntityId(0)])?;
    vtable.column("accel", PrimType::F32, [3], [EntityId(0)])?;
    vtable.column("mag_temp", PrimType::F32, [], [EntityId(0)])?;
    vtable.column("mag_sample", PrimType::U32, [], [EntityId(0)])?;
    vtable.column("baro", PrimType::F32, [], [EntityId(0)])?;

    let vtable = vtable.build();
    let id: [u8; 3] = fastrand::u64(..).to_le_bytes()[..3]
        .try_into()
        .expect("id wrong size");
    let msg = VTableMsg { id, vtable };
    tx.send(msg.to_len_packet()).await.0?;
    let mut port = serialport::new("/dev/ttyACM0", 115200)
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
                    let mut table = LenPacket::table(id, 16);
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
