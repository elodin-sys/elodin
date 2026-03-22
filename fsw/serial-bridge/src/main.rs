use anyhow::Context;
use impeller2::types::{LenPacket, Msg, PacketId, msg_id};
use impeller2_wkt::{
    MsgMetadata, MsgStream, SetComponentMetadata, SetMsgMetadata, log_entry_msg_schema,
};
use roci::tcp::SinkExt;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, time::Duration};
use stellarator::io::{AsyncRead, AsyncWrite};
use stellarator::net::TcpStream;
use stellarator::rent;
use stellarator::{io::SplitExt, struc_con::Joinable};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use blackbox::Record;

const LOG_STREAM_NAME: &str = "aleph.c-blinky.log";
const LOG_STREAM_ID: PacketId = msg_id(LOG_STREAM_NAME);
const LOG_FRAME_PREFIX_LEN: usize = 5;
const LOG_FRAME_MAGIC: [u8; 2] = *b"EL";
const LOG_FRAME_VERSION: u8 = 1;
const LOG_FRAME_KIND_LOG: u8 = 1;
const DEFAULT_SERIAL_PORT: &str = "/dev/ttyTHS1";
const DEFAULT_SERIAL_BAUD: u64 = 115_200;

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}

async fn run() -> anyhow::Result<()> {
    loop {
        if let Err(err) = connect().await {
            println!("error connecting {:?}", err);
            stellarator::sleep(Duration::from_millis(250)).await;
        }
    }
}

#[derive(Serialize, Deserialize, postcard_schema::Schema, Debug, IntoBytes, Immutable)]
struct Command {
    gpios: [bool; 8],
}

#[derive(Serialize)]
struct CblinkyLogEntry {
    level: u8,
    message: String,
}

impl Msg for CblinkyLogEntry {
    const ID: PacketId = LOG_STREAM_ID;
}

enum BridgeFrame<'a> {
    LegacyRecord(&'a [u8]),
    Log { level: u8, message: &'a str },
}

fn parse_bridge_frame(decoded: &[u8]) -> Option<BridgeFrame<'_>> {
    if decoded.len() >= LOG_FRAME_PREFIX_LEN
        && decoded[0] == LOG_FRAME_MAGIC[0]
        && decoded[1] == LOG_FRAME_MAGIC[1]
        && decoded[2] == LOG_FRAME_VERSION
        && decoded[3] == LOG_FRAME_KIND_LOG
    {
        let message = std::str::from_utf8(&decoded[LOG_FRAME_PREFIX_LEN..]).ok()?;
        return Some(BridgeFrame::Log {
            level: decoded[4],
            message,
        });
    }

    if Record::ref_from_bytes(decoded).is_ok() {
        Some(BridgeFrame::LegacyRecord(decoded))
    } else {
        None
    }
}

fn serial_port() -> String {
    env::var("ALEPH_SERIAL_BRIDGE_PORT").unwrap_or_else(|_| DEFAULT_SERIAL_PORT.to_string())
}

fn serial_baud() -> anyhow::Result<u64> {
    match env::var("ALEPH_SERIAL_BRIDGE_BAUD") {
        Ok(value) => value
            .parse()
            .with_context(|| format!("failed to parse ALEPH_SERIAL_BRIDGE_BAUD={value}")),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_SERIAL_BAUD),
        Err(err) => Err(err).context("failed to read ALEPH_SERIAL_BRIDGE_BAUD"),
    }
}

pub async fn connect() -> anyhow::Result<()> {
    let serial_port = serial_port();
    let serial_baud = serial_baud()?;

    println!("opening serial bridge on {serial_port} @ {serial_baud}");

    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stellar::PacketSink::new(tx);
    let mut rx = impeller2_stellar::PacketStream::new(rx);

    let id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.send(&SetComponentMetadata::new("aleph", "aleph"))
        .await
        .0?;
    tx.init_world::<Record>(id).await?;
    tx.send(&SetMsgMetadata {
        id: LOG_STREAM_ID,
        metadata: MsgMetadata {
            name: LOG_STREAM_NAME.to_string(),
            schema: log_entry_msg_schema(),
            metadata: Default::default(),
        },
    })
    .await
    .0?;
    tx.init_msg::<Command>().await?;
    tx.send(&MsgStream {
        msg_id: Command::ID,
    })
    .await
    .0?;
    let mut port = stellarator::serial::SerialPort::open(&serial_port).await?;
    port.set_baud(stellarator::serial::Baud::Other(serial_baud))?;
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
            match frame.push(data) {
                Ok(Some(decoded)) => match parse_bridge_frame(decoded) {
                    Some(BridgeFrame::LegacyRecord(record)) => {
                        let mut table = LenPacket::table(id, 64);
                        table.extend_from_slice(record);
                        tx.send(table).await.0?;
                    }
                    Some(BridgeFrame::Log { level, message }) => {
                        let entry = CblinkyLogEntry {
                            level,
                            message: message.to_owned(),
                        };
                        tx.send(&entry).await.0?;
                    }
                    None => {
                        println!("failed to decode bridge frame");
                    }
                },
                Ok(None) => {}
                Err(err) => {
                    // Keep the DB connection alive even if the serial line contains
                    // garbage or framing is temporarily out of sync.
                    println!("failed to decode serial frame {:?}", err);
                    frame = impeller2_frame::FrameDecoder::<Vec<u8>>::default();
                }
            }
        }
    });
    futures_lite::future::race(async { write.join().await.unwrap() }, async {
        read.join().await.unwrap()
    })
    .await
    .unwrap()
}
