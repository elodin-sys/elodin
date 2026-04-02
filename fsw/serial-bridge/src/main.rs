use anyhow::Context;
use impeller2::types::{LenPacket, Msg, PacketId, msg_id};
use impeller2_stellar::SinkExt;
use impeller2_wkt::{
    MsgMetadata, MsgStream, SetComponentMetadata, SetMsgMetadata, log_entry_msg_schema,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    mem::size_of,
    net::SocketAddr,
    time::{Duration, Instant},
};
use stellarator::io::{AsyncRead, AsyncWrite};
use stellarator::net::TcpStream;
use stellarator::rent;
use stellarator::{io::SplitExt, struc_con::Joinable};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use blackbox::{CompassRecord, GpsRecord, ImuRecord, Record};
use impeller2::types::Timestamp;

#[derive(db_macros::AsVTable, db_macros::Metadatatize)]
#[db(parent = "aleph")]
#[repr(C)]
struct BridgeRecord {
    #[db(timestamp)]
    time: i64,
    baro: f32,
    baro_temp: f32,
    vin: f32,
    vbat: f32,
    aux_current: f32,
    rtc_vbat: f32,
    cpu_temp: f32,
}

#[derive(db_macros::AsVTable, db_macros::Metadatatize)]
#[db(parent = "ublox")]
#[repr(C)]
struct GpsBridgeRecord {
    #[db(timestamp)]
    time: i64,
    unix_epoch_ms: i64,
    itow: u32,
    lat: i32,
    lon: i32,
    alt_msl: i32,
    alt_wgs84: i32,
    vel_ned: [i32; 3],
    ground_speed: u32,
    heading_motion: i32,
    h_acc: u32,
    v_acc: u32,
    s_acc: u32,
    fix_type: u8,
    satellites: u8,
    valid_flags: u8,
    #[db(skip)]
    _pad: u8,
}

#[derive(db_macros::AsVTable, db_macros::Metadatatize)]
#[db(parent = "qmc5883l")]
#[repr(C)]
struct CompassBridgeRecord {
    #[db(timestamp)]
    time: i64,
    mag: [i16; 3],
    status: u8,
    #[db(skip)]
    _pad: u8,
}

#[derive(db_macros::AsVTable, db_macros::Metadatatize)]
#[db(parent = "imu")]
#[repr(C)]
struct ImuBridgeRecord {
    #[db(timestamp)]
    time: i64,
    accel: [f32; 3],
    gyro: [f32; 3],
    mag: [f32; 3],
}

const LOG_STREAM_NAME: &str = "aleph.stm32.log";
const LOG_STREAM_ID: PacketId = msg_id(LOG_STREAM_NAME);
const LOG_FRAME_PREFIX_LEN: usize = 5;
const LOG_FRAME_MAGIC: [u8; 2] = *b"EL";
const LOG_FRAME_VERSION: u8 = 1;
const LOG_FRAME_KIND_LOG: u8 = 1;
const LOG_FRAME_KIND_GPS: u8 = 2;
const LOG_FRAME_KIND_COMPASS: u8 = 3;
const LOG_FRAME_KIND_IMU: u8 = 4;
const DEFAULT_SERIAL_PORT: &str = "/dev/ttyTHS1";
const DEFAULT_SERIAL_BAUD: u64 = 115_200;
const GPS_WAIT_LOG_INTERVAL: u64 = 500;

struct GpsClock {
    anchor_gps_us: i64,
    anchor_instant: Instant,
    last_emitted_us: i64,
}

impl GpsClock {
    fn new(anchor_gps_us: i64) -> Self {
        Self {
            anchor_gps_us,
            anchor_instant: Instant::now(),
            last_emitted_us: anchor_gps_us,
        }
    }

    fn update(&mut self, unix_epoch_ms: i64) {
        self.anchor_gps_us = unix_epoch_ms.saturating_mul(1000);
        self.anchor_instant = Instant::now();
    }

    fn timestamp_us(&self, now: Instant) -> i64 {
        let elapsed = now.duration_since(self.anchor_instant);
        self.anchor_gps_us
            .saturating_add(elapsed.as_micros() as i64)
    }

    fn timestamp(&mut self) -> Timestamp {
        let ts = self.timestamp_us(Instant::now());
        let ts = ts.max(self.last_emitted_us.saturating_add(1));
        self.last_emitted_us = ts;
        Timestamp(ts)
    }

    fn emit_gps_timestamp(&mut self, unix_epoch_ms: i64) -> Option<Timestamp> {
        let candidate_us = unix_epoch_ms.saturating_mul(1000);
        self.update(unix_epoch_ms);
        self.last_emitted_us = self.last_emitted_us.max(candidate_us);
        Some(Timestamp(self.last_emitted_us))
    }
}

struct ClockState {
    gps_mode: bool,
    clock: Option<GpsClock>,
    dropped: u64,
}

impl ClockState {
    fn new(gps_mode: bool) -> Self {
        Self {
            gps_mode,
            clock: None,
            dropped: 0,
        }
    }

    fn sensor_ts(&mut self) -> Option<Timestamp> {
        if !self.gps_mode {
            return Some(Timestamp::now());
        }
        if let Some(clock) = self.clock.as_mut() {
            return Some(clock.timestamp());
        }
        self.dropped += 1;
        if self.dropped % GPS_WAIT_LOG_INTERVAL == 1 {
            println!(
                "waiting for GPS clock lock, dropping non-GPS frame (count={})",
                self.dropped
            );
        }
        None
    }

    fn gps_ts(&mut self, unix_epoch_ms: i64) -> Option<Timestamp> {
        if !self.gps_mode {
            return Some(Timestamp::now());
        }
        if unix_epoch_ms <= 0 {
            self.dropped += 1;
            if self.dropped % GPS_WAIT_LOG_INTERVAL == 1 {
                println!(
                    "waiting for valid GPS UTC time, dropping GPS frame (count={})",
                    self.dropped
                );
            }
            return None;
        }
        if let Some(clock) = self.clock.as_mut() {
            Some(
                clock
                    .emit_gps_timestamp(unix_epoch_ms)
                    .unwrap_or(Timestamp::now()),
            )
        } else {
            let anchor_us = unix_epoch_ms.saturating_mul(1000);
            let mut clock = GpsClock::new(anchor_us);
            let ts = clock
                .emit_gps_timestamp(unix_epoch_ms)
                .unwrap_or(Timestamp(anchor_us));
            self.clock = Some(clock);
            self.dropped = 0;
            println!("GPS clock locked at unix_epoch_ms={}", unix_epoch_ms);
            Some(ts)
        }
    }
}

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
    Gps(&'a [u8]),
    Compass(&'a [u8]),
    Imu(&'a [u8]),
}

fn parse_bridge_frame(decoded: &[u8]) -> Option<BridgeFrame<'_>> {
    if decoded.len() >= LOG_FRAME_PREFIX_LEN
        && decoded[0] == LOG_FRAME_MAGIC[0]
        && decoded[1] == LOG_FRAME_MAGIC[1]
        && decoded[2] == LOG_FRAME_VERSION
    {
        let payload = &decoded[LOG_FRAME_PREFIX_LEN..];
        match decoded[3] {
            LOG_FRAME_KIND_LOG => {
                let message = std::str::from_utf8(payload).ok()?;
                return Some(BridgeFrame::Log {
                    level: decoded[4],
                    message,
                });
            }
            LOG_FRAME_KIND_GPS => {
                if payload.len() == size_of::<GpsRecord>() {
                    return Some(BridgeFrame::Gps(payload));
                }
            }
            LOG_FRAME_KIND_COMPASS => {
                if payload.len() == size_of::<CompassRecord>() {
                    return Some(BridgeFrame::Compass(payload));
                }
            }
            LOG_FRAME_KIND_IMU => {
                if payload.len() == size_of::<ImuRecord>() {
                    return Some(BridgeFrame::Imu(payload));
                }
            }
            _ => {}
        }
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

fn gps_clock_enabled() -> bool {
    env::var("ALEPH_SERIAL_BRIDGE_GPS_CLOCK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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

    let gps_clock_mode = gps_clock_enabled();
    if gps_clock_mode {
        println!("GPS-disciplined timestamping enabled");
    } else {
        println!("GPS-disciplined timestamping disabled");
    }

    let id: PacketId = fastrand::u16(..).to_le_bytes();
    let compass_id: PacketId = fastrand::u16(..).to_le_bytes();
    let imu_id: PacketId = fastrand::u16(..).to_le_bytes();
    tx.send(&SetComponentMetadata::new("aleph", "aleph"))
        .await
        .0?;
    tx.send(&SetComponentMetadata::new("qmc5883l", "qmc5883l"))
        .await
        .0?;
    tx.send(&SetComponentMetadata::new("imu", "imu")).await.0?;
    tx.init_world::<BridgeRecord>(id).await?;
    tx.init_world::<CompassBridgeRecord>(compass_id).await?;
    tx.init_world::<ImuBridgeRecord>(imu_id).await?;

    let gps_id: PacketId = if gps_clock_mode {
        let id: PacketId = fastrand::u16(..).to_le_bytes();
        tx.send(&SetComponentMetadata::new("ublox", "ublox"))
            .await
            .0?;
        tx.init_world::<GpsBridgeRecord>(id).await?;
        id
    } else {
        [0, 0]
    };

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
    port.set_baud(match serial_baud {
        9600 => stellarator::serial::Baud::B9600,
        115200 => stellarator::serial::Baud::B115200,
        _ => stellarator::serial::Baud::Other(serial_baud),
    })?;
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
        let mut buf = vec![0u8; 4096];
        let mut frame = impeller2_frame::FrameDecoder::<Vec<u8>>::default();
        let mut clock_state = ClockState::new(gps_clock_mode);
        let mut imu_rx: u64 = 0;
        let mut imu_sent: u64 = 0;
        let mut imu_dropped_gps: u64 = 0;
        let mut bytes_rx: u64 = 0;
        let mut frames_decoded: u64 = 0;
        let mut cobs_errors: u64 = 0;
        let mut parse_fails: u64 = 0;
        let mut last_diag = Instant::now();
        loop {
            if last_diag.elapsed() >= Duration::from_secs(1) {
                let s = &frame.stats;
                println!(
                    "bridge: {}KB/s rx, {} frames/s, imu: rx={} sent={} dropped={} parse_fail={} cobs_err={} | decoder: {} cobs_err, {} empty, {} oversize, max_buf={}",
                    bytes_rx / 1024,
                    frames_decoded,
                    imu_rx,
                    imu_sent,
                    imu_dropped_gps,
                    parse_fails,
                    cobs_errors,
                    s.cobs_decode_errors,
                    s.empty_frames,
                    s.oversize_frames,
                    s.max_buffer_len,
                );
                imu_rx = 0;
                imu_sent = 0;
                imu_dropped_gps = 0;
                bytes_rx = 0;
                frames_decoded = 0;
                cobs_errors = 0;
                parse_fails = 0;
                frame.stats = Default::default();
                last_diag = Instant::now();
            }
            let n = rent!(port_rx.read(buf).await, buf)?;
            bytes_rx += n as u64;
            let data = &buf[..n];
            let mut offset = 0;
            while offset < data.len() {
                match frame.push(&data[offset..]) {
                    Ok((Some(decoded), consumed)) => {
                        offset += consumed;
                        frames_decoded += 1;
                        match parse_bridge_frame(decoded) {
                            Some(BridgeFrame::LegacyRecord(record)) => {
                                let Some(timestamp) = clock_state.sensor_ts() else {
                                    continue;
                                };
                                let mut table = LenPacket::table(id, 8 + record.len());
                                table.extend_from_slice(&timestamp.0.to_le_bytes());
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
                            Some(BridgeFrame::Gps(payload)) if gps_clock_mode => {
                                let mut unix_epoch_ms_bytes = [0u8; 8];
                                unix_epoch_ms_bytes.copy_from_slice(&payload[..8]);
                                let unix_epoch_ms = i64::from_le_bytes(unix_epoch_ms_bytes);
                                let Some(timestamp) = clock_state.gps_ts(unix_epoch_ms) else {
                                    continue;
                                };
                                let mut table = LenPacket::table(gps_id, 8 + payload.len());
                                table.extend_from_slice(&timestamp.0.to_le_bytes());
                                table.extend_from_slice(payload);
                                tx.send(table).await.0?;
                            }
                            Some(BridgeFrame::Compass(payload)) => {
                                let Some(timestamp) = clock_state.sensor_ts() else {
                                    continue;
                                };
                                let mut table = LenPacket::table(compass_id, 8 + payload.len());
                                table.extend_from_slice(&timestamp.0.to_le_bytes());
                                table.extend_from_slice(payload);
                                tx.send(table).await.0?;
                            }
                            Some(BridgeFrame::Imu(payload)) => {
                                imu_rx += 1;
                                let Some(timestamp) = clock_state.sensor_ts() else {
                                    imu_dropped_gps += 1;
                                    continue;
                                };
                                let mut table = LenPacket::table(imu_id, 8 + payload.len());
                                table.extend_from_slice(&timestamp.0.to_le_bytes());
                                table.extend_from_slice(payload);
                                tx.send(table).await.0?;
                                imu_sent += 1;
                            }
                            Some(BridgeFrame::Gps(_)) => {}
                            None => {
                                parse_fails += 1;
                            }
                        }
                    }
                    Ok((None, consumed)) => {
                        let _ = consumed;
                        break;
                    }
                    Err(_err) => {
                        cobs_errors += 1;
                        frame = impeller2_frame::FrameDecoder::<Vec<u8>>::default();
                        break;
                    }
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
