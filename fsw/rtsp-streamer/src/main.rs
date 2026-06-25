//! RTSP → elodin-db video producer.
//!
//! Pulls an H.264 RTSP stream with [`retina`], reframes each access unit to
//! Annex-B with in-band SPS/PPS, maps timestamps onto the DB timeline, and
//! streams the frames **into** elodin-db over the impeller2 wire protocol —
//! exactly like the `elodinsink` GStreamer plugin and `fsw/video-streamer`.
//!
//! This is a standalone producer: elodin-db stays passive (it never reaches out
//! to a source). Run one process per camera/source, pointed at the DB address.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use futures_lite::StreamExt;
use impeller2::types::{IntoLenPacket, LenPacket, Timestamp, msg_id};
use impeller2_wkt::{
    LastUpdated, MsgMetadata, SetMsgMetadata, SubscribeLastUpdated, opaque_bytes_msg_schema,
};
use retina::client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport};
use retina::codec::{CodecItem, ParametersRef};
use rtsp_ingest::annexb::{AnnexBConverter, ParameterSets, annexb_contains_idr};
use rtsp_ingest::clock::ClockMapper;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tracing::{info, warn};
use url::Url;

/// impeller2 packet header: ty (1) + id (2) + req_id (1).
const PACKET_HEADER_LEN: usize = 4;
const RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// Pull an H.264 RTSP stream and stream it into elodin-db.
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// RTSP URL to pull from (rtsp://[user:pass@]host[:port]/path).
    url: String,

    /// Message-log name to store the video under (e.g. `rtsp-camera`).
    msg_name: String,

    /// Elodin DB address (IP:PORT).
    #[clap(short, long, default_value = "127.0.0.1:2240")]
    db_addr: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    // One clock for the whole process: reconnects re-anchor it but keep
    // `last_written_us`, so the first post-reconnect frame can't repeat or
    // precede an already-stored timestamp (mirrors elodinsink).
    let mut clock = ClockMapper::new(Timestamp::now().0);
    // Latest `last_updated` streamed by the DB; used to anchor the first frame.
    let latest_base = Arc::new(AtomicI64::new(i64::MIN));

    loop {
        match run_once(&args, &latest_base, &mut clock).await {
            Ok(()) => info!("stream ended; reconnecting"),
            Err(e) => warn!(error = %e, "stream error; reconnecting"),
        }
        tokio::time::sleep(RECONNECT_DELAY).await;
    }
}

/// Connects to the DB and the RTSP source, then streams frames until either ends.
async fn run_once(
    args: &Args,
    latest_base: &Arc<AtomicI64>,
    clock: &mut ClockMapper,
) -> Result<()> {
    let id = msg_id(&args.msg_name);
    let mut conn = connect_db(args.db_addr, &args.msg_name, latest_base.clone()).await?;
    let (mut demuxed, video_i) = rtsp_connect(&args.url).await?;

    let mut converter: Option<AnnexBConverter> = None;
    // The stored log must start on a keyframe: the export muxer rejects a
    // leading non-IDR frame and the editor decoder can only seek from an IDR.
    // RTSP sessions routinely begin mid-GOP, so drop frames until the first one.
    let mut seen_keyframe = false;

    info!(url = %redact(&args.url), msg_name = %args.msg_name, "RTSP connected; streaming to elodin-db");
    while let Some(item) = demuxed.next().await {
        let CodecItem::VideoFrame(frame) = item? else {
            continue; // ignore audio / RTCP / application data
        };

        if (converter.is_none() || frame.has_new_parameters())
            && let Some((params, nal_length_size)) = video_parameters(&demuxed, video_i)
        {
            match converter.as_mut() {
                Some(c) => c.update_parameter_sets(params),
                None => {
                    converter =
                        Some(AnnexBConverter::new(params).with_nal_length_size(nal_length_size)?)
                }
            }
        }
        let Some(converter) = converter.as_ref() else {
            continue; // no parameter sets yet; wait for them
        };

        // Skip non-keyframes until the log is anchored, without yet trusting
        // retina's flag: a flagged IDR can still fail to convert below.
        if !seen_keyframe && !frame.is_random_access_point() {
            continue;
        }

        let annexb = match converter.convert(frame.data()) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!(error = %e, "skipping undecodable access unit");
                continue;
            }
        };

        // Open the gate only on a successfully converted access unit that
        // actually carries an IDR, so a dropped keyframe never lets later
        // non-IDR frames be sent without a decodable keyframe at the start.
        if !seen_keyframe {
            if annexb_contains_idr(&annexb) {
                // Anchor here, right before the first send: the DB may have
                // advanced last_updated while we waited for an IDR, so this
                // keeps stored timestamps aligned with live playback.
                // `connect_db` blocks until the DB's first `last_updated`, so
                // this is the DB timeline, never wall-clock time.
                let base = latest_base.load(Ordering::Relaxed);
                if base != i64::MIN {
                    clock.reanchor(base);
                }
                seen_keyframe = true;
            } else {
                continue;
            }
        }

        let ts = frame.timestamp();
        let pts_us = (ts.elapsed() as i128 * 1_000_000 / ts.clock_rate().get() as i128) as i64;

        // Compute the timestamp without committing: the clock must advance only
        // once the frame is durably written, so a fully-failed send leaves no
        // gap in the timeline.
        let out_ts = clock.peek(pts_us);
        let pkt = build_pkt(id, out_ts, &annexb);
        let committed_ts = if conn.write.write_all(&pkt.inner).await.is_ok() {
            out_ts
        } else {
            warn!("elodin-db send failed; reconnecting");
            conn = connect_db(args.db_addr, &args.msg_name, latest_base.clone()).await?;
            // The DB clock may have moved while we were disconnected; re-anchor
            // and recompute so the retried frame lands at a live timestamp
            // instead of a stale one (mirrors elodinsink).
            let base = latest_base.load(Ordering::Relaxed);
            if base != i64::MIN {
                clock.reanchor(base);
            }
            let retry_ts = clock.peek(pts_us);
            let retry_pkt = build_pkt(id, retry_ts, &annexb);
            conn.write
                .write_all(&retry_pkt.inner)
                .await
                .context("resend frame after reconnect")?;
            retry_ts
        };
        // Only now that the write succeeded do we advance the clock.
        clock.commit(pts_us, committed_ts);
    }
    Ok(())
}

/// A live connection to elodin-db: a write half plus a background task that
/// drains streamed `LastUpdated` packets into `latest_base`.
struct DbConn {
    write: OwnedWriteHalf,
    reader: tokio::task::JoinHandle<()>,
}

impl Drop for DbConn {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// Connects to elodin-db, subscribes to `last_updated`, and registers the msg
/// name (so `export-videos` names the file), mirroring `elodinsink` on connect.
async fn connect_db(
    db_addr: SocketAddr,
    msg_name: &str,
    latest_base: Arc<AtomicI64>,
) -> Result<DbConn> {
    let stream = TcpStream::connect(db_addr)
        .await
        .with_context(|| format!("connect to elodin-db at {db_addr}"))?;
    stream.set_nodelay(true).ok();
    let (mut read, mut write) = stream.into_split();

    write
        .write_all(&(&SubscribeLastUpdated).into_len_packet().inner)
        .await
        .context("send SubscribeLastUpdated")?;

    let set_meta = SetMsgMetadata {
        id: msg_id(msg_name),
        metadata: MsgMetadata {
            name: msg_name.to_string(),
            schema: opaque_bytes_msg_schema(),
            metadata: Default::default(),
        },
    };
    write
        .write_all(&(&set_meta).into_len_packet().inner)
        .await
        .context("send SetMsgMetadata")?;

    // Block for the DB's current timeline position before returning: the DB
    // streams `last_updated` immediately on subscribe, so the first stored
    // keyframe anchors to the DB clock rather than wall-clock time.
    loop {
        let buf = read_len_packet(&mut read)
            .await
            .context("read first last_updated from elodin-db")?;
        if let Some(payload) = buf.get(PACKET_HEADER_LEN..)
            && let Ok(last_updated) = postcard::from_bytes::<LastUpdated>(payload)
        {
            latest_base.store(last_updated.0.0, Ordering::Relaxed);
            break;
        }
    }

    // Continuously drain LastUpdated so the DB's send buffer never back-pressures
    // and `latest_base` always holds the DB's current timeline position.
    let reader = tokio::spawn(async move {
        while let Ok(buf) = read_len_packet(&mut read).await {
            if let Some(payload) = buf.get(PACKET_HEADER_LEN..)
                && let Ok(last_updated) = postcard::from_bytes::<LastUpdated>(payload)
            {
                latest_base.store(last_updated.0.0, Ordering::Relaxed);
            }
        }
    });

    Ok(DbConn { write, reader })
}

/// Builds a timestamped opaque-bytes message packet for the video log.
fn build_pkt(id: impeller2::types::PacketId, ts: i64, annexb: &[u8]) -> LenPacket {
    let mut pkt = LenPacket::msg_with_timestamp(id, Timestamp(ts), annexb.len());
    pkt.extend_from_slice(annexb);
    pkt
}

/// Reads one length-prefixed (u32 LE) impeller2 packet body from `read`.
async fn read_len_packet(read: &mut OwnedReadHalf) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    read.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    read.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Opens an RTSP session (TCP-interleaved) and returns the H.264 demuxer.
async fn rtsp_connect(url_str: &str) -> Result<(retina::client::Demuxed, usize)> {
    let mut url = Url::parse(url_str).context("parse RTSP URL")?;
    // Pull any userinfo out of the URL into retina credentials. Accept a
    // username, a password, or both (some cameras use `rtsp://:pass@host`).
    let creds = if !url.username().is_empty() || url.password().is_some() {
        Some(Credentials {
            username: url.username().to_string(),
            password: url.password().unwrap_or("").to_string(),
        })
    } else {
        None
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);

    let options = SessionOptions::default()
        .creds(creds)
        .user_agent("elodin-rtsp-streamer".to_string());
    let mut session = Session::describe(url, options)
        .await
        .context("RTSP DESCRIBE")?;

    let video_i = session
        .streams()
        .iter()
        .position(|s| s.media() == "video" && s.encoding_name().eq_ignore_ascii_case("h264"))
        .context("no H.264 video stream in RTSP presentation")?;

    session
        .setup(
            video_i,
            SetupOptions::default().transport(Transport::Tcp(Default::default())),
        )
        .await
        .context("RTSP SETUP")?;
    let session = session
        .play(PlayOptions::default())
        .await
        .context("RTSP PLAY")?;
    let demuxed = session.demuxed().context("RTSP demux")?;
    Ok((demuxed, video_i))
}

/// Extracts H.264 SPS/PPS and the AVC NAL length size for `stream_id` from
/// retina's out-of-band parameters.
fn video_parameters(
    demuxed: &retina::client::Demuxed,
    stream_id: usize,
) -> Option<(ParameterSets, usize)> {
    match demuxed.streams().get(stream_id)?.parameters()? {
        ParametersRef::Video(params) => parse_avcc(params.extra_data()),
        _ => None,
    }
}

/// Parses SPS/PPS NAL units and the NAL length-prefix size from an
/// `AVCDecoderConfigurationRecord` (avcC), retina's `extra_data` for H.264.
fn parse_avcc(extra: &[u8]) -> Option<(ParameterSets, usize)> {
    // configurationVersion(1) profile(1) compat(1) level(1) lengthSizeMinusOne(1) numSPS(1)
    if extra.len() < 7 || extra[0] != 1 {
        return None;
    }
    // Low 2 bits of byte 4 are lengthSizeMinusOne; the prefix is 1..=4 bytes.
    let nal_length_size = (extra[4] & 0x03) as usize + 1;
    let num_sps = extra[5] & 0x1f;
    let mut i = 6;
    let mut sps = Vec::new();
    for _ in 0..num_sps {
        let len = u16::from_be_bytes([*extra.get(i)?, *extra.get(i + 1)?]) as usize;
        i += 2;
        sps = extra.get(i..i + len)?.to_vec();
        i += len;
    }
    let num_pps = *extra.get(i)?;
    i += 1;
    let mut pps = Vec::new();
    for _ in 0..num_pps {
        let len = u16::from_be_bytes([*extra.get(i)?, *extra.get(i + 1)?]) as usize;
        i += 2;
        pps = extra.get(i..i + len)?.to_vec();
        i += len;
    }
    let params = ParameterSets::new(sps, pps);
    params.is_complete().then_some((params, nal_length_size))
}

/// Strips credentials from an RTSP URL so it is safe to log.
fn redact(url: &str) -> String {
    match Url::parse(url) {
        Ok(mut u) => {
            let _ = u.set_username("");
            let _ = u.set_password(None);
            u.to_string()
        }
        Err(_) => "<invalid url>".to_string(),
    }
}
