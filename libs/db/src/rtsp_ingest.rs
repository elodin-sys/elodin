//! In-DB RTSP H.264 ingest (pull / receive-only).
//!
//! Opt-in (`rtsp` feature, off by default). One reconnecting task per source
//! pulls an H.264 RTSP stream with [`retina`], reframes each access unit to
//! Annex-B with in-band SPS/PPS via [`rtsp_ingest::annexb`], maps timestamps
//! onto the DB timeline via [`rtsp_ingest::clock`], and stores them through the
//! same [`DB::push_msg`] path as `elodinsink`. Everything downstream (playback,
//! scrubbing, export, follower replication) is unchanged.

use std::sync::Arc;
use std::time::Duration;

use futures_lite::StreamExt;
use impeller2::types::{Timestamp, msg_id};
use impeller2_wkt::{MsgMetadata, opaque_bytes_msg_schema};
use retina::client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport};
use retina::codec::{CodecItem, ParametersRef};
use rtsp_ingest::annexb::{AnnexBConverter, ParameterSets};
use rtsp_ingest::clock::ClockMapper;
use rtsp_ingest::config::RtspSource;
use tracing::{info, warn};
use url::Url;

use crate::DB;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// Spawns one reconnecting ingest task per source. Runs for the process lifetime.
pub async fn run(sources: Vec<RtspSource>, db: Arc<DB>) {
    let mut handles = Vec::new();
    for source in sources {
        let db = db.clone();
        handles.push(tokio::spawn(async move { run_source(source, db).await }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

async fn run_source(source: RtspSource, db: Arc<DB>) {
    let id = msg_id(&source.msg_name);
    // Register the friendly name once so `export-videos` names the file
    // correctly, mirroring the SetMsgMetadata `elodinsink` sends on connect.
    if let Err(e) = db.with_state_mut(|s| {
        s.set_msg_metadata(
            id,
            MsgMetadata {
                name: source.msg_name.clone(),
                schema: opaque_bytes_msg_schema(),
                metadata: Default::default(),
            },
            &db.path,
        )
    }) {
        warn!(msg_name = %source.msg_name, error = %e, "failed to set RTSP msg metadata");
    }

    // One clock for the source's whole lifetime: reconnects re-anchor it but
    // keep `last_written_us`, so the first post-reconnect frame can't repeat or
    // precede an already-stored timestamp (elodinsink keeps this state too).
    let mut clock = ClockMapper::new(Timestamp::now().0);

    loop {
        match stream_once(&source, id, &db, &mut clock).await {
            Ok(()) => info!(msg_name = %source.msg_name, "RTSP stream ended; reconnecting"),
            Err(e) => {
                warn!(msg_name = %source.msg_name, error = %e, "RTSP stream error; reconnecting")
            }
        }
        tokio::time::sleep(RECONNECT_DELAY).await;
    }
}

/// Connects, plays, and ingests frames until the stream ends or errors.
async fn stream_once(
    source: &RtspSource,
    id: impeller2::types::PacketId,
    db: &DB,
    clock: &mut ClockMapper,
) -> Result<(), BoxError> {
    let mut url = Url::parse(&source.url)?;
    // Pull any userinfo out of the URL into retina credentials.
    let creds = match url.password() {
        Some(password) if !url.username().is_empty() => Some(Credentials {
            username: url.username().to_string(),
            password: password.to_string(),
        }),
        _ => None,
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);

    let options = SessionOptions::default()
        .creds(creds)
        .user_agent("elodin-db".to_string());
    let mut session = Session::describe(url, options).await?;

    let video_i = session
        .streams()
        .iter()
        .position(|s| s.media() == "video" && s.encoding_name() == "h264")
        .ok_or("no H.264 video stream in RTSP presentation")?;

    session
        .setup(
            video_i,
            SetupOptions::default().transport(Transport::Tcp(Default::default())),
        )
        .await?;
    let session = session.play(PlayOptions::default()).await?;
    let mut demuxed = session.demuxed()?;

    // Anchor to the DB's current time (like elodinsink's last_updated anchor);
    // fall back to wall clock for an otherwise-empty DB. Re-anchoring (vs a fresh
    // mapper) preserves monotonicity across reconnects: the first frame is bumped
    // past the last stored timestamp instead of repeating `last_updated`.
    let base = db.last_updated.latest();
    let base_us = if base.0 == i64::MIN {
        Timestamp::now().0
    } else {
        base.0
    };
    clock.reanchor(base_us);
    let mut converter: Option<AnnexBConverter> = None;
    // The stored log must start on a keyframe: the export muxer rejects a
    // leading non-IDR frame and the editor decoder can only seek from an IDR.
    // RTSP sessions routinely begin mid-GOP, so drop frames until the first one.
    let mut seen_keyframe = false;

    info!(msg_name = %source.msg_name, "RTSP connected");
    while let Some(item) = demuxed.next().await {
        let CodecItem::VideoFrame(frame) = item? else {
            continue; // ignore audio / RTCP / application data
        };

        if (converter.is_none() || frame.has_new_parameters())
            && let Some(params) = video_parameters(&demuxed, video_i)
        {
            match converter.as_mut() {
                Some(c) => c.update_parameter_sets(params),
                None => converter = Some(AnnexBConverter::new(params)),
            }
        }
        let Some(converter) = converter.as_ref() else {
            continue; // no parameter sets yet; wait for them
        };

        if !seen_keyframe {
            if frame.is_random_access_point() {
                seen_keyframe = true;
            } else {
                continue; // wait for the first IDR so the log starts decodable
            }
        }

        let annexb = match converter.convert(frame.data()) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!(msg_name = %source.msg_name, error = %e, "skipping undecodable access unit");
                continue;
            }
        };

        let ts = frame.timestamp();
        let pts_us = (ts.elapsed() as i128 * 1_000_000 / ts.clock_rate().get() as i128) as i64;
        let out_ts = clock.map(pts_us);
        if let Err(e) = db.push_msg(Timestamp(out_ts), id, &annexb) {
            warn!(msg_name = %source.msg_name, error = %e, "push_msg failed");
        }
    }
    Ok(())
}

/// Extracts H.264 SPS/PPS for `stream_id` from retina's out-of-band parameters.
fn video_parameters(demuxed: &retina::client::Demuxed, stream_id: usize) -> Option<ParameterSets> {
    match demuxed.streams().get(stream_id)?.parameters()? {
        ParametersRef::Video(params) => parse_avcc(params.extra_data()),
        _ => None,
    }
}

/// Parses SPS/PPS NAL units from an `AVCDecoderConfigurationRecord` (avcC),
/// retina's `extra_data` for H.264 with length-prefixed framing.
fn parse_avcc(extra: &[u8]) -> Option<ParameterSets> {
    // configurationVersion(1) profile(1) compat(1) level(1) lengthSize(1) numSPS(1)
    if extra.len() < 7 || extra[0] != 1 {
        return None;
    }
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
    params.is_complete().then_some(params)
}
