//! Native RTMP ingest server for Elodin DB.
//!
//! Accepts RTMP publish connections (e.g. from OBS Studio), demuxes
//! FLV-encapsulated H.264 video, converts AVCC NAL units to Annex-B
//! byte-stream format, and stores them via [`DB::push_msg`].
//!
//! The RTMP stream key is used as the message name, which maps to
//! `video_stream` panels in the Elodin Editor via [`impeller2::types::msg_id`].

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use impeller2::types::{PacketId, Timestamp, msg_id};
use scuffle_flv::video::VideoData;
use scuffle_flv::video::body::VideoTagBody;
use scuffle_flv::video::body::enhanced::{
    ExVideoTagBody, VideoPacket, VideoPacketCodedFrames, VideoPacketSequenceStart,
};
use scuffle_flv::video::body::legacy::LegacyVideoTagBody;
use scuffle_flv::video::header::VideoFrameType;
use scuffle_flv::video::header::VideoTagHeaderData;
use scuffle_flv::video::header::legacy::LegacyVideoTagHeader;
use scuffle_rtmp::session::server::{
    ServerSession, ServerSessionError, SessionData, SessionHandler,
};
use tokio::net::TcpListener;
use tracing::{debug, info, warn};

use crate::DB;

/// Annex-B 4-byte start code.
const START_CODE: &[u8] = &[0x00, 0x00, 0x00, 0x01];

/// FLV keyframe frame type value.
const FRAME_TYPE_KEYFRAME: VideoFrameType = VideoFrameType(1);

/// Start an RTMP ingest server on `addr`.
///
/// For each incoming connection a [`ServerSession`] is spawned that
/// processes video data and writes it into `db`.
pub async fn serve(addr: SocketAddr, db: Arc<DB>) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    info!(?addr, "RTMP ingest server listening");

    loop {
        let (stream, peer) = listener.accept().await?;
        info!(?peer, "RTMP connection accepted");

        // Disable Nagle's algorithm so RTMP responses are sent immediately.
        if let Err(err) = stream.set_nodelay(true) {
            warn!(?err, "failed to set TCP_NODELAY on RTMP connection");
        }

        let conn_db = db.clone();
        tokio::spawn(async move {
            let handler = RtmpHandler::new(conn_db);
            let session = ServerSession::new(stream, handler);
            match session.run().await {
                Ok(graceful) => {
                    if graceful {
                        info!(?peer, "RTMP session ended gracefully");
                    } else {
                        info!(?peer, "RTMP session ended (non-graceful disconnect)");
                    }
                }
                Err(err) => {
                    warn!(?peer, ?err, "RTMP session error");
                }
            }
        });
    }
}

/// Per-connection RTMP handler that converts incoming FLV video to
/// Annex-B H.264 and stores it in the database.
struct RtmpHandler {
    db: Arc<DB>,
    /// RTMP stream name (set on publish, used as message name).
    stream_name: Option<String>,
    /// Packet ID derived from `stream_name` via [`msg_id`].
    packet_id: Option<PacketId>,
    /// Cached SPS NAL units (raw bytes, no start code).
    sps_list: Vec<Vec<u8>>,
    /// Cached PPS NAL units (raw bytes, no start code).
    pps_list: Vec<Vec<u8>>,
    /// NAL unit length field size in bytes (from AVCDecoderConfigurationRecord).
    nal_length_size: usize,
    /// Base Elodin timestamp, captured at publish time.
    base_timestamp: Option<Timestamp>,
    /// First RTMP timestamp received (for offset calculation).
    first_rtmp_ts: Option<u32>,
}

impl RtmpHandler {
    fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            stream_name: None,
            packet_id: None,
            sps_list: Vec::new(),
            pps_list: Vec::new(),
            nal_length_size: 4,
            base_timestamp: None,
            first_rtmp_ts: None,
        }
    }

    /// Compute an Elodin timestamp from an RTMP timestamp (ms).
    fn elodin_timestamp(&mut self, rtmp_ts_ms: u32) -> Timestamp {
        // Capture base from DB on first packet.
        let base = *self
            .base_timestamp
            .get_or_insert_with(|| self.db.last_updated.latest());

        let first = *self.first_rtmp_ts.get_or_insert(rtmp_ts_ms);
        let offset_ms = rtmp_ts_ms.wrapping_sub(first) as i64;
        // Convert ms -> µs.
        Timestamp(base.0 + offset_ms * 1000)
    }

    /// Handle an FLV video tag: demux, convert, store.
    ///
    /// Supports both Legacy FLV and Enhanced RTMP (used by OBS Studio 30+).
    fn handle_video(&mut self, timestamp_ms: u32, data: Bytes) -> Result<(), String> {
        let mut cursor = std::io::Cursor::new(data);
        let video_data =
            VideoData::demux(&mut cursor).map_err(|e| format!("FLV demux error: {e}"))?;

        let is_keyframe = video_data.header.frame_type == FRAME_TYPE_KEYFRAME;

        match (&video_data.header.data, video_data.body) {
            // ==== Legacy FLV (older RTMP clients) ====

            // Sequence header: extract SPS/PPS from AVCDecoderConfigurationRecord
            (
                VideoTagHeaderData::Legacy(LegacyVideoTagHeader::AvcPacket(_)),
                VideoTagBody::Legacy(LegacyVideoTagBody::AvcVideoPacketSeqHdr(ref avcc)),
            ) => self.store_avcc_config(avcc),

            // Video frame: convert AVCC NAL units to Annex-B
            (
                VideoTagHeaderData::Legacy(LegacyVideoTagHeader::AvcPacket(_)),
                VideoTagBody::Legacy(LegacyVideoTagBody::Other { data }),
            ) => self.store_avcc_frame(&data, is_keyframe, timestamp_ms),

            // ==== Enhanced RTMP (OBS Studio 30+) ====

            // Enhanced sequence start: extract SPS/PPS (same AVCDecoderConfigurationRecord)
            (
                VideoTagHeaderData::Enhanced(_),
                VideoTagBody::Enhanced(ExVideoTagBody::NoMultitrack {
                    packet: VideoPacket::SequenceStart(VideoPacketSequenceStart::Avc(ref avcc)),
                    ..
                }),
            ) => self.store_avcc_config(avcc),

            // Enhanced coded frames (with composition time offset)
            (
                VideoTagHeaderData::Enhanced(_),
                VideoTagBody::Enhanced(ExVideoTagBody::NoMultitrack {
                    packet: VideoPacket::CodedFrames(VideoPacketCodedFrames::Avc { ref data, .. }),
                    ..
                }),
            ) => self.store_avcc_frame(data, is_keyframe, timestamp_ms),

            // Enhanced coded frames (without composition time offset)
            (
                VideoTagHeaderData::Enhanced(_),
                VideoTagBody::Enhanced(ExVideoTagBody::NoMultitrack {
                    packet: VideoPacket::CodedFramesX { ref data },
                    ..
                }),
            ) => self.store_avcc_frame(data, is_keyframe, timestamp_ms),

            // Ignore commands, sequence end, metadata, and other codecs.
            _ => Ok(()),
        }
    }

    /// Cache SPS/PPS from an AVCDecoderConfigurationRecord.
    fn store_avcc_config(
        &mut self,
        avcc: &scuffle_h264::AVCDecoderConfigurationRecord,
    ) -> Result<(), String> {
        self.nal_length_size = (avcc.length_size_minus_one as usize) + 1;

        self.sps_list.clear();
        for sps in &avcc.sps {
            self.sps_list.push(sps.to_vec());
        }

        self.pps_list.clear();
        for pps in &avcc.pps {
            self.pps_list.push(pps.to_vec());
        }

        debug!(
            sps_count = self.sps_list.len(),
            pps_count = self.pps_list.len(),
            nal_length_size = self.nal_length_size,
            "Received AVC sequence header"
        );
        Ok(())
    }

    /// Convert an AVCC video frame to Annex-B and store it in the database.
    fn store_avcc_frame(
        &mut self,
        avcc_data: &[u8],
        is_keyframe: bool,
        timestamp_ms: u32,
    ) -> Result<(), String> {
        let packet_id = self
            .packet_id
            .ok_or_else(|| "No stream name set (no publish received)".to_string())?;

        let annex_b = self.avcc_to_annex_b(avcc_data, is_keyframe);
        if annex_b.is_empty() {
            return Ok(());
        }

        let ts = self.elodin_timestamp(timestamp_ms);
        self.db
            .push_msg(ts, packet_id, &annex_b)
            .map_err(|e| format!("push_msg error: {e}"))?;

        Ok(())
    }

    /// Convert AVCC-formatted NAL units to Annex-B byte-stream.
    ///
    /// On keyframes, SPS and PPS are prepended so the decoder can
    /// (re-)initialise at any keyframe.
    fn avcc_to_annex_b(&self, avcc_data: &[u8], is_keyframe: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(avcc_data.len() + 128);

        // Prepend SPS/PPS on keyframes.
        if is_keyframe {
            for sps in &self.sps_list {
                out.extend_from_slice(START_CODE);
                out.extend_from_slice(sps);
            }
            for pps in &self.pps_list {
                out.extend_from_slice(START_CODE);
                out.extend_from_slice(pps);
            }
        }

        // Walk AVCC NAL units, replacing length prefixes with start codes.
        let len_size = self.nal_length_size;
        let mut pos = 0;
        while pos + len_size <= avcc_data.len() {
            let nal_len = read_nal_length(&avcc_data[pos..], len_size);
            pos += len_size;
            if nal_len == 0 || pos + nal_len > avcc_data.len() {
                break;
            }
            out.extend_from_slice(START_CODE);
            out.extend_from_slice(&avcc_data[pos..pos + nal_len]);
            pos += nal_len;
        }

        out
    }
}

impl SessionHandler for RtmpHandler {
    async fn on_publish(
        &mut self,
        _stream_id: u32,
        app_name: &str,
        stream_name: &str,
    ) -> Result<(), ServerSessionError> {
        info!(app_name, stream_name, "RTMP stream published");

        self.stream_name = Some(stream_name.to_string());
        self.packet_id = Some(msg_id(stream_name));

        // Reset timestamp state for new publish.
        self.base_timestamp = None;
        self.first_rtmp_ts = None;
        self.sps_list.clear();
        self.pps_list.clear();

        Ok(())
    }

    async fn on_unpublish(&mut self, _stream_id: u32) -> Result<(), ServerSessionError> {
        info!(stream_name = ?self.stream_name, "RTMP stream unpublished");

        self.stream_name = None;
        self.packet_id = None;
        self.base_timestamp = None;
        self.first_rtmp_ts = None;

        Ok(())
    }

    async fn on_data(
        &mut self,
        _stream_id: u32,
        data: SessionData,
    ) -> Result<(), ServerSessionError> {
        match data {
            SessionData::Video {
                timestamp,
                data: payload,
            } => {
                if let Err(err) = self.handle_video(timestamp, payload) {
                    warn!(err, "Error handling RTMP video data");
                }
            }
            // Ignore audio and AMF0 data.
            SessionData::Audio { .. } | SessionData::Amf0 { .. } => {}
        }
        Ok(())
    }
}

/// Read a big-endian NAL unit length from `buf` with the given field size (1-4 bytes).
fn read_nal_length(buf: &[u8], len_size: usize) -> usize {
    let mut val: usize = 0;
    for &byte in buf.iter().take(len_size) {
        val = (val << 8) | (byte as usize);
    }
    val
}
