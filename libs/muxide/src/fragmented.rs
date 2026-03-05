//! Fragmented MP4 (fMP4) support for streaming applications.
//!
//! Fragmented MP4 splits the container into an init segment (ftyp + moov)
//! and media segments (moof + mdat). This is essential for:
//! - DASH streaming
//! - HLS with fMP4
//! - Low-latency live streaming
//!
//! # Example
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use muxide::api::{MuxerBuilder, VideoCodec};
//!
//! // Create a fragmented muxer using the builder API
//! let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
//!     .video(VideoCodec::H264, 1920, 1080, 30.0)
//!     .with_sps(vec![0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11]) // SPS
//!     .with_pps(vec![0x68, 0xce, 0x38, 0x80]) // PPS
//!     .new_with_fragment()?;
//!
//! // Get init segment (write once at start)
//! let init_segment = muxer.init_segment();
//!
//! // Write frames...
//! let data = vec![0u8; 100]; // Your frame data
//! muxer.write_video(0, 0, &data, true)?;
//!
//! // Get media segment when ready
//! if let Some(segment) = muxer.flush_segment() {
//!     // Send segment to client
//! }
//! # Ok(())
//! # }
//! ```

// No imports needed currently - pure Vec-based API

/// Errors that can occur during fragmented MP4 muxing.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentedError {
    /// DTS values must be non-decreasing.
    NonMonotonicDts { prev_dts: u64, curr_dts: u64 },
}

impl std::error::Error for FragmentedError {}

impl std::fmt::Display for FragmentedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentedError::NonMonotonicDts { prev_dts, curr_dts } => {
                write!(
                    f,
                    "DTS values must be non-decreasing: prev={}, curr={}",
                    prev_dts, curr_dts
                )
            }
        }
    }
}

/// Configuration for fragmented MP4 output.
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Media timescale (typically 90000 for video).
    pub timescale: u32,
    /// Target fragment duration in milliseconds.
    pub fragment_duration_ms: u32,
    /// SPS NAL unit (H.264 required for init segment).
    pub sps: Vec<u8>,
    /// PPS NAL unit (H.264 required for init segment).
    pub pps: Vec<u8>,
    /// VPS NAL unit (H.265 required for init segment).
    pub vps: Option<Vec<u8>>,
    /// Sequence Header OBU (AV1 required for init segment).
    pub av1_sequence_header: Option<Vec<u8>>,
    /// VP9 configuration (extracted from first keyframe).
    pub vp9_config: Option<crate::codec::vp9::Vp9Config>,
}

impl Default for FragmentConfig {
    fn default() -> Self {
        // Note: This default provides example SPS/PPS for testing.
        // In production, you must provide actual SPS/PPS from your encoder.
        Self {
            width: 1920,
            height: 1080,
            timescale: 90000,
            fragment_duration_ms: 2000,
            sps: vec![0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11],
            pps: vec![0x68, 0xce, 0x38, 0x80],
            vps: None,
            av1_sequence_header: None,
            vp9_config: None,
        }
    }
}

/// Sample information for fragmented muxing.
#[derive(Debug, Clone)]
struct FragmentSample {
    /// Presentation timestamp in timescale units.
    pts: u64,
    /// Decode timestamp in timescale units.
    dts: u64,
    /// Sample data (AVCC format).
    data: Vec<u8>,
    /// Whether this is a sync sample (keyframe).
    is_sync: bool,
}

/// A fragmented MP4 muxer for streaming applications.
#[derive(Debug)]
pub struct FragmentedMuxer {
    config: FragmentConfig,
    samples: Vec<FragmentSample>,
    sequence_number: u32,
    base_media_decode_time: u64,
    init_segment: Option<Vec<u8>>,
    last_dts: Option<u64>,
}

impl FragmentedMuxer {
    /// Create a new fragmented muxer with the given configuration.
    pub fn new(config: FragmentConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            sequence_number: 1,
            base_media_decode_time: 0,
            init_segment: None,
            last_dts: None,
        }
    }

    /// Get the initialization segment (ftyp + moov).
    /// This should be sent once at the start of a stream.
    pub fn init_segment(&mut self) -> Vec<u8> {
        if let Some(ref init) = self.init_segment {
            return init.clone();
        }

        let mut buf = Vec::new();

        // ftyp box
        let ftyp = build_ftyp_fmp4();
        buf.extend_from_slice(&ftyp);

        // moov box (no sample tables for fMP4)
        let moov = build_moov_fmp4(&self.config);
        buf.extend_from_slice(&moov);

        self.init_segment = Some(buf.clone());
        buf
    }

    /// Queue a video sample for the current fragment.
    ///
    /// - `pts`: Presentation timestamp in timescale units
    /// - `dts`: Decode timestamp in timescale units
    /// - `data`: Sample data in MP4 length-prefixed format (4-byte NAL length prefixes)
    /// - `is_sync`: True if this is a sync sample (keyframe)
    pub fn write_video(
        &mut self,
        pts: u64,
        dts: u64,
        data: &[u8],
        is_sync: bool,
    ) -> Result<(), FragmentedError> {
        // Enforce monotonic DTS
        if let Some(last) = self.last_dts {
            if dts < last {
                return Err(FragmentedError::NonMonotonicDts {
                    prev_dts: last,
                    curr_dts: dts,
                });
            }
        }
        self.last_dts = Some(dts);

        self.samples.push(FragmentSample {
            pts,
            dts,
            data: data.to_vec(),
            is_sync,
        });
        Ok(())
    }

    /// Flush all queued samples as a media segment (moof + mdat).
    /// Returns None if there are no samples to flush.
    pub fn flush_segment(&mut self) -> Option<Vec<u8>> {
        if self.samples.is_empty() {
            return None;
        }

        let samples = std::mem::take(&mut self.samples);
        let segment = build_media_segment(
            &samples,
            self.sequence_number,
            self.base_media_decode_time,
            self.config.timescale,
        );

        // Update state for next segment
        self.sequence_number += 1;
        if let Some(last) = samples.last() {
            // Estimate next base_media_decode_time
            if samples.len() >= 2 {
                let duration_total = last.dts.saturating_sub(samples[0].dts);
                let avg_duration = duration_total / (samples.len() as u64 - 1);
                self.base_media_decode_time = last.dts + avg_duration;
            } else {
                self.base_media_decode_time = last.dts + 3000; // Fallback: 1 frame at 30fps
            }
        }

        Some(segment)
    }

    /// Check if we have enough samples to make a fragment.
    pub fn ready_to_flush(&self) -> bool {
        if self.samples.is_empty() {
            return false;
        }

        if self.samples.len() < 2 {
            return false;
        }

        let first_dts = self.samples[0].dts;
        let last_dts = self.samples.last().unwrap().dts;
        let duration_ticks = last_dts.saturating_sub(first_dts);
        let duration_ms = duration_ticks * 1000 / self.config.timescale as u64;

        duration_ms >= self.config.fragment_duration_ms as u64
    }

    /// Get current fragment duration in milliseconds.
    pub fn current_fragment_duration_ms(&self) -> u64 {
        if self.samples.len() < 2 {
            return 0;
        }
        let first_dts = self.samples[0].dts;
        let last_dts = self.samples.last().unwrap().dts;
        let duration_ticks = last_dts.saturating_sub(first_dts);
        duration_ticks * 1000 / self.config.timescale as u64
    }
}

// ============================================================================
// Box building functions for fMP4
// ============================================================================

fn build_box(typ: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = (8 + payload.len()) as u32;
    let mut buf = Vec::with_capacity(size as usize);
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(typ);
    buf.extend_from_slice(payload);
    buf
}

fn build_ftyp_fmp4() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"iso5"); // Major brand: ISO Base Media File Format v5
    payload.extend_from_slice(&0u32.to_be_bytes()); // Minor version
    payload.extend_from_slice(b"iso5"); // Compatible brands
    payload.extend_from_slice(b"iso6");
    payload.extend_from_slice(b"mp41");
    build_box(b"ftyp", &payload)
}

fn build_moov_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // mvhd (movie header)
    let mvhd = build_mvhd_fmp4(config.timescale);
    payload.extend_from_slice(&mvhd);

    // mvex (movie extends) - required for fragmented MP4
    let mvex = build_mvex();
    payload.extend_from_slice(&mvex);

    // trak (video track)
    let trak = build_trak_fmp4(config);
    payload.extend_from_slice(&trak);

    build_box(b"moov", &payload)
}

fn build_mvhd_fmp4(timescale: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Creation time
    payload.extend_from_slice(&0u32.to_be_bytes()); // Modification time
    payload.extend_from_slice(&timescale.to_be_bytes()); // Timescale
    payload.extend_from_slice(&0u32.to_be_bytes()); // Duration (unknown for live)
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes()); // Rate (1.0)
    payload.extend_from_slice(&0x0100_u16.to_be_bytes()); // Volume (1.0)
    payload.extend_from_slice(&[0u8; 10]); // Reserved
                                           // Unity matrix (36 bytes)
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(&0x4000_0000_u32.to_be_bytes());
    payload.extend_from_slice(&[0u8; 24]); // Pre-defined
    payload.extend_from_slice(&2u32.to_be_bytes()); // Next track ID
    build_box(b"mvhd", &payload)
}

fn build_mvex() -> Vec<u8> {
    // trex (track extends) - default sample flags
    let mut trex_payload = Vec::new();
    trex_payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    trex_payload.extend_from_slice(&1u32.to_be_bytes()); // Track ID
    trex_payload.extend_from_slice(&1u32.to_be_bytes()); // Default sample description index
    trex_payload.extend_from_slice(&0u32.to_be_bytes()); // Default sample duration
    trex_payload.extend_from_slice(&0u32.to_be_bytes()); // Default sample size
    trex_payload.extend_from_slice(&0u32.to_be_bytes()); // Default sample flags
    let trex = build_box(b"trex", &trex_payload);

    build_box(b"mvex", &trex)
}

fn build_trak_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // tkhd (track header)
    let tkhd = build_tkhd_fmp4(config);
    payload.extend_from_slice(&tkhd);

    // mdia (media)
    let mdia = build_mdia_fmp4(config);
    payload.extend_from_slice(&mdia);

    build_box(b"trak", &payload)
}

fn build_tkhd_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0x0000_0003_u32.to_be_bytes()); // Version 0, flags: enabled + in_movie
    payload.extend_from_slice(&0u32.to_be_bytes()); // Creation time
    payload.extend_from_slice(&0u32.to_be_bytes()); // Modification time
    payload.extend_from_slice(&1u32.to_be_bytes()); // Track ID
    payload.extend_from_slice(&0u32.to_be_bytes()); // Reserved
    payload.extend_from_slice(&0u32.to_be_bytes()); // Duration
    payload.extend_from_slice(&[0u8; 8]); // Reserved
    payload.extend_from_slice(&0u16.to_be_bytes()); // Layer
    payload.extend_from_slice(&0u16.to_be_bytes()); // Alternate group
    payload.extend_from_slice(&0u16.to_be_bytes()); // Volume (0 for video)
    payload.extend_from_slice(&0u16.to_be_bytes()); // Reserved
                                                    // Unity matrix (36 bytes)
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(&0x4000_0000_u32.to_be_bytes());
    // Width and height in fixed-point 16.16
    payload.extend_from_slice(&((config.width) << 16).to_be_bytes());
    payload.extend_from_slice(&((config.height) << 16).to_be_bytes());
    build_box(b"tkhd", &payload)
}

fn build_mdia_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // mdhd (media header)
    let mdhd = build_mdhd_fmp4(config.timescale, None);
    payload.extend_from_slice(&mdhd);

    // hdlr (handler)
    let hdlr = build_hdlr_video();
    payload.extend_from_slice(&hdlr);

    // minf (media info)
    let minf = build_minf_fmp4(config);
    payload.extend_from_slice(&minf);

    build_box(b"mdia", &payload)
}

fn encode_language_code(language: &str) -> [u8; 2] {
    // ISO 639-2/T language codes are packed into 16 bits as (c1<<10) | (c2<<5) | c3
    // where each character is offset by 0x60
    let chars: Vec<char> = language.chars().take(3).collect();
    let c1 = chars.first().copied().unwrap_or('u') as u16;
    let c2 = chars.get(1).copied().unwrap_or('n') as u16;
    let c3 = chars.get(2).copied().unwrap_or('d') as u16;

    let packed = ((c1.saturating_sub(0x60) & 0x1F) << 10)
        | ((c2.saturating_sub(0x60) & 0x1F) << 5)
        | (c3.saturating_sub(0x60) & 0x1F);

    packed.to_be_bytes()
}

fn build_mdhd_fmp4(timescale: u32, language: Option<&str>) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Creation time
    payload.extend_from_slice(&0u32.to_be_bytes()); // Modification time
    payload.extend_from_slice(&timescale.to_be_bytes()); // Timescale
    payload.extend_from_slice(&0u32.to_be_bytes()); // Duration (unknown)
    payload.extend_from_slice(&encode_language_code(language.unwrap_or("und"))); // Language
    payload.extend_from_slice(&0u16.to_be_bytes()); // Quality
    build_box(b"mdhd", &payload)
}

fn build_hdlr_video() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Pre-defined
    payload.extend_from_slice(b"vide"); // Handler type
    payload.extend_from_slice(&[0u8; 12]); // Reserved
    payload.extend_from_slice(b"VideoHandler\0"); // Name
    build_box(b"hdlr", &payload)
}

fn build_minf_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // vmhd (video media header)
    let vmhd = build_vmhd();
    payload.extend_from_slice(&vmhd);

    // dinf (data information)
    let dinf = build_dinf();
    payload.extend_from_slice(&dinf);

    // stbl (sample table) - minimal for fMP4
    let stbl = build_stbl_fmp4(config);
    payload.extend_from_slice(&stbl);

    build_box(b"minf", &payload)
}

fn build_vmhd() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0x0000_0001_u32.to_be_bytes()); // Version 0, flags: 1
    payload.extend_from_slice(&[0u8; 8]); // Graphics mode + op color
    build_box(b"vmhd", &payload)
}

fn build_dinf() -> Vec<u8> {
    // dref with self-contained data reference
    let mut dref_payload = Vec::new();
    dref_payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    dref_payload.extend_from_slice(&1u32.to_be_bytes()); // Entry count
                                                         // url box (self-contained)
    let url_payload = [0x00, 0x00, 0x00, 0x01]; // Flags: self-contained
    let url_box = build_box(b"url ", &url_payload);
    dref_payload.extend_from_slice(&url_box);
    let dref = build_box(b"dref", &dref_payload);

    build_box(b"dinf", &dref)
}

fn build_stbl_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // stsd (sample description)
    let stsd = build_stsd_fmp4(config);
    payload.extend_from_slice(&stsd);

    // Empty stts (time-to-sample) - actual data in moof
    let stts = build_empty_stts();
    payload.extend_from_slice(&stts);

    // Empty stsc (sample-to-chunk)
    let stsc = build_empty_stsc();
    payload.extend_from_slice(&stsc);

    // Empty stsz (sample size)
    let stsz = build_empty_stsz();
    payload.extend_from_slice(&stsz);

    // Empty stco (chunk offset)
    let stco = build_empty_stco();
    payload.extend_from_slice(&stco);

    build_box(b"stbl", &payload)
}

fn build_stsd_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let sample_entry = if config.av1_sequence_header.is_some() {
        build_av01_fmp4(config)
    } else if config.vp9_config.is_some() {
        build_vp09_fmp4(config)
    } else if config.vps.is_some() {
        build_hvc1_fmp4(config)
    } else {
        build_avc1_fmp4(config)
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&1u32.to_be_bytes()); // Entry count
    payload.extend_from_slice(&sample_entry);
    build_box(b"stsd", &payload)
}

fn build_avc1_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Data reference index
    payload.extend_from_slice(&0u16.to_be_bytes()); // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes()); // Reserved
    payload.extend_from_slice(&[0u8; 12]); // Pre-defined
    payload.extend_from_slice(&(config.width as u16).to_be_bytes());
    payload.extend_from_slice(&(config.height as u16).to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Horizontal resolution (72 dpi)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Vertical resolution (72 dpi)
    payload.extend_from_slice(&0u32.to_be_bytes()); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Frame count
    payload.extend_from_slice(&[0u8; 32]); // Compressor name
    payload.extend_from_slice(&0x0018_u16.to_be_bytes()); // Depth: 24-bit color
    payload.extend_from_slice(&0xffff_u16.to_be_bytes()); // Pre-defined (-1)

    // avcC (AVC Configuration)
    let avcc = build_avcc_fmp4(config);
    payload.extend_from_slice(&avcc);

    build_box(b"avc1", &payload)
}

fn build_hvc1_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Data reference index
    payload.extend_from_slice(&0u16.to_be_bytes()); // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes()); // Reserved
    payload.extend_from_slice(&[0u8; 12]); // Pre-defined
    payload.extend_from_slice(&(config.width as u16).to_be_bytes());
    payload.extend_from_slice(&(config.height as u16).to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Horizontal resolution (72 dpi)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Vertical resolution (72 dpi)
    payload.extend_from_slice(&0u32.to_be_bytes()); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Frame count
    payload.extend_from_slice(&[0u8; 32]); // Compressor name
    payload.extend_from_slice(&0x0018_u16.to_be_bytes()); // Depth: 24-bit color
    payload.extend_from_slice(&0xffff_u16.to_be_bytes()); // Pre-defined (-1)

    // hvcC (HEVC Configuration)
    let hvcc = build_hvcc_fmp4(config);
    payload.extend_from_slice(&hvcc);

    build_box(b"hvc1", &payload)
}

fn build_avcc_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = vec![
        1,                                          // Configuration version
        config.sps.get(1).copied().unwrap_or(0x42), // Profile
        config.sps.get(2).copied().unwrap_or(0x00), // Profile compatibility
        config.sps.get(3).copied().unwrap_or(0x1e), // Level
        0xff, // 6 bits reserved + 2 bits NAL unit length - 1 (3 = 4 bytes)
        0xe1, // 3 bits reserved + 5 bits number of SPS
    ];
    payload.extend_from_slice(&(config.sps.len() as u16).to_be_bytes());
    payload.extend_from_slice(&config.sps);
    payload.push(1); // Number of PPS
    payload.extend_from_slice(&(config.pps.len() as u16).to_be_bytes());
    payload.extend_from_slice(&config.pps);
    build_box(b"avcC", &payload)
}

fn build_empty_stts() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Entry count
    build_box(b"stts", &payload)
}

fn build_empty_stsc() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Entry count
    build_box(b"stsc", &payload)
}

fn build_empty_stsz() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Sample size (0 = variable)
    payload.extend_from_slice(&0u32.to_be_bytes()); // Sample count
    build_box(b"stsz", &payload)
}

fn build_empty_stco() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // Entry count
    build_box(b"stco", &payload)
}

fn build_hvcc_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let num_arrays: u8 = if config.vps.is_some() { 3 } else { 2 };

    let mut payload = vec![
        1, // configuration_version
        0, // general_profile_space (2 bits), general_tier_flag (1 bit), general_profile_idc (5 bits) - using defaults
        0, 0, 0, 0, // general_profile_compatibility_flags
        0, 0, 0, 0, 0, 0, // general_constraint_indicator_flags
        0, // general_level_idc - using default
        0, 0, // min_spatial_segmentation_idc
        0, // parallelismType
        0, // chromaFormat
        0, // bitDepthLumaMinus8
        0, // bitDepthChromaMinus8
        0, 0,          // avgFrameRate
        0x07, // constantFrameRate=0, numTemporalLayers=0, temporalIdNested=1, lengthSizeMinusOne=3 (4-byte lengths)
        num_arrays, // numOfArrays
    ];

    // VPS array
    if let Some(vps) = &config.vps {
        payload.push(0b10100000); // array_completeness=1, reserved=0, nal_unit_type=32 (VPS)
        payload.extend_from_slice(&(1u16).to_be_bytes()); // numNalus
        payload.extend_from_slice(&(vps.len() as u16).to_be_bytes()); // nalUnitLength
        payload.extend_from_slice(vps);
    }

    // SPS array
    payload.push(0b10100001); // array_completeness=1, reserved=0, nal_unit_type=33 (SPS)
    payload.extend_from_slice(&(1u16).to_be_bytes()); // numNalus
    payload.extend_from_slice(&(config.sps.len() as u16).to_be_bytes()); // nalUnitLength
    payload.extend_from_slice(&config.sps);

    // PPS array
    payload.push(0b10100010); // array_completeness=1, reserved=0, nal_unit_type=34 (PPS)
    payload.extend_from_slice(&(1u16).to_be_bytes()); // numNalus
    payload.extend_from_slice(&(config.pps.len() as u16).to_be_bytes()); // nalUnitLength
    payload.extend_from_slice(&config.pps);

    build_box(b"hvcC", &payload)
}

fn build_av01_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Data reference index
    payload.extend_from_slice(&0u16.to_be_bytes()); // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes()); // Reserved
    payload.extend_from_slice(&[0u8; 12]); // Pre-defined
    payload.extend_from_slice(&(config.width as u16).to_be_bytes());
    payload.extend_from_slice(&(config.height as u16).to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Horizontal resolution (72 dpi)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Vertical resolution (72 dpi)
    payload.extend_from_slice(&0u32.to_be_bytes()); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Frame count
    payload.extend_from_slice(&[0u8; 32]); // Compressor name
    payload.extend_from_slice(&0x0018_u16.to_be_bytes()); // Depth: 24-bit color
    payload.extend_from_slice(&0xffff_u16.to_be_bytes()); // Pre-defined (-1)

    // av1C (AV1 Configuration)
    let av1c = build_av1c_fmp4(config);
    payload.extend_from_slice(&av1c);

    build_box(b"av01", &payload)
}

fn build_av1c_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.push(1); // version
    payload.push(0); // seq_profile, seq_level_idx_0, seq_tier_0, high_bitdepth, twelve_bit, monochrome, chroma_subsampling_x, chroma_subsampling_y, chroma_sample_position, reserved
    payload.push(0); // initial_presentation_delay_present, reserved
    if let Some(seq_header) = &config.av1_sequence_header {
        payload.extend_from_slice(seq_header);
    }
    build_box(b"av1C", &payload)
}

fn build_vp09_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Data reference index
    payload.extend_from_slice(&0u16.to_be_bytes()); // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes()); // Reserved
    payload.extend_from_slice(&[0u8; 12]); // Pre-defined
    payload.extend_from_slice(&(config.width as u16).to_be_bytes());
    payload.extend_from_slice(&(config.height as u16).to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Horizontal resolution (72 dpi)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // Vertical resolution (72 dpi)
    payload.extend_from_slice(&0u32.to_be_bytes()); // Reserved
    payload.extend_from_slice(&1u16.to_be_bytes()); // Frame count
    payload.extend_from_slice(&[0u8; 32]); // Compressor name
    payload.extend_from_slice(&0x0018_u16.to_be_bytes()); // Depth: 24-bit color
    payload.extend_from_slice(&0xffff_u16.to_be_bytes()); // Pre-defined (-1)

    // vpcC (VP9 Configuration)
    let vpcc = build_vpcc_fmp4(config);
    payload.extend_from_slice(&vpcc);

    build_box(b"vp09", &payload)
}

fn build_vpcc_fmp4(config: &FragmentConfig) -> Vec<u8> {
    let mut payload = Vec::new();
    if let Some(vp9_config) = &config.vp9_config {
        payload.push(1); // version
        payload.push(vp9_config.profile); // profile
        payload.push(vp9_config.level); // level
        payload.push(vp9_config.bit_depth); // bit_depth
        payload.push(vp9_config.color_space); // color_space
        payload.push(vp9_config.transfer_function); // transfer_function
        payload.push(vp9_config.matrix_coefficients); // matrix_coefficients
        payload.push(vp9_config.full_range_flag); // full_range_flag
    }
    build_box(b"vpcC", &payload)
}

// ============================================================================
// Media segment building (moof + mdat)
// ============================================================================

fn build_media_segment(
    samples: &[FragmentSample],
    sequence_number: u32,
    base_media_decode_time: u64,
    _timescale: u32, // Reserved for future use (duration calculations)
) -> Vec<u8> {
    // Calculate total mdat size
    let mdat_payload_size: usize = samples.iter().map(|s| s.data.len()).sum();

    // Build moof first to get its size
    let moof = build_moof(samples, sequence_number, base_media_decode_time);
    let moof_size = moof.len() as u32;

    // Data offset is moof_size + mdat_header(8)
    let data_offset = moof_size + 8;

    // Rebuild moof with correct data offset
    let moof = build_moof_with_offset(
        samples,
        sequence_number,
        base_media_decode_time,
        data_offset,
    );

    // Build mdat
    let mut segment = Vec::with_capacity(moof.len() + 8 + mdat_payload_size);
    segment.extend_from_slice(&moof);

    // mdat header
    let mdat_size = (8 + mdat_payload_size) as u32;
    segment.extend_from_slice(&mdat_size.to_be_bytes());
    segment.extend_from_slice(b"mdat");

    // mdat payload (all sample data)
    for sample in samples {
        segment.extend_from_slice(&sample.data);
    }

    segment
}

fn build_moof(
    samples: &[FragmentSample],
    sequence_number: u32,
    base_media_decode_time: u64,
) -> Vec<u8> {
    build_moof_with_offset(samples, sequence_number, base_media_decode_time, 0)
}

fn build_moof_with_offset(
    samples: &[FragmentSample],
    sequence_number: u32,
    base_media_decode_time: u64,
    data_offset: u32,
) -> Vec<u8> {
    let mut payload = Vec::new();

    // mfhd (movie fragment header)
    let mfhd = build_mfhd(sequence_number);
    payload.extend_from_slice(&mfhd);

    // traf (track fragment)
    let traf = build_traf(samples, base_media_decode_time, data_offset);
    payload.extend_from_slice(&traf);

    build_box(b"moof", &payload)
}

fn build_mfhd(sequence_number: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // Version + flags
    payload.extend_from_slice(&sequence_number.to_be_bytes());
    build_box(b"mfhd", &payload)
}

fn build_traf(
    samples: &[FragmentSample],
    base_media_decode_time: u64,
    data_offset: u32,
) -> Vec<u8> {
    let mut payload = Vec::new();

    // tfhd (track fragment header)
    let tfhd = build_tfhd();
    payload.extend_from_slice(&tfhd);

    // tfdt (track fragment decode time)
    let tfdt = build_tfdt(base_media_decode_time);
    payload.extend_from_slice(&tfdt);

    // trun (track run)
    let trun = build_trun(samples, data_offset);
    payload.extend_from_slice(&trun);

    build_box(b"traf", &payload)
}

fn build_tfhd() -> Vec<u8> {
    // Flags: 0x020000 = default-base-is-moof
    let mut payload = Vec::new();
    payload.extend_from_slice(&0x0002_0000_u32.to_be_bytes()); // Version 0 + flags
    payload.extend_from_slice(&1u32.to_be_bytes()); // Track ID
    build_box(b"tfhd", &payload)
}

fn build_tfdt(base_media_decode_time: u64) -> Vec<u8> {
    // Version 1 for 64-bit decode time
    let mut payload = Vec::new();
    payload.extend_from_slice(&0x0100_0000_u32.to_be_bytes()); // Version 1 + flags
    payload.extend_from_slice(&base_media_decode_time.to_be_bytes());
    build_box(b"tfdt", &payload)
}

fn build_trun(samples: &[FragmentSample], data_offset: u32) -> Vec<u8> {
    // Flags:
    // 0x000001 = data-offset-present
    // 0x000100 = sample-duration-present
    // 0x000200 = sample-size-present
    // 0x000400 = sample-flags-present
    // 0x000800 = sample-composition-time-offset-present
    let flags: u32 = 0x000001 | 0x000100 | 0x000200 | 0x000400 | 0x000800;

    let mut payload = Vec::new();
    // Version 1 for signed composition time offsets
    payload.extend_from_slice(&(0x0100_0000 | flags).to_be_bytes());
    payload.extend_from_slice(&(samples.len() as u32).to_be_bytes()); // Sample count
    payload.extend_from_slice(&data_offset.to_be_bytes()); // Data offset

    // Per-sample data
    for (i, sample) in samples.iter().enumerate() {
        // Sample duration (estimate from DTS delta)
        let duration = if i + 1 < samples.len() {
            (samples[i + 1].dts - sample.dts) as u32
        } else if i > 0 {
            // Use previous duration for last sample
            (sample.dts - samples[i - 1].dts) as u32
        } else {
            3000 // Default: 1 frame at 30fps
        };
        payload.extend_from_slice(&duration.to_be_bytes());

        // Sample size
        payload.extend_from_slice(&(sample.data.len() as u32).to_be_bytes());

        // Sample flags
        // Bits 24-25: depends_on (2 = no other samples)
        // Bit 16: is_non_sync_sample
        let flags = if sample.is_sync {
            0x0200_0000_u32 // depends_on = 2, is_non_sync = 0
        } else {
            0x0101_0000_u32 // depends_on = 1, is_non_sync = 1
        };
        payload.extend_from_slice(&flags.to_be_bytes());

        // Composition time offset (signed, pts - dts)
        let cts = (sample.pts as i64 - sample.dts as i64) as i32;
        payload.extend_from_slice(&cts.to_be_bytes());
    }

    build_box(b"trun", &payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_box_offset(data: &[u8], typ: &[u8; 4]) -> Option<usize> {
        data.windows(4)
            .position(|w| w == typ)
            .and_then(|pos| pos.checked_sub(4))
    }

    fn read_u32_be(data: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
    }

    fn read_u64_be(data: &[u8], offset: usize) -> u64 {
        u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn init_segment_contains_ftyp_moov() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);
        let init = muxer.init_segment();

        // Check ftyp
        assert_eq!(&init[4..8], b"ftyp");

        // Find moov
        let ftyp_size = u32::from_be_bytes(init[0..4].try_into().unwrap()) as usize;
        assert_eq!(&init[ftyp_size + 4..ftyp_size + 8], b"moov");
    }

    #[test]
    fn init_segment_is_cached_after_first_call() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);
        let init1 = muxer.init_segment();
        let init2 = muxer.init_segment();
        assert_eq!(init1, init2);
    }

    #[test]
    fn media_segment_contains_moof_mdat() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);

        // Write some samples
        let sample_data = vec![0x00, 0x00, 0x00, 0x05, 0x65, 0xaa, 0xbb, 0xcc, 0xdd];
        muxer.write_video(0, 0, &sample_data, true).unwrap();
        muxer.write_video(3000, 3000, &sample_data, false).unwrap();

        let segment = muxer.flush_segment().unwrap();

        // Check moof
        assert_eq!(&segment[4..8], b"moof");

        // Find mdat
        let moof_size = u32::from_be_bytes(segment[0..4].try_into().unwrap()) as usize;
        assert_eq!(&segment[moof_size + 4..moof_size + 8], b"mdat");
    }

    #[test]
    fn ready_to_flush_tracks_sample_count_and_duration() {
        let config = FragmentConfig {
            fragment_duration_ms: 1,
            ..Default::default()
        };
        let mut muxer = FragmentedMuxer::new(config);

        assert!(!muxer.ready_to_flush(), "empty should not be ready");

        let sample_data = vec![0, 0, 0, 5, 0x65, 1, 2, 3, 4];
        muxer.write_video(0, 0, &sample_data, true).unwrap();
        assert!(!muxer.ready_to_flush(), "single sample should not be ready");

        // 1ms at 90kHz timescale is 90 ticks.
        muxer.write_video(90, 90, &sample_data, false).unwrap();
        assert!(
            muxer.ready_to_flush(),
            "two samples reaching duration should be ready"
        );
    }

    #[test]
    fn tfdt_base_decode_time_advances_between_segments() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);

        let sample_data = vec![0, 0, 0, 5, 0x65, 1, 2, 3, 4];
        muxer.write_video(0, 0, &sample_data, true).unwrap();
        muxer.write_video(3000, 3000, &sample_data, false).unwrap();
        let _seg1 = muxer.flush_segment().unwrap();

        // base_media_decode_time should now be last.dts + avg_duration = 3000 + 3000 = 6000.
        muxer.write_video(6000, 6000, &sample_data, true).unwrap();
        let seg2 = muxer.flush_segment().unwrap();

        let tfdt_off = find_box_offset(&seg2, b"tfdt").expect("tfdt box");
        let tfdt_size = read_u32_be(&seg2, tfdt_off) as usize;
        assert!(tfdt_size >= 8 + 12);
        // payload: version+flags (4), baseMediaDecodeTime (8)
        let base = read_u64_be(&seg2, tfdt_off + 8 + 4);
        assert_eq!(base, 6000);
    }

    #[test]
    fn trun_single_sample_uses_default_duration_3000() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);

        let sample_data = vec![0, 0, 0, 5, 0x65, 1, 2, 3, 4];
        muxer.write_video(0, 0, &sample_data, true).unwrap();
        let seg = muxer.flush_segment().unwrap();

        let trun_off = find_box_offset(&seg, b"trun").expect("trun box");
        // payload begins after header (8): version+flags(4), sample_count(4), data_offset(4), then sample_duration(4)
        let duration = read_u32_be(&seg, trun_off + 8 + 12);
        assert_eq!(duration, 3000);
    }

    #[test]
    fn flush_returns_none_when_empty() {
        let config = FragmentConfig::default();
        let mut muxer = FragmentedMuxer::new(config);
        assert!(muxer.flush_segment().is_none());
    }
}
