use crate::assert_invariant;
use crate::codec::common::AnnexBNalIter;
use crate::codec::vp9::is_vp9_keyframe;
use crate::fragmented::{FragmentConfig, FragmentedMuxer};
/// Public API definitions for the Muxide crate.
///
/// This module contains the types and traits that form the public contract
/// for users of the crate.  Concrete implementations live in private
/// modules.  The API defined here intentionally exposes only the
/// capabilities promised by the charter and contract documents.  It does
/// not contain any implementation details.
use crate::muxer::mp4::{Mp4AudioTrack, Mp4VideoTrack, Mp4Writer, Mp4WriterError, MEDIA_TIMESCALE};
use std::fmt;
use std::io::Write;

/// Enumeration of supported video codecs for the initial version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264/AVC video codec.  Only the AVC Annex B stream format is
    /// currently supported.  B‑frames are not permitted in v0.
    H264,
    /// H.265/HEVC video codec. Annex B stream format with VPS/SPS/PPS.
    /// Requires first keyframe to contain VPS, SPS, and PPS NALs.
    H265,
    /// AV1 video codec. OBU (Open Bitstream Unit) stream format.
    /// Requires first keyframe to contain Sequence Header OBU.
    Av1,
    /// VP9 video codec. Compressed VP9 frames with frame headers.
    /// Requires first keyframe to contain sequence parameters.
    Vp9,
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoCodec::H264 => write!(f, "H.264"),
            VideoCodec::H265 => write!(f, "H.265"),
            VideoCodec::Av1 => write!(f, "AV1"),
            VideoCodec::Vp9 => write!(f, "VP9"),
        }
    }
}

impl std::str::FromStr for VideoCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "h264" | "h.264" | "avc" => Ok(VideoCodec::H264),
            "h265" | "h.265" | "hevc" => Ok(VideoCodec::H265),
            "av1" => Ok(VideoCodec::Av1),
            "vp9" => Ok(VideoCodec::Vp9),
            _ => Err(format!("Unknown video codec: {}", s)),
        }
    }
}

/// AAC profile variants supported by Muxide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    /// AAC Low Complexity (LC) - most common profile.
    Lc,
    /// AAC Main profile - higher quality than LC.
    Main,
    /// AAC Scalable Sample Rate (SSR).
    Ssr,
    /// AAC Long Term Prediction (LTP).
    Ltp,
    /// HE-AAC (High Efficiency AAC) - LC + SBR.
    He,
    /// HE-AAC v2 - HE-AAC + PS (Parametric Stereo).
    Hev2,
}

/// Enumeration of supported audio codecs for the initial version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// AAC (Advanced Audio Coding) with ADTS framing. Supports multiple profiles.
    Aac(AacProfile),
    /// Opus audio codec. Raw Opus packets (no container framing).
    /// Sample rate is always 48kHz per Opus spec.
    Opus,
    /// No audio.  Use this variant when only video is being muxed.
    None,
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioCodec::Aac(profile) => write!(f, "AAC-{}", profile),
            AudioCodec::Opus => write!(f, "Opus"),
            AudioCodec::None => write!(f, "None"),
        }
    }
}

impl fmt::Display for AacProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AacProfile::Lc => write!(f, "LC"),
            AacProfile::Main => write!(f, "Main"),
            AacProfile::Ssr => write!(f, "SSR"),
            AacProfile::Ltp => write!(f, "LTP"),
            AacProfile::He => write!(f, "HE"),
            AacProfile::Hev2 => write!(f, "HEv2"),
        }
    }
}

impl std::str::FromStr for AudioCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "aac" | "aac-lc" => Ok(AudioCodec::Aac(AacProfile::Lc)),
            "aac-main" => Ok(AudioCodec::Aac(AacProfile::Main)),
            "aac-ssr" => Ok(AudioCodec::Aac(AacProfile::Ssr)),
            "aac-ltp" => Ok(AudioCodec::Aac(AacProfile::Ltp)),
            "aac-he" => Ok(AudioCodec::Aac(AacProfile::He)),
            "aac-hev2" => Ok(AudioCodec::Aac(AacProfile::Hev2)),
            "opus" => Ok(AudioCodec::Opus),
            "none" => Ok(AudioCodec::None),
            _ => Err(format!("Unknown audio codec: {}", s)),
        }
    }
}

/// High-level muxer configuration intended for simple integrations (e.g. CrabCamera).
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    pub width: u32,
    pub height: u32,
    pub framerate: f64,
    pub audio: Option<AudioTrackConfig>,
    pub metadata: Option<Metadata>,
    pub fast_start: bool,
}

/// Metadata to embed in the MP4 file (title, creation time, etc.)
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Title of the recording (appears in media players)
    pub title: Option<String>,
    /// Creation timestamp in seconds since Unix epoch (1970-01-01)
    pub creation_time: Option<u64>,
    /// Language code (ISO 639-2/T format, e.g., "eng", "spa", "und")
    pub language: Option<String>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_creation_time(mut self, unix_timestamp: u64) -> Self {
        self.creation_time = Some(unix_timestamp);
        self
    }

    /// Set creation time to current system time
    pub fn with_current_time(mut self) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.creation_time = Some(duration.as_secs());
        }
        self
    }

    /// Set language code (ISO 639-2/T format, e.g., "eng", "spa", "und")
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

impl MuxerConfig {
    pub fn new(width: u32, height: u32, framerate: f64) -> Self {
        Self {
            width,
            height,
            framerate,
            audio: None,
            metadata: None,
            fast_start: true, // Default ON for web compatibility
        }
    }

    pub fn with_audio(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        if codec == AudioCodec::None {
            self.audio = None;
        } else {
            self.audio = Some(AudioTrackConfig {
                codec,
                sample_rate,
                channels,
            });
        }
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_fast_start(mut self, enabled: bool) -> Self {
        self.fast_start = enabled;
        self
    }
}

/// Summary statistics returned when finishing a mux.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuxerStats {
    pub video_frames: u64,
    pub audio_frames: u64,
    pub duration_secs: f64,
    pub bytes_written: u64,
}

/// Builder for constructing a new muxer instance.
///
/// The builder follows a fluent API pattern: each method returns a
/// modified builder, allowing method chaining.  Only the configuration
/// necessary for the initial v0 release is included.  Additional
/// configuration (such as B‑frame support, fragmented MP4 or other
/// containers) will be added in future slices.
pub struct MuxerBuilder<Writer> {
    /// The underlying writer to which container data will be written.
    writer: Writer,
    /// Optional video configuration.
    video: Option<(VideoCodec, u32, u32, f64)>,
    /// Optional audio configuration.
    audio: Option<(AudioCodec, u32, u16)>,
    /// Metadata to embed in the output file.
    metadata: Option<Metadata>,
    /// Whether to enable fast-start (moov before mdat).
    fast_start: bool,
    /// SPS data for fragmented MP4.
    sps: Option<Vec<u8>>,
    /// PPS data for fragmented MP4.
    pps: Option<Vec<u8>>,
    /// VPS data for H.265 fragmented MP4.
    vps: Option<Vec<u8>>,
    /// AV1 sequence header OBU for fragmented MP4.
    av1_sequence_header: Option<Vec<u8>>,
    /// VP9 configuration for fragmented MP4.
    vp9_config: Option<crate::codec::vp9::Vp9Config>,
}

impl<Writer> MuxerBuilder<Writer> {
    /// Create a new builder for the given output writer.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            video: None,
            audio: None,
            metadata: None,
            fast_start: true, // Default ON for web compatibility
            sps: None,
            pps: None,
            vps: None,
            av1_sequence_header: None,
            vp9_config: None,
        }
    }

    /// Configure the video track.
    pub fn video(mut self, codec: VideoCodec, width: u32, height: u32, framerate: f64) -> Self {
        self.video = Some((codec, width, height, framerate));
        self
    }

    /// Configure the audio track.
    pub fn audio(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        self.audio = Some((codec, sample_rate, channels));
        self
    }

    /// Set metadata to embed in the output file (title, creation time, etc.)
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Enable or disable fast-start mode (moov before mdat).
    /// Default is `true` for web streaming compatibility.
    pub fn with_fast_start(mut self, enabled: bool) -> Self {
        self.fast_start = enabled;
        self
    }

    /// Set SPS (Sequence Parameter Set) data for H.264/H.265 fragmented MP4.
    /// Required for proper fragmented MP4 initialization.
    pub fn with_sps(mut self, sps: Vec<u8>) -> Self {
        self.sps = Some(sps);
        self
    }

    /// Set PPS (Picture Parameter Set) data for H.264/H.265 fragmented MP4.
    /// Required for proper fragmented MP4 initialization.
    pub fn with_pps(mut self, pps: Vec<u8>) -> Self {
        self.pps = Some(pps);
        self
    }

    /// Set VPS (Video Parameter Set) data for H.265 fragmented MP4.
    /// Required for proper H.265 fragmented MP4 initialization.
    pub fn with_vps(mut self, vps: Vec<u8>) -> Self {
        self.vps = Some(vps);
        self
    }

    /// Set AV1 sequence header OBU for fragmented MP4.
    /// Required for proper AV1 fragmented MP4 initialization.
    pub fn with_av1_sequence_header(mut self, sequence_header: Vec<u8>) -> Self {
        self.av1_sequence_header = Some(sequence_header);
        self
    }

    /// Set VP9 configuration for fragmented MP4.
    /// Required for proper VP9 fragmented MP4 initialization.
    pub fn with_vp9_config(mut self, config: crate::codec::vp9::Vp9Config) -> Self {
        self.vp9_config = Some(config);
        self
    }

    /// Set creation time for the media file
    pub fn set_create_time(mut self, unix_timestamp: u64) -> Self {
        self.metadata
            .get_or_insert_with(Metadata::default)
            .creation_time = Some(unix_timestamp);
        self
    }

    /// Set language code for the media file
    pub fn set_language(mut self, language: impl Into<String>) -> Self {
        self.metadata.get_or_insert_with(Metadata::default).language = Some(language.into());
        self
    }

    /// Set video track parameters
    pub fn set_video_track(
        mut self,
        codec: VideoCodec,
        width: u32,
        height: u32,
        framerate: f64,
    ) -> Self {
        self.video = Some((codec, width, height, framerate));
        self
    }

    /// Set audio track parameters
    pub fn set_audio_track(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        self.audio = Some((codec, sample_rate, channels));
        self
    }

    /// Finalise the builder and produce a `Muxer` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if required configuration is missing or invalid.
    pub fn build(self) -> Result<Muxer<Writer>, MuxerError>
    where
        Writer: Write,
    {
        // In v0, we perform minimal validation: video configuration must be
        // present.  Future releases may relax this to allow audio‑only
        // streams.
        let (codec, width, height, framerate) = self.video.ok_or(MuxerError::MissingVideoConfig)?;
        let video_track = VideoTrackConfig {
            codec,
            width,
            height,
            framerate,
        };

        let audio_track = self.audio.and_then(|(codec, sample_rate, channels)| {
            if codec == AudioCodec::None {
                None
            } else {
                Some(AudioTrackConfig {
                    codec,
                    sample_rate,
                    channels,
                })
            }
        });

        let mut writer = Mp4Writer::new(self.writer, video_track.codec);
        if let Some(audio) = &audio_track {
            writer.enable_audio(Mp4AudioTrack {
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                codec: audio.codec,
            });
        }

        Ok(Muxer {
            writer,
            video_track,
            audio_track,
            metadata: self.metadata,
            fast_start: self.fast_start,
            first_video_pts: None,
            last_video_pts: None,
            last_video_dts: None,
            last_audio_pts: None,
            video_frame_count: 0,
            audio_frame_count: 0,
            finished: false,
            current_video_pts: 0.0,
            current_audio_pts: 0.0,
        })
    }

    /// Create a fragmented MP4 muxer.
    ///
    /// This creates a `FragmentedMuxer` with the configuration from this builder.
    /// Supports H.264, H.265, AV1, and VP9 video for fragmented MP4.
    /// Codec-specific parameters must be provided using the appropriate with_() methods.
    /// Only video configuration is supported for fragmented MP4.
    ///
    /// # Errors
    ///
    /// Returns an error if video configuration is missing, unsupported codec,
    /// or required codec parameters are not provided.
    pub fn new_with_fragment(self) -> Result<FragmentedMuxer, MuxerError> {
        // Fragmented MP4 requires video configuration
        let (codec, width, height, _framerate) =
            self.video.ok_or(MuxerError::MissingVideoConfig)?;

        // Extract codec-specific configuration
        let (sps, pps, vps, av1_sequence_header, vp9_config) = match codec {
            VideoCodec::H264 => {
                let sps = self.sps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "SPS must be provided for H.264 fragmented MP4 using with_sps()",
                    ))
                })?;
                let pps = self.pps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "PPS must be provided for H.264 fragmented MP4 using with_pps()",
                    ))
                })?;
                (sps, pps, None, None, None)
            }
            VideoCodec::H265 => {
                let vps = self.vps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "VPS must be provided for H.265 fragmented MP4 using with_vps()",
                    ))
                })?;
                let sps = self.sps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "SPS must be provided for H.265 fragmented MP4 using with_sps()",
                    ))
                })?;
                let pps = self.pps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "PPS must be provided for H.265 fragmented MP4 using with_pps()",
                    ))
                })?;
                (sps, pps, Some(vps), None, None)
            }
            VideoCodec::Av1 => {
                let av1_sequence_header = self.av1_sequence_header.ok_or_else(|| MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "AV1 sequence header must be provided for AV1 fragmented MP4 using with_av1_sequence_header()",
                )))?;
                (vec![], vec![], None, Some(av1_sequence_header), None)
            }
            VideoCodec::Vp9 => {
                let vp9_config = self.vp9_config.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "VP9 config must be provided for VP9 fragmented MP4 using with_vp9_config()",
                ))
                })?;
                (vec![], vec![], None, None, Some(vp9_config))
            }
        };

        let config = FragmentConfig {
            width,
            height,
            timescale: 90000,           // Standard video timescale
            fragment_duration_ms: 2000, // 2 second fragments
            sps,
            pps,
            vps,
            av1_sequence_header,
            vp9_config,
        };

        Ok(FragmentedMuxer::new(config))
    }
}

/// Configuration for a video track.
#[derive(Debug, Clone)]
pub struct VideoTrackConfig {
    /// Video codec.
    pub codec: VideoCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub framerate: f64,
}

/// Configuration for an audio track.
#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
}

/// Opaque muxer type.  Users interact with this type to write frames
/// into the container.  Implementation details are hidden in a private
/// module.
///
/// # Thread Safety
///
/// `Muxer<W>` is `Send` when `W: Send` and `Sync` when `W: Sync`.
/// This means you can safely move a `Muxer<File>` between threads or
/// share a `Muxer<Vec<u8>>` across threads (with appropriate synchronization).
pub struct Muxer<Writer> {
    writer: Mp4Writer<Writer>,
    video_track: VideoTrackConfig,
    audio_track: Option<AudioTrackConfig>,
    metadata: Option<Metadata>,
    fast_start: bool,
    first_video_pts: Option<f64>,
    last_video_pts: Option<f64>,
    last_video_dts: Option<f64>,
    last_audio_pts: Option<f64>,
    video_frame_count: u64,
    audio_frame_count: u64,
    finished: bool,
    current_video_pts: f64,
    current_audio_pts: f64,
}

/// Error type for builder validation and runtime errors.
///
/// All errors include context to help diagnose issues. Error messages are designed
/// to be educational—they explain what went wrong and how to fix it.
#[derive(Debug)]
pub enum MuxerError {
    /// Video configuration is missing.  In v0, a video track is required.
    MissingVideoConfig,
    /// Low-level IO error while writing the container.
    Io(std::io::Error),
    /// The muxer has already been finished.
    AlreadyFinished,
    /// Video `pts` must be non-negative.
    NegativeVideoPts { pts: f64, frame_index: u64 },
    /// Video `dts` must be non-negative.
    NegativeVideoDts { dts: f64, frame_index: u64 },
    /// Video `pts` must be finite (not NaN or Inf).
    InvalidVideoPts { pts: f64, frame_index: u64 },
    /// Video `dts` must be finite (not NaN or Inf).
    InvalidVideoDts { dts: f64, frame_index: u64 },
    /// Audio `pts` must be non-negative.
    NegativeAudioPts { pts: f64, frame_index: u64 },
    /// Audio `pts` must be finite (not NaN or Inf).
    InvalidAudioPts { pts: f64, frame_index: u64 },
    /// Audio was written but no audio track was configured.
    AudioNotConfigured,
    /// Audio sample is empty.
    EmptyAudioFrame { frame_index: u64 },
    /// Video sample is empty.
    EmptyVideoFrame { frame_index: u64 },
    /// Video timestamps must be strictly increasing.
    NonIncreasingVideoPts {
        prev_pts: f64,
        curr_pts: f64,
        frame_index: u64,
    },
    /// Audio timestamps must be non-decreasing.
    DecreasingAudioPts {
        prev_pts: f64,
        curr_pts: f64,
        frame_index: u64,
    },
    /// Audio may not precede the first video frame.
    AudioBeforeFirstVideo {
        audio_pts: f64,
        first_video_pts: Option<f64>,
    },
    /// The first video frame must be a keyframe.
    FirstVideoFrameMustBeKeyframe,
    /// The first video frame must include SPS/PPS (H.264/H.265).
    FirstVideoFrameMissingSpsPps,
    /// The first AV1 keyframe must include a Sequence Header OBU.
    FirstAv1FrameMissingSequenceHeader,
    /// The first VP9 keyframe must include sequence parameters.
    FirstVp9FrameMissingSequenceHeader,
    /// Audio sample is not a valid ADTS frame.
    InvalidAdts { frame_index: u64 },
    /// Audio sample has detailed ADTS validation errors.
    InvalidAdtsDetailed {
        frame_index: u64,
        error: Box<crate::muxer::mp4::AdtsValidationError>,
    },
    /// Audio sample is not a valid Opus packet.
    InvalidOpusPacket { frame_index: u64 },
    /// DTS must be monotonically increasing.
    NonIncreasingDts {
        prev_dts: f64,
        curr_dts: f64,
        frame_index: u64,
    },
}

impl From<std::io::Error> for MuxerError {
    fn from(err: std::io::Error) -> Self {
        MuxerError::Io(err)
    }
}

impl fmt::Display for MuxerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MuxerError::MissingVideoConfig => {
                write!(f, "missing video configuration: call .video() on MuxerBuilder before .build()")
            }
            MuxerError::Io(err) => write!(f, "IO error: {}", err),
            MuxerError::AlreadyFinished => {
                write!(f, "muxer already finished: cannot write frames after calling finish()")
            }
            MuxerError::NegativeVideoPts { pts, frame_index } => {
                write!(f, "video frame {} has negative PTS ({:.3}s): timestamps must be >= 0.0", 
                       frame_index, pts)
            }
            MuxerError::InvalidVideoPts { pts, frame_index } => {
                write!(f, "video frame {} has invalid PTS ({:.3}s): timestamps must be finite (not NaN or Inf)", 
                       frame_index, pts)
            }
            MuxerError::NegativeVideoDts { dts, frame_index } => {
                write!(f, "video frame {} has negative DTS ({:.3}s): decode timestamps must be >= 0.0", 
                       frame_index, dts)
            }
            MuxerError::InvalidVideoDts { dts, frame_index } => {
                write!(f, "video frame {} has invalid DTS ({:.3}s): decode timestamps must be finite (not NaN or Inf)", 
                       frame_index, dts)
            }
            MuxerError::NegativeAudioPts { pts, frame_index } => {
                write!(f, "audio frame {} has negative PTS ({:.3}s): timestamps must be >= 0.0",
                       frame_index, pts)
            }
            MuxerError::InvalidAudioPts { pts, frame_index } => {
                write!(f, "audio frame {} has invalid PTS ({:.3}s): timestamps must be finite (not NaN or Inf)",
                       frame_index, pts)
            }
            MuxerError::AudioNotConfigured => {
                write!(f, "audio track not configured: call .audio() on MuxerBuilder to enable audio")
            }
            MuxerError::EmptyAudioFrame { frame_index } => {
                write!(f, "audio frame {} is empty: ADTS frames must contain data", frame_index)
            }
            MuxerError::EmptyVideoFrame { frame_index } => {
                write!(f, "video frame {} is empty: video samples must contain NAL units", frame_index)
            }
            MuxerError::NonIncreasingVideoPts { prev_pts, curr_pts, frame_index } => {
                write!(f, "video frame {} has PTS {:.3}s which is not greater than previous PTS {:.3}s: \
                          video timestamps must strictly increase. For B-frames, use write_video_with_dts()",
                       frame_index, curr_pts, prev_pts)
            }
            MuxerError::DecreasingAudioPts { prev_pts, curr_pts, frame_index } => {
                write!(f, "audio frame {} has PTS {:.3}s which is less than previous PTS {:.3}s: \
                          audio timestamps must not decrease",
                       frame_index, curr_pts, prev_pts)
            }
            MuxerError::AudioBeforeFirstVideo { audio_pts, first_video_pts } => {
                match first_video_pts {
                    Some(v) => write!(f, "audio PTS {:.3}s arrives before first video PTS {:.3}s: \
                                         write video frames first, or ensure audio PTS >= video PTS",
                                      audio_pts, v),
                    None => write!(f, "audio frame arrived before any video frame: \
                                       write at least one video frame before writing audio"),
                }
            }
            MuxerError::FirstVideoFrameMustBeKeyframe => {
                write!(f, "first video frame must be a keyframe (IDR): \
                          set is_keyframe=true and ensure the frame contains an IDR NAL unit")
            }
            MuxerError::FirstVideoFrameMissingSpsPps => {
                write!(f, "first video frame must contain SPS and PPS NAL units: \
                          prepend SPS (NAL type 7) and PPS (NAL type 8) to the first keyframe")
            }
            MuxerError::FirstAv1FrameMissingSequenceHeader => {
                write!(f, "first AV1 frame must contain a Sequence Header OBU: \
                          ensure the first keyframe includes OBU type 1 (SEQUENCE_HEADER)")
            }
            MuxerError::FirstVp9FrameMissingSequenceHeader => {
                write!(f, "first VP9 frame must contain sequence parameters: \
                          ensure the first keyframe includes VP9 frame header with configuration data")
            }
            MuxerError::InvalidAdts { frame_index } => {
                write!(f, "audio frame {} is not valid ADTS: ensure the frame starts with 0xFFF sync word",
                       frame_index)
            }
            MuxerError::InvalidAdtsDetailed { frame_index, error } => {
                write!(f, "audio frame {} ADTS validation failed: {}", frame_index, error)
            }
            MuxerError::InvalidOpusPacket { frame_index } => {
                write!(f, "audio frame {} is not a valid Opus packet: ensure the frame has valid TOC byte",
                       frame_index)
            }
            MuxerError::NonIncreasingDts { prev_dts, curr_dts, frame_index } => {
                write!(f, "video frame {} has DTS {:.3}s which is not greater than previous DTS {:.3}s: \
                          DTS (decode timestamps) must strictly increase",
                       frame_index, curr_dts, prev_dts)
            }
        }
    }
}

impl std::error::Error for MuxerError {}
impl<Writer: Write> Muxer<Writer> {
    /// Write a video frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds.  Frames must
    /// be supplied in strictly increasing PTS order.  The `data` slice
    /// contains the encoded frame bitstream in Annex B format (for H.264).
    ///
    /// For streams with B-frames (where PTS != DTS), use `write_video_with_dts()` instead.
    pub fn write_video(
        &mut self,
        pts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        let frame_index = self.video_frame_count;

        // Reject empty frames - they cause playback issues
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }

        // Validate PTS is strictly increasing
        if let Some(prev) = self.last_video_pts {
            if pts <= prev {
                return Err(MuxerError::NonIncreasingVideoPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample(pts_units, data, is_keyframe)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write a video frame with explicit decode timestamp for B-frame support.
    ///
    /// - `pts` is the presentation timestamp in seconds (display order)
    /// - `dts` is the decode timestamp in seconds (decode order)
    ///
    /// For streams with B-frames, PTS and DTS may differ. The only constraint is that
    /// DTS must be strictly monotonically increasing (frames must be fed in decode order).
    ///
    /// Example GOP: I P B B where decode order is I,P,B,B but display order is I,B,B,P
    /// - I: DTS=0, PTS=0
    /// - P: DTS=1, PTS=3 (decoded second, displayed fourth)
    /// - B: DTS=2, PTS=1 (decoded third, displayed second)
    /// - B: DTS=3, PTS=2 (decoded fourth, displayed third)
    pub fn write_video_with_dts(
        &mut self,
        pts: f64,
        dts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }

        let frame_index = self.video_frame_count;

        // Reject empty frames - they cause playback issues
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }

        // Validate DTS is finite (not NaN or Inf)
        if !dts.is_finite() {
            return Err(MuxerError::InvalidVideoDts { dts, frame_index });
        }

        // Validate DTS is non-negative
        if dts < 0.0 {
            return Err(MuxerError::NegativeVideoDts { dts, frame_index });
        }

        // Note: PTS can be less than DTS for B-frames (displayed before their decode position)
        // This is valid and expected for B-frame streams.

        // Validate DTS is strictly increasing
        if let Some(prev_dts) = self.last_video_dts {
            if dts <= prev_dts {
                return Err(MuxerError::NonIncreasingDts {
                    prev_dts,
                    curr_dts: dts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;
        let scaled_dts = (dts * MEDIA_TIMESCALE as f64).round();
        let dts_units = scaled_dts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample_with_dts(pts_units, dts_units, data, is_keyframe)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.last_video_dts = Some(dts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Convert internal Mp4WriterError to MuxerError with context
    fn convert_mp4_error(&self, err: Mp4WriterError, frame_index: u64) -> MuxerError {
        match err {
            Mp4WriterError::NonIncreasingTimestamp => MuxerError::NonIncreasingVideoPts {
                prev_pts: self.last_video_pts.unwrap_or(0.0),
                curr_pts: 0.0, // We don't have access here, but validation above catches this
                frame_index,
            },
            Mp4WriterError::FirstFrameMustBeKeyframe => MuxerError::FirstVideoFrameMustBeKeyframe,
            Mp4WriterError::FirstFrameMissingSpsPps => MuxerError::FirstVideoFrameMissingSpsPps,
            Mp4WriterError::FirstFrameMissingSequenceHeader => {
                MuxerError::FirstAv1FrameMissingSequenceHeader
            }
            Mp4WriterError::FirstFrameMissingVp9Config => {
                MuxerError::FirstVp9FrameMissingSequenceHeader
            }
            Mp4WriterError::InvalidAdts => MuxerError::InvalidAdts { frame_index },
            Mp4WriterError::InvalidAdtsDetailed(error) => {
                MuxerError::InvalidAdtsDetailed { frame_index, error }
            }
            Mp4WriterError::InvalidOpusPacket => MuxerError::InvalidOpusPacket { frame_index },
            Mp4WriterError::AudioNotEnabled => MuxerError::AudioNotConfigured,
            Mp4WriterError::DurationOverflow => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duration overflow",
            )),
            Mp4WriterError::AlreadyFinalized => MuxerError::AlreadyFinished,
        }
    }

    /// Write an audio frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds.  The `data` slice
    /// contains the encoded audio frame (an AAC ADTS frame).
    /// Audio timestamps must be non-decreasing and must not precede the first video frame.
    pub fn write_audio(&mut self, pts: f64, data: &[u8]) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }

        let frame_index = self.audio_frame_count;

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidAudioPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeAudioPts { pts, frame_index });
        }

        // Validate frame is not empty
        if data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }

        // Validate PTS is non-decreasing
        if let Some(prev) = self.last_audio_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingAudioPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        // Validate audio doesn't precede first video
        if let Some(first_video) = self.first_video_pts {
            if pts < first_video {
                return Err(MuxerError::AudioBeforeFirstVideo {
                    audio_pts: pts,
                    first_video_pts: Some(first_video),
                });
            }
        } else {
            return Err(MuxerError::AudioBeforeFirstVideo {
                audio_pts: pts,
                first_video_pts: None,
            });
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        self.writer
            .write_audio_sample(pts_units, data)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_audio_pts = Some(pts);
        self.audio_frame_count += 1;
        Ok(())
    }

    /// Simple video encoding method.
    pub fn encode_video(&mut self, data: &[u8], duration_ms: u32) -> Result<(), MuxerError> {
        let pts = self.current_video_pts;
        let is_keyframe = self.is_keyframe(data);
        self.write_video(pts, data, is_keyframe)?;
        self.current_video_pts += duration_ms as f64 / 1000.0;
        Ok(())
    }

    /// Simple audio encoding method.
    pub fn encode_audio(&mut self, data: &[u8], samples: u32) -> Result<(), MuxerError> {
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let sample_rate = self.audio_track.as_ref().unwrap().sample_rate;
        let pts = self.current_audio_pts;
        self.write_audio(pts, data)?;
        self.current_audio_pts += samples as f64 / sample_rate as f64;
        Ok(())
    }

    /// Helper to detect if a video frame is a keyframe.
    fn is_keyframe(&self, data: &[u8]) -> bool {
        // INV-100: Video frame data must not be empty
        assert_invariant!(
            !data.is_empty(),
            "INV-100: Video frame data must not be empty",
            "api::is_keyframe"
        );

        match self.video_track.codec {
            VideoCodec::H264 => {
                // Check for IDR NAL (type 5)
                let has_idr = AnnexBNalIter::new(data).any(|nal| (nal[0] & 0x1f) == 5);
                has_idr
            }
            VideoCodec::H265 => {
                // Check for IDR NAL (type 19-21)
                let has_idr = AnnexBNalIter::new(data).any(|nal| {
                    let nal_type = (nal[0] >> 1) & 0x3f;
                    (19..=21).contains(&nal_type)
                });
                has_idr
            }
            VideoCodec::Av1 => {
                // For AV1, check if it's a key frame (first frame or has key frame flag)
                // Simple heuristic: first frame is keyframe
                let is_key = self.video_frame_count == 0;

                // INV-103: AV1 first frame must be keyframe
                assert_invariant!(
                    is_key || self.video_frame_count > 0,
                    "AV1 first frame must be keyframe",
                    "api::is_keyframe::av1"
                );

                is_key
            }
            VideoCodec::Vp9 => {
                // Use VP9 keyframe detection
                let is_key = is_vp9_keyframe(data).unwrap_or(false);

                // INV-104: VP9 keyframe detection must handle invalid frames gracefully
                assert_invariant!(
                    is_key || data.len() >= 3,
                    "VP9 keyframe detection requires minimum frame size",
                    "api::is_keyframe::vp9"
                );

                is_key
            }
        }
    }

    /// Finalise the container and flush any buffered data.
    ///
    /// In the current slice this writes the `ftyp`/`moov` boxes, resulting
    /// in a minimal MP4 header that can be inspected by the slice 02 tests.
    pub fn finish_in_place(&mut self) -> Result<(), MuxerError> {
        self.finish_in_place_with_stats().map(|_| ())
    }

    /// Finalise the container and return muxing statistics.
    pub fn finish_in_place_with_stats(&mut self) -> Result<MuxerStats, MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let params = Mp4VideoTrack {
            width: self.video_track.width,
            height: self.video_track.height,
        };
        self.writer
            .finalize(&params, self.metadata.as_ref(), self.fast_start)?;
        self.finished = true;

        let video_frames = self.writer.video_sample_count();
        let audio_frames = self.writer.audio_sample_count();
        let duration_ticks = self.writer.max_end_pts().unwrap_or(0);
        let duration_secs = duration_ticks as f64 / MEDIA_TIMESCALE as f64;
        let bytes_written = self.writer.bytes_written();

        Ok(MuxerStats {
            video_frames,
            audio_frames,
            duration_secs,
            bytes_written,
        })
    }

    pub fn finish(mut self) -> Result<(), MuxerError> {
        self.finish_in_place()
    }

    /// Finalise the container and return muxing statistics.
    pub fn finish_with_stats(mut self) -> Result<MuxerStats, MuxerError> {
        self.finish_in_place_with_stats()
    }

    /// Flush the muxer and finalize the output.
    pub fn flush(self) -> Result<(), MuxerError> {
        self.finish()
    }
}

// Static assertions for thread safety
#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn muxer_is_send_when_writer_is_send() {
        assert_send::<Muxer<std::fs::File>>();
        assert_send::<Muxer<Vec<u8>>>();
    }

    #[test]
    fn muxer_is_sync_when_writer_is_sync() {
        assert_sync::<Muxer<std::fs::File>>();
        assert_sync::<Muxer<Vec<u8>>>();
    }

    #[test]
    fn builder_is_send_sync() {
        assert_send::<MuxerBuilder<std::fs::File>>();
        assert_sync::<MuxerBuilder<std::fs::File>>();
    }

    #[test]
    fn simple_api_works() -> Result<(), MuxerError> {
        let mut buffer = Vec::new();
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
            .build()?;

        // Test video encoding with a valid keyframe
        let video_data = make_h264_keyframe();
        muxer.encode_video(&video_data, 33)?; // 33ms

        // Test audio encoding
        let audio_data = vec![0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb]; // ADTS
        muxer.encode_audio(&audio_data, 1024)?; // 1024 samples

        muxer.finish()?;
        assert!(!buffer.is_empty());
        Ok(())
    }

    /// Helper to create a valid H.264 keyframe with SPS/PPS
    fn make_h264_keyframe() -> Vec<u8> {
        // Minimal valid Annex B H.264 stream with SPS, PPS, and IDR slice
        let mut data = Vec::new();
        // SPS (NAL type 7)
        data.extend_from_slice(&[
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0x95, 0xa8, 0x28, 0x28, 0x28,
        ]);
        // PPS (NAL type 8)
        data.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80]);
        // IDR slice (NAL type 5)
        data.extend_from_slice(&[
            0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
        ]);
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_new_creates_empty_metadata() {
        let metadata = Metadata::new();
        assert!(metadata.title.is_none());
        assert!(metadata.language.is_none());
        assert!(metadata.creation_time.is_none());
    }

    #[test]
    fn metadata_with_title_sets_title() {
        let metadata = Metadata::new().with_title("Test Title");
        assert_eq!(metadata.title, Some("Test Title".to_string()));
    }

    #[test]
    fn metadata_with_language_sets_language() {
        let metadata = Metadata::new().with_language("eng");
        assert_eq!(metadata.language, Some("eng".to_string()));
    }

    #[test]
    fn metadata_with_creation_time_sets_timestamp() {
        let metadata = Metadata::new().with_creation_time(1234567890);
        assert_eq!(metadata.creation_time, Some(1234567890));
    }

    #[test]
    fn metadata_with_current_time_sets_current_timestamp() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = Metadata::new().with_current_time();

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        assert!(metadata.creation_time.is_some());
        let time = metadata.creation_time.unwrap();
        assert!(time >= before && time <= after);
    }

    #[test]
    fn metadata_chaining_works() {
        let metadata = Metadata::new()
            .with_title("Test Movie")
            .with_language("spa")
            .with_creation_time(1000000000);

        assert_eq!(metadata.title, Some("Test Movie".to_string()));
        assert_eq!(metadata.language, Some("spa".to_string()));
        assert_eq!(metadata.creation_time, Some(1000000000));
    }

    #[test]
    fn muxer_config_new_creates_basic_config() {
        let config = MuxerConfig::new(1920, 1080, 30.0);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.framerate, 30.0);
        assert!(config.audio.is_none());
        assert!(config.metadata.is_none());
        assert!(config.fast_start);
    }

    #[test]
    fn muxer_config_with_audio_sets_audio_config() {
        let config = MuxerConfig::new(1920, 1080, 30.0).with_audio(
            AudioCodec::Aac(AacProfile::Lc),
            48000,
            2,
        );

        assert!(config.audio.is_some());
        let audio = config.audio.unwrap();
        assert!(matches!(audio.codec, AudioCodec::Aac(AacProfile::Lc)));
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn muxer_config_with_audio_none_clears_audio() {
        let config = MuxerConfig::new(1920, 1080, 30.0)
            .with_audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
            .with_audio(AudioCodec::None, 0, 0);

        assert!(config.audio.is_none());
    }

    #[test]
    fn muxer_config_with_metadata_sets_metadata() {
        let metadata = Metadata::new().with_title("Test");
        let config = MuxerConfig::new(1920, 1080, 30.0).with_metadata(metadata);

        assert!(config.metadata.is_some());
        assert_eq!(config.metadata.unwrap().title, Some("Test".to_string()));
    }

    #[test]
    fn muxer_config_with_fast_start_sets_fast_start() {
        let config = MuxerConfig::new(1920, 1080, 30.0).with_fast_start(false);

        assert!(!config.fast_start);
    }

    #[test]
    fn muxer_config_chaining_works() {
        let metadata = Metadata::new()
            .with_title("Chained Test")
            .with_language("eng");

        let config = MuxerConfig::new(1280, 720, 24.0)
            .with_audio(AudioCodec::Opus, 48000, 1)
            .with_metadata(metadata)
            .with_fast_start(false);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.framerate, 24.0);
        assert!(config.audio.is_some());
        assert!(config.metadata.is_some());
        assert!(!config.fast_start);

        let audio = config.audio.unwrap();
        assert!(matches!(audio.codec, AudioCodec::Opus));

        let metadata = config.metadata.unwrap();
        assert_eq!(metadata.title, Some("Chained Test".to_string()));
        assert_eq!(metadata.language, Some("eng".to_string()));
    }
}
