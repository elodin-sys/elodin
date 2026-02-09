/// Shared track configuration used by the API and muxer modules.
///
/// These structs are intentionally minimal and may expand in future slices
/// as additional track metadata is required for the encoder.
#[derive(Debug, Clone)]
pub struct VideoTrackConfig {
    /// Video codec.
    pub codec: crate::api::VideoCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub framerate: f64,
}

#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Audio codec.
    pub codec: crate::api::AudioCodec,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
}
