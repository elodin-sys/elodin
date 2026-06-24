//! Configuration for RTSP ingest sources.

use crate::Error;

/// A single RTSP source mapped to a DB message-log name.
///
/// The URL may embed credentials (`rtsp://user:pass@host/path`); the session
/// layer extracts them for digest auth and connects with a cleaned URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtspSource {
    /// Message-log name the ingested video is stored under (e.g. `rtsp-camera`).
    pub msg_name: String,
    /// RTSP URL to pull from.
    pub url: String,
}

impl RtspSource {
    /// Parses a `NAME=URL` spec (as passed via a repeatable `--rtsp-source` flag).
    pub fn parse(spec: &str) -> Result<Self, Error> {
        let (name, url) = spec.split_once('=').ok_or(Error::InvalidSourceSpec)?;
        let msg_name = name.trim();
        let url = url.trim();
        if msg_name.is_empty() || url.is_empty() {
            return Err(Error::InvalidSourceSpec);
        }
        Ok(Self {
            msg_name: msg_name.to_string(),
            url: url.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_and_url() {
        let s = RtspSource::parse("rtsp-camera=rtsp://127.0.0.1:8554/test").unwrap();
        assert_eq!(s.msg_name, "rtsp-camera");
        assert_eq!(s.url, "rtsp://127.0.0.1:8554/test");
    }

    #[test]
    fn trims_whitespace() {
        let s = RtspSource::parse("  cam = rtsp://h/s ").unwrap();
        assert_eq!(s.msg_name, "cam");
        assert_eq!(s.url, "rtsp://h/s");
    }

    #[test]
    fn url_with_embedded_equals_is_kept() {
        let s = RtspSource::parse("cam=rtsp://h/s?a=b&c=d").unwrap();
        assert_eq!(s.url, "rtsp://h/s?a=b&c=d");
    }

    #[test]
    fn rejects_missing_separator_or_empty() {
        assert_eq!(
            RtspSource::parse("noequals").unwrap_err(),
            Error::InvalidSourceSpec
        );
        assert_eq!(
            RtspSource::parse("=rtsp://h").unwrap_err(),
            Error::InvalidSourceSpec
        );
        assert_eq!(
            RtspSource::parse("name=").unwrap_err(),
            Error::InvalidSourceSpec
        );
    }
}
