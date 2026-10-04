//! AVC (length-prefixed) → Annex-B conversion with in-band SPS/PPS injection.
//!
//! `retina` yields depacketized H.264 access units in AVC form (each NAL unit
//! prefixed by a big-endian length) and exposes the parameter sets out-of-band
//! (from SDP). Elodin-DB's storage contract instead requires **Annex-B** access
//! units with start codes and **SPS/PPS repeated in-band ahead of each IDR**, so
//! that `export_videos::find_sps_nal` and the editor decoder can start on any
//! keyframe (equivalent to `h264parse config-interval=-1`).

use crate::Error;

/// 4-byte Annex-B start code prefixed before every emitted NAL unit.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// H.264 NAL unit types relevant to the storage contract.
pub mod nal_type {
    /// Coded slice of a non-IDR picture.
    pub const NON_IDR_SLICE: u8 = 1;
    /// Coded slice of an IDR picture (keyframe).
    pub const IDR_SLICE: u8 = 5;
    /// Sequence parameter set.
    pub const SPS: u8 = 7;
    /// Picture parameter set.
    pub const PPS: u8 = 8;
}

/// Returns the NAL unit type (lower 5 bits of the header byte).
#[inline]
pub fn nal_unit_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| b & 0x1f)
}

/// Raw H.264 parameter set NAL units (no start code, no length prefix).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParameterSets {
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

impl ParameterSets {
    pub fn new(sps: Vec<u8>, pps: Vec<u8>) -> Self {
        Self { sps, pps }
    }

    /// True unless both an SPS and a PPS are present.
    pub fn is_complete(&self) -> bool {
        !self.sps.is_empty() && !self.pps.is_empty()
    }
}

/// Splits an AVC-framed buffer into its constituent NAL unit slices.
///
/// `nal_length_size` is the number of length-prefix bytes (1..=4); `retina`
/// uses 4 by default.
pub fn split_avc_nals(buf: &[u8], nal_length_size: usize) -> Result<Vec<&[u8]>, Error> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(Error::InvalidNalLengthSize(nal_length_size));
    }
    let mut nals = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        if pos + nal_length_size > buf.len() {
            return Err(Error::TruncatedNal);
        }
        let mut len = 0usize;
        for &b in &buf[pos..pos + nal_length_size] {
            len = (len << 8) | b as usize;
        }
        pos += nal_length_size;
        if len == 0 {
            return Err(Error::ZeroLengthNal);
        }
        let end = pos.checked_add(len).ok_or(Error::TruncatedNal)?;
        if end > buf.len() {
            return Err(Error::TruncatedNal);
        }
        nals.push(&buf[pos..end]);
        pos = end;
    }
    if nals.is_empty() {
        return Err(Error::EmptyAccessUnit);
    }
    Ok(nals)
}

/// Returns true if any NAL unit in the Annex-B buffer is an IDR slice.
///
/// Matches the editor's keyframe scan (NAL type 5 after a start code).
pub fn annexb_contains_idr(buf: &[u8]) -> bool {
    split_annexb_nals(buf)
        .iter()
        .any(|nal| nal_unit_type(nal) == Some(nal_type::IDR_SLICE))
}

/// Splits an Annex-B buffer into NAL unit payloads (handles 3- and 4-byte start codes).
pub fn split_annexb_nals(buf: &[u8]) -> Vec<&[u8]> {
    // For each NAL, track where its start code begins and where its payload starts.
    let mut code_begins = Vec::new();
    let mut payload_starts = Vec::new();
    let mut i = 0;
    while i + 3 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 {
            // A leading zero before `00 00 01` belongs to a 4-byte start code.
            let begin = if i > 0 && buf[i - 1] == 0 { i - 1 } else { i };
            code_begins.push(begin);
            payload_starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let total = buf.len();
    let mut nals = Vec::with_capacity(payload_starts.len());
    for (idx, &start) in payload_starts.iter().enumerate() {
        // A NAL ends where the next start code begins (or at the buffer end).
        let end = code_begins.get(idx + 1).copied().unwrap_or(total);
        if end >= start {
            nals.push(&buf[start..end]);
        }
    }
    nals
}

/// Converts AVC-framed access units to Annex-B, injecting SPS/PPS ahead of IDRs.
#[derive(Clone, Debug)]
pub struct AnnexBConverter {
    params: ParameterSets,
    nal_length_size: usize,
}

impl AnnexBConverter {
    /// Creates a converter with the stream's out-of-band parameter sets and the
    /// default 4-byte NAL length size.
    pub fn new(params: ParameterSets) -> Self {
        Self {
            params,
            nal_length_size: 4,
        }
    }

    /// Overrides the AVC NAL length size (1..=4).
    pub fn with_nal_length_size(mut self, n: usize) -> Result<Self, Error> {
        if !(1..=4).contains(&n) {
            return Err(Error::InvalidNalLengthSize(n));
        }
        self.nal_length_size = n;
        Ok(self)
    }

    /// Replaces the parameter sets and NAL length size, e.g. after a mid-stream
    /// resolution change where a fresh avcC may use a different length prefix.
    pub fn update_parameter_sets(
        &mut self,
        params: ParameterSets,
        nal_length_size: usize,
    ) -> Result<(), Error> {
        if !(1..=4).contains(&nal_length_size) {
            return Err(Error::InvalidNalLengthSize(nal_length_size));
        }
        self.params = params;
        self.nal_length_size = nal_length_size;
        Ok(())
    }

    /// Converts one AVC-framed access unit into a self-contained Annex-B access
    /// unit. If the AU contains an IDR slice and does not already carry an SPS
    /// in-band, the stored SPS/PPS are injected ahead of the picture so the
    /// keyframe is independently decodable.
    pub fn convert(&self, avc_au: &[u8]) -> Result<Vec<u8>, Error> {
        let nals = split_avc_nals(avc_au, self.nal_length_size)?;
        let has_idr = nals
            .iter()
            .any(|n| nal_unit_type(n) == Some(nal_type::IDR_SLICE));
        let has_inband_sps = nals.iter().any(|n| nal_unit_type(n) == Some(nal_type::SPS));

        let mut out = Vec::with_capacity(avc_au.len() + nals.len() * START_CODE.len() + 16);
        if has_idr && !has_inband_sps {
            if !self.params.is_complete() {
                return Err(Error::MissingParameterSets);
            }
            push_nal(&mut out, &self.params.sps);
            push_nal(&mut out, &self.params.pps);
        }
        for nal in nals {
            push_nal(&mut out, nal);
        }
        Ok(out)
    }
}

fn push_nal(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&START_CODE);
    out.extend_from_slice(nal);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an AVC access unit by 4-byte-length-prefixing each NAL.
    fn avc_au(nals: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        for nal in nals {
            buf.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            buf.extend_from_slice(nal);
        }
        buf
    }

    fn nal(ty: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![ty & 0x1f];
        v.extend_from_slice(body);
        v
    }

    fn params() -> ParameterSets {
        ParameterSets::new(nal(nal_type::SPS, b"sps"), nal(nal_type::PPS, b"pps"))
    }

    #[test]
    fn non_idr_passes_through_without_injection() {
        let slice = nal(nal_type::NON_IDR_SLICE, b"frame");
        let au = avc_au(&[&slice]);
        let out = AnnexBConverter::new(params()).convert(&au).unwrap();

        let mut expected = START_CODE.to_vec();
        expected.extend_from_slice(&slice);
        assert_eq!(out, expected);
        assert!(!annexb_contains_idr(&out));
    }

    #[test]
    fn idr_injects_sps_pps_ahead_of_keyframe() {
        let idr = nal(nal_type::IDR_SLICE, b"keyframe");
        let au = avc_au(&[&idr]);
        let p = params();
        let out = AnnexBConverter::new(p.clone()).convert(&au).unwrap();

        let nals = split_annexb_nals(&out);
        assert_eq!(nals.len(), 3, "expected SPS + PPS + IDR");
        assert_eq!(nal_unit_type(nals[0]), Some(nal_type::SPS));
        assert_eq!(nal_unit_type(nals[1]), Some(nal_type::PPS));
        assert_eq!(nal_unit_type(nals[2]), Some(nal_type::IDR_SLICE));
        assert_eq!(nals[0], p.sps.as_slice());
        assert!(annexb_contains_idr(&out));
    }

    #[test]
    fn idr_with_inband_sps_is_not_double_injected() {
        let sps = nal(nal_type::SPS, b"sps");
        let pps = nal(nal_type::PPS, b"pps");
        let idr = nal(nal_type::IDR_SLICE, b"keyframe");
        let au = avc_au(&[&sps, &pps, &idr]);
        let out = AnnexBConverter::new(params()).convert(&au).unwrap();

        let sps_count = split_annexb_nals(&out)
            .into_iter()
            .filter(|n| nal_unit_type(n) == Some(nal_type::SPS))
            .count();
        assert_eq!(sps_count, 1, "must not duplicate an already in-band SPS");
    }

    #[test]
    fn idr_without_parameter_sets_errors() {
        let idr = nal(nal_type::IDR_SLICE, b"keyframe");
        let au = avc_au(&[&idr]);
        let err = AnnexBConverter::new(ParameterSets::default())
            .convert(&au)
            .unwrap_err();
        assert_eq!(err, Error::MissingParameterSets);
    }

    #[test]
    fn multi_nal_access_unit_preserves_order() {
        let sei = nal(6, b"sei");
        let slice = nal(nal_type::NON_IDR_SLICE, b"frame");
        let au = avc_au(&[&sei, &slice]);
        let out = AnnexBConverter::new(params()).convert(&au).unwrap();

        let nals = split_annexb_nals(&out);
        assert_eq!(nals.len(), 2);
        assert_eq!(nal_unit_type(nals[0]), Some(6));
        assert_eq!(nal_unit_type(nals[1]), Some(nal_type::NON_IDR_SLICE));
    }

    #[test]
    fn truncated_length_prefix_errors() {
        let buf = vec![0u8, 0, 0]; // 3 bytes, can't hold a 4-byte length
        assert_eq!(split_avc_nals(&buf, 4).unwrap_err(), Error::TruncatedNal);
    }

    #[test]
    fn declared_length_overruns_buffer_errors() {
        let mut buf = 99u32.to_be_bytes().to_vec();
        buf.extend_from_slice(b"short");
        assert_eq!(split_avc_nals(&buf, 4).unwrap_err(), Error::TruncatedNal);
    }

    #[test]
    fn zero_length_nal_errors() {
        let buf = 0u32.to_be_bytes().to_vec();
        assert_eq!(split_avc_nals(&buf, 4).unwrap_err(), Error::ZeroLengthNal);
    }

    #[test]
    fn empty_buffer_errors() {
        assert_eq!(split_avc_nals(&[], 4).unwrap_err(), Error::EmptyAccessUnit);
    }

    #[test]
    fn invalid_nal_length_size_errors() {
        let au = avc_au(&[&nal(nal_type::NON_IDR_SLICE, b"x")]);
        assert_eq!(
            split_avc_nals(&au, 0).unwrap_err(),
            Error::InvalidNalLengthSize(0)
        );
        assert!(
            AnnexBConverter::new(params())
                .with_nal_length_size(5)
                .is_err()
        );
    }

    #[test]
    fn two_byte_nal_length_size_is_supported() {
        let slice = nal(nal_type::NON_IDR_SLICE, b"frame");
        let mut au = (slice.len() as u16).to_be_bytes().to_vec();
        au.extend_from_slice(&slice);
        let out = AnnexBConverter::new(params())
            .with_nal_length_size(2)
            .unwrap()
            .convert(&au)
            .unwrap();
        let nals = split_annexb_nals(&out);
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0], slice.as_slice());
    }
}
