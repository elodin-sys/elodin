//! Annex-B NAL helpers. These run on every target so a native demuxer and a
//! wasm decoder share one implementation.

/// NAL units from an Annex-B buffer, without start codes.
pub fn annexb_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut nal_start: Option<usize> = None;
    let mut i = 0;
    while i + 3 < data.len() {
        let start_code_len = if data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && i + 3 < data.len()
            && data[i + 3] == 1
        {
            4
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            3
        } else {
            i += 1;
            continue;
        };
        if let Some(start) = nal_start {
            nals.push(&data[start..i]);
        }
        nal_start = Some(i + start_code_len);
        i += start_code_len;
    }
    if let Some(start) = nal_start
        && start < data.len()
    {
        nals.push(&data[start..]);
    }
    nals
}

pub fn nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|byte| byte & 0x1F)
}

/// An access unit is a seek point when it contains an IDR picture (NAL type 5).
pub fn is_keyframe(annexb: &[u8]) -> bool {
    annexb_nals(annexb)
        .iter()
        .any(|nal| nal_type(nal) == Some(5))
}

/// `avc1.PPCCLL` from an SPS NAL, including its header byte.
pub fn avc1_codec_string(sps: &[u8]) -> Option<String> {
    if sps.len() < 4 {
        return None;
    }
    Some(format!("avc1.{:02X}{:02X}{:02X}", sps[1], sps[2], sps[3]))
}

/// NALs joined with 4-byte start codes. This is the sample format `avc1`
/// decoders accept when configured without a `description`.
pub fn nals_to_annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nals.iter().map(|nal| nal.len() + 4).sum());
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb_round_trip_drops_access_unit_delimiter() {
        let aud = [0x09, 0x10];
        let sps = [0x67, 0x42, 0xC0, 0x1F];
        let pps = [0x68, 0xCE, 0x06, 0xE2];
        let idr = [0x65, 0x88, 0x84];
        let mut annexb = vec![0, 0, 0, 1];
        annexb.extend_from_slice(&aud);
        for nal in [sps.as_slice(), pps.as_slice(), idr.as_slice()] {
            annexb.extend_from_slice(&[0, 0, 1]);
            annexb.extend_from_slice(nal);
        }
        let nals = annexb_nals(&annexb);
        assert_eq!(nals.len(), 4);
        assert!(is_keyframe(&annexb));
        let sps_nal = nals.iter().find(|nal| nal_type(nal) == Some(7)).unwrap();
        assert_eq!(avc1_codec_string(sps_nal).as_deref(), Some("avc1.42C01F"));
        let kept: Vec<&[u8]> = nals
            .into_iter()
            .filter(|nal| nal_type(nal) != Some(9))
            .collect();
        let sample = nals_to_annexb(&kept);
        let mut expected = Vec::new();
        for nal in [sps.as_slice(), pps.as_slice(), idr.as_slice()] {
            expected.extend_from_slice(&[0, 0, 0, 1]);
            expected.extend_from_slice(nal);
        }
        assert_eq!(sample, expected);
    }
}
