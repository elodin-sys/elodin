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

/// AVCDecoderConfigurationRecord (`avcC`) for one SPS and one PPS.
/// WebCodecs requires this in `VideoDecoderConfig.description` for `avc1`.
pub fn avc_decoder_configuration_record(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut record = Vec::with_capacity(11 + sps.len() + pps.len());
    record.push(1);
    record.push(sps.get(1).copied().unwrap_or(0));
    record.push(sps.get(2).copied().unwrap_or(0));
    record.push(sps.get(3).copied().unwrap_or(0));
    record.push(0xFF); // lengthSizeMinusOne = 3
    record.push(0xE1);
    record.extend_from_slice(&(u16::try_from(sps.len()).unwrap_or(u16::MAX)).to_be_bytes());
    record.extend_from_slice(sps);
    record.push(1);
    record.extend_from_slice(&(u16::try_from(pps.len()).unwrap_or(u16::MAX)).to_be_bytes());
    record.extend_from_slice(pps);
    record
}

/// Length-prefixed NALs. This is the sample format `avc1` decoders accept.
pub fn nals_to_avcc(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        let len = u32::try_from(nal.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb_round_trip_strips_parameter_sets() {
        let sps = [0x67, 0x42, 0xC0, 0x1F];
        let pps = [0x68, 0xCE, 0x06, 0xE2];
        let idr = [0x65, 0x88, 0x84];
        let mut annexb = Vec::new();
        for nal in [sps.as_slice(), pps.as_slice(), idr.as_slice()] {
            annexb.extend_from_slice(&[0, 0, 0, 1]);
            annexb.extend_from_slice(nal);
        }
        let nals = annexb_nals(&annexb);
        assert_eq!(nals.len(), 3);
        assert!(is_keyframe(&annexb));
        let sps_nal = nals.iter().find(|nal| nal_type(nal) == Some(7)).unwrap();
        let pps_nal = nals.iter().find(|nal| nal_type(nal) == Some(8)).unwrap();
        let record = avc_decoder_configuration_record(sps_nal, pps_nal);
        assert_eq!(record[1], 0x42);
        assert_eq!(&record[8..12], &sps);
        assert_eq!(avc1_codec_string(sps_nal).as_deref(), Some("avc1.42C01F"));
        let kept: Vec<&[u8]> = nals
            .into_iter()
            .filter(|nal| !matches!(nal_type(nal), Some(7 | 8)))
            .collect();
        let avcc = nals_to_avcc(&kept);
        assert_eq!(&avcc[..4], &3u32.to_be_bytes());
        assert_eq!(&avcc[4..], &idr);
    }
}
