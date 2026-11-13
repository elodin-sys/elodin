//! Wire-level replay keys, shared by both browser sockets.
use impeller::types::Msg;
use impeller_wkt::{ErrorResponse, GetMsgs, GetTimeSeries, GetTimeSeriesPredecessor};

pub fn valid_frame(frame: &[u8]) -> bool {
    frame.len() >= 8
        && u32::from_le_bytes(frame[..4].try_into().expect("length checked")) as usize
            == frame.len() - 4
}
pub fn cacheable(frame: &[u8]) -> bool {
    valid_frame(frame)
        && frame[4] == 0
        && [GetTimeSeries::ID, GetMsgs::ID, GetTimeSeriesPredecessor::ID]
            .contains(&[frame[5], frame[6]])
}
pub fn storable(frame: &[u8]) -> bool {
    valid_frame(frame) && !(frame[4] == 0 && [frame[5], frame[6]] == ErrorResponse::ID)
}
pub fn canonical(frame: &[u8]) -> Vec<u8> {
    let mut bytes = frame.to_vec();
    bytes[7] = 0;
    bytes
}
pub fn patch(frame: &[u8], request: u8) -> Option<Vec<u8>> {
    if !storable(frame) {
        return None;
    }
    let mut bytes = frame.to_vec();
    bytes[7] = request;
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller::types::{LenPacket, PacketTy};
    use impeller_wkt::{SQLQuery, Stream};
    #[test]
    fn only_recording_queries_are_cacheable() {
        for id in [GetTimeSeries::ID, GetMsgs::ID, GetTimeSeriesPredecessor::ID] {
            assert!(cacheable(&LenPacket::new(PacketTy::Msg, id, 0).inner));
        }
        for id in [Stream::ID, SQLQuery::ID] {
            assert!(!cacheable(&LenPacket::new(PacketTy::Msg, id, 0).inner));
        }
        assert!(!storable(
            &LenPacket::new(PacketTy::Msg, ErrorResponse::ID, 0).inner
        ));
    }
    #[test]
    fn canonicalization_and_replay_only_change_request_id() {
        let mut frame = LenPacket::new(PacketTy::Msg, GetMsgs::ID, 0).inner;
        frame[7] = 17;
        let key = canonical(&frame);
        assert_eq!(key[7], 0);
        assert_eq!(&key[..7], &frame[..7]);
        assert_eq!(&key[8..], &frame[8..]);
        assert_eq!(patch(&key, 42).unwrap()[7], 42);
    }
}
