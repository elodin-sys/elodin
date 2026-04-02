#![no_std]
extern crate alloc;

use impeller2::buf::Buf;
use impeller2::error::Error;

#[derive(Default)]
pub struct FrameDecoderStats {
    pub cobs_decode_errors: u64,
    pub empty_frames: u64,
    pub oversize_frames: u64,
    pub max_buffer_len: usize,
}

#[derive(Default)]
pub struct FrameDecoder<B: Buf<u8> = alloc::vec::Vec<u8>> {
    frame: B,
    state: FrameDecoderState,
    pub stats: FrameDecoderStats,
}

#[derive(Default, Debug)]
enum FrameDecoderState {
    #[default]
    Finding,
    Building,
    Decoded,
}

impl<B: Buf<u8>> FrameDecoder<B> {
    /// Decode the next COBS frame from `data`. Returns `(decoded_frame, bytes_consumed)`.
    /// Call in a loop, advancing the input slice by `consumed` each time, until
    /// `None` is returned (meaning the remaining bytes are an incomplete frame
    /// buffered internally).
    pub fn push<'a>(&'a mut self, data: &[u8]) -> Result<(Option<&'a [u8]>, usize), Error> {
        let mut n = 0;
        while let Some(data) = data.get(n..) {
            if data.is_empty() {
                break;
            }
            match self.state {
                FrameDecoderState::Finding => {
                    let Some((frame_start, _)) = data.iter().enumerate().find(|(_, b)| **b == 0x00)
                    else {
                        n += data.len();
                        continue;
                    };
                    n += frame_start + 1;
                    self.state = FrameDecoderState::Building;
                    continue;
                }
                FrameDecoderState::Building => {
                    match data.iter().enumerate().find(|(_, b)| **b == 0x00) {
                        Some((frame_end, _)) => {
                            if frame_end >= 255 {
                                n += frame_end + 1;
                                self.stats.oversize_frames += 1;
                                self.clear();
                                continue;
                            }
                            let _ = self.frame.extend_from_slice(&data[..frame_end]);
                            n += frame_end + 1;
                            let buf_len = self.frame.as_slice().len();
                            if buf_len > self.stats.max_buffer_len {
                                self.stats.max_buffer_len = buf_len;
                            }
                            let Ok(len) = cobs::decode_in_place(self.frame.as_mut_slice()) else {
                                self.stats.cobs_decode_errors += 1;
                                self.clear();
                                continue;
                            };
                            if len == 0 {
                                self.stats.empty_frames += 1;
                                self.frame.clear();
                                continue;
                            }
                            self.state = FrameDecoderState::Decoded;
                            return Ok((Some(&self.frame.as_slice()[..len]), n));
                        }
                        None => {
                            n += data.len();
                            if self.frame.extend_from_slice(data).is_err() {
                                self.clear();
                                break;
                            }
                        }
                    };
                }
                FrameDecoderState::Decoded => {
                    self.frame.clear();
                    self.state = FrameDecoderState::Building;
                    continue;
                }
            }
        }
        Ok((None, n))
    }

    pub fn clear(&mut self) {
        self.frame.clear();
        self.state = FrameDecoderState::Finding;
    }
}
