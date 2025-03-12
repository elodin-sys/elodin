use impeller2::buf::Buf;
use impeller2::error::Error;

#[derive(Default)]
pub struct FrameDecoder<B: Buf<u8> = Vec<u8>> {
    frame: B,
    state: FrameDecoderState,
}

#[derive(Default, Debug)]
enum FrameDecoderState {
    #[default]
    Finding,
    Building,
    Decoded,
}

impl<B: Buf<u8>> FrameDecoder<B> {
    pub fn push<'a>(&'a mut self, data: &[u8]) -> Result<Option<&'a [u8]>, Error> {
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
                                self.clear();
                                continue;
                            }
                            let _ = self.frame.extend_from_slice(&data[..frame_end]);
                            n += frame_end.saturating_sub(1);
                            let Ok(len) = cobs::decode_in_place(self.frame.as_mut_slice()) else {
                                self.clear();
                                continue;
                            };
                            if len == 0 {
                                self.clear();
                                continue;
                            }
                            self.state = FrameDecoderState::Decoded;
                            return Ok(Some(&self.frame.as_slice()[..len]));
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
                    self.clear();
                    continue;
                }
            }
        }
        Ok(None)
    }

    pub fn clear(&mut self) {
        self.frame.clear();
        self.state = FrameDecoderState::Finding;
    }
}
