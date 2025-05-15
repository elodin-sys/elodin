// Dummy implementation for non-Apple platforms

use crate::{DecoderOptions, DecodedFrame, Error, H264Decoder, Result};

/// Dummy H264 decoder for non-Apple platforms
#[derive(Debug)]
pub struct VideoToolboxDecoder;

impl H264Decoder for VideoToolboxDecoder {
    fn new(_options: DecoderOptions) -> Result<Self> {
        Err(Error::UnsupportedPlatform)
    }
    
    fn decode_nal(&mut self, _nal_data: &[u8], _pts: i64) -> Result<Option<DecodedFrame>> {
        Err(Error::UnsupportedPlatform)
    }
    
    fn decode(&mut self, _packet: &[u8], _pts: i64) -> Result<Option<DecodedFrame>> {
        Err(Error::UnsupportedPlatform)
    }
    
    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        Err(Error::UnsupportedPlatform)
    }
    
    fn reset(&mut self) -> Result<()> {
        Err(Error::UnsupportedPlatform)
    }
}