use alloc::boxed::Box;
use embedded_io::{ErrorType, Read, ReadReady, Write, WriteReady};
use fugit::MicrosDuration;
use hal::usart::UartError;
use modular_bitfield::prelude::*;

use crate::monotonic::Instant;

pub const CRSF_BAUDRATE: u32 = 420000;
const CRSF_SYNC_BYTE: u8 = 0xC8;
const CRSF_FRAME_SIZE_MAX: usize = 64;
const CRSF_MAX_CHANNEL: usize = 16;

const RC_MIN: u16 = 172;
const RC_MAX: u16 = 1811;
const RC_MID: u16 = (RC_MIN + RC_MAX) / 2;

// CRSF protocol:
// - 420000 baud
// - not inverted
// - 8 data bits, no parity, 1 stop bit
// - big endian
//
// 420000 bit/s = 46667 byte/s (including stop bit) = 21.43µs per byte
// Max frame size: 64 bytes
// A frame can be sent every 1372µs; round up to 1750µs

const LINK_STATUS_UPDATE_TIMEOUT: MicrosDuration<u64> = MicrosDuration::<u64>::millis(250);
const CRSF_TIME_NEEDED_PER_FRAME: MicrosDuration<u64> = MicrosDuration::<u64>::micros(1750);

#[derive(Debug, Clone, Copy, defmt::Format)]
#[repr(u8)]
pub enum FrameType {
    GpsFrame = 0x02,
    BatteryFrame = 0x08,
    LinkStatistics = 0x14,
    RcChannelsPacked = 0x16,
    SubsetRcChannelsPacked = 0x17,
    LinkStatisticsRx = 0x1C,
    LinkStatisticsTx = 0x1D,
    Attitude = 0x1E,
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    Io(UartError),
    InvalidFrameType,
    InvalidFrameLength,
    InvalidCrc,
    InvalidSync,
    BufferOverflow,
}

#[bitfield(bits = 176)] // 16 channels * 11 bits = 176 bits = 22 bytes
#[derive(Default, Debug, Clone, Copy, defmt::Format)]
pub struct RcChannels {
    ch1: B11,
    ch2: B11,
    ch3: B11,
    ch4: B11,
    ch5: B11,
    ch6: B11,
    ch7: B11,
    ch8: B11,
    ch9: B11,
    ch10: B11,
    ch11: B11,
    ch12: B11,
    ch13: B11,
    ch14: B11,
    ch15: B11,
    ch16: B11,
}

#[derive(Debug, Clone, Copy, defmt::Format)]
pub struct Control {
    pub aileron: f32,
    pub elevator: f32,
    pub throttle: f32,
    pub rudder: f32,
    pub aux: [bool; 12],
}

impl Control {
    pub fn armed(&self) -> bool {
        self.aux[3]
    }
}

impl From<RcChannels> for [u16; CRSF_MAX_CHANNEL] {
    fn from(value: RcChannels) -> Self {
        [
            value.ch1(),
            value.ch2(),
            value.ch3(),
            value.ch4(),
            value.ch5(),
            value.ch6(),
            value.ch7(),
            value.ch8(),
            value.ch9(),
            value.ch10(),
            value.ch11(),
            value.ch12(),
            value.ch13(),
            value.ch14(),
            value.ch15(),
            value.ch16(),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InvalidFrameType;

impl From<InvalidFrameType> for Error {
    fn from(_: InvalidFrameType) -> Self {
        Error::InvalidFrameType
    }
}

impl TryFrom<u8> for FrameType {
    type Error = InvalidFrameType;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x02 => Ok(FrameType::GpsFrame),
            0x08 => Ok(FrameType::BatteryFrame),
            0x14 => Ok(FrameType::LinkStatistics),
            0x16 => Ok(FrameType::RcChannelsPacked),
            0x17 => Ok(FrameType::SubsetRcChannelsPacked),
            0x1C => Ok(FrameType::LinkStatisticsRx),
            0x1D => Ok(FrameType::LinkStatisticsTx),
            0x1E => Ok(FrameType::Attitude),
            _ => Err(InvalidFrameType),
        }
    }
}

pub struct CrsfReceiver {
    io: Box<dyn SerialIo>,
    frame_buffer: [u8; CRSF_FRAME_SIZE_MAX],
    frame_pos: usize,
    frame_start_time: Instant,
    channel_data: [u16; CRSF_MAX_CHANNEL],
    last_frame_time: Instant,
}

pub trait SerialIo:
    Read + Write + ReadReady + WriteReady + ErrorType<Error = UartError> + 'static
{
}

impl<IO> SerialIo for IO where
    IO: Read + Write + ReadReady + WriteReady + ErrorType<Error = UartError> + 'static
{
}

impl CrsfReceiver {
    pub fn new(io: Box<dyn SerialIo>) -> Self {
        Self {
            io,
            frame_buffer: [0; CRSF_FRAME_SIZE_MAX],
            frame_pos: 0,
            frame_start_time: Instant::from_ticks(0),
            channel_data: [RC_MID; CRSF_MAX_CHANNEL],
            last_frame_time: Instant::from_ticks(0),
        }
    }

    /// FrSky/Futaba/Hitec channel map (AETR1234)
    /// This is also the Betaflight default channel map
    pub fn frsky(&self) -> Control {
        Control {
            aileron: self.channel(0),
            elevator: self.channel(1),
            throttle: self.channel(2),
            rudder: self.channel(3),
            aux: self.aux(),
        }
    }

    /// Spektrum/Graupned/JR channel map (TAER1234)
    pub fn spektrum(&self) -> Control {
        Control {
            throttle: self.channel(0),
            aileron: self.channel(1),
            elevator: self.channel(2),
            rudder: self.channel(3),
            aux: self.aux(),
        }
    }

    fn channel(&self, index: usize) -> f32 {
        (self.channel_data[index] - RC_MIN) as f32 / (RC_MAX - RC_MIN) as f32
    }

    fn aux(&self) -> [bool; 12] {
        [
            self.channel_data[4] > RC_MID,
            self.channel_data[5] > RC_MID,
            self.channel_data[6] > RC_MID,
            self.channel_data[7] > RC_MID,
            self.channel_data[8] > RC_MID,
            self.channel_data[9] > RC_MID,
            self.channel_data[10] > RC_MID,
            self.channel_data[11] > RC_MID,
            self.channel_data[12] > RC_MID,
            self.channel_data[13] > RC_MID,
            self.channel_data[14] > RC_MID,
            self.channel_data[15] > RC_MID,
        ]
    }

    fn frame_length(&self) -> usize {
        if self.frame_pos > 2 {
            let frame_length = self.frame_buffer[1] as usize;
            (frame_length + 2).min(CRSF_FRAME_SIZE_MAX)
        } else {
            // Assume frame is 5 bytes long until we have received the frame length.
            5
        }
    }

    pub fn update(&mut self, now: Instant) {
        if let Err(err) = self.try_update(now) {
            if self.last_frame_time.ticks() != 0 {
                defmt::warn!("CRSF error: {}", err);
            }
            self.frame_pos = 0;
        }
    }

    pub fn try_update(&mut self, now: Instant) -> Result<bool, Error> {
        // Check if we need to reset frame parsing due to timeout
        if self.frame_pos != 0
            && now
                .checked_duration_since(self.frame_start_time)
                .map_or(true, |d| d > CRSF_TIME_NEEDED_PER_FRAME)
        {
            defmt::warn!("Resetting frame due to CRSF timeout");
            self.frame_pos = 0;
        }

        let mut frame_length = self.frame_length();
        let read_buffer = &mut self.frame_buffer[self.frame_pos..frame_length];

        if !self.io.read_ready().map_err(Error::Io)? {
            return Ok(false);
        }
        match self.io.read(read_buffer) {
            Ok(n) => {
                defmt::trace!("Read {} bytes", n);
                assert!(n <= frame_length - self.frame_pos);
                if self.frame_pos == 0 {
                    // Check if new frame starts with sync byte
                    if self.frame_buffer[0] != CRSF_SYNC_BYTE {
                        self.frame_pos = 0;
                        defmt::debug!("Invalid sync byte: {=u8:x}", self.frame_buffer[0]);
                        return Ok(false);
                    }
                    self.frame_start_time = now;
                }

                self.frame_pos += n;
                // Re-calculate frame length after reading more data
                frame_length = self.frame_length();

                // Check if we have a complete frame
                if self.frame_pos >= frame_length {
                    defmt::trace!("Processing frame of length {}", frame_length);
                    self.process_frame(now)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Err(err) => Err(Error::Io(err)),
        }
    }

    fn process_frame(&mut self, now: Instant) -> Result<(), Error> {
        let frame_type: FrameType = self.frame_buffer[2].try_into()?;
        let frame_length = self.frame_length();
        defmt::trace!("Frame type: {:?}, length: {}", frame_type, frame_length);

        // Check CRC
        let crc = self.frame_buffer[frame_length - 1];
        let payload = &self.frame_buffer[2..frame_length - 1];
        if crc != crc8_dvb_s2(payload) {
            return Err(Error::InvalidCrc);
        }

        if let FrameType::RcChannelsPacked = frame_type {
            // RC data is 22 bytes
            let payload = (&self.frame_buffer[3..25]).try_into().unwrap();
            let rc_channels = RcChannels::from_bytes(payload);
            self.channel_data = rc_channels.into();
            defmt::trace!("RC channels: {}", self.channel_data);
        }
        self.last_frame_time = now;
        self.frame_pos = 0;
        Ok(())
    }

    pub fn is_connected(&self, now: Instant) -> bool {
        now.checked_duration_since(self.last_frame_time)
            .map_or(false, |d| d < LINK_STATUS_UPDATE_TIMEOUT)
    }
}

fn crc8_dvb_s2(data: &[u8]) -> u8 {
    crc(data, 0xD5)
}

fn crc(data: &[u8], poly: u8) -> u8 {
    let mut crc = 0;
    for byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}
