use alloc::boxed::Box;
use embedded_io::{ErrorType, Read, ReadReady, Write, WriteReady};
use hal::usart::UartError;

#[cfg(feature = "gps-38400")]
pub const GPS_BAUDRATE: u32 = 38400;
#[cfg(not(feature = "gps-38400"))]
pub const GPS_BAUDRATE: u32 = 9600;

const SYNC1: u8 = 0xB5;
const SYNC2: u8 = 0x62;
const NAV_CLASS: u8 = 0x01;
const NAV_PVT_ID: u8 = 0x07;
const NAV_PVT_LEN: u16 = 92;
const CFG_CLASS: u8 = 0x06;
const CFG_VALSET_ID: u8 = 0x8A;
const MAX_PAYLOAD: usize = 96;

pub trait UbxIo:
    Read + Write + ReadReady + WriteReady + ErrorType<Error = UartError> + 'static
{
}
impl<IO> UbxIo for IO where
    IO: Read + Write + ReadReady + WriteReady + ErrorType<Error = UartError> + 'static
{
}

#[derive(Debug, Clone, Copy, Default, defmt::Format)]
pub struct GpsData {
    pub itow: u32,
    pub unix_epoch_ms: i64,
    pub fix_type: u8,
    pub satellites: u8,
    pub valid_flags: u8,
    pub lat: i32,
    pub lon: i32,
    pub alt_msl: i32,
    pub alt_wgs84: i32,
    pub vel_n: i32,
    pub vel_e: i32,
    pub vel_d: i32,
    pub ground_speed: i32,
    pub heading_motion: i32,
    pub h_acc: u32,
    pub v_acc: u32,
    pub s_acc: u32,
}

fn days_from_civil(year: i32, month: u8, day: u8) -> i64 {
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = month as i32;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

fn utc_to_unix_epoch_ms(
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    nano: i32,
) -> i64 {
    let leap_second = if second == 60 { 1_i64 } else { 0_i64 };
    let sec = core::cmp::min(second, 59) as i64;
    let days = days_from_civil(year as i32, month, day);
    let secs = days * 86_400 + (hour as i64) * 3_600 + (minute as i64) * 60 + sec + leap_second;
    let nanos_ms = (nano as i64).div_euclid(1_000_000);
    secs * 1_000 + nanos_ms
}

enum ParserState {
    WaitSync1,
    WaitSync2,
    ReadClass,
    ReadId,
    ReadLen1,
    ReadLen2,
    ReadPayload,
    ReadCkA,
    ReadCkB,
}

pub struct Ubx {
    io: Box<dyn UbxIo>,
    state: ParserState,
    msg_class: u8,
    msg_id: u8,
    payload_len: u16,
    payload_idx: u16,
    payload: [u8; MAX_PAYLOAD],
    ck_a: u8,
    ck_b: u8,
    pub data: GpsData,
    pub fix_count: u32,
}

impl Ubx {
    pub fn new(mut io: Box<dyn UbxIo>, delay: &mut impl embedded_hal::delay::DelayNs) -> Self {
        send_config(&mut *io, delay);

        Self {
            io,
            state: ParserState::WaitSync1,
            msg_class: 0,
            msg_id: 0,
            payload_len: 0,
            payload_idx: 0,
            payload: [0; MAX_PAYLOAD],
            ck_a: 0,
            ck_b: 0,
            data: GpsData::default(),
            fix_count: 0,
        }
    }

    /// Poll the UART for incoming UBX frames. Returns true when a new NAV-PVT
    /// fix has been parsed and `self.data` updated.
    pub fn update(&mut self) -> bool {
        let mut got_fix = false;
        let mut byte = [0u8; 1];
        while let Ok(true) = self.io.read_ready() {
            match self.io.read(&mut byte) {
                Ok(1) => {
                    if self.parse_byte(byte[0]) {
                        got_fix = true;
                    }
                }
                _ => break,
            }
        }
        got_fix
    }

    fn parse_byte(&mut self, b: u8) -> bool {
        match self.state {
            ParserState::WaitSync1 => {
                if b == SYNC1 {
                    self.state = ParserState::WaitSync2;
                }
                false
            }
            ParserState::WaitSync2 => {
                if b == SYNC2 {
                    self.state = ParserState::ReadClass;
                    self.ck_a = 0;
                    self.ck_b = 0;
                } else if b == SYNC1 {
                    // stay in WaitSync2 for consecutive 0xB5 bytes
                } else {
                    self.state = ParserState::WaitSync1;
                }
                false
            }
            ParserState::ReadClass => {
                self.msg_class = b;
                self.checksum_byte(b);
                self.state = ParserState::ReadId;
                false
            }
            ParserState::ReadId => {
                self.msg_id = b;
                self.checksum_byte(b);
                self.state = ParserState::ReadLen1;
                false
            }
            ParserState::ReadLen1 => {
                self.payload_len = b as u16;
                self.checksum_byte(b);
                self.state = ParserState::ReadLen2;
                false
            }
            ParserState::ReadLen2 => {
                self.payload_len |= (b as u16) << 8;
                self.checksum_byte(b);
                self.payload_idx = 0;
                if self.payload_len == 0 {
                    self.state = ParserState::ReadCkA;
                } else if self.payload_len as usize > MAX_PAYLOAD {
                    self.state = ParserState::WaitSync1;
                } else {
                    self.state = ParserState::ReadPayload;
                }
                false
            }
            ParserState::ReadPayload => {
                self.payload[self.payload_idx as usize] = b;
                self.checksum_byte(b);
                self.payload_idx += 1;
                if self.payload_idx >= self.payload_len {
                    self.state = ParserState::ReadCkA;
                }
                false
            }
            ParserState::ReadCkA => {
                self.state = if b == self.ck_a {
                    ParserState::ReadCkB
                } else {
                    ParserState::WaitSync1
                };
                false
            }
            ParserState::ReadCkB => {
                self.state = ParserState::WaitSync1;
                if b == self.ck_b {
                    self.process_message()
                } else {
                    false
                }
            }
        }
    }

    fn checksum_byte(&mut self, b: u8) {
        self.ck_a = self.ck_a.wrapping_add(b);
        self.ck_b = self.ck_b.wrapping_add(self.ck_a);
    }

    fn process_message(&mut self) -> bool {
        if self.msg_class == NAV_CLASS
            && self.msg_id == NAV_PVT_ID
            && self.payload_len >= NAV_PVT_LEN
        {
            self.parse_nav_pvt();
            return true;
        }
        false
    }

    fn parse_nav_pvt(&mut self) {
        let p = &self.payload;
        let i32_at = |off: usize| i32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]);
        let u16_at = |off: usize| u16::from_le_bytes([p[off], p[off + 1]]);
        let u32_at = |off: usize| u32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]);
        let valid_flags = p[11];
        let unix_epoch_ms = if valid_flags & 0x03 == 0x03 {
            utc_to_unix_epoch_ms(u16_at(4), p[6], p[7], p[8], p[9], p[10], i32_at(16))
        } else {
            0
        };

        self.data = GpsData {
            itow: u32_at(0),
            unix_epoch_ms,
            fix_type: p[20],
            valid_flags,
            satellites: p[23],
            lon: i32_at(24),
            lat: i32_at(28),
            alt_wgs84: i32_at(32),
            alt_msl: i32_at(36),
            h_acc: u32_at(40),
            v_acc: u32_at(44),
            vel_n: i32_at(48),
            vel_e: i32_at(52),
            vel_d: i32_at(56),
            ground_speed: i32_at(60),
            heading_motion: i32_at(64),
            s_acc: u32_at(68),
        };
        self.fix_count += 1;
    }
}

fn send_ubx_msg(io: &mut dyn UbxIo, class: u8, id: u8, payload: &[u8]) {
    let len = payload.len() as u16;
    let len_bytes = len.to_le_bytes();

    let mut ck_a: u8 = 0;
    let mut ck_b: u8 = 0;
    for &b in &[class, id, len_bytes[0], len_bytes[1]] {
        ck_a = ck_a.wrapping_add(b);
        ck_b = ck_b.wrapping_add(ck_a);
    }
    for &b in payload {
        ck_a = ck_a.wrapping_add(b);
        ck_b = ck_b.wrapping_add(ck_a);
    }

    let _ = io.write_all(&[SYNC1, SYNC2, class, id, len_bytes[0], len_bytes[1]]);
    let _ = io.write_all(payload);
    let _ = io.write_all(&[ck_a, ck_b]);
    let _ = io.flush();
}

fn drain_rx(io: &mut dyn UbxIo) {
    let mut junk = [0u8; 64];
    for _ in 0..16 {
        if io.read_ready().unwrap_or(false) {
            let _ = io.read(&mut junk);
        } else {
            break;
        }
    }
}

fn send_config(io: &mut dyn UbxIo, delay: &mut impl embedded_hal::delay::DelayNs) {
    // Allow the GPS module time to boot before sending configuration.
    // M9N and M10 modules need different boot times; 500 ms covers both.
    delay.delay_ms(500);
    drain_rx(io);

    #[rustfmt::skip]
    let payload: [u8; 45] = [
        0x00,       // version (0 for CFG-VALSET, same for M9 and M10)
        0x01,       // layers = RAM
        0x00, 0x00, // reserved

        // CFG-RATE-MEAS = 200ms (5 Hz)
        0x01, 0x00, 0x21, 0x30, 0xC8, 0x00,

        // CFG-MSGOUT-UBX_NAV_PVT_UART1 = 1 (every measurement)
        0x07, 0x00, 0x91, 0x20, 0x01,

        // Disable default NMEA sentences on UART1
        0xBB, 0x00, 0x91, 0x20, 0x00, // GGA
        0xCA, 0x00, 0x91, 0x20, 0x00, // GLL
        0xC0, 0x00, 0x91, 0x20, 0x00, // GSA
        0xC5, 0x00, 0x91, 0x20, 0x00, // GSV
        0xAB, 0x00, 0x91, 0x20, 0x00, // RMC
        0xB1, 0x00, 0x91, 0x20, 0x00, // VTG
    ];

    // Send config twice for reliability -- the first may arrive while the
    // module is still processing boot-up NMEA and get lost.
    send_ubx_msg(io, CFG_CLASS, CFG_VALSET_ID, &payload);
    delay.delay_ms(200);
    drain_rx(io);
    send_ubx_msg(io, CFG_CLASS, CFG_VALSET_ID, &payload);
    delay.delay_ms(200);
    drain_rx(io);

    // Poll UBX-MON-VER to identify the module (response is logged by the
    // parser if we extend it, but for now just triggers activity).
    send_ubx_msg(io, 0x0A, 0x04, &[]);
    delay.delay_ms(100);
}
