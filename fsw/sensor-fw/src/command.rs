use crate::blackbox::{CompassRecord, GpsRecord, ImuRecord, Record};
use alloc::boxed::Box;
use core::ops::Deref;
use embedded_io::{Read, ReadReady, Write};
use hal::pac;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

const EL_MAGIC: [u8; 2] = *b"EL";
const EL_VERSION: u8 = 1;
const EL_KIND_LOG: u8 = 1;
const EL_KIND_GPS: u8 = 2;
const EL_KIND_COMPASS: u8 = 3;
const EL_KIND_IMU: u8 = 4;
const EL_HEADER_LEN: usize = 5;

#[derive(TryFromBytes, Immutable, Clone, defmt::Format)]
pub struct Command {
    gpios: [bool; 8],
}

impl Command {
    pub fn apply(&self, pins: &mut [hal::gpio::Pin; 8]) {
        for (pin, state) in pins.iter_mut().zip(self.gpios.iter()) {
            if *state {
                pin.set_high();
            } else {
                pin.set_low();
            }
        }
    }
}

pub struct CommandBridge<R> {
    uart: Box<crate::healing_usart::HealingUsart<R>>,
    buf: [u8; 128],
    n: usize,
}

impl<R> CommandBridge<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock> + hal::RccPeriph,
    hal::usart::Usart<R>: Write<Error = hal::usart::UartError>,
    hal::usart::Usart<R>: Read<Error = hal::usart::UartError>,
    hal::usart::Usart<R>: ReadReady<Error = hal::usart::UartError>,
{
    pub fn new(uart: Box<crate::healing_usart::HealingUsart<R>>) -> Self {
        Self {
            uart,
            buf: [0u8; 128],
            n: 0,
        }
    }

    pub fn write_record(&mut self, record: &Record) -> (bool, usize) {
        let mut cobs = [0u8; 128];
        let len = cobs::encode(record.as_bytes(), &mut cobs[1..]);
        let write = &cobs[0..=len];
        let ok = self.uart.write_all(write).is_ok();
        (ok, write.len())
    }

    pub fn write_gps_record(&mut self, record: &GpsRecord) -> (bool, usize) {
        self.write_el_frame(EL_KIND_GPS, record.as_bytes())
    }

    pub fn write_compass_record(&mut self, record: &CompassRecord) -> (bool, usize) {
        self.write_el_frame(EL_KIND_COMPASS, record.as_bytes())
    }

    pub fn write_imu_record(&mut self, record: &ImuRecord) -> (bool, usize) {
        self.write_el_frame(EL_KIND_IMU, record.as_bytes())
    }

    pub fn write_log(&mut self, msg: &[u8]) -> (bool, usize) {
        let level: u8 = 0;
        let mut frame = [0u8; 96];
        frame[0] = EL_MAGIC[0];
        frame[1] = EL_MAGIC[1];
        frame[2] = EL_VERSION;
        frame[3] = EL_KIND_LOG;
        frame[4] = level;
        let end = (EL_HEADER_LEN + msg.len()).min(frame.len());
        let msg_len = end - EL_HEADER_LEN;
        frame[EL_HEADER_LEN..end].copy_from_slice(&msg[..msg_len]);

        let mut cobs_buf = [0u8; 128];
        let len = cobs::encode(&frame[..end], &mut cobs_buf[1..]);
        let write = &cobs_buf[0..=len];
        let ok = self.uart.write_all(write).is_ok();
        (ok, write.len())
    }

    fn write_el_frame(&mut self, kind: u8, payload: &[u8]) -> (bool, usize) {
        let mut frame = [0u8; 96];
        frame[0] = EL_MAGIC[0];
        frame[1] = EL_MAGIC[1];
        frame[2] = EL_VERSION;
        frame[3] = kind;
        frame[4] = 0;
        let end = EL_HEADER_LEN + payload.len();
        frame[EL_HEADER_LEN..end].copy_from_slice(payload);

        let mut cobs_buf = [0u8; 128];
        let len = cobs::encode(&frame[..end], &mut cobs_buf[1..]);
        let write = &cobs_buf[0..=len];
        let ok = self.uart.write_all(write).is_ok();
        (ok, write.len())
    }

    pub fn read(&mut self) -> Option<Command> {
        while self.uart.read_ready().ok()? {
            self.n += self.uart.read(&mut self.buf[self.n..]).ok()?;
            let mut decode = [0u8; 128];
            let Some(len) = cobs::decode(&self.buf[..self.n], &mut decode).ok() else {
                continue;
            };
            let cmd = Command::try_read_from_bytes(&decode[..len]).ok()?;
            self.n = 0;
            return Some(cmd);
        }
        None
    }
}
