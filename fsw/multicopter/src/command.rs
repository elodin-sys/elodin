use crate::blackbox::Record;
use alloc::boxed::Box;
use core::ops::Deref;
use embedded_io::{Read, ReadReady, Write};
use hal::pac;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

#[derive(TryFromBytes, Immutable, Clone, defmt::Format)]
pub struct Command {
    gpios: [bool; 8],
}

impl Command {
    pub fn apply(&self, mut pins: [&mut hal::gpio::Pin; 8]) {
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
    R: Deref<Target = pac::usart1::RegisterBlock>,
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

    pub fn write_record(&mut self, record: &Record) {
        let mut cobs = [0u8; 128];
        let len = cobs::encode(record.as_bytes(), &mut cobs[1..]);
        let write = &cobs[0..=len];
        let _ = self.uart.write_all(write);
        let _ = self.uart.flush();
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
