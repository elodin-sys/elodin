use core::ops::Deref;

use embedded_io::{ErrorType, Read, ReadReady, Write, WriteReady};
use hal::{pac, usart};

/// A wrapper around an `embedded_io` traits that records errors and resets the USART peripheral
/// if too many errors occur. This is useful for quickly recovering from desync issues caused by
/// physical disconnects + reconnections.
pub struct HealingUsart<R> {
    usart: usart::Usart<R>,
    error_score: u16,
    error_threshold: u16,
}

impl<R> HealingUsart<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock>,
{
    pub fn new(usart: usart::Usart<R>) -> Self {
        Self {
            usart,
            error_score: 0,
            error_threshold: 100,
        }
    }

    fn record_error(&mut self, err: &usart::UartError) {
        if matches!(*err, usart::UartError::Overrun | usart::UartError::Parity) {
            // Ignore overrun and parity errors as they don't indicate a hardware issue
            // that can be recovered from by resetting the peripheral
            return;
        }

        self.error_score += 10;
        if self.error_score > self.error_threshold {
            defmt::warn!(
                "Resetting USART peripheral due to error score ({} > 100)",
                self.error_score
            );
            self.usart.disable();
            self.usart.enable();
            self.error_score = 0;
        }
    }

    fn record_success(&mut self) {
        self.error_score = self.error_score.saturating_sub(1);
    }
}

impl<R> ErrorType for HealingUsart<R> {
    type Error = usart::UartError;
}

impl<R> Read for HealingUsart<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock>,
    usart::Usart<R>: Read<Error = Self::Error>,
{
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        Read::read(&mut self.usart, buf)
            .inspect(|_| self.record_success())
            .inspect_err(|err| self.record_error(err))
    }
}

impl<R> ReadReady for HealingUsart<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock>,
    usart::Usart<R>: ReadReady<Error = Self::Error>,
{
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        self.usart
            .read_ready()
            .inspect_err(|err| self.record_error(err))
    }
}

impl<R> Write for HealingUsart<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock>,
    usart::Usart<R>: Write<Error = Self::Error>,
{
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        // Ignore write errors
        Write::write(&mut self.usart, buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Ignore write errors
        Write::flush(&mut self.usart)
    }
}

impl<R> WriteReady for HealingUsart<R>
where
    R: Deref<Target = pac::usart1::RegisterBlock>,
    usart::Usart<R>: WriteReady<Error = Self::Error>,
{
    fn write_ready(&mut self) -> Result<bool, Self::Error> {
        // Ignore write errors
        self.usart.write_ready()
    }
}
