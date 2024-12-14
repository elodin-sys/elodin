use hal::gpio;

use crate::monotonic::Instant;

pub struct PeriodicLed {
    led: gpio::Pin,
    period: fugit::MillisDuration<u32>,
    last_update: Option<Instant>,
}

impl PeriodicLed {
    pub fn new(led: gpio::Pin, period: fugit::MillisDuration<u32>) -> Self {
        Self {
            led,
            period,
            last_update: None,
        }
    }

    pub fn update(&mut self, now: Instant) {
        let Some(last_update) = self.last_update else {
            self.led.toggle();
            self.last_update = Some(now);
            return;
        };

        if now
            .checked_duration_since(last_update)
            .map_or(false, |d| d > (self.period / 2))
        {
            defmt::trace!("Toggling LED");
            self.led.toggle();
            self.last_update = Some(now);
        }
    }
}
