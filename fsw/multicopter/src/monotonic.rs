use fugit::RateExtU32 as _;
use hal::{clocks, timer};

use crate::peripheral::*;

pub type Instant = fugit::TimerInstant<u64, 1_000_000>;

/// Monotonic timer
pub struct Monotonic<T>
where
    T: HalTimerRegExt,
    timer::Timer<T>: HalTimerExt<HalTimerReg = T>,
{
    timer: timer::Timer<T>,
    wrap_count: u32,
}

impl<T> Monotonic<T>
where
    T: HalTimerRegExt,
    timer::Timer<T>: HalTimerExt<HalTimerReg = T>,
{
    pub fn new(timer_reg: T, clocks: &clocks::Clocks) -> Self {
        let timer_clock_speed = timer_reg.clock_speed(clocks);
        // Set prescaler such that timer clock is 1MHz
        let psc = (timer_clock_speed.to_Hz() / 1_000_000 - 1) as u16;
        let timer_cfg = timer::TimerConfig {
            update_request_source: timer::UpdateReqSrc::OverUnderFlow,
            ..Default::default()
        };
        let mut timer = timer_reg.timer(1.Hz(), timer_cfg, clocks);
        // Set auto-reload to maximum possible value to minimize overflow frequency
        // At 1MHz, u32 overflows every ~1.2 hours
        timer.update_psc_arr(psc, u32::MAX);
        timer.enable();
        Self {
            timer,
            wrap_count: 0,
        }
    }

    /// Returns elapsed ticks (1 tick = 1 us).
    /// Automatically handles overflow by incrementing internal wrap count, but can't detect
    /// multiple overflows between calls. This isn't an issue because it requires a 1.2 hour
    /// gap between calls to this function and the monotonic guarantee is still maintained.
    /// For long time horizon time-keeping, use RTC instead.
    pub fn elapsed_ticks(&mut self) -> u64 {
        if self.timer.overflow() {
            self.wrap_count += 1;
        }
        self.timer.read_count() as u64 + self.wrap_count as u64 * u32::MAX as u64
    }

    /// Returns elapsed time.
    pub fn now(&mut self) -> Instant {
        fugit::TimerInstant::<u64, 1_000_000>::from_ticks(self.elapsed_ticks())
    }
}
