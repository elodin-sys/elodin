use cortex_m::peripheral::DWT;
use hal::clocks::Clocks;

type Duration = fugit::Duration<u32, 1, 1_000_000>;

pub struct DwtTimer {
    core_frequency: u32,
}

impl DwtTimer {
    pub fn new(clocks: &Clocks) -> Self {
        DwtTimer {
            core_frequency: clocks.d1cpreclk(),
        }
    }

    #[inline(always)]
    pub fn now(&self) -> CycleCount {
        CycleCount {
            cycle_count: DWT::cycle_count(),
            core_frequency: self.core_frequency,
        }
    }
}

pub struct CycleCount {
    cycle_count: u32,
    core_frequency: u32,
}

impl CycleCount {
    pub fn elapsed(&self) -> Duration {
        let elapsed_microseconds = DWT::cycle_count().wrapping_sub(self.cycle_count);
        let cycles_per_microsecond = self.core_frequency / 1_000_000;
        let elapsed_microseconds = elapsed_microseconds / cycles_per_microsecond;
        Duration::micros(elapsed_microseconds)
    }
}
