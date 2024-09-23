#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use fugit::RateExtU32 as _;
use hal::{clocks::Clocks, pac};

use roci_multicopter::{arena::DmaAlloc, peripheral::*, pin::*};

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");

    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();
    defmt::info!("Configured peripherals");

    let clock_cfg = Clocks::default();
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick());
    defmt::info!("Configured clocks");

    // Generate a 600kHz PWM signal on TIM3
    let mut pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    PC6::set(&pwm_timer.ch1());
    PC7::set(&pwm_timer.ch2());
    PC8::set(&pwm_timer.ch3());
    PC9::set(&pwm_timer.ch4());
    pwm_timer.enable_pwm();
    let lo = pwm_timer.lo();
    let hi = pwm_timer.hi();
    defmt::info!("Configured PWM timer");

    let mut dma1 = dp.DMA1.dma();
    let mut dma1_ch1 = dma1.ch1();
    dma1_ch1.mux(&mut pwm_timer);
    let mut alloc = DmaAlloc::take();
    let mut dma_buf = dma1_ch1.buf::<17, _>(&mut alloc);
    defmt::info!("Configured DMA");

    loop {
        delay.delay_us(100);
        defmt::trace!("Generating frame");

        dma_buf.write(&[
            lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, 0,
        ]);
        pwm_timer.write(&mut dma_buf);
    }
}
