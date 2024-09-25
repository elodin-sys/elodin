#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use fugit::RateExtU32 as _;
use hal::{clocks::Clocks, pac};

use roci_multicopter::{arena::DmaAlloc, dshot, peripheral::*, pin::*};

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");

    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();
    defmt::info!("Configured peripherals");

    let clock_cfg = Clocks::default();
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    // Generate a 600kHz PWM signal on TIM3
    let pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    PC6::set(&pwm_timer.ch1());
    PC7::set(&pwm_timer.ch2());
    PC8::set(&pwm_timer.ch3());
    PC9::set(&pwm_timer.ch4());
    defmt::info!("Configured PWM timer");

    let mut dma1 = dp.DMA1.dma();
    let dma1_ch1 = dma1.ch1();
    let mut alloc = DmaAlloc::take();

    let mut dshot_driver = dshot::Driver::new(pwm_timer, dma1_ch1, &mut alloc);

    let throttle = 0.3;
    dshot_driver.arm_motors(&mut delay);
    loop {
        dshot_driver.write_throttle([throttle.into(); 4]);
        delay.delay_us(100);
    }
}
