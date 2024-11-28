#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use fugit::RateExtU32 as _;
use hal::{i2c, pac};

use roci_multicopter::bsp::aleph as bsp;
use roci_multicopter::{arena::ArenaAlloc, bmm350, dshot, peripheral::*};

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");

    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();
    let _pins = bsp::Pins::take().unwrap();
    defmt::info!("Configured peripherals");

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    let i2c = i2c::I2c::new(
        dp.I2C1,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
        &clock_cfg,
    );
    let mut bmm350 = bmm350::Bmm350::new(i2c, bmm350::Address::Low, &mut delay).unwrap();
    defmt::info!("Configured BMM350");

    // Generate a 600kHz PWM signal on TIM3
    let pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    defmt::info!("Configured PWM timer");

    let [_, dma1_ch1, ..] = dp.DMA1.split();
    let mut alloc = ArenaAlloc::take();

    let mut dshot_driver = dshot::Driver::new(pwm_timer, dma1_ch1, &mut alloc);

    let throttle = 0.3;
    dshot_driver.arm_motors(&mut delay);
    let mut tick = 0;
    loop {
        tick += 1;
        if tick % 100 == 0 {
            let mag_data = bmm350.read_data().unwrap();
            defmt::info!("BMM350: {}", mag_data);
        }
        dshot_driver.write_throttle([throttle.into(); 4]);
        delay.delay_us(100);
    }
}
