#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use hal::{i2c, pac};

use roci_multicopter::bsp::aleph as bsp;
use roci_multicopter::{bmm350, dma::*, i2c_dma::*, monotonic};

#[cortex_m_rt::entry]
fn main() -> ! {
    roci_multicopter::init_heap();

    let cp = cortex_m::Peripherals::take().unwrap();
    let mut dp = pac::Peripherals::take().unwrap();
    let _pins = bsp::Pins::take().unwrap();
    defmt::info!("Configured peripherals");

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    let mut monotonic = monotonic::Monotonic::new(dp.TIM2, &clock_cfg);

    let [i2c1_rx, ..] = dp.DMA1.split();

    defmt::debug!("Initializing I2C + DMA");
    let mut i2c1_dma = I2cDma::new(
        dp.I2C1,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
        i2c1_rx,
        &clock_cfg,
        &mut dp.DMAMUX1,
        &mut dp.DMAMUX2,
    );
    let mut bmm350 = bmm350::Bmm350::new(&mut i2c1_dma, bmm350::Address::Low, &mut delay).unwrap();

    defmt::info!("x,y,z");
    loop {
        let now = monotonic.now();

        delay.delay_ms(2);
        bmm350.update(&mut i2c1_dma, now);
        defmt::info!(
            "{},{},{}",
            bmm350.data.mag[0],
            bmm350.data.mag[1],
            bmm350.data.mag[2]
        );
    }
}
