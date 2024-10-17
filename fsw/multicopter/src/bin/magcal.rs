#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use hal::{gpio, i2c, pac};

use roci_multicopter::bsp::aleph as bsp;
use roci_multicopter::{bmm350, pin::*};

#[cortex_m_rt::entry]
fn main() -> ! {
    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();

    PB6::set(&dp.I2C4).output_type(gpio::OutputType::OpenDrain);
    PB7::set(&dp.I2C4).output_type(gpio::OutputType::OpenDrain);
    let i2c = i2c::I2c::new(
        dp.I2C4,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
        &clock_cfg,
    );
    let mut bmm350 = bmm350::Bmm350::new(i2c, bmm350::Address::Low, &mut delay).unwrap();

    defmt::info!("x,y,z");
    loop {
        delay.delay_ms(2);
        let data = bmm350.read_data().unwrap();
        defmt::info!("{},{},{}", data.mag[0], data.mag[1], data.mag[2]);
    }
}
