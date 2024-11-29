#![no_main]
#![no_std]

use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use hal::{i2c, pac};

use roci_multicopter::bmm350;
use roci_multicopter::bsp::aleph as bsp;

#[global_allocator]
static HEAP: embedded_alloc::TlsfHeap = embedded_alloc::TlsfHeap::empty();

#[cortex_m_rt::entry]
fn main() -> ! {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) };
        defmt::info!("Configured heap with {} bytes", HEAP_SIZE);
    }

    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();
    let _pins = bsp::Pins::take().unwrap();

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();

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
