#![no_main]
#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use fugit::RateExtU32 as _;
use hal::{i2c, pac, usart};

use roci_multicopter::bsp::aleph as bsp;
use roci_multicopter::{
    arena::ArenaAlloc, bmm350, crsf, dshot, healing_usart, monotonic, peripheral::*,
};

#[global_allocator]
static HEAP: embedded_alloc::TlsfHeap = embedded_alloc::TlsfHeap::empty();

const ELRS_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(8000);
const MAG_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(800);

const ELRS_PERIOD: fugit::MicrosDuration<u64> = ELRS_RATE.into_duration();
const MAG_PERIOD: fugit::MicrosDuration<u64> = MAG_RATE.into_duration();

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");
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
    defmt::info!("Configured peripherals");

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    let mut monotonic = monotonic::Monotonic::new(dp.TIM2, &clock_cfg);
    defmt::info!("Configured monotonic timer");

    let elrs_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART3,
        crsf::CRSF_BAUDRATE,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured ELRS UART");
    let mut crsf = crsf::CrsfReceiver::new(elrs_uart);

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

    let mut last_elrs_update = monotonic.now();
    let mut last_mag_update = monotonic.now();
    let mut last_dshot_update = monotonic.now();

    loop {
        let now = monotonic.now();
        let ts = now.duration_since_epoch();

        if now.checked_duration_since(last_elrs_update).unwrap() > ELRS_PERIOD {
            last_elrs_update = now;
            defmt::trace!("{}: Reading ELRS data", ts);

            crsf.update(monotonic.now());
        } else if now.checked_duration_since(last_mag_update).unwrap() > MAG_PERIOD {
            last_mag_update = now;
            defmt::trace!("{}: Reading BMM350 data", ts);

            match bmm350.read_data() {
                Ok(mag_data) => defmt::trace!("BMM350: {}", mag_data),
                Err(err) => defmt::error!("BMM350 error: {}", err),
            }
        } else if now.checked_duration_since(last_dshot_update).unwrap() > dshot::UPDATE_PERIOD {
            last_dshot_update = now;
            defmt::trace!("{}: Sending DSHOT data", ts);

            let control = crsf.frsky();
            let armed = control.armed();
            dshot_driver.write_throttle([control.throttle.into(); 4], armed, now);
        }

        delay.delay_us(10);
    }
}
