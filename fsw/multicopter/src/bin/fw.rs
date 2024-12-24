#![no_main]
#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use fugit::{ExtU32 as _, RateExtU32 as _};
use hal::{i2c, pac, usart};

use roci_multicopter::bsp::aleph as bsp;
use roci_multicopter::{dma::*, i2c_dma::*, peripheral::*, *};

const ELRS_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(8000);
const ELRS_PERIOD: fugit::MicrosDuration<u64> = ELRS_RATE.into_duration();

const CAN_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(10);
const CAN_PERIOD: fugit::MicrosDuration<u64> = CAN_RATE.into_duration();

const SD_LOG_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(100);
const SD_LOG_PERIOD: fugit::MicrosDuration<u64> = SD_LOG_RATE.into_duration();

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");
    roci_multicopter::init_heap();

    let pins = bsp::Pins::take().unwrap();
    let bsp::Pins {
        pd11: led_sr0,
        pd10: led_sg0,
        pb15: mut _led_sb0,
        pb14: mut led_sa0,
        ..
    } = pins;
    defmt::info!("Configured peripherals");
    led_sa0.set_high();

    let cp = cortex_m::Peripherals::take().unwrap();
    let mut dp = pac::Peripherals::take().unwrap();

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    let mut monotonic = monotonic::Monotonic::new(dp.TIM2, &clock_cfg);
    defmt::info!("Configured monotonic timer");

    let mut running_led = led::PeriodicLed::new(led_sg0, 100u32.millis());

    let elrs_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART3,
        crsf::CRSF_BAUDRATE,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured ELRS UART");
    let mut crsf = crsf::CrsfReceiver::new(elrs_uart);

    // Generate a 600kHz PWM signal on TIM3
    let pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    defmt::info!("Configured PWM timer");

    let [i2c1_rx, dshot_tx, ..] = dp.DMA1.split();

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

    let can = can::setup_can(dp.FDCAN1, &dp.RCC);
    let mut can = dronecan::DroneCan::new(can);
    defmt::info!("Configured DroneCAN");

    let mut dshot_driver = dshot::Driver::new(pwm_timer, dshot_tx, &mut dp.DMAMUX1);
    defmt::info!("Configured DSHOT driver");

    let sd = sdmmc::Sdmmc::new(&dp.RCC, dp.SDMMC1, &clock_cfg).unwrap();
    let volume_mgr = sd.volume_manager(rtc::FakeTime {});
    defmt::info!("Configured SDMMC");

    let mut blackbox = blackbox::Blackbox::new("aleph", volume_mgr, led_sr0);
    blackbox.arm("test").unwrap();
    defmt::info!("Configured blackbox");

    let mut last_elrs_update = monotonic.now();
    let mut last_dshot_update = monotonic.now();
    let mut last_can_update = monotonic.now();
    let mut last_sd_log = monotonic.now();

    led_sa0.set_low();
    loop {
        let now = monotonic.now();
        let ts = now.duration_since_epoch();

        if now.checked_duration_since(last_elrs_update).unwrap() > ELRS_PERIOD {
            last_elrs_update = now;
            defmt::trace!("{}: Reading ELRS data", ts);

            crsf.update(monotonic.now());
        } else if now.checked_duration_since(last_dshot_update).unwrap() > dshot::UPDATE_PERIOD {
            last_dshot_update = now;
            defmt::trace!("{}: Sending DSHOT data", ts);

            let control = crsf.frsky();
            let armed = control.armed();
            dshot_driver.write_throttle([control.throttle.into(); 4], armed, now);
        } else if now.checked_duration_since(last_can_update).unwrap() > CAN_PERIOD {
            last_can_update = now;
            defmt::trace!("{}: CAN update", ts);
            if let Some(msg) = can.read(now) {
                blackbox.write_can(&msg);
                let msg = dronecan::Message::try_from(msg).unwrap();
                defmt::debug!("{}: Received message: {}", ts, msg);
            }
        } else if now.checked_duration_since(last_sd_log).unwrap() > SD_LOG_PERIOD {
            last_sd_log = now;
            defmt::trace!("{}: Logging to SD card", ts);

            let record = blackbox::Record {
                ts: ts.to_millis() as u32,
                mag: bmm350.data.mag,
                gyro: [0f32; 3],
                accel: [0f32; 3],
                mag_temp: bmm350.data.temp,
                mag_sample: bmm350.data.sample,
            };
            blackbox.write_record(record);
        }

        running_led.update(now);

        let mag_updated = bmm350.update(&mut i2c1_dma, now);
        if mag_updated && bmm350.data.sample % 400 == 0 {
            defmt::info!(
                "{}: BMM350 sample {}: {}",
                ts,
                bmm350.data.sample,
                bmm350.data
            );
        }
        delay.delay_ns(1);
    }
}
