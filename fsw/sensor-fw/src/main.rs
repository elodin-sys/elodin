#![no_main]
#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use fugit::{ExtU32 as _, RateExtU32 as _};
use hal::{i2c, pac, usart};

use sensor_fw::bsp::aleph as bsp;
use sensor_fw::{dma::*, i2c_dma::*, peripheral::*, *};

const ELRS_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(8000);
const ELRS_PERIOD: fugit::MicrosDuration<u64> = ELRS_RATE.into_duration();

const CAN_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(10);
const CAN_PERIOD: fugit::MicrosDuration<u64> = CAN_RATE.into_duration();

const USB_LOG_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(50);
const USB_LOG_PERIOD: fugit::MicrosDuration<u64> = USB_LOG_RATE.into_duration();

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");
    sensor_fw::init_heap();

    let pins = bsp::Pins::take().unwrap();
    let bsp::Pins {
        pd11: led_sr0,
        pd10: led_sg0,
        pb15: mut _led_sb0,
        pb14: mut led_sa0,
        mut gpio_connector,
        ..
    } = pins;
    defmt::info!("Configured peripherals");
    led_sa0.set_high();

    let mut cp = cortex_m::Peripherals::take().unwrap();
    let mut dp = pac::Peripherals::take().unwrap();

    // Enable tracing and debugging in general
    cp.DCB.enable_trace();
    // Enable the cycle counter
    cp.DWT.enable_cycle_counter();

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

    let uart_bridge = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.UART8,
        115200,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured UART bridge");

    // Generate a 600kHz PWM signal on TIM3
    let pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    defmt::info!("Configured PWM timer");

    let [i2c1_rx, i2c2_rx, dshot_tx, ..] = dp.DMA1.split();

    let sd = sdmmc::Sdmmc::new(&dp.RCC, dp.SDMMC1, &clock_cfg).unwrap();
    defmt::info!("Configured SDMMC");
    let mut sdmmc_fs = sd.fatfs(led_sr0);
    let mut blackbox = sdmmc_fs.blackbox(monotonic.now());
    defmt::info!("Configured blackbox");

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
    let mut i2c2_dma = I2cDma::new(
        dp.I2C2,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
        i2c2_rx,
        &clock_cfg,
        &mut dp.DMAMUX1,
        &mut dp.DMAMUX2,
    );
    let mut bmm350 = bmm350::Bmm350::new(&mut i2c1_dma, bmm350::Address::Low, &mut delay).unwrap();
    defmt::info!("Configured BMM350");
    let mut bmp581 = bmp581::Bmp581::new(&mut i2c1_dma, bmp581::Address::Low, &mut delay).unwrap();
    defmt::info!("Configured BMP581");
    let mut bmi270 = bmi270::Bmi270::new(&mut i2c2_dma, bmi270::Address::Low, &mut delay).unwrap();
    defmt::info!("Configured BMI270");

    let can = can::setup_can(dp.FDCAN1, &dp.RCC);
    let mut can = dronecan::DroneCan::new(can);
    defmt::info!("Configured DroneCAN");

    let mut dshot_driver = dshot::Driver::new(pwm_timer, dshot_tx, &mut dp.DMAMUX1);
    defmt::info!("Configured DSHOT driver");

    let mut cmd_bridge = command::CommandBridge::new(uart_bridge);

    let mut last_elrs_update = monotonic.now();
    let mut last_dshot_update = monotonic.now();
    let mut last_can_update = monotonic.now();
    let mut last_usb_log = monotonic.now();

    led_sa0.set_low();
    loop {
        // usb_serial.poll();
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
                if let Ok(msg) = dronecan::Message::try_from(msg) {
                    defmt::debug!("{}: Received message: {}", ts, msg);
                }
            }
        } else if now.checked_duration_since(last_usb_log).unwrap() > USB_LOG_PERIOD {
            last_usb_log = now;
            let record = blackbox::Record {
                ts: ts.to_millis() as u32,
                mag: bmm350.data.mag,
                gyro: bmi270.gyro_dps,
                accel: bmi270.accel_g,
                mag_temp: bmm350.data.temp,
                mag_sample: bmm350.data.sample,
                baro: bmp581.data.pressure_pascal,
                baro_temp: bmp581.data.temp_c,
            };
            cmd_bridge.write_record(&record);
        }

        if let Some(command) = cmd_bridge.read() {
            defmt::info!("Received command: {:?}", command);
            command.apply(gpio_connector.each_mut())
        }

        running_led.update(now);
        let mag_updated = bmm350.update(&mut i2c1_dma, now);
        let _ = bmi270.update(&mut i2c2_dma, now);
        let _ = bmp581.update(&mut i2c1_dma, now);

        if mag_updated {
            let record = blackbox::Record {
                ts: ts.to_millis() as u32,
                mag: bmm350.data.mag,
                gyro: bmi270.gyro_dps,
                accel: bmi270.accel_g,
                mag_temp: bmm350.data.temp,
                mag_sample: bmm350.data.sample,
                baro: bmp581.data.pressure_pascal,
                baro_temp: bmp581.data.temp_c,
            };
            blackbox.write_record(record);
        }

        if mag_updated && bmm350.data.sample % 400 == 0 {
            defmt::info!(
                "{}: BMM350 sample {}: {}",
                ts,
                bmm350.data.sample,
                bmm350.data
            );
        }
        delay.delay_ns(0);
    }
}
