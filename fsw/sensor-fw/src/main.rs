#![no_main]
#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use cortex_m::delay::Delay;
use embedded_hal::delay::DelayNs;
use embedded_hal_compat::ForwardCompat;
use embedded_io::Write;
use fugit::{ExtU32 as _, RateExtU32 as _};
use hal::{i2c, pac, usart};

use sensor_fw::bsp::aleph::{self as bsp};
use sensor_fw::{dma::*, i2c_dma::*, peripheral::*, *};

const ELRS_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(8000);
const ELRS_PERIOD: fugit::MicrosDuration<u64> = ELRS_RATE.into_duration();

const CAN_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(10);
const CAN_PERIOD: fugit::MicrosDuration<u64> = CAN_RATE.into_duration();

const USB_LOG_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(50);
const USB_LOG_PERIOD: fugit::MicrosDuration<u64> = USB_LOG_RATE.into_duration();

/// Tracks which hardware components initialized successfully
#[derive(Default)]
struct HardwareStatus {
    sdcard: bool,
    usb_hub: bool,
    bmm350: bool,
    bmp581: bool,
    bmi270: bool,
    fram: bool,
}

impl HardwareStatus {
    fn print_report(&self) {
        defmt::info!("========================================");
        defmt::info!("Hardware Initialization Report");
        defmt::info!("========================================");
        defmt::info!(
            "SD Card:        {}",
            if self.sdcard {
                "✓ Ready"
            } else {
                "✗ Not detected"
            }
        );
        defmt::info!(
            "USB Hub:        {}",
            if self.usb_hub {
                "✓ Ready"
            } else {
                "✗ Not detected"
            }
        );
        defmt::info!(
            "BMM350 (Mag):   {}",
            if self.bmm350 {
                "✓ Ready"
            } else {
                "✗ Failed"
            }
        );
        defmt::info!(
            "BMP581 (Baro):  {}",
            if self.bmp581 {
                "✓ Ready"
            } else {
                "✗ Failed"
            }
        );
        defmt::info!(
            "BMI270 (IMU):   {}",
            if self.bmi270 {
                "✓ Ready"
            } else {
                "✗ Failed"
            }
        );
        defmt::info!(
            "FRAM:           {}",
            if self.fram {
                "✓ Ready"
            } else {
                "✗ Not detected"
            }
        );
        defmt::info!("========================================");

        let ready_count = [
            self.sdcard,
            self.usb_hub,
            self.bmm350,
            self.bmp581,
            self.bmi270,
            self.fram,
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        defmt::info!("Ready: {}/6 devices", ready_count);
    }
}

#[cortex_m_rt::entry]
fn main() -> ! {
    defmt::info!("Starting");
    sensor_fw::init_heap();

    let mut hw_status = HardwareStatus::default();

    let pins = bsp::Pins::take().unwrap();
    let bsp::Pins {
        red_led,
        green_led,
        blue_led,
        mut amber_led,

        // GPIO connector pins
        mut gpio,
        ..
    } = pins;
    defmt::info!("Configured peripherals");
    sensor_fw::set_panic_led(red_led);
    amber_led.set_high();

    let mut cp = cortex_m::Peripherals::take().unwrap();
    let mut dp = pac::Peripherals::take().unwrap();

    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    let clock_cfg = bsp::clock_cfg(dp.PWR);
    clock_cfg.setup().unwrap();
    let mut delay = Delay::new(cp.SYST, clock_cfg.systick()).forward();
    defmt::info!("Configured clocks");

    let mut monotonic = monotonic::Monotonic::new(dp.TIM2, &clock_cfg);
    defmt::info!("Configured monotonic timer");

    let mut blue_led = led::PeriodicLed::new(blue_led, 200u32.millis());

    let elrs_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART2,
        crsf::CRSF_BAUDRATE,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured ELRS UART");
    let mut crsf = crsf::CrsfReceiver::new(elrs_uart);

    let uart_bridge = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART1,
        1000000,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured UART bridge");

    let mut debug_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART6,
        115200,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured debug UART");

    // Generate a 600kHz PWM signal on TIM3
    let pwm_timer = dp.TIM3.timer(600.kHz(), Default::default(), &clock_cfg);
    defmt::info!("Configured PWM timer");

    let [_i2c1_rx, i2c2_rx, i2c3_rx, dshot_tx, ..] = dp.DMA1.split();

    let sd = match sdmmc::Sdmmc::new(&dp.RCC, dp.SDMMC1, &clock_cfg) {
        Ok(sd) => {
            defmt::info!("Configured SDMMC");
            Some(sd)
        }
        Err(e) => {
            defmt::warn!("Failed to configure SDMMC: {:?}", e);
            None
        }
    };
    let mut sdmmc_fs = sd.map(|s| s.fatfs(green_led));
    let mut blackbox = sdmmc_fs.as_mut().map(|fs| fs.blackbox(monotonic.now()));
    hw_status.sdcard = blackbox.is_some();
    if hw_status.sdcard {
        defmt::info!("Configured blackbox");
    }

    defmt::debug!("Initializing I2C + DMA");
    let mut i2c3_dma = I2cDma::new(
        dp.I2C3,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
        i2c3_rx,
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
    // Create I2C1 at 100kHz for USB hub configuration
    let i2c1_low_speed = i2c::I2c::new(
        dp.I2C1,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::Standard100K,
            ..Default::default()
        },
        &clock_cfg,
    );
    defmt::info!("Configured I2C1 at 100kHz for USB hub");

    // Convert to type-erased for USB hub
    let i2c::I2c { regs, cfg } = i2c1_low_speed;
    let mut i2c1_low_speed_erased = i2c::I2c {
        regs: regs.into(),
        cfg,
    };

    let usb_hub = usb2513b::Usb2513b::default();
    hw_status.usb_hub = match usb_hub.configure_if_needed(&mut i2c1_low_speed_erased) {
        Ok(_) => {
            defmt::info!("USB2513B hub configuration complete");
            true
        }
        Err(e) => {
            defmt::warn!(
                "USB2513B hub configuration failed (may not be present): {:?}",
                e
            );
            false
        }
    };

    // Extract and reconfigure I2C1 with higher speed for FRAM
    let i2c::I2c { regs, .. } = i2c1_low_speed_erased;

    let mut i2c1_high_speed = i2c::I2c {
        regs,
        cfg: i2c::I2cConfig {
            speed: i2c::I2cSpeed::FastPlus1M,
            ..Default::default()
        },
    };
    defmt::info!("Reconfigured I2C1 to 1MHz for FRAM");

    let mut bmm350 = match bmm350::Bmm350::new(&mut i2c3_dma, bmm350::Address::Low, &mut delay) {
        Ok(sensor) => {
            defmt::info!("Configured BMM350");
            hw_status.bmm350 = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize BMM350: {:?}", e);
            None
        }
    };

    let mut bmp581 = match bmp581::Bmp581::new(&mut i2c3_dma, bmp581::Address::Low, &mut delay) {
        Ok(sensor) => {
            defmt::info!("Configured BMP581");
            hw_status.bmp581 = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize BMP581: {:?}", e);
            None
        }
    };

    let mut bmi270 = match bmi270::Bmi270::new(&mut i2c2_dma, bmi270::Address::Low, &mut delay) {
        Ok(sensor) => {
            defmt::info!("Configured BMI270");
            hw_status.bmi270 = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize BMI270: {:?}", e);
            None
        }
    };

    let mut fram = match fm24cl16b::Fm24cl16b::new(&mut i2c1_high_speed) {
        Ok(f) => {
            defmt::info!("Configured FRAM");
            Some(f)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize FRAM: {:?}", e);
            None
        }
    };

    let can = can::setup_can(dp.FDCAN1, &dp.RCC);
    let mut can = dronecan::DroneCan::new(can);
    defmt::info!("Configured DroneCAN");

    let mut dshot_driver = dshot::Driver::new(pwm_timer, dshot_tx, &mut dp.DMAMUX1);
    defmt::info!("Configured DSHOT driver");

    // Initialize monitor for voltage and current readings
    let mut monitor = monitor::Monitor::new(
        dp.ADC1,
        dp.ADC3,
        &dp.RCC,
        bsp::monitor::VIN_CHANNEL,
        bsp::monitor::VBAT_CHANNEL,
        bsp::monitor::AUX_CURRENT_CHANNEL,
        bsp::monitor::VIN_DIVIDER,
        bsp::monitor::VBAT_DIVIDER,
        bsp::monitor::AUX_CURRENT_GAIN,
    );
    defmt::info!("Configured voltage/current monitor");

    // Run FRAM self-test if present
    if let Some(ref mut f) = fram {
        hw_status.fram = match f.self_test(&mut i2c1_high_speed) {
            Ok(_) => {
                defmt::info!("FRAM self-test passed");
                true
            }
            Err(e) => {
                defmt::warn!("FRAM self-test failed: {:?}", e);
                false
            }
        };
    }

    // Print hardware initialization report
    hw_status.print_report();

    let mut cmd_bridge = command::CommandBridge::new(uart_bridge);

    let mut last_elrs_update = monotonic.now();
    let mut last_dshot_update = monotonic.now();
    let mut last_can_update = monotonic.now();
    let mut last_usb_log = monotonic.now();

    let usb_alloc = usb_serial::usb_bus(
        dp.OTG2_HS_GLOBAL,
        dp.OTG2_HS_DEVICE,
        dp.OTG2_HS_PWRCLK,
        &clock_cfg,
    );
    let mut usb = usb_serial::UsbSerial::new(&usb_alloc);

    amber_led.set_low();
    loop {
        let now = monotonic.now();
        let ts = now.duration_since_epoch();
        usb.poll();

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
                mag: bmm350.as_ref().map_or([0.0; 3], |s| s.data.mag),
                gyro: bmi270.as_ref().map_or([0.0; 3], |s| s.gyro_dps),
                accel: bmi270.as_ref().map_or([0.0; 3], |s| s.accel_g),
                mag_temp: bmm350.as_ref().map_or(0.0, |s| s.data.temp),
                mag_sample: bmm350.as_ref().map_or(0, |s| s.data.sample),
                baro: bmp581.as_ref().map_or(0.0, |s| s.data.pressure_pascal),
                baro_temp: bmp581.as_ref().map_or(0.0, |s| s.data.temp_c),
                vin: monitor.data.vin,
                vbat: monitor.data.vbat,
                aux_current: monitor.data.aux_current,
                rtc_vbat: monitor.data.rtc_vbat,
                cpu_temp: monitor.data.cpu_temp,
            };
            cmd_bridge.write_record(&record);
        }

        if let Some(command) = cmd_bridge.read() {
            defmt::info!("Received command: {:?}", command);
            command.apply(&mut gpio)
        }

        blue_led.update(now);
        let mag_updated = bmm350
            .as_mut()
            .is_some_and(|s| s.update(&mut i2c3_dma, now));
        if let Some(ref mut s) = bmi270 {
            let _ = s.update(&mut i2c2_dma, now);
        }
        if let Some(ref mut s) = bmp581 {
            let _ = s.update(&mut i2c3_dma, now);
        }
        let _ = monitor.update(now);

        if mag_updated {
            let record = blackbox::Record {
                ts: ts.to_millis() as u32,
                mag: bmm350.as_ref().map_or([0.0; 3], |s| s.data.mag),
                gyro: bmi270.as_ref().map_or([0.0; 3], |s| s.gyro_dps),
                accel: bmi270.as_ref().map_or([0.0; 3], |s| s.accel_g),
                mag_temp: bmm350.as_ref().map_or(0.0, |s| s.data.temp),
                mag_sample: bmm350.as_ref().map_or(0, |s| s.data.sample),
                baro: bmp581.as_ref().map_or(0.0, |s| s.data.pressure_pascal),
                baro_temp: bmp581.as_ref().map_or(0.0, |s| s.data.temp_c),
                vin: monitor.data.vin,
                vbat: monitor.data.vbat,
                aux_current: monitor.data.aux_current,
                rtc_vbat: monitor.data.rtc_vbat,
                cpu_temp: monitor.data.cpu_temp,
            };
            if let Some(ref mut bb) = blackbox {
                bb.write_record(record);
            }
        }

        if mag_updated && bmm350.as_ref().is_some_and(|s| s.data.sample % 400 == 0) {
            defmt::info!(
                "{}: mag: {}, mag_temp: {}, gyro: {}, accel: {}, baro: {}, baro_temp: {}°C",
                ts,
                bmm350.as_ref().map_or([0.0; 3], |s| s.data.mag),
                bmm350.as_ref().map_or(0.0, |s| s.data.temp),
                bmi270.as_ref().map_or([0.0; 3], |s| s.gyro_dps),
                bmi270.as_ref().map_or([0.0; 3], |s| s.accel_g),
                bmp581.as_ref().map_or(0.0, |s| s.data.pressure_pascal),
                bmp581.as_ref().map_or(0.0, |s| s.data.temp_c),
            );
            defmt::info!(
                "{}: vin: {}V, vbat: {}V, current: {}A, rtc_vbat: {}V, cpu_temp: {}°C",
                ts,
                monitor.data.vin,
                monitor.data.vbat,
                monitor.data.aux_current,
                monitor.data.rtc_vbat,
                monitor.data.cpu_temp,
            );
            debug_uart.write_fmt(format_args!("{}\r\n", ts)).unwrap();
        }
        delay.delay_ns(0);
    }
}
