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

const USB_LOG_RATE: fugit::Hertz<u64> = fugit::Hertz::<u64>::Hz(10);
const USB_LOG_PERIOD: fugit::MicrosDuration<u64> = USB_LOG_RATE.into_duration();

/// Tracks which hardware components initialized successfully
#[derive(Default)]
struct HardwareStatus {
    sdcard: bool,
    usb_hub: bool,
    bmm350: bool,
    bmp581: bool,
    bmi270_i2c: bool,
    bmi270_spi: bool,
    fram: bool,
    gps: bool,
    compass: bool,
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
            "BMI270 I2C:     {}",
            if self.bmi270_i2c {
                "✓ Ready"
            } else {
                "✗ Not detected"
            }
        );
        defmt::info!(
            "BMI270 SPI:     {}",
            if self.bmi270_spi {
                "✓ Ready"
            } else {
                "✗ Not detected"
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
        defmt::info!(
            "GPS (UBX):      {}",
            if self.gps {
                "✓ UART ready"
            } else {
                "✗ Not configured"
            }
        );
        defmt::info!(
            "Compass (Ext):  {}",
            if self.compass {
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
            self.bmi270_i2c,
            self.bmi270_spi,
            self.fram,
            self.gps,
            self.compass,
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        defmt::info!("Ready: {}/9 devices", ready_count);
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
        pk1: spi5_cs,

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
        dp.UART8,
        crsf::CRSF_BAUDRATE,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured ELRS UART (UART8)");
    let mut crsf = crsf::CrsfReceiver::new(elrs_uart);

    let uart_bridge = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART1,
        1_000_000,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured UART bridge (2 Mbaud)");

    let mut debug_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART6,
        115200,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured debug UART");

    let gps_uart = Box::new(healing_usart::HealingUsart::new(usart::Usart::new(
        dp.USART2,
        ubx::GPS_BAUDRATE,
        usart::UsartConfig::default(),
        &clock_cfg,
    )));
    defmt::info!("Configured GPS UART (USART2 @ {})", ubx::GPS_BAUDRATE);

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
            defmt::info!("Configured BMI270 (I2C)");
            hw_status.bmi270_i2c = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize BMI270 I2C: {:?}", e);
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

    let mut bmi270_spi = match bmi270_spi::Bmi270Spi::new(dp.SPI5, spi5_cs, &mut delay) {
        Ok(sensor) => {
            defmt::info!("Configured BMI270 (SPI)");
            hw_status.bmi270_spi = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize BMI270 SPI: {:?}", e);
            None
        }
    };

    // I2C4 for external compass (QMC5883L on J7)
    let i2c4_raw = i2c::I2c::new(
        dp.I2C4,
        i2c::I2cConfig {
            speed: i2c::I2cSpeed::Standard100K,
            ..Default::default()
        },
        &clock_cfg,
    );
    let i2c::I2c { regs, cfg } = i2c4_raw;
    let mut i2c4 = i2c::I2c {
        regs: regs.into(),
        cfg,
    };
    defmt::info!("Configured I2C4 for external compass");

    // GPS via UBX on USART2 (J7) -- liveness tracked at runtime
    let mut gps = ubx::Ubx::new(gps_uart, &mut delay);
    defmt::info!("Configured UBX GPS driver");

    // Print hardware initialization report (before compass, so bridge is ready for diagnostics)
    let mut cmd_bridge = command::CommandBridge::new(uart_bridge);

    let mut ext_compass = match qmc5883l::Qmc5883l::new(&mut i2c4, &mut delay) {
        Ok(sensor) => {
            defmt::info!("Configured QMC5883L external compass");
            hw_status.compass = true;
            Some(sensor)
        }
        Err(e) => {
            defmt::warn!("Failed to initialize QMC5883L: {:?}", e);
            None
        }
    };

    hw_status.print_report();

    let imu_count = bmi270.is_some() as u32 + bmi270_spi.is_some() as u32;
    let decimation = if imu_count >= 2 { 4 } else { 2 };
    let mut integrator = coning_sculling::ConingScullingIntegrator::new(decimation);
    let mut last_imu_time = monotonic.now();
    defmt::info!(
        "Coning/sculling: {} IMU(s), N={}, target ~800 Hz",
        imu_count,
        decimation
    );

    let mut i2c_samples_sec: u32 = 0;
    let mut spi_samples_sec: u32 = 0;
    let mut imu_out_sec: u32 = 0;
    let mut last_diag = monotonic.now();
    let mut last_i2c_gyro: [f32; 3] = [0.0; 3];
    let mut last_i2c_accel: [f32; 3] = [0.0; 3];
    let mut last_spi_gyro: [f32; 3] = [0.0; 3];
    let mut last_spi_accel: [f32; 3] = [0.0; 3];
    let mut last_out_gyro: [f32; 3] = [0.0; 3];
    let mut last_out_accel: [f32; 3] = [0.0; 3];
    let mut last_dt_us: u64 = 0;
    let mut loop_count: u32 = 0;
    let mut max_loop_cycles: u32 = 0;
    let mut uart_tx_cycles: u32 = 0;
    let mut loop_start_cycles: u32 = cortex_m::peripheral::DWT::cycle_count();
    let mut tx_ok: u32 = 0;
    let mut tx_err: u32 = 0;
    let mut tx_bytes: u64 = 0;
    let mut repoll_samples: u32 = 0;
    let mut repoll_emits: u32 = 0;
    let mut last_fix_count: u32 = 0;
    let mut gps_ever_seen: bool = false;

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
        let cyc = cortex_m::peripheral::DWT::cycle_count();
        let iter_cycles = cyc.wrapping_sub(loop_start_cycles);
        loop_start_cycles = cyc;
        if iter_cycles > max_loop_cycles {
            max_loop_cycles = iter_cycles;
        }
        loop_count += 1;

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
            if let Some(msg) = can.read(now)
                && let Ok(msg) = dronecan::Message::try_from(msg)
            {
                defmt::debug!("{}: Received message: {}", ts, msg);
            }
        } else if now.checked_duration_since(last_usb_log).unwrap() > USB_LOG_PERIOD {
            last_usb_log = now;
            let record = blackbox::Record {
                baro: bmp581.as_ref().map_or(0.0, |s| s.data.pressure_pascal),
                baro_temp: bmp581.as_ref().map_or(0.0, |s| s.data.temp_c),
                vin: monitor.data.vin,
                vbat: monitor.data.vbat,
                aux_current: monitor.data.aux_current,
                rtc_vbat: monitor.data.rtc_vbat,
                cpu_temp: monitor.data.cpu_temp,
            };
            let _ = cmd_bridge.write_record(&record);
        }

        if let Some(command) = cmd_bridge.read() {
            defmt::info!("Received command: {:?}", command);
            command.apply(&mut gpio)
        }

        blue_led.update(now);
        let mag_updated = bmm350
            .as_mut()
            .is_some_and(|s| s.update(&mut i2c3_dma, now));
        // IMU: poll both BMI270s, feed into coning/sculling integrator, emit at ~800 Hz
        let i2c_imu_updated = bmi270
            .as_mut()
            .is_some_and(|s| s.update(&mut i2c2_dma, now));
        let spi_imu_updated = bmi270_spi.as_mut().is_some_and(|s| s.update(now));

        let mag = bmm350.as_ref().map_or([0.0; 3], |s| s.data.mag);

        macro_rules! feed_integrator {
            ($gyro:expr, $accel:expr) => {{
                let imu_now = monotonic.now();
                let dt_us = imu_now
                    .checked_duration_since(last_imu_time)
                    .map_or(1, |d| d.ticks().max(1));
                last_imu_time = imu_now;
                last_dt_us = dt_us;
                let dt = dt_us as f32 / 1_000_000.0;
                if let Some(out) = integrator.push($gyro, $accel, dt) {
                    imu_out_sec += 1;
                    last_out_gyro = out.gyro;
                    last_out_accel = out.accel;
                    let record = blackbox::ImuRecord {
                        accel: out.accel,
                        gyro: out.gyro,
                        mag,
                    };
                    let c0 = cortex_m::peripheral::DWT::cycle_count();
                    let (ok, len) = cmd_bridge.write_imu_record(&record);
                    uart_tx_cycles += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(c0);
                    if ok {
                        tx_ok += 1;
                        tx_bytes += len as u64;
                    } else {
                        tx_err += 1;
                    }
                    true
                } else {
                    false
                }
            }};
        }

        macro_rules! poll_i2c {
            () => {
                if bmi270
                    .as_mut()
                    .is_some_and(|s| s.update(&mut i2c2_dma, monotonic.now()))
                {
                    let s = bmi270.as_ref().unwrap();
                    i2c_samples_sec += 1;
                    let gyro = [s.gyro_dps[0], s.gyro_dps[1], -s.gyro_dps[2]];
                    let accel = [s.accel_g[0], s.accel_g[1], -s.accel_g[2]];
                    last_i2c_gyro = gyro;
                    last_i2c_accel = accel;
                    repoll_samples += 1;
                    if feed_integrator!(gyro, accel) {
                        repoll_emits += 1;
                    }
                }
            };
        }

        if i2c_imu_updated {
            let s = bmi270.as_ref().unwrap();
            i2c_samples_sec += 1;
            // I2C BMI270 (U16, back of PCB): convert to FRD body frame
            let gyro = [s.gyro_dps[0], s.gyro_dps[1], -s.gyro_dps[2]];
            let accel = [s.accel_g[0], s.accel_g[1], -s.accel_g[2]];
            last_i2c_gyro = gyro;
            last_i2c_accel = accel;
            if feed_integrator!(gyro, accel) {
                poll_i2c!();
            }
        }
        if spi_imu_updated {
            let s = bmi270_spi.as_ref().unwrap();
            spi_samples_sec += 1;
            // SPI BMI270 (U18, front of PCB): align to I2C + convert to FRD
            let gyro = [-s.gyro_dps[0], s.gyro_dps[1], s.gyro_dps[2]];
            let accel = [-s.accel_g[0], s.accel_g[1], s.accel_g[2]];
            last_spi_gyro = gyro;
            last_spi_accel = accel;
            if feed_integrator!(gyro, accel) {
                poll_i2c!();
            }
        }

        if now.checked_duration_since(last_diag).unwrap() > fugit::MicrosDuration::<u64>::secs(1) {
            last_diag = now;
            use core::fmt::Write as FmtWrite;
            let mut log_buf = heapless::String::<86>::new();
            let _ = write!(
                log_buf,
                "i2c={}/s spi={}/s out={}/s dt={}us",
                i2c_samples_sec, spi_samples_sec, imu_out_sec, last_dt_us,
            );
            let _ = cmd_bridge.write_log(log_buf.as_bytes());

            log_buf.clear();
            let _ = write!(
                log_buf,
                "I2C g=[{:.1},{:.1},{:.1}] a=[{:.3},{:.3},{:.3}]",
                last_i2c_gyro[0],
                last_i2c_gyro[1],
                last_i2c_gyro[2],
                last_i2c_accel[0],
                last_i2c_accel[1],
                last_i2c_accel[2],
            );
            let _ = cmd_bridge.write_log(log_buf.as_bytes());

            if spi_samples_sec > 0 {
                log_buf.clear();
                let _ = write!(
                    log_buf,
                    "SPI g=[{:.1},{:.1},{:.1}] a=[{:.3},{:.3},{:.3}]",
                    last_spi_gyro[0],
                    last_spi_gyro[1],
                    last_spi_gyro[2],
                    last_spi_accel[0],
                    last_spi_accel[1],
                    last_spi_accel[2],
                );
                let _ = cmd_bridge.write_log(log_buf.as_bytes());
            }

            log_buf.clear();
            let _ = write!(
                log_buf,
                "OUT g=[{:.1},{:.1},{:.1}] a=[{:.3},{:.3},{:.3}]",
                last_out_gyro[0],
                last_out_gyro[1],
                last_out_gyro[2],
                last_out_accel[0],
                last_out_accel[1],
                last_out_accel[2],
            );
            let _ = cmd_bridge.write_log(log_buf.as_bytes());

            log_buf.clear();
            let _ = write!(
                log_buf,
                "loops={}/s max={}us tx={}us",
                loop_count,
                max_loop_cycles / 400,
                uart_tx_cycles / 400,
            );
            let _ = cmd_bridge.write_log(log_buf.as_bytes());

            let gps_alive = gps.fix_count > last_fix_count;
            if gps_alive {
                gps_ever_seen = true;
            }
            last_fix_count = gps.fix_count;

            log_buf.clear();
            let _ = write!(
                log_buf,
                "tx: {}ok {}err {}KB/s repoll: {}/{}",
                tx_ok,
                tx_err,
                tx_bytes / 1024,
                repoll_samples,
                repoll_emits,
            );
            if gps_ever_seen {
                let _ = write!(log_buf, " gps:{}", if gps_alive { "ok" } else { "--" });
            }
            let _ = cmd_bridge.write_log(log_buf.as_bytes());

            i2c_samples_sec = 0;
            spi_samples_sec = 0;
            imu_out_sec = 0;
            loop_count = 0;
            max_loop_cycles = 0;
            uart_tx_cycles = 0;
            tx_ok = 0;
            tx_err = 0;
            tx_bytes = 0;
            repoll_samples = 0;
            repoll_emits = 0;
        }

        if let Some(ref mut s) = bmp581 {
            let _ = s.update(&mut i2c3_dma, now);
        }
        let _ = monitor.update(now);

        // GPS: poll UART for new UBX frames
        if gps.update() {
            let d = &gps.data;
            let record = blackbox::GpsRecord {
                unix_epoch_ms: d.unix_epoch_ms,
                itow: d.itow,
                lat: d.lat,
                lon: d.lon,
                alt_msl: d.alt_msl,
                alt_wgs84: d.alt_wgs84,
                vel_ned: [d.vel_n, d.vel_e, d.vel_d],
                ground_speed: d.ground_speed.max(0) as u32,
                heading_motion: d.heading_motion,
                h_acc: d.h_acc,
                v_acc: d.v_acc,
                s_acc: d.s_acc,
                fix_type: d.fix_type,
                satellites: d.satellites,
                valid_flags: d.valid_flags,
                _pad: 0,
            };
            let _ = cmd_bridge.write_gps_record(&record);
            if gps.fix_count % 5 == 1 {
                defmt::info!(
                    "GPS fix={} sats={} lat={} lon={} h_acc={}mm",
                    d.fix_type,
                    d.satellites,
                    d.lat,
                    d.lon,
                    d.h_acc,
                );
            }
        }

        // External compass: poll I2C4
        let compass_updated = ext_compass
            .as_mut()
            .is_some_and(|s| s.update(&mut i2c4, now));
        if compass_updated {
            let c = &ext_compass.as_ref().unwrap().data;
            let record = blackbox::CompassRecord {
                mag: [c.mag_x, c.mag_y, c.mag_z],
                status: c.status,
                _pad: 0,
            };
            let _ = cmd_bridge.write_compass_record(&record);
        }

        if mag_updated {
            let record = blackbox::Record {
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
