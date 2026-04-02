use core::ops::Deref;

use embedded_hal::delay::DelayNs;
use fugit::{Hertz, MicrosDuration};
use hal::gpio::Pin;
use hal::pac;
use hal::spi::{self, BaudRate, Spi, SpiConfig};

use crate::monotonic::Instant;

type Duration = fugit::Duration<u32, 1, 1_000_000>;

const CONFIG_FILE: &[u8; 8192] = include_bytes!("bmi270_config.bin");

const CHIP_ID: u8 = 0x24;

const POWER_CONF_REG_VAL: u8 = 0b000;
const POWER_CTRL_REG_VAL: u8 = 0b1110;
const START_INITIALIZATION: u8 = 0x00;
const STOP_INITIALIZATION: u8 = 0x01;
const ACC_CONF_REG_VAL: u8 = 0b10101100;
const ACC_RANGE_REG_VAL: u8 = 0x02;
const GYR_CONF_REG_VAL: u8 = 0b11101100;
const GYR_RANGE_REG_VAL: u8 = 0x00;
const GYR_OFFSET_XYZ_VAL: u8 = 0x40;

const POR_DELAY_MICRO_SECONDS: Duration = Duration::micros(550);
const CONFIG_DELAY_MILLI_SECONDS: Duration = Duration::millis(50);
const WRITE_2_WRITE_DELAY_MICROS_SECONDS: Duration = Duration::micros(50);

const POLL_RATE_HZ: Hertz<u32> = Hertz::<u32>::Hz(1600);
const POLL_PERIOD: MicrosDuration<u32> = POLL_RATE_HZ.into_duration();
const SENSOR_TIME_HZ: Hertz<u32> = Hertz::<u32>::Hz(25600);
const GYRO_ODR_HZ: Hertz<u32> = Hertz::<u32>::Hz(1600);
const TIME_PER_SAMPLE: u32 = SENSOR_TIME_HZ.to_Hz() / GYRO_ODR_HZ.to_Hz();

#[repr(u8)]
enum Reg {
    ChipId = 0x00,
    AccelXLsb = 0x0C,
    InternalStatus = 0x21,
    AccelConf = 0x40,
    AccelRange = 0x41,
    GyrConf = 0x42,
    GyrRange = 0x43,
    InitCtrl = 0x59,
    InitAdd0 = 0x5B,
    InitData = 0x5E,
    AccelOffsetX = 0x71,
    AccelOffsetY = 0x72,
    AccelOffsetZ = 0x73,
    GyrOffsetXYZ89 = 0x77,
    PowerConfig = 0x7C,
    PowerCtrl = 0x7D,
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    Spi(spi::SpiError),
    InvalidChipId,
    ConfigError,
}

impl From<spi::SpiError> for Error {
    fn from(err: spi::SpiError) -> Self {
        Error::Spi(err)
    }
}

/// Newtype wrapper so we can implement `RccPeriph` for SPI5 (missing from HAL).
pub struct Spi5Regs(pub pac::SPI5);

impl Deref for Spi5Regs {
    type Target = pac::spi1::RegisterBlock;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl hal::RccPeriph for Spi5Regs {
    fn en_reset(rcc: &pac::rcc::RegisterBlock) {
        rcc.apb2enr.modify(|_, w| w.spi5en().set_bit());
        rcc.apb2rstr.modify(|_, w| w.spi5rst().set_bit());
        rcc.apb2rstr.modify(|_, w| w.spi5rst().clear_bit());
    }
}

pub struct Bmi270Spi {
    spi: Spi<Spi5Regs>,
    cs: Pin,
    accel_scale: f32,
    gyr_scale: f32,
    pub gyro_dps: [f32; 3],
    pub accel_g: [f32; 3],
    pub sample_count: u32,
    pub next_update: Instant,
}

impl Bmi270Spi {
    pub fn new<D: DelayNs>(spi5: pac::SPI5, cs: Pin, delay: &mut D) -> Result<Self, Error> {
        let spi = Spi::new(Spi5Regs(spi5), SpiConfig::default(), BaudRate::Div16);
        let mut s = Self {
            spi,
            cs,
            accel_scale: 0.0,
            gyr_scale: 0.0,
            gyro_dps: [0.0; 3],
            accel_g: [0.0; 3],
            sample_count: 0,
            next_update: Instant::from_ticks(0),
        };
        s.cs.set_high();
        s.init(delay)?;
        Ok(s)
    }

    fn write_register(&mut self, reg: Reg, data: u8) -> Result<(), Error> {
        self.cs.set_low();
        let buf = [(reg as u8) & 0x7F, data];
        let r = self.spi.write(&buf);
        self.cs.set_high();
        r?;
        Ok(())
    }

    fn read_register(&mut self, reg: Reg) -> Result<u8, Error> {
        self.cs.set_low();
        let mut buf = [(reg as u8) | 0x80, 0x00, 0x00];
        let r = self.spi.transfer(&mut buf);
        self.cs.set_high();
        r?;
        Ok(buf[2])
    }

    fn read_bytes(&mut self, reg: Reg, out: &mut [u8]) -> Result<(), Error> {
        self.cs.set_low();
        let mut header = [(reg as u8) | 0x80, 0x00];
        let _ = self.spi.transfer(&mut header);
        let r = self.spi.transfer(out);
        self.cs.set_high();
        r?;
        Ok(())
    }

    fn burst_write(&mut self, data: &[u8]) {
        let config_size = data.len();
        let mut index = 0;
        let chunk_len = 254;

        let remain = config_size % chunk_len;

        while index < config_size {
            let current_chunk_size = if remain == 0 || index < config_size - remain {
                chunk_len
            } else {
                usize::min(2, config_size - index)
            };

            let addr = (index / 2) as u16;
            let addr_array = [(addr & 0x0F) as u8, ((addr >> 4) & 0xFF) as u8];

            self.cs.set_low();
            let _ = self
                .spi
                .write(&[(Reg::InitAdd0 as u8) & 0x7F, addr_array[0], addr_array[1]]);
            self.cs.set_high();

            self.cs.set_low();
            let _ = self.spi.write(&[(Reg::InitData as u8) & 0x7F]);
            let _ = self.spi.write(&data[index..index + current_chunk_size]);
            self.cs.set_high();

            index += current_chunk_size;
        }
    }

    fn init<D: DelayNs>(&mut self, delay: &mut D) -> Result<(), Error> {
        defmt::debug!("Initializing BMI270 via SPI");

        // Rising edge on CS selects SPI mode after power-on
        self.cs.set_low();
        delay.delay_us(1);
        self.cs.set_high();
        delay.delay_ms(1);

        let chip_id = self.read_register(Reg::ChipId)?;
        defmt::debug!("BMI270 SPI chip id {:#04X}", chip_id);
        if chip_id != CHIP_ID {
            return Err(Error::InvalidChipId);
        }

        self.write_register(Reg::PowerConfig, POWER_CONF_REG_VAL)?;
        delay.delay_us(POR_DELAY_MICRO_SECONDS.to_micros());
        self.write_register(Reg::InitCtrl, START_INITIALIZATION)?;
        self.burst_write(CONFIG_FILE);
        self.write_register(Reg::InitCtrl, STOP_INITIALIZATION)?;
        delay.delay_ms(CONFIG_DELAY_MILLI_SECONDS.to_millis());

        let status = self.read_register(Reg::InternalStatus)?;
        defmt::debug!("BMI270 SPI internal status: {=u8}", status);
        if status != 1 {
            return Err(Error::ConfigError);
        }

        self.write_register(Reg::PowerCtrl, POWER_CTRL_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(Reg::AccelConf, ACC_CONF_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());

        self.write_register(Reg::AccelRange, ACC_RANGE_REG_VAL)?;
        self.accel_scale = 2.442002e-4;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());

        self.write_register(Reg::GyrConf, GYR_CONF_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(Reg::GyrRange, GYR_RANGE_REG_VAL)?;
        self.gyr_scale = 6.097561e-2;

        self.write_register(Reg::AccelOffsetX, 0x01)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(Reg::AccelOffsetY, 0x01)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(Reg::AccelOffsetZ, 0x01)?;

        self.write_register(Reg::GyrOffsetXYZ89, GYR_OFFSET_XYZ_VAL)?;

        defmt::info!("BMI270 SPI initialized");
        Ok(())
    }

    pub fn update(&mut self, now: Instant) -> bool {
        if now < self.next_update {
            return false;
        }
        match self.try_update(now) {
            Ok(updated) => updated,
            Err(err) => {
                defmt::trace!("BMI270 SPI error: {}", err);
                self.next_update = now + POLL_PERIOD;
                false
            }
        }
    }

    fn try_update(&mut self, now: Instant) -> Result<bool, Error> {
        let mut raw = [0u8; 15];
        self.read_bytes(Reg::AccelXLsb, &mut raw)?;

        let time = (raw[12] as u32) | ((raw[13] as u32) << 8) | ((raw[14] as u32) << 16);

        let sample = time / TIME_PER_SAMPLE;
        if sample == self.sample_count {
            return Ok(false);
        } else if sample > self.sample_count + 1 && self.sample_count != 0 {
            defmt::trace!(
                "Dropped {} BMI270 SPI sample(s)",
                sample - self.sample_count - 1
            );
        }

        while now >= self.next_update {
            self.next_update += POLL_PERIOD;
        }

        self.sample_count = sample;

        let ax = (i16::from(raw[0]) | (i16::from(raw[1]) << 8)) as f32 * self.accel_scale;
        let ay = (i16::from(raw[2]) | (i16::from(raw[3]) << 8)) as f32 * self.accel_scale;
        let az = (i16::from(raw[4]) | (i16::from(raw[5]) << 8)) as f32 * self.accel_scale;
        self.accel_g = [ax, ay, az];

        let gx = (i16::from(raw[6]) | (i16::from(raw[7]) << 8)) as f32 * self.gyr_scale;
        let gy = (i16::from(raw[8]) | (i16::from(raw[9]) << 8)) as f32 * self.gyr_scale;
        let gz = (i16::from(raw[10]) | (i16::from(raw[11]) << 8)) as f32 * self.gyr_scale;
        self.gyro_dps = [gx, gy, gz];

        Ok(true)
    }
}
