use core::ops::DerefMut;

use embedded_hal::delay::DelayNs;
use fugit::{ExtU32 as _, Hertz, MicrosDuration};

use hal::i2c;

use crate::i2c_dma;
use crate::monotonic::Instant;

type Duration = fugit::Duration<u32, 1, 1_000_000>;

// the file below from: https://github.com/boschsensortec/BMI270_SensorAPI/blob/master/bmi270.c#L51
const CONFIG_FILE: &[u8; 8192] = include_bytes!("bmi270_config.bin");

const CHIP_ID: u8 = 0x24;

const POWER_CONF_REG_VAL: u8 = 0b000; // fast power up = disabled, FIFO read = disabled, advanced power mode = disabled (bits are in order)
const POWER_CTRL_REG_VAL: u8 = 0b1110; // temp on, acc on, gyr on, aux off
const START_INITIALIZATION: u8 = 0x00;
const STOP_INITIALIZATION: u8 = 0x01;
const ACC_CONF_REG_VAL: u8 = 0b10101100; // acc_filter perf = high performance mode
                                         // acc_bwp = filter in normal mode, avberage 8 samples
                                         // acc_odr = 1.6 kHz
const ACC_RANGE_REG_VAL: u8 = 0x02; // acc_range = ±8g
const GYR_CONF_REG_VAL: u8 = 0b11101110; // gyr_filter_perf = high performance mode
                                         // gyr_noise_perf = high performance mdode
                                         // gyr_bwp = normal mode
                                         // gyr_odr = 6.4 kHz
const GYR_RANGE_REG_VAL: u8 = 0x00; // gyr_range = ±2000 dps
const GYR_OFFSET_XYZ_VAL: u8 = 0x40;

const POR_DELAY_MICRO_SECONDS: Duration = Duration::micros(550); // YESSS 501, not 500, 501
const CONFIG_DELAY_MILLI_SECONDS: Duration = Duration::millis(50);
const WRITE_2_WRITE_DELAY_MICROS_SECONDS: Duration = Duration::micros(50);

const SENSOR_ODR_HZ: Hertz<u32> = Hertz::<u32>::Hz(6400);
const SENSORPERIOD_MICRO_SECONDS: MicrosDuration<u32> = SENSOR_ODR_HZ.into_duration();
const SENSOR_TIME_HZ: u32 = 25600;

#[repr(u8)]
pub enum Registers {
    ChipId = 0x00,
    Error = 0x02,
    AccelXLsb = 0x0C,
    AccelXMsb = 0x0D,
    AccelYLsb = 0x0E,
    AccelYMsb = 0x0F,
    AccelZLsb = 0x10,
    AccelZMsb = 0x11,
    GyrXLsb = 0x12,
    GyrXMsb = 0x13,
    GyrYLsb = 0x14,
    GyrYMsb = 0x15,
    GyrZLsb = 0x16,
    GyrZMsb = 0x17,
    InternalStatus = 0x21,
    GyrCas = 0x3C,
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
    GyrOffsetX07 = 0x74,   // bits 0-7 of the 10 bit offset
    GyrOffsetY07 = 0x75,   // bits 0-7 of the 10 bit offset
    GyrOffsetZ07 = 0x76,   // bits 0-7 of the 10 bit offset
    GyrOffsetXYZ89 = 0x77, // bits 8 & 9 of the 10 bit offsets
    PowerConfg = 0x7C,
    PowerCtrl = 0x7D,
}

#[repr(u8)]
pub enum Address {
    Low = 0x68,
    High = 0x69,
}

#[repr(u8)]
enum _AccelOdr {
    Hz0p78 = 0x01,
    Hz1p5 = 0x02,
    Hz3p1 = 0x03,
    Hz6p25 = 0x04,
    Hz12p5 = 0x05,
    Hz25 = 0x06,
    Hz50 = 0x07,
    Hz100 = 0x08,
    Hz200 = 0x09,
    Hz400 = 0x0A,
    Hz800 = 0x0B,
    Hz1k6 = 0x0C,
}

#[repr(u8)]
enum _GyrOdr {
    Hz25 = 0x06,
    Hz50 = 0x07,
    Hz100 = 0x08,
    Hz200 = 0x09,
    Hz400 = 0x0A,
    Hz800 = 0x0B,
    Hz1k6 = 0x0C,
    Hz3k2 = 0x0D,
    Hz6k4 = 0x0E,
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2cDma(i2c_dma::Error),
    InvalidChipId,
    ConfigError,
}

impl From<i2c::Error> for Error {
    fn from(err: i2c::Error) -> Self {
        Error::I2cDma(i2c_dma::Error::I2c(err))
    }
}

impl From<i2c_dma::Error> for Error {
    fn from(err: i2c_dma::Error) -> Self {
        Error::I2cDma(err)
    }
}

pub struct Bmi270 {
    i2c_address: u8,
    accel_scale: f32,
    gyr_scale: f32,
    pub gyro_dps: [f32; 3],
    pub accel_g: [f32; 3],
    sample_count: u32,
    raw_data: [i16; 6], //[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    next_update: Instant,
}

impl Bmi270 {
    pub fn new<D: DelayNs>(
        i2c_dma_perph: &mut i2c_dma::I2cDma,
        i2c_address: Address,
        delay: &mut D,
    ) -> Result<Self, Error> {
        let mut bmi270 = Bmi270 {
            i2c_address: i2c_address as u8,
            accel_scale: 0.0,
            gyr_scale: 0.0,
            gyro_dps: [0f32; 3],
            accel_g: [0f32; 3],
            sample_count: 0,
            raw_data: [0; 6],
            next_update: Instant::from_ticks(0),
        };

        bmi270.init(i2c_dma_perph, delay)?;
        Ok(bmi270)
    }

    fn write_register<D: Into<u8>>(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        register: Registers,
        data: D,
    ) -> Result<(), Error> {
        embedded_hal::i2c::I2c::write(
            i2c_dma.deref_mut(),
            self.i2c_address,
            &[register as u8, data.into()],
        )?;
        Ok(())
    }

    fn read_register(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        register: Registers,
    ) -> Result<u8, Error> {
        let mut data = [0];
        i2c_dma.write_read(self.i2c_address, &[register as u8], &mut data)?;
        Ok(data[0])
    }

    pub fn update(&mut self, i2c_dma: &mut i2c_dma::I2cDma, now: Instant) -> bool {
        if now < self.next_update {
            return false;
        }
        match self.try_update(i2c_dma, now) {
            Ok(updated) => updated,
            Err(err) => {
                defmt::trace!("BMI270 error: {}", err);
                false
            }
        }
    }

    fn try_update(&mut self, i2c_dma: &mut i2c_dma::I2cDma, now: Instant) -> Result<bool, Error> {
        match i2c_dma.state()? {
            i2c_dma::State::Idle => {
                defmt::trace!("Starting BMI270 DMA read");
                i2c_dma.begin_read(self.i2c_address, Registers::AccelXLsb as u8, 15)?;
                // Wait at least 150us for the data to be ready
                self.next_update += 150u32.micros();
                return Ok(false);
            }
            i2c_dma::State::Reading => {
                return Ok(false);
            }
            i2c_dma::State::Done => {}
        };
        let raw_data = i2c_dma.finish_read()?;

        let time =
            (raw_data[12] as u32) | ((raw_data[13] as u32) << 8) | ((raw_data[14] as u32) << 16);

        // The output data rate is 6.4 kHz and sensortime is incremented at 25.6kHz
        // So, a new sample is available every (4) sensortime ticks
        const TIME_PER_SAMPLE: u32 = SENSOR_TIME_HZ / 6400;

        let sample = time / TIME_PER_SAMPLE;
        if sample == self.sample_count {
            defmt::trace!("No new BMI270 data");
            return Ok(false);
        } else if sample > self.sample_count + 1 && self.sample_count != 0 {
            let dropped_samples = sample - self.sample_count - 1;
            defmt::trace!("Dropped {} BMI270 sample(s)", dropped_samples);
        }

        // Wait a little bit less than the mag ODR to avoid skipping samples
        while now >= self.next_update {
            self.next_update += SENSORPERIOD_MICRO_SECONDS;
        }

        self.raw_data = [
            i16::from(raw_data[0]) | (i16::from(raw_data[1]) << 8),
            i16::from(raw_data[2]) | (i16::from(raw_data[3]) << 8),
            i16::from(raw_data[4]) | (i16::from(raw_data[5]) << 8),
            i16::from(raw_data[6]) | (i16::from(raw_data[7]) << 8),
            i16::from(raw_data[8]) | (i16::from(raw_data[9]) << 8),
            i16::from(raw_data[10]) | (i16::from(raw_data[11]) << 8),
        ];

        self.sample_count = sample;

        let accel_x_g = self.raw_data[0] as f32 * self.accel_scale;
        let accel_y_g = self.raw_data[1] as f32 * self.accel_scale;
        let accel_z_g = self.raw_data[2] as f32 * self.accel_scale;

        self.accel_g = [accel_x_g, accel_y_g, accel_z_g];

        let gyro_x_dps = self.raw_data[3] as f32 * self.gyr_scale;
        let gyro_y_dps = self.raw_data[4] as f32 * self.gyr_scale;
        let gyro_z_dps = self.raw_data[5] as f32 * self.gyr_scale;

        self.gyro_dps = [gyro_x_dps, gyro_y_dps, gyro_z_dps];

        Ok(true)
    }

    fn burst_write<D: DelayNs>(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        data: &[u8],
        _delay: &mut D,
    ) {
        let config_size = data.len();
        let mut index = 0;
        let read_write_len = 254;

        let remain = config_size % read_write_len;

        let mut buffer = [0u8; 255];

        while index < config_size {
            let current_chunk_size = if remain == 0 || index < config_size - remain {
                read_write_len
            } else {
                usize::min(2, config_size - index)
            };

            let addr = (index / 2) as u16;
            let addr_array = [(addr & 0x0F) as u8, ((addr >> 4) & 0xFF) as u8];

            let addr_write_buf = [Registers::InitAdd0 as u8, addr_array[0], addr_array[1]];

            let _ = i2c_dma.write(self.i2c_address, &addr_write_buf);

            buffer[0] = Registers::InitData as u8;
            buffer[1..(current_chunk_size + 1)]
                .copy_from_slice(&data[index..index + current_chunk_size]);

            let _ = i2c_dma.write(self.i2c_address, &buffer[..(current_chunk_size + 1)]);

            index += current_chunk_size;
        }
    }

    pub fn init<D: DelayNs>(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        delay: &mut D,
    ) -> Result<(), Error> {
        defmt::debug!(
            "Initializing BMI270 at i2c address: {:#04X}",
            self.i2c_address
        );

        let chip_id = self.read_register(i2c_dma, Registers::ChipId)?;
        defmt::debug!("chip id {:#04X}", chip_id);
        if chip_id != CHIP_ID {
            return Err(Error::InvalidChipId);
        }

        defmt::debug!("Configuring BMI270");
        self.write_register(i2c_dma, Registers::PowerConfg, POWER_CONF_REG_VAL)?;
        delay.delay_us(POR_DELAY_MICRO_SECONDS.to_micros());
        self.write_register(i2c_dma, Registers::InitCtrl, START_INITIALIZATION)?;
        self.burst_write(i2c_dma, CONFIG_FILE, delay);
        self.write_register(i2c_dma, Registers::InitCtrl, STOP_INITIALIZATION)?;
        delay.delay_ms(CONFIG_DELAY_MILLI_SECONDS.to_millis());

        let status = self.read_register(i2c_dma, Registers::InternalStatus)?;
        defmt::debug!("Internal status: {=u8}", status);
        if status != 1 {
            return Err(Error::ConfigError);
        }

        self.write_register(i2c_dma, Registers::PowerCtrl, POWER_CTRL_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(i2c_dma, Registers::AccelConf, ACC_CONF_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());

        self.write_register(i2c_dma, Registers::AccelRange, ACC_RANGE_REG_VAL)?;
        self.accel_scale = 2.442002e-4;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());

        self.write_register(i2c_dma, Registers::GyrConf, GYR_CONF_REG_VAL)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(i2c_dma, Registers::GyrRange, GYR_RANGE_REG_VAL)?;
        self.gyr_scale = 6.097561e-2;

        self.write_register(i2c_dma, Registers::AccelOffsetX, 0x01)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(i2c_dma, Registers::AccelOffsetY, 0x01)?;
        delay.delay_us(WRITE_2_WRITE_DELAY_MICROS_SECONDS.to_micros());
        self.write_register(i2c_dma, Registers::AccelOffsetZ, 0x01)?;

        self.write_register(i2c_dma, Registers::GyrOffsetXYZ89, GYR_OFFSET_XYZ_VAL)?;

        Ok(())
    }
}
