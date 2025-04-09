use core::ops::DerefMut;

use embedded_hal::delay::DelayNs;
use fugit::{ExtU32 as _, Hertz, MicrosDuration};

use hal::i2c;

use crate::i2c_dma;
use crate::monotonic::Instant;

const CHIP_ID: u8 = 0x50;

const BARO_ODR: Hertz<u32> = Hertz::<u32>::Hz(280);
const BARO_PERIOD: MicrosDuration<u32> = BARO_ODR.into_duration();

const OSR_CONFIG_VAL: u8 = 0x40;
const ODR_CONFIG_CAL: u8 = 0x81;

const PRESSURE_RAW_TO_PASCAL_CONVERSION_FACTOR: f32 = 1f32 / 64f32;
const TEMP_RAW_TO_CELSIUS_CONVERSION_FACTOR: f32 = 1f32 / 65536f32;

#[repr(u8)]
pub enum Register {
    ChipId = 0x01,
    TempDataXlsb = 0x1D,
    _TempDataLsb = 0x1E,
    _TempDataMsb = 0x1F,
    _PressDataXlsb = 0x20,
    _PressDataLsb = 0x21,
    _PressDataMsb = 0x22,
    IntStatus = 0x27,
    Status = 0x28,
    OsrConfig = 0x36,
    OdrConfig = 0x37,
}

#[repr(u8)]
pub enum Address {
    Low = 0x46,
    High = 0x47,
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2cDma(i2c_dma::Error),
    InvalidChipId,
    NvmErr,
    IntStatusErr,
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

#[derive(Debug, Clone, Copy, defmt::Format, Default)]
struct RawData {
    pressure: u32,
    temp: u32,
}

#[derive(Debug, Clone, Copy, defmt::Format, Default)]
pub struct Data {
    pub pressure_pascal: f32,
    pub temp_c: f32,
}

pub struct Bmp581 {
    address: u8,
    next_update: Instant,
    raw_data: RawData,
    pub data: Data,
}

impl Bmp581 {
    pub fn new<D: DelayNs>(
        i2c_dma: &mut i2c_dma::I2cDma,
        address: Address,
        delay: &mut D,
    ) -> Result<Self, Error> {
        let mut bmp581 = Bmp581 {
            address: address as u8,
            next_update: Instant::from_ticks(0),
            raw_data: RawData::default(),
            data: Data::default(),
        };

        bmp581.init(i2c_dma, delay)?;

        Ok(bmp581)
    }

    fn write_register<D: Into<u8>>(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        register: Register,
        data: D,
    ) -> Result<(), Error> {
        embedded_hal::i2c::I2c::write(
            i2c_dma.deref_mut(),
            self.address,
            &[register as u8, data.into()],
        )?;

        Ok(())
    }

    fn read_register(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        register: Register,
    ) -> Result<u8, Error> {
        // Read 2 more bytes than necessary as per BST-BMM350-DS001-25 #9.2.3.
        let mut data = [0];
        i2c_dma.write_read(self.address, &[register as u8], &mut data)?;
        Ok(data[0])
    }

    fn try_update(&mut self, i2c_dma: &mut i2c_dma::I2cDma, now: Instant) -> Result<bool, Error> {
        match i2c_dma.state()?.0 {
            i2c_dma::State::Idle => {
                defmt::trace!("Starting BMP581 DMA read");
                i2c_dma.begin_read(self.address, Register::TempDataXlsb as u8, 6)?;
                // Wait at least 1000us (1 ms) for the data to be ready
                self.next_update += 1000u32.micros();
                return Ok(false);
            }
            i2c_dma::State::Reading => {
                return Ok(false);
            }
            i2c_dma::State::Done => {}
        };

        if i2c_dma.state()?.1 == self.address {
            let data = &i2c_dma.finish_read()?;
            let raw_temp = (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);
            let raw_pressure =
                (data[3] as u32) | ((data[4] as u32) << 8) | ((data[5] as u32) << 16);

            // Wait a little bit less than the mag ODR to avoid skipping samples
            while now >= self.next_update {
                self.next_update += BARO_PERIOD - 300u32.micros();
            }

            self.raw_data = RawData {
                pressure: raw_pressure,
                temp: raw_temp,
            };

            let pressure_pascal = (raw_pressure as f32) * PRESSURE_RAW_TO_PASCAL_CONVERSION_FACTOR;
            let temp_c = raw_temp as f32 * TEMP_RAW_TO_CELSIUS_CONVERSION_FACTOR;

            self.data = Data {
                pressure_pascal,
                temp_c,
            };
        } else {
            return Ok(false);
        }

        Ok(true)
    }

    pub fn update(&mut self, i2c_dma: &mut i2c_dma::I2cDma, now: Instant) -> bool {
        if now < self.next_update {
            return false;
        }
        match self.try_update(i2c_dma, now) {
            Ok(updated) => updated,
            Err(err) => {
                defmt::warn!("BMP581 error: {}", err);
                false
            }
        }
    }

    fn init<D: DelayNs>(
        &mut self,
        i2c_dma: &mut i2c_dma::I2cDma,
        delay: &mut D,
    ) -> Result<(), Error> {
        defmt::debug!("Initializing BMP581");
        delay.delay_ms(50);

        defmt::debug!("Reading BMP581 chip ID");
        let chip_id = self.read_register(i2c_dma, Register::ChipId)?;
        if chip_id != CHIP_ID {
            return Err(Error::InvalidChipId);
        }

        self.write_register(i2c_dma, Register::OsrConfig, OSR_CONFIG_VAL)?;

        self.write_register(i2c_dma, Register::OdrConfig, ODR_CONFIG_CAL)?;

        Ok(())
    }
}
