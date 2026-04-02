use crate::i2c_dma::I2cRegs;
use crate::monotonic::Instant;
use embedded_hal::i2c::{I2c as I2cTrait, Operation};
use fugit::Hertz;
use hal::i2c;

const QMC5883L_ADDR: u8 = 0x0D;
const HMC5883L_ADDR: u8 = 0x1E;
const CHIP_ID_REG: u8 = 0x0D;
const EXPECTED_CHIP_ID: u8 = 0xFF;

const DATA_OUT_REG: u8 = 0x00;
const STATUS_REG: u8 = 0x06;
const CONTROL_REG1: u8 = 0x09;
const CONTROL_REG2: u8 = 0x0A;
const SET_RESET_PERIOD: u8 = 0x0B;

// Control Register 1: continuous mode, 200 Hz ODR, 8 Gauss, 512x oversampling
const CR1_VALUE: u8 = 0x1D;
const CR2_DISABLE_INT: u8 = 0x01;
const INIT_RETRIES: u8 = 3;

const COMPASS_ODR: Hertz<u32> = Hertz::<u32>::Hz(50);
const COMPASS_PERIOD: fugit::MicrosDuration<u32> = COMPASS_ODR.into_duration();

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2c(i2c::Error),
    WrongChipId(u8),
}

impl From<i2c::Error> for Error {
    fn from(err: i2c::Error) -> Self {
        Error::I2c(err)
    }
}

#[derive(Debug, Clone, Copy, Default, defmt::Format)]
pub struct CompassData {
    pub mag_x: i16,
    pub mag_y: i16,
    pub mag_z: i16,
    pub status: u8,
}

pub struct Qmc5883l {
    addr: u8,
    pub data: CompassData,
    pub sample_count: u32,
    next_update: Instant,
}

impl Qmc5883l {
    pub fn new(
        i2c: &mut i2c::I2c<I2cRegs>,
        delay: &mut impl embedded_hal::delay::DelayNs,
    ) -> Result<Self, Error> {
        // Give the module time to power up after 5V-AUX enable
        delay.delay_ms(200);

        let mut last_err = Error::I2c(i2c::Error::Hardware);
        for attempt in 0..INIT_RETRIES {
            // Try QMC5883L at 0x0D first, then HMC5883L-compatible at 0x1E
            for &addr in &[QMC5883L_ADDR, HMC5883L_ADDR] {
                match Self::try_init(i2c, delay, addr) {
                    Ok(s) => return Ok(s),
                    Err(e) => {
                        defmt::warn!(
                            "Compass init attempt {} addr 0x{:02X}: {:?}",
                            attempt + 1,
                            addr,
                            e
                        );
                        last_err = e;
                    }
                }
            }
            delay.delay_ms(100);
        }
        Err(last_err)
    }

    fn try_init(
        i2c: &mut i2c::I2c<I2cRegs>,
        delay: &mut impl embedded_hal::delay::DelayNs,
        addr: u8,
    ) -> Result<Self, Error> {
        // Verify chip ID before touching any state
        let reg = [CHIP_ID_REG];
        let mut id = [0u8];
        i2c.transaction(
            addr,
            &mut [Operation::Write(&reg), Operation::Read(&mut id)],
        )?;
        if id[0] != EXPECTED_CHIP_ID {
            return Err(Error::WrongChipId(id[0]));
        }

        // Soft reset
        let buf = [CONTROL_REG2, 0x80];
        i2c.transaction(addr, &mut [Operation::Write(&buf)])?;
        delay.delay_ms(50);

        // SET/RESET period (recommended by datasheet)
        let buf = [SET_RESET_PERIOD, 0x01];
        i2c.transaction(addr, &mut [Operation::Write(&buf)])?;

        // Configure: continuous mode, 200 Hz ODR, 8 Gauss, 512x oversampling
        let buf = [CONTROL_REG1, CR1_VALUE];
        i2c.transaction(addr, &mut [Operation::Write(&buf)])?;

        // Disable interrupt pin
        let buf = [CONTROL_REG2, CR2_DISABLE_INT];
        i2c.transaction(addr, &mut [Operation::Write(&buf)])?;

        defmt::info!(
            "QMC5883L initialized at 0x{:02X} (chip ID 0x{:02X})",
            addr,
            id[0]
        );

        Ok(Self {
            addr,
            data: CompassData::default(),
            sample_count: 0,
            next_update: Instant::from_ticks(0),
        })
    }

    /// Poll for new compass data. Returns true when a new sample was read.
    pub fn update(&mut self, i2c: &mut i2c::I2c<I2cRegs>, now: Instant) -> bool {
        if now.checked_duration_since(self.next_update).is_none() {
            return false;
        }

        match self.try_read(i2c) {
            Ok(true) => {
                self.next_update = now + COMPASS_PERIOD;
                true
            }
            Ok(false) => false,
            Err(_) => {
                self.next_update = now + COMPASS_PERIOD;
                false
            }
        }
    }

    fn try_read(&mut self, i2c: &mut i2c::I2c<I2cRegs>) -> Result<bool, Error> {
        // Check DRDY bit in status register
        let reg = [STATUS_REG];
        let mut status = [0u8];
        i2c.transaction(
            self.addr,
            &mut [Operation::Write(&reg), Operation::Read(&mut status)],
        )?;

        if status[0] & 0x01 == 0 {
            return Ok(false);
        }

        // Read 6 data bytes (X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB)
        let reg = [DATA_OUT_REG];
        let mut buf = [0u8; 6];
        i2c.transaction(
            self.addr,
            &mut [Operation::Write(&reg), Operation::Read(&mut buf)],
        )?;

        self.data = CompassData {
            mag_x: i16::from_le_bytes([buf[0], buf[1]]),
            mag_y: i16::from_le_bytes([buf[2], buf[3]]),
            mag_z: i16::from_le_bytes([buf[4], buf[5]]),
            status: status[0],
        };
        self.sample_count += 1;
        Ok(true)
    }
}
