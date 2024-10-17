use bitfield_struct::bitfield;
use embedded_hal::{delay::DelayNs, i2c::I2c};

type Duration = fugit::Duration<u32, 1, 1_000_000>;

const ADDR_LOW: u8 = 0x14;
const ADDR_HIGH: u8 = 0x15;
const CHIP_ID: u8 = 0x33;

const START_UP_TIME_FROM_POR: Duration = Duration::millis(3);
const SOFT_RESET_DELAY: Duration = Duration::millis(24);
const GOTO_SUSPEND_DELAY: Duration = Duration::millis(6);
const SUSPEND_TO_NORMAL_DELAY: Duration = Duration::millis(38);
const UPD_OAE_DELAY: Duration = Duration::millis(1);
const BR_DELAY: Duration = Duration::millis(14);
const FGR_DELAY: Duration = Duration::millis(18);

const OTP_DATA_LEN: usize = 32;
const OTP_TEMP_OFF_SENS: usize = 0x0D;
const OTP_MAG_OFFSET_X: usize = 0x0E;
const OTP_MAG_OFFSET_Y: usize = 0x0F;
const OTP_MAG_OFFSET_Z: usize = 0x10;
const OTP_MAG_SENS_X: usize = 0x10;
const OTP_MAG_SENS_Y: usize = 0x11;
const OTP_MAG_SENS_Z: usize = 0x11;
const OTP_MAG_TCO_X: usize = 0x12;
const OTP_MAG_TCO_Y: usize = 0x13;
const OTP_MAG_TCO_Z: usize = 0x14;
const OTP_MAG_TCS_X: usize = 0x12;
const OTP_MAG_TCS_Y: usize = 0x13;
const OTP_MAG_TCS_Z: usize = 0x14;
const OTP_MAG_DUT_T0: usize = 0x18;
const OTP_CROSS_X_Y: usize = 0x15;
const OTP_CROSS_Y_X: usize = 0x15;
const OTP_CROSS_Z_X: usize = 0x16;
const OTP_CROSS_Z_Y: usize = 0x16;

const SENS_CORR_Y: f32 = 0.01;
const TCS_CORR_Z: f32 = 0.0001;
const TEMP_OFFSET_SCALE: f32 = 5.0;
const TEMP_SENS_SCALE: f32 = 512.0;
const MAG_SENS_SCALE: f32 = 256.0;
const TCO_SCALE: f32 = 32.0;
const TCS_SCALE: f32 = 16384.0;
const DUT_T0_SCALE: f32 = 512.0;
const DUT_T0_OFFSET: f32 = 23.0;
const CROSS_AXIS_SCALE: f32 = 800.0;

const MAG_SENS: [f32; 3] = [14.55, 14.55, 9.0];
const MAG_GAIN_TARGET: [f32; 3] = [19.46, 19.46, 31.0];
const TEMP_SENS: f32 = 0.00204;
const ADC_GAIN: f32 = 1.0 / 1.5;
const LUT_GAIN: f32 = 0.714_607_24;
const MEGA_BINARY_TO_DECIMAL: f32 = 1000000.0 / 1048576.0;
const SENSOR_TIME_HZ: u32 = 25600;

#[derive(Debug, Clone, Copy)]
enum Register {
    ChipId = 0x00,
    _RevId = 0x01,
    _ErrReg = 0x02,
    _PadCtrl = 0x03,
    PmuCmdAggrSet = 0x04,
    _PmuCmdAxisEn = 0x05,
    PmuCmd = 0x06,
    PmuCmdStatus0 = 0x07,
    _PmuCmdStatus1 = 0x08,
    _I3cErr = 0x09,
    _I2cWdtSet = 0x0A,
    _TrsdcrRevId = 0x0D,
    _TcSyncTu = 0x21,
    _TcSyncOdr = 0x22,
    _TcSyncTph1 = 0x23,
    _TcSyncTph2 = 0x24,
    _TcSyncDt = 0x25,
    _TcSyncSt0 = 0x26,
    _TcSyncSt1 = 0x27,
    _TcSyncSt2 = 0x28,
    _TcSyncStatus = 0x29,
    _IntCtrl = 0x2E,
    _IntCtrlIbi = 0x2F,
    _IntStatus = 0x30,
    MagXXlsb = 0x31,
    _MagXLsb = 0x32,
    _MagXMsb = 0x33,
    _MagYXlsb = 0x34,
    _MagYLsb = 0x35,
    _MagYMsb = 0x36,
    _MagZXlsb = 0x37,
    _MagZLsb = 0x38,
    _MagZMsb = 0x39,
    _TempXlsb = 0x3A,
    _TempLsb = 0x3B,
    _TempMsb = 0x3C,
    _SensortimeXlsb = 0x3D,
    _SensortimeLsb = 0x3E,
    _SensortimeMsb = 0x3F,
    OtpCmdReg = 0x50,
    OtpDataMsbReg = 0x52,
    OtpDataLsbReg = 0x53,
    OtpStatusReg = 0x55,
    _TmrSelftestUser = 0x60,
    _CtrlUser = 0x61,
    Cmd = 0x7E,
}

impl Register {
    fn addr(&self) -> u8 {
        *self as u8
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Command {
    _NoOp = 0x00,
    SoftReset = 0xB6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, defmt::Format)]
#[repr(u8)]
enum Pmu {
    SuspendMode = 0x00,
    NormalMode = 0x01,
    UpdateOdrAndAvg = 0x02,
    ForcedMode = 0x03,
    ForcedModeFast = 0x04,
    FluxGuideReset = 0x05,
    FluxGuideResetFast = 0x06,
    BitReset = 0x07,
    BitResetFast = 0x08,
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Otp {
    DirectRead = 0x20,
    _DirectProgram1Byte = 0x40,
    _DirectProgram = 0x60,
    PowerOff = 0x80,
    _ExternalRead = 0xA0,
    _ExternalProgram = 0xE0,
}

#[derive(Debug, Clone, Copy, defmt::Format)]
#[repr(u8)]
enum Odr {
    Hz400 = 0x2,
    Hz200 = 0x3,
    Hz100 = 0x4,
    Hz50 = 0x5,
    Hz25 = 0x6,
    Hz12_5 = 0x7,
    Hz6_25 = 0x8,
    Hz3_125 = 0x9,
    Hz1_5625 = 0xA,
}

impl Odr {
    const fn from_bits(bits: u8) -> Self {
        match bits {
            0x2 => Self::Hz400,
            0x3 => Self::Hz200,
            0x4 => Self::Hz100,
            0x5 => Self::Hz50,
            0x6 => Self::Hz25,
            0x7 => Self::Hz12_5,
            0x8 => Self::Hz6_25,
            0x9 => Self::Hz3_125,
            0xA => Self::Hz1_5625,
            _ => Self::Hz400,
        }
    }

    const fn into_bits(self) -> u8 {
        self as _
    }
}

#[derive(Debug, Clone, Copy, defmt::Format, PartialEq, Eq)]
#[repr(u8)]
enum Averaging {
    None = 0x0,
    Avg2 = 0x1,
    Avg4 = 0x2,
    Avg8 = 0x3,
}

impl Averaging {
    const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::None,
            1 => Self::Avg2,
            2 => Self::Avg4,
            3 => Self::Avg8,
            _ => Self::None,
        }
    }

    const fn into_bits(self) -> u8 {
        self as _
    }
}

#[bitfield(u8)]
#[derive(defmt::Format)]
struct PmuCmdStatus0 {
    pmu_cmd_busy: bool,
    odr_ovwr: bool,
    avr_ovwr: bool,
    pwr_mode_is_normal: bool,
    cmd_is_illegal: bool,
    #[bits(3)]
    pmu_cmd_value: u8,
}

#[bitfield(u8)]
struct OdrAverage {
    #[bits(4)]
    odr: Odr,
    #[bits(2)]
    avg: Averaging,
    #[bits(2)]
    _reserved: u8,
}

impl From<Command> for u8 {
    fn from(command: Command) -> u8 {
        command as u8
    }
}

impl From<Pmu> for u8 {
    fn from(pmu: Pmu) -> u8 {
        pmu as u8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InvalidPmu;

impl TryFrom<u8> for Pmu {
    type Error = InvalidPmu;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Pmu::SuspendMode),
            0x01 => Ok(Pmu::NormalMode),
            0x02 => Ok(Pmu::UpdateOdrAndAvg),
            0x03 => Ok(Pmu::ForcedMode),
            0x04 => Ok(Pmu::ForcedModeFast),
            0x05 => Ok(Pmu::FluxGuideReset),
            0x06 => Ok(Pmu::FluxGuideResetFast),
            0x07 => Ok(Pmu::BitReset),
            0x08 => Ok(Pmu::BitResetFast),
            _ => Err(InvalidPmu),
        }
    }
}

impl From<Otp> for u8 {
    fn from(otp: Otp) -> u8 {
        otp as u8
    }
}

pub enum Address {
    Low,
    High,
}

impl From<Address> for u8 {
    fn from(address: Address) -> u8 {
        match address {
            Address::Low => ADDR_LOW,
            Address::High => ADDR_HIGH,
        }
    }
}

#[derive(Debug)]
pub enum Error<E> {
    I2c(E),
    InvalidChipId,
    InvalidPmu,
    ModeNotSupported,
    BitResetFailed,
    FluxGuideResetFailed,
    OtpTimeout,
}

impl<E> From<InvalidPmu> for Error<E> {
    fn from(_: InvalidPmu) -> Self {
        Error::InvalidPmu
    }
}

trait SignedExt {
    fn to_i32<const BITS: usize>(self) -> i32;
    fn to_f32<const BITS: usize>(self) -> f32;
}

impl SignedExt for u16 {
    fn to_i32<const BITS: usize>(self) -> i32 {
        (self as u32).to_i32::<BITS>()
    }
    fn to_f32<const BITS: usize>(self) -> f32 {
        self.to_i32::<BITS>() as f32
    }
}

impl SignedExt for u32 {
    fn to_i32<const BITS: usize>(self) -> i32 {
        let mask = (1 << BITS) - 1;
        let power = 1 << (BITS - 1);
        let mut ret = (self & mask) as i32;
        if ret >= power {
            ret -= power << 1;
        }
        ret
    }
    fn to_f32<const BITS: usize>(self) -> f32 {
        self.to_i32::<BITS>() as f32
    }
}

pub struct Bmm350<I2C> {
    i2c: I2C,
    address: u8,
    calibration: Calibration,
}

#[derive(Debug, Clone, Copy, defmt::Format)]
struct RawData {
    mag: [f32; 3],
    temp: f32,
    time: u32,
}

#[derive(Debug, Clone, Copy, defmt::Format)]
pub struct CalibratedData {
    pub mag: [f32; 3],
    pub temp: f32,
    pub sample: u32,
}

#[derive(Debug, Clone, Copy, defmt::Format, Default)]
struct Calibration {
    mag_offset: [f32; 3],
    temp_offset: f32,
    mag_sensitivity: [f32; 3],
    temp_sensitivity: f32,
    dut_tco: [f32; 3],
    dut_tcs: [f32; 3],
    dut_t0: f32,
    cross_axis: [f32; 4],
}

impl<I2C: I2c> Bmm350<I2C> {
    pub fn new<A: Into<u8>, D: DelayNs>(
        i2c: I2C,
        address: A,
        delay: &mut D,
    ) -> Result<Self, Error<I2C::Error>> {
        let mut bmm350 = Bmm350 {
            i2c,
            address: address.into(),
            calibration: Calibration::default(),
        };
        bmm350.init(delay)?;
        Ok(bmm350)
    }

    fn read_raw_data(&mut self) -> Result<RawData, Error<I2C::Error>> {
        let data = self.read_registers::<15>(Register::MagXXlsb)?;
        let mag = [
            (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16),
            (data[3] as u32) | ((data[4] as u32) << 8) | ((data[5] as u32) << 16),
            (data[6] as u32) | ((data[7] as u32) << 8) | ((data[8] as u32) << 16),
        ];
        let temp = (data[9] as u32) | ((data[10] as u32) << 8) | ((data[11] as u32) << 16);
        let time = (data[12] as u32) | ((data[13] as u32) << 8) | ((data[14] as u32) << 16);

        let mag = mag.map(SignedExt::to_f32::<24>);
        let temp = temp.to_f32::<24>();
        Ok(RawData { mag, temp, time })
    }

    // Based on the following reference implementation:
    // https://github.com/boschsensortec/BMM350_SensorAPI/blob/4127534c6262f903683ee210b36b85ab2bb504c8/bmm350.c#L899
    pub fn read_data(&mut self) -> Result<CalibratedData, Error<I2C::Error>> {
        let raw_data = self.read_raw_data()?;

        // Convert raw ADC values to micro-Tesla and degrees Celsius
        let mut mag = raw_data.mag;
        mag.iter_mut().enumerate().for_each(|(i, x)| {
            *x *= MEGA_BINARY_TO_DECIMAL / (MAG_SENS[i] * MAG_GAIN_TARGET[i] * ADC_GAIN * LUT_GAIN)
        });
        let mut temp = raw_data.temp / (TEMP_SENS * ADC_GAIN * LUT_GAIN * 1048576.0);
        if temp > 0.0 {
            temp -= 25.49;
        } else if temp < 0.0 {
            temp += 25.49;
        }
        // The output data rate is 400Hz and sensortime is incremented at 25.6kHz
        // So, a new sample is available every (25600 / 400) sensortime ticks
        const TIME_PER_SAMPLE: u32 = SENSOR_TIME_HZ / 400;
        let sample = raw_data.time / TIME_PER_SAMPLE;

        // Apply compensation values from OTP
        let cal = &self.calibration;
        temp *= 1.0 + cal.temp_sensitivity;
        temp += cal.temp_offset;
        mag.iter_mut().enumerate().for_each(|(i, m)| {
            *m *= 1.0 + cal.mag_sensitivity[i];
            *m += cal.mag_offset[i];
            *m += cal.dut_tco[i] * (temp - cal.dut_t0);
            *m /= 1.0 + cal.dut_tcs[i] * (temp - cal.dut_t0);
        });

        // Apply cross-axis sensitivity compensation
        let [cross_x_y, cross_y_x, cross_z_x, cross_z_y] = cal.cross_axis;
        let denom = 1.0 - cross_y_x * cross_x_y;
        mag = [
            (mag[0] - cross_x_y * mag[1]) / denom,
            (mag[1] - cross_y_x * mag[0]) / denom,
            mag[2]
                + (mag[0] * (cross_y_x * cross_z_y - cross_z_x)
                    - mag[1] * (cross_z_y - cross_x_y * cross_z_x))
                    / denom,
        ];
        Ok(CalibratedData { mag, temp, sample })
    }

    fn write_register<D: Into<u8>>(
        &mut self,
        register: Register,
        data: D,
    ) -> Result<(), Error<I2C::Error>> {
        self.i2c
            .write(self.address, &[register.addr(), data.into()])
            .map_err(Error::I2c)
    }

    fn read_registers<const N: usize>(
        &mut self,
        register: Register,
    ) -> Result<[u8; N], Error<I2C::Error>> {
        // Read 2 more bytes than necessary as per BST-BMM350-DS001-25 #9.2.3.
        const MAX_LEN: usize = 16;
        const { assert!(N <= MAX_LEN) }
        let temp_data = &mut [0; MAX_LEN + 2][..N + 2];
        self.i2c
            .write_read(self.address, &[register.addr()], temp_data)
            .map_err(Error::I2c)?;
        let mut out = [0; N];
        out.copy_from_slice(&temp_data[2..]);
        Ok(out)
    }

    fn read_register(&mut self, register: Register) -> Result<u8, Error<I2C::Error>> {
        // Read 2 more bytes than necessary as per BST-BMM350-DS001-25 #9.2.3.
        let mut data = [0; 3];
        self.i2c
            .write_read(self.address, &[register.addr()], &mut data)
            .map_err(Error::I2c)?;
        Ok(data[2])
    }

    fn read_calibration_data(&mut self) -> Result<(), Error<I2C::Error>> {
        defmt::debug!("Downloading BMM350 OTP memory");
        let mut data = [0u16; OTP_DATA_LEN];
        for (i, w) in data.iter_mut().enumerate() {
            *w = self.read_otp_word(i as u8)?;
        }
        defmt::debug!("Powering off BMM350 OTP memory");
        self.write_register(Register::OtpCmdReg, Otp::PowerOff)?;
        self.calibration = Calibration::new(data);
        defmt::debug!("BMM350 calibration data: {}", self.calibration);
        Ok(())
    }

    fn read_otp_word(&mut self, addr: u8) -> Result<u16, Error<I2C::Error>> {
        let otp_cmd = Otp::DirectRead as u8 | (addr & 0x1F);
        self.write_register(Register::OtpCmdReg, otp_cmd)?;

        // Wait for OTP to be ready
        let mut attempts = 100usize;
        loop {
            let otp_status = self.read_register(Register::OtpStatusReg)?;
            if otp_status & 0x01 != 0 {
                break;
            }
            attempts -= 1;
            if attempts == 0 {
                return Err(Error::OtpTimeout);
            }
        }

        let msb = self.read_register(Register::OtpDataMsbReg)?;
        let lsb = self.read_register(Register::OtpDataLsbReg)?;
        Ok(((msb as u16) << 8) | lsb as u16)
    }

    fn read_pmu_cmd_status_0(&mut self) -> Result<PmuCmdStatus0, Error<I2C::Error>> {
        let reg_data = self.read_register(Register::PmuCmdStatus0)?;
        Ok(PmuCmdStatus0::from_bits(reg_data))
    }

    fn set_power_mode<D: DelayNs>(
        &mut self,
        mode: Pmu,
        delay: &mut D,
    ) -> Result<(), Error<I2C::Error>> {
        defmt::debug!("Setting power mode to {}", mode);
        self.write_register(Register::PmuCmd, mode)?;
        let wait = match mode {
            Pmu::SuspendMode => GOTO_SUSPEND_DELAY,
            Pmu::NormalMode => SUSPEND_TO_NORMAL_DELAY,
            Pmu::UpdateOdrAndAvg => UPD_OAE_DELAY,
            _ => return Err(Error::ModeNotSupported),
        };
        delay.delay_ms(wait.to_millis());

        Ok(())
    }

    fn mag_reset<D: DelayNs>(&mut self, delay: &mut D) -> Result<(), Error<I2C::Error>> {
        defmt::debug!("Mag resetting BMM350");
        defmt::debug!("Setting BR (Bit Reset)");
        self.write_register(Register::PmuCmd, Pmu::BitReset)?;
        delay.delay_ms(BR_DELAY.to_millis());
        if self.read_pmu_cmd_status_0()?.pmu_cmd_value() != Pmu::BitReset as u8 {
            return Err(Error::BitResetFailed);
        }

        defmt::debug!("Setting FGR (Flux Guide Reset)");
        self.write_register(Register::PmuCmd, Pmu::FluxGuideReset)?;
        delay.delay_ms(FGR_DELAY.to_millis());
        if self.read_pmu_cmd_status_0()?.pmu_cmd_value() != Pmu::FluxGuideReset as u8 {
            return Err(Error::FluxGuideResetFailed);
        }
        Ok(())
    }

    fn init<D: DelayNs>(&mut self, delay: &mut D) -> Result<(), Error<I2C::Error>> {
        defmt::debug!("Initializing BMM350");
        delay.delay_ms(START_UP_TIME_FROM_POR.to_millis());

        defmt::debug!("Soft resetting BMM350");
        self.write_register(Register::Cmd, Command::SoftReset)?;
        delay.delay_ms(SOFT_RESET_DELAY.to_millis());

        defmt::debug!("Reading BMM350 chip ID");
        let chip_id = self.read_register(Register::ChipId)?;
        if chip_id != CHIP_ID {
            return Err(Error::InvalidChipId);
        }

        self.read_calibration_data()?;

        self.mag_reset(delay)?;

        let reg_data = OdrAverage::new()
            .with_odr(Odr::Hz400)
            .with_avg(Averaging::None);
        defmt::debug!(
            "Setting ODR to {} and averaging to {}",
            reg_data.odr(),
            reg_data.avg()
        );
        self.write_register(Register::PmuCmdAggrSet, reg_data)?;
        self.set_power_mode(Pmu::UpdateOdrAndAvg, delay)?;

        // Suspend before changing power mode
        self.set_power_mode(Pmu::SuspendMode, delay)?;
        self.set_power_mode(Pmu::NormalMode, delay)?;

        defmt::debug!("Finished initializing BMM350");
        Ok(())
    }
}

impl Calibration {
    fn new(otp: [u16; OTP_DATA_LEN]) -> Self {
        // Parse magnetometer offsets
        let mag_off_msb = [
            otp[OTP_MAG_OFFSET_X] & 0x0F00,
            otp[OTP_MAG_OFFSET_X] & 0xF000 >> 4,
            otp[OTP_MAG_OFFSET_Y] & 0x0F00,
        ];
        let mag_off_lsb = [
            otp[OTP_MAG_OFFSET_X] & 0x00FF,
            otp[OTP_MAG_OFFSET_Y] & 0x00FF,
            otp[OTP_MAG_OFFSET_Z] & 0x00FF,
        ];
        let mag_off = [
            (mag_off_msb[0] | mag_off_lsb[0]),
            (mag_off_msb[1] | mag_off_lsb[1]),
            (mag_off_msb[2] | mag_off_lsb[2]),
        ];

        // Parse magnetometer sensitivities
        let mag_sen = [
            otp[OTP_MAG_SENS_X] & 0xFF00 >> 8,
            otp[OTP_MAG_SENS_Y] & 0x00FF,
            otp[OTP_MAG_SENS_Z] & 0xFF00 >> 8,
        ];

        // Parse temperature offset and sensitivity
        let temp_sen = otp[OTP_TEMP_OFF_SENS] & 0xFF00 >> 8;
        let temp_off = otp[OTP_TEMP_OFF_SENS] & 0x00FF;

        // Parse TCO
        let tco = [
            otp[OTP_MAG_TCO_X] & 0x00FF,
            otp[OTP_MAG_TCO_Y] & 0x00FF,
            otp[OTP_MAG_TCO_Z] & 0x00FF,
        ];

        // Parse TCS
        let tcs = [
            otp[OTP_MAG_TCS_X] & 0xFF00 >> 8,
            otp[OTP_MAG_TCS_Y] & 0xFF00 >> 8,
            otp[OTP_MAG_TCS_Z] & 0xFF00 >> 8,
        ];

        // Parse DUT T0
        let dut_t0 = otp[OTP_MAG_DUT_T0];

        // Parse cross-axis coefficients
        let cross_axis = [
            otp[OTP_CROSS_X_Y] & 0x00FF,
            otp[OTP_CROSS_Y_X] & 0xFF00 >> 8,
            otp[OTP_CROSS_Z_X] & 0x00FF,
            otp[OTP_CROSS_Z_Y] & 0xFF00 >> 8,
        ];

        let mut comp = Calibration {
            mag_offset: mag_off.map(SignedExt::to_f32::<12>),
            temp_offset: temp_off.to_f32::<8>(),
            mag_sensitivity: mag_sen.map(SignedExt::to_f32::<8>),
            temp_sensitivity: temp_sen.to_f32::<8>(),
            dut_tco: tco.map(SignedExt::to_f32::<8>),
            dut_tcs: tcs.map(SignedExt::to_f32::<8>),
            dut_t0: dut_t0.to_f32::<8>(),
            cross_axis: cross_axis.map(SignedExt::to_f32::<8>),
        };

        // Scale the calibration values
        comp.mag_sensitivity
            .iter_mut()
            .for_each(|sens| *sens /= MAG_SENS_SCALE);
        comp.dut_tco.iter_mut().for_each(|tco| *tco /= TCO_SCALE);
        comp.dut_tcs.iter_mut().for_each(|tcs| *tcs /= TCS_SCALE);
        comp.cross_axis
            .iter_mut()
            .for_each(|cross| *cross /= CROSS_AXIS_SCALE);

        comp.temp_offset /= TEMP_OFFSET_SCALE;
        comp.temp_sensitivity /= TEMP_SENS_SCALE;
        comp.dut_t0 /= DUT_T0_SCALE;

        // Apply offsets, corrections
        comp.dut_t0 += DUT_T0_OFFSET;
        comp.mag_sensitivity[1] += SENS_CORR_Y;
        comp.dut_tcs[2] -= TCS_CORR_Z;

        comp
    }
}
