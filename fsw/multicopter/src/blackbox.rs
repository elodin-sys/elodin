//! The blackbox stores sensor data, control inputs, and other relevant flight data
//! onto the SD card. This data can be used for debugging, tuning, and analysis.
//!
//! The blackbox only writes data to the SD card when the vehicle is armed. On each arming
//! event, a new folder is created under the "/<fc_name>/blackbox" directory with the current
//! date and time (formatted as "YYYYMMDD_HHmmss_SSS").
//!
//! Successful initialization of the blackbox can be used as a gating condition for arming the
//! vehicle.

use hal::gpio::Pin;
use zerocopy::IntoBytes;

use crate::rtc::FakeTime;
use crate::sdmmc;

#[derive(zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
#[repr(C)]
pub struct Record {
    pub ts: u32, // in milliseconds
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
}

pub struct Blackbox<'a> {
    volume_mgr: embedded_sdmmc::VolumeManager<sdmmc::Sdmmc, FakeTime, 4, 4, 1>,
    fc_name: &'a str,
    data_file: Option<embedded_sdmmc::RawFile>,
    led: Option<Pin>,
}

#[derive(Debug, Copy, Clone, defmt::Format)]
pub enum Error {
    NotFormatted,
    SdmmcError,
    DeviceError(sdmmc::Error),
}

impl From<embedded_sdmmc::Error<sdmmc::Error>> for Error {
    fn from(err: embedded_sdmmc::Error<sdmmc::Error>) -> Self {
        match err {
            embedded_sdmmc::Error::DeviceError(err) => Error::DeviceError(err),
            _ => Error::SdmmcError,
        }
    }
}

impl<'a> Blackbox<'a> {
    pub fn new(
        fc_name: &'a str,
        volume_mgr: embedded_sdmmc::VolumeManager<sdmmc::Sdmmc, FakeTime, 4, 4, 1>,
        led: Option<Pin>,
    ) -> Self {
        Self {
            volume_mgr,
            fc_name,
            data_file: None,
            led,
        }
    }

    pub fn arm(&mut self, dir_name: &str) -> Result<(), Error> {
        if !self.volume_mgr.device().connected() {
            defmt::warn!("SD card not connected");
            return Ok(());
        }
        let mut volume = self.volume_mgr.open_volume(embedded_sdmmc::VolumeIdx(0))?;
        let mut root_dir = volume.open_root_dir()?;
        let mut dir = mkdir_if_not_exists(&mut root_dir, self.fc_name)?;
        let mut dir = mkdir_if_not_exists(&mut dir, "blackbox")?;
        let mut dir = mkdir_if_not_exists(&mut dir, dir_name)?;
        let file =
            dir.open_file_in_dir("data.bin", embedded_sdmmc::Mode::ReadWriteCreateOrAppend)?;
        defmt::info!(
            "Opened /{}/blackbox/{}/data.bin for writing telemetry data",
            self.fc_name,
            dir_name
        );
        self.data_file = Some(file.to_raw_file());
        Ok(())
    }

    pub fn write_record(&mut self, record: Record) {
        let record = record.as_bytes();
        if let Some(data_file) = &mut self.data_file {
            if let Some(led) = self.led.as_mut() {
                led.set_high();
            }
            let mut data_file = data_file.to_file(&mut self.volume_mgr);
            if let Err(err) = data_file.write(record) {
                defmt::error!("Failed to write to SD card: {}", Error::from(err));
                self.data_file = None;
            } else {
                self.data_file = Some(data_file.to_raw_file());
            }
        }
        if let Some(led) = self.led.as_mut() {
            led.set_low();
        }
    }

    pub fn flush(&mut self) {
        if let Some(data_file) = &mut self.data_file {
            let mut data_file = data_file.to_file(&mut self.volume_mgr);
            if let Err(err) = data_file.flush() {
                defmt::error!("Failed to flush SD card: {}", Error::from(err));
                self.data_file = None;
            }
            self.data_file = Some(data_file.to_raw_file());
        }
    }

    pub fn disarm(&mut self) -> Result<(), Error> {
        if let Some(data_file) = self.data_file.take() {
            let mut data_file = data_file.to_file(&mut self.volume_mgr);
            data_file.flush()?;
            data_file.close()?;
        }
        Ok(())
    }
}

fn mkdir_if_not_exists<'a>(
    parent_dir: &'a mut embedded_sdmmc::Directory<sdmmc::Sdmmc, FakeTime, 4, 4, 1>,
    name: &str,
) -> Result<embedded_sdmmc::Directory<'a, sdmmc::Sdmmc, FakeTime, 4, 4, 1>, Error> {
    match parent_dir.make_dir_in_dir(name) {
        Ok(_) => defmt::info!("Created {} directory on SD card", name),
        Err(embedded_sdmmc::Error::DirAlreadyExists) => {}
        Err(err) => return Err(err.into()),
    }
    let dir = parent_dir.open_dir(name)?;
    Ok(dir)
}
