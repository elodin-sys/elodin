//! The blackbox stores sensor data, control inputs, and other relevant flight data
//! onto the SD card. This data can be used for debugging, tuning, and analysis.
//!
//! The blackbox only writes data to the SD card when the vehicle is armed. On each arming
//! event, a new folder is created under the "/<fc_name>/blackbox" directory with the current
//! date and time (formatted as "YYYYMMDD_HHmmss_SSS").
//!
//! Successful initialization of the blackbox can be used as a gating condition for arming the
//! vehicle.

use alloc::boxed::Box;
use hal::gpio::Pin;
use zerocopy::IntoBytes;

use crate::dronecan;
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
    data_file: BufferedFile,
    can_file: BufferedFile,
    led: Pin,
}

pub struct BufferedFile {
    file: Option<embedded_sdmmc::RawFile>,
    buf: Box<[u8]>,
    buf_len: usize,
}

impl BufferedFile {
    fn buf_remaining(&self) -> usize {
        self.buf.len() - self.buf_len
    }

    fn buf_push(&mut self, data: &[u8]) {
        let len = core::cmp::min(data.len(), self.buf_remaining());
        self.buf[self.buf_len..self.buf_len + len].copy_from_slice(&data[..len]);
        self.buf_len += len;
    }

    fn buf(&self) -> &[u8] {
        &self.buf[..self.buf_len]
    }

    fn write(
        &mut self,
        data: &[u8],
        volume_mgr: &mut embedded_sdmmc::VolumeManager<sdmmc::Sdmmc, FakeTime, 4, 4, 1>,
        led: &mut Pin,
    ) {
        if data.len() > self.buf_remaining() {
            self.flush(volume_mgr, led);
        }
        self.buf_push(data);
    }

    fn flush(
        &mut self,
        volume_mgr: &mut embedded_sdmmc::VolumeManager<sdmmc::Sdmmc, FakeTime, 4, 4, 1>,
        led: &mut Pin,
    ) {
        if let Some(file) = self.file.as_mut() {
            led.set_high();
            defmt::trace!("Flushing buffered data to SD card");
            let mut data_file = file.to_file(volume_mgr);
            match data_file.write(self.buf()).and_then(|_| data_file.flush()) {
                Ok(_) => {
                    defmt::trace!("Flushed {} bytes to SD card", self.buf_len);
                    self.file = Some(data_file.to_raw_file());
                }
                Err(err) => {
                    defmt::error!("Failed to write to SD card: {}", Error::from(err));
                    self.file = None;
                }
            }
        }
        led.set_low();
        self.buf_len = 0;
    }
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
        led: Pin,
    ) -> Self {
        Self {
            volume_mgr,
            fc_name,
            data_file: BufferedFile {
                file: None,
                buf: alloc::vec![0; 4 * 1024].into_boxed_slice(),
                buf_len: 0,
            },
            can_file: BufferedFile {
                file: None,
                buf: alloc::vec![0; 1024].into_boxed_slice(),
                buf_len: 0,
            },
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
        let data_file =
            dir.open_file_in_dir("data.bin", embedded_sdmmc::Mode::ReadWriteCreateOrAppend)?;
        defmt::info!(
            "Opened /{}/blackbox/{}/data.bin for writing telemetry data",
            self.fc_name,
            dir_name
        );
        self.data_file.file = Some(data_file.to_raw_file());
        let can_file =
            dir.open_file_in_dir("can.bin", embedded_sdmmc::Mode::ReadWriteCreateOrAppend)?;
        defmt::info!(
            "Opened /{}/blackbox/{}/can.bin for writing CAN data",
            self.fc_name,
            dir_name
        );
        self.can_file.file = Some(can_file.to_raw_file());
        Ok(())
    }

    pub fn write_record(&mut self, record: Record) {
        self.data_file
            .write(record.as_bytes(), &mut self.volume_mgr, &mut self.led)
    }

    pub fn write_can(&mut self, msg: &dronecan::RawMessage) {
        let id_bytes = msg.id.into_bytes();
        let len = id_bytes.len() + msg.buf.len();
        if (len + 4) > self.can_file.buf_remaining() {
            self.can_file.flush(&mut self.volume_mgr, &mut self.led);
        }
        self.can_file.buf_push(&(len as u32).to_le_bytes());
        self.can_file.buf_push(&id_bytes);
        self.can_file.buf_push(msg.buf);
    }

    pub fn disarm(&mut self) -> Result<(), Error> {
        self.data_file.flush(&mut self.volume_mgr, &mut self.led);
        self.can_file.flush(&mut self.volume_mgr, &mut self.led);
        if let Some(data_file) = self.data_file.file.take() {
            let data_file = data_file.to_file(&mut self.volume_mgr);
            data_file.close()?;
        }
        if let Some(can_file) = self.can_file.file.take() {
            let can_file = can_file.to_file(&mut self.volume_mgr);
            can_file.close()?;
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
