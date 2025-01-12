//! The blackbox stores sensor data, control inputs, and other relevant flight data
//! onto the SD card. This data can be used for debugging, tuning, and analysis.

use alloc::boxed::Box;

use fatfs::{FileSystem, IoError, Seek, Write};
use hal::gpio::Pin;
use zerocopy::IntoBytes;

use crate::dwt::DwtTimer;
use crate::monotonic::Instant;
use crate::sdmmc;

type Dir<'a> = fatfs::Dir<'a, sdmmc::Sdmmc, fatfs::DefaultTimeProvider, fatfs::LossyOemCpConverter>;
type File<'a> =
    fatfs::File<'a, sdmmc::Sdmmc, fatfs::DefaultTimeProvider, fatfs::LossyOemCpConverter>;

const RESET_PERIOD: fugit::MicrosDuration<u32> = fugit::MicrosDuration::<u32>::millis(100);
const MIN_CLUSTER_SIZE: usize = 512 * 8;

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

pub struct SdmmcFs {
    sdmmc: sdmmc::Sdmmc,
    fs: Option<FileSystem<sdmmc::Sdmmc>>,
    led: Pin,
}

pub struct Blackbox<'a> {
    files: Option<Files<'a>>,
    led: Pin,
    created_at: Instant,
}

struct Files<'a> {
    data_file: BufferedFile<'a>,
}

struct BufferedFile<'a> {
    file: File<'a>,
    buf: Box<[u8]>,
    buf_len: usize,
}

impl<'a> BufferedFile<'a> {
    fn new(file: File<'a>, buf_size: usize) -> Self {
        Self {
            file,
            buf: alloc::vec![0; buf_size].into_boxed_slice(),
            buf_len: 0,
        }
    }
}

impl BufferedFile<'_> {
    fn write(&mut self, data: &[u8], led: &mut Pin) -> Result<(), Error> {
        self.try_flush(led)?;
        if data.len() > self.buf.len() - self.buf_len {
            return Err(Error::BufferOverrun);
        }
        self.buf[self.buf_len..self.buf_len + data.len()].copy_from_slice(data);
        self.buf_len += data.len();
        Ok(())
    }

    fn try_flush(&mut self, led: &mut Pin) -> Result<(), Error> {
        let timer = DwtTimer {
            core_frequency: 400_000_000,
        };
        if self.buf_len < 512 {
            return Ok(());
        }

        led.set_high();
        let start = timer.now();
        match self.file.write(&self.buf[..self.buf_len]) {
            Ok(0) => {}
            Ok(n) => {
                self.buf_len -= n;
                self.buf.copy_within(n.., 0);
                self.file.flush()?;
            }
            Err(ref err) if err.is_interrupted() => {}
            Err(err) => return Err(err.into()),
        }
        let elapsed = start.elapsed();
        if elapsed > fugit::MillisDuration::<u32>::millis(5) {
            defmt::warn!(
                "Write took {}, remaining buf len: {}",
                elapsed,
                self.buf_len
            );
        }
        led.set_low();
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, defmt::Format, PartialEq, Eq)]
pub enum Error {
    NotFormatted,
    DeviceError(sdmmc::Error),
    InvalidInput,
    NotFound,
    AlreadyExists,
    DirectoryIsNotEmpty,
    CorruptedFileSystem,
    NotEnoughSpace,
    InvalidFileNameLength,
    UnsupportedFileNameCharacter,
    BufferOverrun,
    UnknownFsError,
}

impl From<sdmmc::Error> for Error {
    fn from(err: sdmmc::Error) -> Self {
        Self::DeviceError(err)
    }
}

impl From<fatfs::Error<sdmmc::Error>> for Error {
    fn from(err: fatfs::Error<sdmmc::Error>) -> Self {
        match err {
            fatfs::Error::Io(err) => Error::DeviceError(err),
            fatfs::Error::WriteZero => Error::DeviceError(sdmmc::Error::WriteZero),
            fatfs::Error::UnexpectedEof => Error::DeviceError(sdmmc::Error::UnexpectedEof),
            fatfs::Error::InvalidInput => Error::InvalidInput,
            fatfs::Error::NotFound => Error::NotFound,
            fatfs::Error::AlreadyExists => Error::AlreadyExists,
            fatfs::Error::DirectoryIsNotEmpty => Error::DirectoryIsNotEmpty,
            fatfs::Error::CorruptedFileSystem => Error::CorruptedFileSystem,
            fatfs::Error::NotEnoughSpace => Error::NotEnoughSpace,
            fatfs::Error::InvalidFileNameLength => Error::InvalidFileNameLength,
            fatfs::Error::UnsupportedFileNameCharacter => Error::UnsupportedFileNameCharacter,
            _ => Error::UnknownFsError,
        }
    }
}

impl SdmmcFs {
    pub fn new(sdmmc: sdmmc::Sdmmc, led: Pin) -> Self {
        Self {
            sdmmc,
            fs: None,
            led,
        }
    }

    pub fn blackbox(&mut self, now: Instant) -> Blackbox {
        // Try connecting to the SD card and initializing the file system
        if self.fs.is_none() || !self.sdmmc.connected() {
            self.fs = None;
            match self.sdmmc.try_connect() {
                Ok(_) => {
                    match FileSystem::new(self.sdmmc.clone(), fatfs::FsOptions::new()) {
                        Ok(new_fs) => self.fs = Some(new_fs),
                        Err(err) => {
                            defmt::debug!("Failed to initialize FatFS: {}", Error::from(err))
                        }
                    };
                }
                Err(err) => {
                    defmt::debug!("Failed to connect to SD card: {}", err);
                    self.sdmmc.disconnect();
                }
            }
        }
        // Try opening the data and CAN files
        let files = self.fs.as_ref().and_then(|fs| {
            let volume_id = fs.volume_id();
            let volume_label = fs.volume_label_as_bytes();
            let volume_label = core::str::from_utf8(volume_label).unwrap();
            let cluster_size = fs.cluster_size();
            defmt::debug!(
                "Found FatFS volume id: 0x{:08X}, label: {}, cluster size: {}",
                volume_id,
                volume_label,
                cluster_size
            );
            let root_dir = fs.root_dir();
            let data_file = append(
                &root_dir,
                "data.bin",
                MIN_CLUSTER_SIZE * 8,
                core::mem::size_of::<Record>(),
            )
            .inspect_err(|err| defmt::error!("Failed to create /data.bin: {}", err))
            .ok()?;
            defmt::info!("Opened data.bin and can.bin for writing");
            Some(Files { data_file })
        });
        Blackbox {
            files,
            led: self.led.clone(),
            created_at: now,
        }
    }
}

fn append<'a>(
    dir: &Dir<'a>,
    file_name: &str,
    buf_size: usize,
    record_size: usize,
) -> Result<BufferedFile<'a>, Error> {
    let mut file = dir.create_file(file_name)?;
    let cursor = file.seek(fatfs::SeekFrom::End(0))? as usize;
    let aligned_cursor = cursor / record_size * record_size;
    if cursor != aligned_cursor {
        let offset = aligned_cursor as i64 - cursor as i64;
        defmt::debug!("Seeking {} bytes to aligned cursor", offset);
        file.seek(fatfs::SeekFrom::Current(offset))?;
    }
    Ok(BufferedFile::new(file, buf_size))
}

impl Blackbox<'_> {
    pub fn write_record(&mut self, record: Record) {
        if let Some(Files { data_file, .. }) = &mut self.files {
            if let Err(err) = data_file.write(record.as_bytes(), &mut self.led) {
                self.led.set_low();
                if err == Error::BufferOverrun {
                    // Clear the buffer
                    let dropped_records = data_file.buf_len / core::mem::size_of::<Record>();
                    data_file.buf_len %= core::mem::size_of::<Record>();
                    defmt::warn!("Dropped {} records due to buffer overrun", dropped_records);
                } else {
                    self.files = None;
                    defmt::warn!("Failed to write record: {}", err);
                }
            }
        }
    }

    pub fn needs_reset(&self, now: Instant) -> bool {
        self.files.is_none() && now.checked_duration_since(self.created_at).unwrap() > RESET_PERIOD
    }
}
