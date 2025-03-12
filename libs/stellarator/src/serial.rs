//! Async serial ports
use std::{io, path::Path};

use crate::{
    BufResult, Error,
    buf::{IoBuf, IoBufMut},
    fs::{File, OpenOptions},
    io::{AsyncRead, AsyncWrite},
};
use rustix::termios::{self, Termios};

/// Serial Port Baud Rate
///
/// This enum contains a set of serial port baud rates, specified in bits per second
pub enum Baud {
    /// 921600 Baud
    #[cfg(target_os = "linux")]
    B921600,
    #[cfg(target_os = "linux")]
    /// 460800 Baud
    B460800,
    /// 115200 Baud
    B115200,
    /// 9600 Baud
    B9600,
    /// A custom baud rate
    Other(u64),
}

impl Baud {
    /// The fastest baud rate available on the platform
    ///
    /// This is 921600 on Linux, and 115200 on macOS
    #[cfg(target_os = "linux")]
    pub fn fastest() -> Self {
        Baud::B921600
    }
    /// The fastest baud rate available on the platform
    ///
    /// This is 921600 on Linux, and 115200 on macOS
    #[cfg(not(target_os = "linux"))]
    pub fn fastest() -> Self {
        Baud::B115200
    }

    fn termios_baud(&self) -> u32 {
        match self {
            #[cfg(target_os = "linux")]
            Baud::B921600 => termios::speed::B921600,
            #[cfg(target_os = "linux")]
            Baud::B460800 => termios::speed::B460800,
            Baud::B115200 => termios::speed::B115200,
            Baud::B9600 => termios::speed::B9600,
            Baud::Other(b) => *b as _,
        }
    }
}

/// A handle to an opened serial port
pub struct SerialPort {
    termios: Termios,
    file: File,
}

impl SerialPort {
    /// Open a new serial port located at `path`
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut options = OpenOptions::default();
        options.write(true).read(true).custom_flags(libc::O_NOCTTY);

        let file = File::open_with(path, &options).await?;

        let termios = {
            let mut termios = termios::tcgetattr(&file)?;
            if cfg!(target_os = "linux") {
                termios.control_modes |=
                    termios::ControlModes::CLOCAL | termios::ControlModes::CREAD;
            }

            // Make raw
            termios.make_raw();

            // Configure serial port settings using rustix::termios structs
            termios.control_modes.remove(termios::ControlModes::CSTOPB);
            termios.control_modes.remove(termios::ControlModes::CRTSCTS);
            termios.local_modes.remove(termios::LocalModes::ICANON);
            termios
                .input_modes
                .remove(termios::InputModes::IXON | termios::InputModes::IXOFF);
            termios
                .control_modes
                .remove(termios::ControlModes::PARENB | termios::ControlModes::PARODD);
            termios.input_modes.remove(termios::InputModes::INPCK);
            termios.input_modes.insert(termios::InputModes::IGNPAR);
            termios.control_modes.remove(termios::ControlModes::CSIZE);
            termios.control_modes.insert(termios::ControlModes::CS8);

            termios::tcsetattr(&file, termios::OptionalActions::Now, &termios)?;

            termios
        };

        Ok(SerialPort { termios, file })
    }

    /// Set the baud rate for the serial port
    pub fn set_baud(&mut self, baud: impl Into<Baud>) -> io::Result<()> {
        let baud = baud.into();
        let speed = baud.termios_baud();
        self.termios.set_speed(speed)?;
        termios::tcsetattr(&self.file, termios::OptionalActions::Now, &self.termios)?;

        Ok(())
    }

    /// Write data to the serial port
    pub async fn write<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        self.file.write(buf).await
    }

    /// Read data from the serial port
    pub async fn read<B: IoBufMut>(&self, buf: B) -> BufResult<usize, B> {
        self.file.read(buf).await
    }
}

impl AsyncRead for SerialPort {
    fn read<B: IoBufMut>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.read(buf)
    }
}

impl AsyncWrite for SerialPort {
    fn write<B: IoBuf>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.write(buf)
    }
}

impl std::os::fd::AsRawFd for SerialPort {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.file.as_raw_fd()
    }
}
