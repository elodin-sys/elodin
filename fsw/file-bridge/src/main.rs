use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Duration;
use zerocopy::FromBytes;

fn main() -> anyhow::Result<()> {
    std::fs::create_dir_all("/sensors")?;
    let mut mag = fifo("/sensors/mag");
    let mut gyro = fifo("/sensors/gyro");
    let mut accel = fifo("/sensors/accel");
    let mut baro = fifo("/sensors/accel");
    println!("reading");
    let mut port = serialport::new("/dev/ttyACM0", 115200)
        .timeout(Duration::MAX)
        .open()?;
    let mut read = vec![0; 256];
    while let Ok(n) = port.read(&mut read) {
        let mut i = 0;
        let buf = &read[..n];
        'decode: loop {
            match buf.get(i) {
                Some(0) => {
                    let Ok(decode) = cobs::decode_vec(&buf[(i + 1)..]) else {
                        break 'decode;
                    };
                    i += decode.len();
                    let Ok(data) = <Record>::read_from_bytes(&decode) else {
                        continue 'decode;
                    };
                    println!("record = {:?}", &data);
                    let mag_str =
                        format!("{:?},{:?},{:?}\n", data.mag[0], data.mag[1], data.mag[2]);
                    let _ = mag.write_all(mag_str.as_bytes());
                    let gyro_str =
                        format!("{:?},{:?},{:?}\n", data.gyro[0], data.gyro[1], data.gyro[2]);
                    let _ = gyro.write_all(gyro_str.as_bytes());
                    let accel_str = format!(
                        "{:?},{:?},{:?}\n",
                        data.accel[0], data.accel[1], data.accel[2]
                    );
                    let _ = accel.write_all(accel_str.as_bytes());
                    let baro_str = format!("{:?}\n", data.baro);
                    let _ = baro.write_all(baro_str.as_bytes());
                }
                Some(_) => {
                    i += 1;
                }
                None => {
                    break 'decode;
                }
            }
        }
    }
    Ok(())
}

fn fifo(path: impl AsRef<Path>) -> File {
    let path = path.as_ref();
    let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
    unsafe { libc::mkfifo(c_path.as_ptr(), 0o666) };
    OpenOptions::new()
        .append(true)
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .unwrap()
}

#[derive(zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, Clone, Debug)]
#[repr(C)]
pub struct Record {
    pub ts: u32, // in milliseconds
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
    pub baro: f32,
    pub baro_temp: f32,
}
