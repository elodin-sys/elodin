use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

use arrayvec::ArrayVec;
use enumflags2::{bitflags, BitFlags};
use nox::{ArrayRepr, SpatialTransform, MRP};
use roci::Metadatatize;
use roci::{
    drivers::{os_sleep_driver, Driver, Hz},
    tokio, Componentize, Decomponentize, System,
};

pub struct Mcu<const H: usize> {
    port: Box<dyn serialport::SerialPort>,
}

impl<const H: usize> Mcu<H> {
    pub fn new(mut port: Box<dyn serialport::SerialPort>) -> Self {
        writeln!(port, "a.fmt=547;a.stream=0").unwrap();
        writeln!(port, "bb.begin").unwrap();
        writeln!(port, "bb.send={:?}", "NO_COLOR=1 RUST_LOG=trace /root/adcs").unwrap();
        Self { port }
    }
}

static EXIT_FLAG: AtomicBool = AtomicBool::new(false);

#[derive(Default, Componentize, Decomponentize, Debug, Metadatatize)]
pub struct World {
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: SpatialTransform<f64, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "css_value")]
    pub css_inputs: [f64; 6],
    #[roci(entity_id = 0, component_id = "mag_value")]
    pub mag_value: [f64; 3],
    #[roci(entity_id = 0, component_id = "gyro_omega")]
    pub omega: [f64; 3],
    #[roci(entity_id = 0, component_id = "rw_speed")]
    pub rw_speed: [f64; 3],
    #[roci(entity_id = 0, component_id = "att_mrp_bn")]
    pub att_mrp_bn: [f64; 3],
}

impl<const H: usize> System for Mcu<H> {
    type World = World;

    type Driver = Hz<HZ>;

    fn init_world(&mut self) -> Self::World {
        Default::default()
    }

    fn update(&mut self, world: &mut Self::World) {
        writeln!(self.port, "adcs;").unwrap();
        let mut buf = BufReader::new(&mut self.port);
        let mut line = String::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        while buf.read_line(&mut line).is_ok() {
            'inner: for line in line.lines() {
                println!("{}", line);
                if line.get(..6) == Some("[adcs]") {
                    let data = match SensorData::from_str(&line[6..]) {
                        Ok(data) => data,
                        Err(err) => {
                            println!("error parsing sensor data {:?} {:?}", err, line);
                            continue 'inner;
                        }
                    };
                    world.att_mrp_bn = data.mrp;
                    world.css_inputs = data.css_side_avg;
                    world.omega = data.gyro.map(|x| x.to_radians());
                    world.rw_speed = data.reaction_wheel_rpm;
                    world.mag_value = data.mag.map(|x| x as f64);
                    let mrp = MRP::<_, ArrayRepr>(nox::Tensor::from_buf(data.mrp));
                    let quat = nox::Quaternion::from(&mrp);
                    world.inertial_pos = SpatialTransform::from_angular(quat);
                }
            }
            line.clear();
        }
        if EXIT_FLAG.load(SeqCst) {
            writeln!(self.port, "bb.send={:?}", "exit-adcs").unwrap();
            self.port.flush().unwrap();
            std::process::exit(0);
        }
    }
}

const HZ: usize = 50;

fn main() {
    ctrlc::set_handler(move || EXIT_FLAG.store(true, SeqCst))
        .expect("Error setting Ctrl-C handler");

    let (tx, _rx) =
        tokio::tcp_connect::<Hz<HZ>>("127.0.0.1:2240".parse().unwrap(), &[], World::metadata());
    let chamber = Mcu::<HZ>::new(
        serialport::new("/dev/tty.usbmodem2079328738311", 115200)
            .open()
            .unwrap(),
    );

    os_sleep_driver(chamber.pipe(tx)).run();
}

#[bitflags]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AdcsFormat {
    IncludeMag = 0b00_0000_0001,
    IncludeGyro = 0b00_0000_0010,
    IncludeAccel = 0b00_0000_0100,
    IncludeCss = 0b00_0010_0000,
    IncludeReactionWheelRpm = 0b1_0000_0000,
    IncludeMrp = 0b10_0000_0000,
}

#[derive(Debug, Default, PartialEq)]
pub struct SensorData {
    pub mag: [i64; 3], // milligauss
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
    pub css_side_avg: [f64; 6],
    pub css_vertex_avg: [f64; 8],
    pub reaction_wheel_rpm: [f64; 3],
    pub adcs_format: BitFlags<AdcsFormat>,
    pub mrp: [f64; 3],
}

#[derive(Debug, thiserror::Error)]
pub enum SensorParseError {
    #[error("Insufficient parts")]
    InsufficientParts,
    #[error("Invalid ADCS format enum")]
    InvalidAdcsFormat,
    #[error("Invalid integer: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("Invalid float: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
}

impl FromStr for SensorData {
    type Err = SensorParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // skip any leading tags in the string such as "[ADCS]"
        // the tags are assumed to always be enclosed in square brackets
        let last_tag = s.rfind(']').map(|i| i + 1).unwrap_or(0);
        let (_tags, s) = s.split_at(last_tag);

        let mut s = s.split('|');
        let adcs_format = s.next().ok_or(SensorParseError::InsufficientParts)?;
        let adcs_format = adcs_format.parse::<u32>()?;
        let adcs_format = BitFlags::<AdcsFormat>::from_bits(adcs_format)
            .map_err(|_| SensorParseError::InvalidAdcsFormat)?;

        let mut sensor_data = SensorData {
            adcs_format,
            ..Default::default()
        };

        if sensor_data.adcs_format.contains(AdcsFormat::IncludeMag) {
            sensor_data.mag = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(3)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 3>, _>>()?
                .into_inner()
                .map_err(|_| SensorParseError::InsufficientParts)?;
        }

        if sensor_data.adcs_format.contains(AdcsFormat::IncludeGyro) {
            sensor_data.gyro = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(3)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 3>, _>>()?
                .into_inner()
                .map_err(|_| SensorParseError::InsufficientParts)?;
        }

        if sensor_data.adcs_format.contains(AdcsFormat::IncludeAccel) {
            sensor_data.accel = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(3)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 3>, _>>()?
                .into_inner()
                .map_err(|_| SensorParseError::InsufficientParts)?;
        }

        if sensor_data.adcs_format.contains(AdcsFormat::IncludeCss) {
            sensor_data.css_side_avg = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(6)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 6>, _>>()?
                .into_inner()
                .map_err(|_| SensorParseError::InsufficientParts)?;
        }

        if sensor_data
            .adcs_format
            .contains(AdcsFormat::IncludeReactionWheelRpm)
        {
            sensor_data.reaction_wheel_rpm = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(3)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 3>, _>>()?
                .into_inner()
                .unwrap_or_default();
        }

        if sensor_data.adcs_format.contains(AdcsFormat::IncludeMrp) {
            sensor_data.mrp = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .take(3)
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse)
                .collect::<Result<ArrayVec<_, 3>, _>>()?
                .into_inner()
                .unwrap_or_default();
        }

        Ok(sensor_data)
    }
}
