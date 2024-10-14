use std::f64::consts::PI;
use std::io::{self, Read, Write};
use std::str::FromStr;
use std::time::Duration;

use arrayvec::ArrayVec;
use enumflags2::{bitflags, BitFlags};
use nox::{ArrayRepr, Tensor, Vector};
use roci::System;
use roci::{drivers::Hz, Componentize, Decomponentize};
use serialport::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilder, StopBits};
use tracing::{error, info, trace, warn};

use crate::{determination, stdio};

pub struct McuDriver<const HZ: usize, H: System> {
    port: Port,
    read_buf: Vec<u8>,
    config: McuConfig,
    gnc_system: H,
    pub com_type: ComType,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct McuConfig {
    #[serde(default)]
    pub arm_reaction_wheels: bool,
    #[serde(default)]
    pub enable_gps: bool,
    pub baud_rate: Option<u32>,
    pub adcs_format: BitFlags<AdcsFormat>,
    pub gps_update_interval: u32,
    pub gps_cycle_duration: u32,
    pub gps_cycle_interval: u32,
    #[serde(flatten)]
    pub com_type: ComType,
}

enum Port {
    Serial {
        port: Box<dyn SerialPort>,
        builder: SerialPortBuilder,
    },
    Stdio {
        stdout: thingbuf::mpsc::blocking::Sender<Vec<u8>>,
        stdin: thingbuf::mpsc::blocking::Receiver<Vec<u8>>,
    },
}
impl Port {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        match self {
            Port::Serial { port, .. } => port.read(buf),
            Port::Stdio { stdin, .. } => {
                if let Ok(recv) = stdin.try_recv_ref() {
                    let len = recv.len().min(buf.len());
                    buf[..len].copy_from_slice(&recv[..len]);
                    Ok(len)
                } else {
                    Ok(0)
                }
            }
        }
    }

    fn set_timeout(&mut self, from_secs: Duration) -> serialport::Result<()> {
        match self {
            Port::Serial { port, .. } => port.set_timeout(from_secs),
            Port::Stdio { .. } => Ok(()),
        }
    }

    fn write_all(&mut self, arg: &[u8]) -> io::Result<()> {
        match self {
            Port::Serial { port, .. } => port.write_all(arg),
            Port::Stdio { stdout, .. } => stdout
                .send(arg.to_vec())
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "")),
        }
    }

    fn reconnect(&mut self) {
        match self {
            Port::Serial {
                ref mut port,
                builder,
            } => {
                *port = match builder.clone().open() {
                    Ok(port) => port,
                    Err(err) => {
                        error!(?err, "error reopening port");
                        return;
                    }
                }
            }
            Port::Stdio { .. } => {}
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Port::Serial { port, .. } => port.flush(),
            Port::Stdio { .. } => Ok(()),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
#[serde(tag = "com_type", content = "path", rename_all = "snake_case")]
pub enum ComType {
    Serial(String),
    Radio(String),
    Stdio,
}

impl ComType {
    fn port(&self) -> io::Result<Port> {
        match self {
            ComType::Serial(path) | ComType::Radio(path) => {
                let builder = serialport::new(path.clone(), 115200)
                    .data_bits(DataBits::Eight)
                    .flow_control(FlowControl::Software)
                    .parity(Parity::None)
                    .stop_bits(StopBits::One);
                let port = builder.clone().open()?;
                Ok(Port::Serial { port, builder })
            }
            ComType::Stdio => {
                let (handle_in, handle_out, stdout, stdin) = stdio::pair();
                handle_in.spawn();
                handle_out.spawn();
                Ok(Port::Stdio { stdin, stdout })
            }
        }
    }
}

#[derive(Default, Componentize, Decomponentize)]
pub struct ControlWorld {
    #[roci(entity_id = 0, component_id = "rw_pwm_setpoint")]
    rw_pwm_setpoint: [f64; 3],
}

impl<const HZ: usize, H: System> McuDriver<HZ, H> {
    pub fn new(config: McuConfig, gnc_system: H) -> io::Result<Self> {
        let com_type = config.com_type.clone();
        let read_buf = Vec::with_capacity(1024);
        let mut mcu_driver = Self {
            port: com_type.port()?,
            read_buf,
            config,
            gnc_system,
            com_type,
        };
        mcu_driver.init_mcu();
        Ok(mcu_driver)
    }

    pub fn try_read_lines(&mut self) -> io::Result<Vec<String>> {
        let mut lines = Vec::default();
        let mut buf = [0u8; 1024];
        loop {
            let bytes_read = match self.port.read(&mut buf) {
                Ok(n) => n,
                Err(ref e) if e.kind() == io::ErrorKind::TimedOut => 0,
                Err(e) => return Err(e),
            };
            trace!(bytes_read, "read bytes");
            if bytes_read == 0 {
                break;
            }
            self.read_buf.extend_from_slice(&buf[..bytes_read]);
            while let Some(pos) = self.read_buf.iter().position(|&b| b == b'\n') {
                let line = std::str::from_utf8(&self.read_buf[..pos + 1])
                    .map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Could not parse line as UTF-8 string",
                        )
                    })?
                    .trim_end()
                    .to_string();
                lines.push(line);
                self.read_buf.drain(..pos + 1);
            }
        }
        Ok(lines)
    }

    pub fn print_info(&mut self) -> io::Result<()> {
        self.write_cmd("info")?;
        Ok(())
        // loop {
        //     for line in self.try_read_lines()? {
        //         println!("{}", line);
        //         if line == "|info|" {
        //             return Ok(());
        //         }
        //     }
        // }
    }

    fn init_mcu(&mut self) {
        let McuConfig {
            adcs_format,
            gps_update_interval,
            gps_cycle_duration,
            gps_cycle_interval,
            enable_gps,
            ..
        } = self.config.clone();
        self.init_adcs(adcs_format).unwrap();
        if enable_gps {
            if let Err(err) =
                self.init_gps(gps_update_interval, gps_cycle_duration, gps_cycle_interval)
            {
                error!(?err, "failed to initialize GPS");
            }
        }
    }

    fn init_adcs(&mut self, adcs_format: BitFlags<AdcsFormat>) -> io::Result<()> {
        let cmd = adcs_init_command(adcs_format);
        trace!(?cmd, "sending adcs init cmd");
        self.write_cmd(&cmd)?;
        self.port.set_timeout(Duration::from_secs(1))?;
        Ok(())
    }

    fn init_gps(
        &mut self,
        update_interval: u32,
        cycle_duration: u32,
        cycle_interval: u32,
    ) -> io::Result<()> {
        let cmd = gps_init_command(update_interval, cycle_duration, cycle_interval);
        trace!(?cmd, "sending gps init cmd");
        self.write_cmd(&cmd)?;
        self.port.set_timeout(Duration::from_secs(1))?;
        let ack_msg = format!("|{}|", cmd.replace(';', "|").replace('=', ":"));
        let nack_msg = "|!KEY|";
        loop {
            for line in self.try_read_lines()? {
                trace!(?line, "recv gps init response");
                if line == ack_msg {
                    info!("gps initialized success");
                    return Ok(());
                } else if line == nack_msg {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "received NACK from MCU",
                    ));
                }
            }
        }
    }

    fn set_controls(&mut self, control_world: &mut ControlWorld) {
        // map rad/sec to RPM
        let rpms = control_world
            .rw_pwm_setpoint
            .map(|pwm| pwm.clamp(-4096., 4096.) as i32);
        let cmd = set_rw_pwm_command(rpms);
        trace!(?cmd, "sending rw pwm cmd");
        if self.config.arm_reaction_wheels {
            self.write_cmd(&cmd).unwrap();
        } else {
            trace!("skip rw command")
        }
    }

    fn send_mrp(&mut self, mrp: [f64; 3]) {
        let cmd = format!("mrp={},{},{}", mrp[0], mrp[1], mrp[2]);
        self.write_cmd(&cmd).unwrap();
    }

    fn read_sensor_data(&mut self, world: &mut determination::World) {
        if let Err(err) = self.write_cmd("adcs") {
            error!(?err, "error writing adcs command");
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
        let lines = match self.try_read_lines() {
            Ok(lines) => lines,
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => {
                error!("broken pipe, attempting to reopen port");
                self.port.reconnect();
                info!("port reopened successfully");
                self.init_mcu();
                return;
            }
            Err(err) => {
                error!(?err, "error reading from mcu");
                return;
            }
        };
        for line in lines {
            trace!(?line, "recv line");
            if line == "exit-adcs" {
                info!("received exit command");
                std::process::exit(0);
            }
            if let Some(msg) = line.strip_prefix("[adcs]") {
                if msg.starts_with('#') {
                    continue;
                }
                let sensor_data = match SensorData::from_str(msg) {
                    Ok(sensor_data) => sensor_data,
                    Err(err) => {
                        error!(?line, ?err, "error parsing sensor data");
                        continue;
                    }
                };

                let side_lum = sensor_data
                    .css_side_avg
                    .map(|v| v / 2u64.pow(12) as f64) // 12-bit ADC
                    .map(|v| v.clamp(0.0, 1.0));

                let max_cos = side_lum.iter().copied().fold(0.0, f64::max);
                if max_cos > f64::EPSILON {
                    world.css_inputs = side_lum;
                }

                world.mag_value = sensor_data.mag.map(|v| v as f64);
                // swap x and y
                // world.mag_value = [world.mag_value[1], world.mag_value[0], world.mag_value[2]];
                world.omega =
                    Vector::<f64, 3, ArrayRepr>::from_buf(sensor_data.gyro) * (PI / 180.0f64);

                // convert RPM to rad/sec
                if self
                    .config
                    .adcs_format
                    .contains(AdcsFormat::IncludeReactionWheelRpm)
                {
                    world.rw_speed = sensor_data
                        .reaction_wheel_rpm
                        .map(|v| v * 2.0 * PI / 60.0 + 0.01);
                }
            } else if let Some(msg) = line.strip_prefix("[FILE][gps]") {
                trace!(?msg, "received gps message");
                let gps_data = match GpsData::from_str(msg) {
                    Ok(gps_data) => gps_data,
                    Err(err) => {
                        error!(?err, "error parsing gps data");
                        continue;
                    }
                };
                trace!(
                    lat = ?gps_data.latitude, long = ?gps_data.longitude, gps_data.altitude,
                    "received gps"
                );
                world.gps_inputs.lat = gps_data.latitude;
                world.gps_inputs.long = gps_data.longitude;
                world.gps_inputs.alt = gps_data.altitude;
            }
        }
    }

    fn write_cmd(&mut self, cmd: &str) -> io::Result<()> {
        match self.com_type {
            ComType::Serial(_) => {}
            ComType::Radio(_) => {
                self.port.write_all(b">")?;
            }
            ComType::Stdio => {
                self.port.write_all(b"~")?;
            }
        }
        self.port.flush()?;
        match self.port.write_all(cmd.as_bytes()) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                warn!("serial port timeout");
            }
            Err(e) => return Err(e),
        }
        self.port.flush()?;
        match self.port.write_all(b"\n") {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                warn!("serial port timeout");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl<const HZ: usize, H: System<Driver = Hz<{ HZ }>>> System for McuDriver<HZ, H> {
    type World = (determination::World, H::World, ControlWorld);
    type Driver = Hz<HZ>;

    fn init_world(&mut self) -> Self::World {
        let world = determination::World {
            css_inputs: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            mag_value: [0.0, 0.0, 1.0],
            sun_ref: Tensor::from_buf([0.0, 1.0, 0.0]),
            mag_ref: Tensor::from_buf([0.0, 0.0, 1.0]),
            gps_inputs: determination::GpsInputs {
                lat: 40.1,
                long: -111.8,
                alt: 0.0,
            },
            rw_speed: [0.01, 0.01, 0.01],
            ..Default::default()
        };
        let gnc_world = self.gnc_system.init_world();
        let control_world = ControlWorld::default();
        (world, gnc_world, control_world)
    }

    fn update(&mut self, (world, gnc_world, control_world): &mut Self::World) {
        self.port.set_timeout(Duration::from_secs(0)).unwrap();
        self.read_sensor_data(world);
        world.sink_columns(gnc_world);
        self.gnc_system.update(gnc_world);
        gnc_world.sink_columns(control_world);
        gnc_world.sink_columns(world);
        //std::thread::sleep(Duration::from_millis(100));
        self.set_controls(control_world);
        self.send_mrp(world.nav_out.att_mrp_bn);
    }
}

#[bitflags]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AdcsFormat {
    IncludeMag = 0b00000001,
    IncludeGyro = 0b00000010,
    IncludeAccel = 0b00000100,
    IncludeCss = 0b00100000,
    IncludeReactionWheelRpm = 0b1_00000000,
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
}

#[derive(Debug, Default, PartialEq)]
pub struct GpsData {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub unix_time: u64,
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

        Ok(sensor_data)
    }
}

impl FromStr for GpsData {
    type Err = SensorParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // skip any leading tags in the string such as "[GPS]"
        // the tags are assumed to always be enclosed in square brackets
        let last_tag = s.rfind(']').map(|i| i + 1).unwrap_or(0);
        let (_tags, s) = s.split_at(last_tag);

        let mut s = s.split(',');
        let latitude = s
            .next()
            .ok_or(SensorParseError::InsufficientParts)?
            .parse()?;
        let longitude = s
            .next()
            .ok_or(SensorParseError::InsufficientParts)?
            .parse()?;
        let altitude = s
            .next()
            .ok_or(SensorParseError::InsufficientParts)?
            .parse()?;
        let unix_time = s
            .next()
            .ok_or(SensorParseError::InsufficientParts)?
            .parse()?;

        Ok(GpsData {
            latitude,
            longitude,
            altitude,
            unix_time,
        })
    }
}

// adcs_format - a bitfield that specifies which sensor values are included in the ADCS cycle
// Returns a MCU command string that configures the ADCS cycle.
// E.g: a.fmt=7;a.upd=1000;a.dur=10;a.int=100
pub fn adcs_init_command(adcs_format: BitFlags<AdcsFormat>) -> String {
    format!("a.fmt={};a.stream=0;rw.begin", adcs_format.bits())
}

// update_interval (msec) - how often GPS sensor values are sent by the MCU during an active cycle
// cycle_duration (sec) - how long the GPS cycle lasts
// cycle_interval (sec) - how long the GPS cycle is inactive before the next cycle starts
// Returns a MCU command string that configures the PS cycle.
// E.g: gps.fmt=0;gps.upd=1000;gps.dur=10;gps.int=100
pub fn gps_init_command(update_interval: u32, cycle_duration: u32, cycle_interval: u32) -> String {
    format!(
        "g.fmt=0;g.upd={};g.dur={};g.int={}",
        update_interval, cycle_duration, cycle_interval
    )
}

// Returns a MCU command string that sets the reaction wheel RPM values.
pub fn set_rw_pwm_command(pwm: [i32; 3]) -> String {
    format!("rw.pwm={},{},{}", pwm[0], pwm[1], pwm[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_adcs_cycle() {
        assert_eq!(
            adcs_init_command(
                AdcsFormat::IncludeMag | AdcsFormat::IncludeGyro | AdcsFormat::IncludeAccel
            ),
            "a.fmt=7;a.stream=0;rw.begin"
        );

        assert_eq!(
            adcs_init_command(
                AdcsFormat::IncludeMag
                    | AdcsFormat::IncludeGyro
                    | AdcsFormat::IncludeAccel
                    | AdcsFormat::IncludeCss
            ),
            "a.fmt=39;a.stream=0;rw.begin"
        );
    }

    #[test]
    fn test_config_gps_cycle() {
        assert_eq!(
            gps_init_command(1000, 10, 100),
            "g.fmt=0;g.upd=1000;g.dur=10;g.int=100"
        );
    }

    #[test]
    fn test_parse_sensor_data() {
        let sensor_data = "7|0,0,0|0,0,0|0,0,0".parse::<SensorData>().unwrap();
        let expected_sensor_data = SensorData {
            adcs_format: AdcsFormat::IncludeMag
                | AdcsFormat::IncludeGyro
                | AdcsFormat::IncludeAccel,
            ..Default::default()
        };
        assert_eq!(sensor_data, expected_sensor_data);

        // ignore leading tags
        let sensor_data = "[FILE][adcs]7|-824,532,37|-0.12,0.54,0.12|0.18,-0,10.02"
            .parse::<SensorData>()
            .unwrap();
        let expected_sensor_data = SensorData {
            adcs_format: AdcsFormat::IncludeMag
                | AdcsFormat::IncludeGyro
                | AdcsFormat::IncludeAccel,
            mag: [-824, 532, 37],
            gyro: [-0.12, 0.54, 0.12],
            accel: [0.18, 0.0, 10.02],
            ..Default::default()
        };
        assert_eq!(sensor_data, expected_sensor_data);

        let sensor_data =
            "[FILE][adcs]39|-810,538,38|-0.28,-0.02,0.15|0.21,0.01,9.91|,,,,65535,65535,,,,,,,,"
                .parse::<SensorData>()
                .unwrap();
        let expected_sensor_data = SensorData {
            adcs_format: AdcsFormat::IncludeMag
                | AdcsFormat::IncludeGyro
                | AdcsFormat::IncludeAccel
                | AdcsFormat::IncludeCss,
            mag: [-810, 538, 38],
            gyro: [-0.28, -0.02, 0.15],
            accel: [0.21, 0.01, 9.91],
            css_side_avg: [0.0, 0.0, 0.0, 0.0, 65535.0, 65535.0],
            css_vertex_avg: [0.0; 8],
            ..Default::default()
        };
        assert_eq!(sensor_data, expected_sensor_data);
    }

    #[test]
    fn test_parse_gps_data() {
        let gps_data = "0,0,0,0".parse::<GpsData>().unwrap();
        let expected_gps_data = GpsData::default();
        assert_eq!(gps_data, expected_gps_data);

        let gps_data = "40.1,-111.8,800.0,1719068896".parse::<GpsData>().unwrap();
        let expected_gps_data = GpsData {
            latitude: 40.1,
            longitude: -111.8,
            altitude: 800.0,
            unix_time: 1719068896,
        };
        assert_eq!(gps_data, expected_gps_data);
    }
}
