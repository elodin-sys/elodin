use std::f64::consts::PI;
use std::io::Write;
use std::str::FromStr;
use std::time::Duration;

use arrayvec::ArrayVec;
use enumflags2::{bitflags, BitFlags};
use nox::{ArrayRepr, Tensor, Vector};
use roci::drivers::Hz;
use roci::System;
use serialport::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilder, StopBits};

use crate::determination::{GpsInputs, World};

pub struct McuDriver {
    port_builder: SerialPortBuilder,
    port: Box<dyn SerialPort>,
    read_buf: Vec<u8>,
    config: McuConfig,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct McuConfig {
    pub path: String,
    pub baud_rate: Option<u32>,
    pub adcs_format: BitFlags<AdcsFormat>,
    pub adcs_update_interval: u32,
    pub adcs_cycle_duration: u32,
    pub adcs_cycle_interval: u32,
    pub gps_update_interval: u32,
    pub gps_cycle_duration: u32,
    pub gps_cycle_interval: u32,
}

impl McuDriver {
    pub fn new(config: McuConfig) -> std::io::Result<Self> {
        let port_builder = serialport::new(&config.path, config.baud_rate.unwrap_or(9600))
            .data_bits(DataBits::Eight)
            .flow_control(FlowControl::Software)
            .parity(Parity::None)
            .stop_bits(StopBits::One);
        let port = port_builder.clone().open()?;
        let read_buf = Vec::with_capacity(1024);
        let mut mcu_driver = Self {
            port_builder,
            port,
            read_buf,
            config,
        };
        mcu_driver.init_mcu();
        Ok(mcu_driver)
    }

    pub fn try_read_lines(&mut self) -> std::io::Result<Vec<String>> {
        let mut lines = Vec::default();
        let mut buf = [0u8; 1024];
        let bytes_read = match self.port.read(&mut buf) {
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => 0,
            Err(e) => return Err(e),
        };
        if bytes_read > 0 {
            self.read_buf.extend_from_slice(&buf[..bytes_read]);
            while let Some(pos) = self.read_buf.iter().position(|&b| b == b'\n') {
                let line = std::str::from_utf8(&self.read_buf[..pos + 1])
                    .map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
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

    pub fn print_info(&mut self) -> std::io::Result<()> {
        writeln!(&mut self.port, "info")?;
        self.port.set_timeout(Duration::from_secs(1))?;
        loop {
            for line in self.try_read_lines()? {
                println!("{}", line);
                if line == "|info|" {
                    return Ok(());
                }
            }
        }
    }

    fn init_mcu(&mut self) {
        let McuConfig {
            adcs_format,
            adcs_update_interval,
            adcs_cycle_duration,
            adcs_cycle_interval,
            gps_update_interval,
            gps_cycle_duration,
            gps_cycle_interval,
            ..
        } = self.config.clone();
        self.init_adcs(
            adcs_format,
            adcs_update_interval,
            adcs_cycle_duration,
            adcs_cycle_interval,
        )
        .unwrap();
        if let Err(e) = self.init_gps(gps_update_interval, gps_cycle_duration, gps_cycle_interval) {
            eprintln!("Failed to initialize GPS: {}", e);
        }
    }

    fn init_adcs(
        &mut self,
        adcs_format: BitFlags<AdcsFormat>,
        update_interval: u32,
        cycle_duration: u32,
        cycle_interval: u32,
    ) -> std::io::Result<()> {
        let cmd = adcs_init_command(update_interval, cycle_duration, cycle_interval, adcs_format);
        println!("-> {}", cmd);
        writeln!(&mut self.port, "{}", cmd)?;
        self.port.set_timeout(Duration::from_secs(1))?;
        let ack_msg = format!("|{}|", cmd.replace(';', "|").replace('=', ":"));
        let nack_msg = "|!KEY|";
        loop {
            for line in self.try_read_lines()? {
                println!("<- {}", line);
                if line == ack_msg {
                    println!("ADCS initialized success");
                    return Ok(());
                } else if line == nack_msg {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "received NACK from MCU",
                    ));
                }
            }
        }
    }

    fn init_gps(
        &mut self,
        update_interval: u32,
        cycle_duration: u32,
        cycle_interval: u32,
    ) -> std::io::Result<()> {
        let cmd = gps_init_command(update_interval, cycle_duration, cycle_interval);
        println!("-> {}", cmd);
        writeln!(&mut self.port, "{}", cmd)?;
        self.port.set_timeout(Duration::from_secs(1))?;
        let ack_msg = format!("|{}|", cmd.replace(';', "|").replace('=', ":"));
        let nack_msg = "|!KEY|";
        loop {
            for line in self.try_read_lines()? {
                println!("<- {}", line);
                if line == ack_msg {
                    println!("GPS initialized success");
                    return Ok(());
                } else if line == nack_msg {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "received NACK from MCU",
                    ));
                }
            }
        }
    }
}

impl System for McuDriver {
    type World = World;
    type Driver = Hz<100>;

    fn init_world(&mut self) -> Self::World {
        World {
            css_inputs: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            mag_value: [0.0, 0.0, 1.0],
            sun_ref: Tensor::from_buf([0.0, 1.0, 0.0]),
            mag_ref: Tensor::from_buf([0.0, 0.0, 1.0]),
            gps_inputs: GpsInputs {
                lat: 40.1,
                long: -111.8,
                alt: 0.0,
            },
            ..Default::default()
        }
    }

    fn update(&mut self, world: &mut Self::World) {
        self.port.set_timeout(Duration::from_secs(0)).unwrap();
        let lines = match self.try_read_lines() {
            Ok(lines) => lines,
            Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => {
                eprintln!("broken pipe, attempting to reopen port");
                let port = match self.port_builder.clone().open() {
                    Ok(port) => port,
                    Err(err) => {
                        eprintln!("error reopening port: {}", err);
                        return;
                    }
                };
                eprintln!("port reopened successfully");
                self.port = port;
                self.init_mcu();
                return;
            }
            Err(err) => {
                eprintln!("error reading from MCU: {}", err);
                return;
            }
        };
        for line in lines {
            // println!("<- {}", line);
            if let Some(msg) = line.strip_prefix("[FILE][adcs]") {
                if msg.starts_with('#') {
                    continue;
                }
                let sensor_data = match SensorData::from_str(msg) {
                    Ok(sensor_data) => sensor_data,
                    Err(err) => {
                        eprintln!("error parsing sensor data: {}", err);
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
                world.omega =
                    Vector::<f64, 3, ArrayRepr>::from_buf(sensor_data.gyro) * (PI / 180.0f64);
            } else if let Some(msg) = line.strip_prefix("[FILE][gps]") {
                println!("<- received gps message: {}", msg);
                let gps_data = match GpsData::from_str(msg) {
                    Ok(gps_data) => gps_data,
                    Err(err) => {
                        eprintln!("error parsing gps data: {}", err);
                        continue;
                    }
                };
                println!(
                    "<- received gps, lat: {}, long: {}, alt: {}",
                    gps_data.latitute, gps_data.longitude, gps_data.altitude
                );
                world.gps_inputs.lat = gps_data.latitute;
                world.gps_inputs.long = gps_data.longitude;
                world.gps_inputs.alt = gps_data.altitude;
            }
        }
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
}

#[derive(Debug, Default, PartialEq)]
pub struct SensorData {
    pub mag: [i64; 3], // milligauss
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
    pub css_side_avg: [f64; 6],
    pub css_vertex_avg: [f64; 8],
    pub reaction_wheel_rpm: [i64; 3],
    pub adcs_format: BitFlags<AdcsFormat>,
}

#[derive(Debug, Default, PartialEq)]
pub struct GpsData {
    pub latitute: f64,
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
                .map_err(|_| SensorParseError::InsufficientParts)?;
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
        let latitute = s
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
            latitute,
            longitude,
            altitude,
            unix_time,
        })
    }
}

// update_interval (msec) - how often ADCS sensor values are sent by the MCU during an active cycle
// cycle_duration (sec) - how long the ADCS cycle lasts
// cycle_interval (sec) - how long the ADCS cycle is inactive before the next cycle starts
// adcs_format - a bitfield that specifies which sensor values are included in the ADCS cycle
// Returns a MCU command string that configures the ADCS cycle.
// E.g: a.fmt=7;a.upd=1000;a.dur=10;a.int=100
pub fn adcs_init_command(
    update_interval: u32,
    cycle_duration: u32,
    cycle_interval: u32,
    adcs_format: BitFlags<AdcsFormat>,
) -> String {
    format!(
        "a.fmt={};a.upd={};a.dur={};a.int={}",
        adcs_format.bits(),
        update_interval,
        cycle_duration,
        cycle_interval
    )
}

// update_interval (msec) - how often GPS sensor values are sent by the MCU during an active cycle
// cycle_duration (sec) - how long the GPS cycle lasts
// cycle_interval (sec) - how long the GPS cycle is inactive before the next cycle starts
// Returns a MCU command string that configures the GPS cycle.
// E.g: gps.fmt=0;gps.upd=1000;gps.dur=10;gps.int=100
pub fn gps_init_command(update_interval: u32, cycle_duration: u32, cycle_interval: u32) -> String {
    format!(
        "gps.fmt=0;gps.upd={};gps.dur={};gps.int={}",
        update_interval, cycle_duration, cycle_interval
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_adcs_cycle() {
        assert_eq!(
            adcs_init_command(
                1000,
                10,
                100,
                AdcsFormat::IncludeMag | AdcsFormat::IncludeGyro | AdcsFormat::IncludeAccel
            ),
            "a.fmt=7;a.upd=1000;a.dur=10;a.int=100"
        );

        assert_eq!(
            adcs_init_command(
                1000,
                10,
                100,
                AdcsFormat::IncludeMag
                    | AdcsFormat::IncludeGyro
                    | AdcsFormat::IncludeAccel
                    | AdcsFormat::IncludeCss
            ),
            "a.fmt=39;a.upd=1000;a.dur=10;a.int=100"
        );
    }

    #[test]
    fn test_config_gps_cycle() {
        assert_eq!(
            gps_init_command(1000, 10, 100),
            "gps.fmt=0;gps.upd=1000;gps.dur=10;gps.int=100"
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
            latitute: 40.1,
            longitude: -111.8,
            altitude: 800.0,
            unix_time: 1719068896,
        };
        assert_eq!(gps_data, expected_gps_data);
    }
}
