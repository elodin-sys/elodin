use std::str::FromStr;
use std::{borrow::Cow, io::Write};

use arrayvec::ArrayVec;
use enumflags2::{bitflags, BitFlags};
use nox::Tensor;
use roci::drivers::Hz;
use roci::System;
use serialport::{DataBits, FlowControl, Parity, SerialPort, StopBits};

use crate::determination::World;

pub struct McuDriver {
    port: Box<dyn SerialPort>,
    read_buf: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct McuConfig {
    pub path: String,
    pub baud_rate: Option<u32>,
}

impl McuDriver {
    pub fn new<'a>(path: impl Into<Cow<'a, str>>, baud_rate: Option<u32>) -> std::io::Result<Self> {
        let mut port = serialport::new(path.into(), baud_rate.unwrap_or(9600)).open()?;
        port.set_data_bits(DataBits::Eight)?;
        port.set_flow_control(FlowControl::Software)?;
        port.set_parity(Parity::None)?;
        port.set_stop_bits(StopBits::One)?;
        let read_buf = Vec::with_capacity(1024);
        Ok(Self { port, read_buf })
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
        self.port.set_timeout(std::time::Duration::from_secs(1))?;
        loop {
            for line in self.try_read_lines()? {
                println!("{}", line);
                if line == "|info|" {
                    return Ok(());
                }
            }
        }
    }

    pub fn init_adcs(
        &mut self,
        update_interval: u32,
        cycle_duration: u32,
        cycle_interval: u32,
        adcs_format: BitFlags<AdcsFormat>,
    ) -> std::io::Result<()> {
        let cmd = adcs_init_command(update_interval, cycle_duration, cycle_interval, adcs_format);
        writeln!(&mut self.port, "{}", cmd)?;
        self.port.set_timeout(std::time::Duration::from_secs(1))?;
        let ack_msg = format!("|{}|", cmd.replace(';', "|").replace('=', ":"));
        loop {
            for line in self.try_read_lines()? {
                if line == ack_msg {
                    return Ok(());
                }
            }
        }
    }

    pub fn try_get_sensor_data(&mut self) -> std::io::Result<Option<SensorData>> {
        self.port.set_timeout(std::time::Duration::from_secs(0))?;
        self.try_read_lines()?
            .into_iter()
            .filter(|l| l.starts_with("[FILE][adcs]"))
            .filter(|l| !l.starts_with("[FILE][adcs]#"))
            .last()
            .map(|l| {
                SensorData::from_str(&l)
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
            })
            .transpose()
    }
}

impl System for McuDriver {
    type World = World;
    type Driver = Hz<100>;

    fn init_world(&mut self) -> Self::World {
        let mut world = World::default();
        world.css_inputs.css_0 = 0.0;
        world.css_inputs.css_1 = 1.0;
        world.css_inputs.css_2 = 0.0;
        world.mag_value = [0.0, 0.0, 1.0];
        world.sun_ref = Tensor::from_buf([0.0, 1.0, 0.0]);
        world.mag_ref = Tensor::from_buf([0.0, 0.0, 1.0]);
        world
    }

    fn update(&mut self, world: &mut Self::World) {
        match self.try_get_sensor_data() {
            Ok(Some(sensor_data)) => {
                let side_lum = sensor_data
                    .css_side_avg
                    .map(|v| v / 2u64.pow(12) as f64) // 12-bit ADC
                    .map(|v| v.clamp(0.0, 1.0));

                let [x_p, x_n, y_p, y_n, z_p, z_n] = side_lum;
                // high delta = facing sum = low cosine
                let xyz_lum = [x_p - x_n, y_p - y_n, z_p - z_n];
                println!("<- received css: {:?}", xyz_lum);

                let max_cos = xyz_lum.iter().copied().fold(-f64::INFINITY, f64::max);
                if max_cos > f64::EPSILON {
                    world.css_inputs.css_0 = xyz_lum[0];
                    world.css_inputs.css_1 = xyz_lum[1];
                    world.css_inputs.css_2 = xyz_lum[2];
                } else {
                    println!("-> no sun detected");
                }

                world.mag_value = Tensor::<f64, _, _>::from_buf(sensor_data.mag.map(|v| v as f64))
                    .normalize()
                    .into_buf();
                println!("<- received mag: {:?}", world.mag_value);
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!("error getting sensor data: {}", err);
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
            let css_data = s
                .next()
                .ok_or(SensorParseError::InsufficientParts)?
                .split(',')
                .map(|v| if v.is_empty() { "0" } else { v })
                .map(str::parse);

            sensor_data.css_side_avg = css_data
                .clone()
                .take(6)
                .collect::<Result<ArrayVec<_, 6>, _>>()?
                .into_inner()
                .map_err(|_| SensorParseError::InsufficientParts)?;

            sensor_data.css_vertex_avg = css_data
                .skip(6)
                .take(8)
                .collect::<Result<ArrayVec<_, 8>, _>>()?
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
}
