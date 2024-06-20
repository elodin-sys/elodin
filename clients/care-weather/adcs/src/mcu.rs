use std::io::BufRead;
use std::str::FromStr;
use std::{borrow::Cow, io::Write};

use arrayvec::ArrayVec;
use enumflags2::{bitflags, BitFlags};
use serialport::{DataBits, FlowControl, Parity, SerialPort, StopBits};

pub struct McuDriver {
    serial_port: Box<dyn SerialPort>,
}

impl McuDriver {
    pub fn new<'a>(
        path: impl Into<Cow<'a, str>>,
        baud_rate: Option<u32>,
    ) -> Result<Self, serialport::Error> {
        let mut port = serialport::new(path.into(), baud_rate.unwrap_or(9600)).open()?;
        port.set_data_bits(DataBits::Eight)?;
        port.set_flow_control(FlowControl::Software)?;
        port.set_parity(Parity::None)?;
        port.set_stop_bits(StopBits::One)?;
        port.set_timeout(std::time::Duration::from_secs(1))?;
        Ok(Self { serial_port: port })
    }

    pub fn info(&mut self) -> std::io::Result<String> {
        writeln!(&mut self.serial_port, "info")?;
        let mut buf_reader = std::io::BufReader::new(&mut self.serial_port);
        let mut info = String::new();
        while !info.trim().ends_with("|info|") {
            buf_reader.read_line(&mut info)?;
        }
        Ok(info)
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
    IncludeReactionWheelRpm = 0b10000000,
}

#[derive(Debug, Default, PartialEq)]
pub struct SensorData {
    pub mag: [i64; 3], // milligauss
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
    pub css_side_avg: [f64; 6],
    pub css_vertex_avg: [f64; 8],
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
        Ok(sensor_data)
    }
}

// update_interval (msec) - how often ADCS sensor values are sent by the MCU during an active cycle
// cycle_duration (sec) - how long the ADCS cycle lasts
// cycle_interval (sec) - how long the ADCS cycle is inactive before the next cycle starts
// adcs_format - a bitfield that specifies which sensor values are included in the ADCS cycle
// Returns a MCU command string that configures the ADCS cycle.
// E.g: a.fmt=7;a.upd=1000;a.dur=10;a.int=100
pub fn config_adcs_cycle(
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
            config_adcs_cycle(
                1000,
                10,
                100,
                AdcsFormat::IncludeMag | AdcsFormat::IncludeGyro | AdcsFormat::IncludeAccel
            ),
            "a.fmt=7;a.upd=1000;a.dur=10;a.int=100"
        );

        assert_eq!(
            config_adcs_cycle(
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
        };
        assert_eq!(sensor_data, expected_sensor_data);
    }
}
