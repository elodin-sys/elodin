use std::{str::FromStr, sync::OnceLock};

const FINALS_2000A: &str = include_str!("../finals2000A.data");

pub struct IERS {
    min_mjd: usize,
    pub entries: Vec<IERSEntry>,
}

impl Default for IERS {
    fn default() -> Self {
        let mut entries = vec![];
        let mut min = f64::INFINITY;
        for line in FINALS_2000A.lines() {
            let Ok(entry) = IERSEntry::from_str(line) else {
                continue;
            };
            if entry.mjd < min {
                min = entry.mjd;
            }
            entries.push(entry);
        }
        IERS {
            entries,
            min_mjd: min as usize,
        }
    }
}

impl IERS {
    pub fn global() -> &'static IERS {
        static GLOBAL_IERS: OnceLock<IERS> = OnceLock::new();
        GLOBAL_IERS.get_or_init(IERS::default)
    }

    pub fn get_ceil(&self, mjd: f64) -> Option<&IERSEntry> {
        let idx = mjd.ceil() as usize - self.min_mjd;
        self.entries.get(idx)
    }

    pub fn get_floor(&self, mjd: f64) -> Option<&IERSEntry> {
        let idx = mjd.floor() as usize - self.min_mjd;
        self.entries.get(idx)
    }

    pub fn get_pm(&self, mjd: f64) -> Option<[f64; 2]> {
        let a = self.get_floor(mjd)?;
        let b = self.get_ceil(mjd)?;
        if a.mjd == b.mjd {
            return Some([a.pm.x, a.pm.y]);
        }
        let t = (mjd - a.mjd) / (b.mjd - a.mjd);
        let x = lerp(a.pm.x, b.pm.x, t);
        let y = lerp(a.pm.y, b.pm.y, t);
        Some([x, y])
    }

    pub fn get_ut1_utc(&self, mjd: f64) -> Option<f64> {
        let a = self.get_floor(mjd)?;
        let b = self.get_ceil(mjd)?;
        let t = (mjd - a.mjd) / (b.mjd - a.mjd);
        match (a.ut1_utc.as_ref(), b.ut1_utc.as_ref()) {
            (None, None) => None,
            (None, Some(ut1_utc)) | (Some(ut1_utc), None) => Some(ut1_utc.x),
            (Some(a), Some(b)) => Some(lerp(a.x, b.x, t)),
        }
    }

    pub fn get_nutation(&self, mjd: f64) -> Option<[f64; 2]> {
        let a = self.get_floor(mjd)?;
        let b = self.get_ceil(mjd)?;
        let t = (mjd - a.mjd) / (b.mjd - a.mjd);
        match (a.nutation.as_ref(), b.nutation.as_ref()) {
            (None, None) => None,
            (None, Some(nutation)) | (Some(nutation), None) => Some([nutation.x, nutation.y]),
            (Some(a), Some(b)) => Some([lerp(a.x, b.x, t), lerp(a.y, b.y, t)]),
        }
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[derive(Default, Debug)]
pub struct IERSEntry {
    pub mjd: f64,

    pub pm: IERSValue,
    pub ut1_utc: Option<IERSValue>,
    pub nutation: Option<IERSValue>,
}

impl FromStr for IERSEntry {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const ARC_SEC_2_RAD: f64 = std::f64::consts::PI / 180.0 / 3600.0;
        let mjd = f64::from_str(s.get(7..15).ok_or(ParseError::BufferUnderflow)?)?;
        let pm = {
            let prediction_flag =
                PredictionFlag::from_str(s.get(16..17).ok_or(ParseError::BufferUnderflow)?)?;
            let x =
                f64::from_str(s.get(19..27).ok_or(ParseError::BufferUnderflow)?)? * ARC_SEC_2_RAD;
            let x_error = f64::from_str(s.get(28..35).ok_or(ParseError::BufferUnderflow)?)
                .ok()
                .map(|err| err * ARC_SEC_2_RAD);
            let y =
                f64::from_str(s.get(38..45).ok_or(ParseError::BufferUnderflow)?)? * ARC_SEC_2_RAD;
            let y_error = f64::from_str(s.get(47..55).ok_or(ParseError::BufferUnderflow)?)
                .ok()
                .map(|err| err * ARC_SEC_2_RAD);
            IERSValue {
                prediction_flag,
                x,
                x_error,
                y,
                y_error,
            }
        };
        let ut1_utc =
            if let Ok(x) = f64::from_str(s.get(58..68).ok_or(ParseError::BufferUnderflow)?) {
                let y = f64::from_str(s.get(80..86).ok_or(ParseError::BufferUnderflow)?)
                    .unwrap_or_default();
                let prediction_flag =
                    PredictionFlag::from_str(s.get(57..58).ok_or(ParseError::BufferUnderflow)?)?;
                let x_error = f64::from_str(s.get(69..78).ok_or(ParseError::BufferUnderflow)?).ok();
                let y_error = f64::from_str(s.get(87..93).ok_or(ParseError::BufferUnderflow)?).ok();
                Some(IERSValue {
                    prediction_flag,
                    x,
                    x_error,
                    y,
                    y_error,
                })
            } else {
                None
            };
        let nutation = if let Ok(prediction_flag) =
            PredictionFlag::from_str(s.get(95..96).ok_or(ParseError::BufferUnderflow)?)
        {
            let x = f64::from_str(s.get(100..106).ok_or(ParseError::BufferUnderflow)?.trim())?
                * 1.0e-3
                * ARC_SEC_2_RAD;
            let x_error =
                f64::from_str(s.get(110..115).ok_or(ParseError::BufferUnderflow)?.trim()).ok();
            let y = f64::from_str(s.get(119..125).ok_or(ParseError::BufferUnderflow)?.trim())?
                * 1.0e-3
                * ARC_SEC_2_RAD;
            let y_error =
                f64::from_str(s.get(129..133).ok_or(ParseError::BufferUnderflow)?.trim()).ok();
            Some(IERSValue {
                prediction_flag,
                x,
                x_error,
                y,
                y_error,
            })
        } else {
            None
        };

        Ok(IERSEntry {
            mjd,
            pm,
            ut1_utc,
            nutation,
        })
    }
}

#[derive(Debug)]
pub enum ParseError {
    BufferUnderflow,
    InvalidPredictionFlag,
    Float(std::num::ParseFloatError),
}

impl From<std::num::ParseFloatError> for ParseError {
    fn from(e: std::num::ParseFloatError) -> Self {
        ParseError::Float(e)
    }
}

#[derive(Default, Debug)]
pub struct IERSValue {
    pub prediction_flag: PredictionFlag,
    pub x: f64,
    pub x_error: Option<f64>,
    pub y: f64,
    pub y_error: Option<f64>,
}

#[derive(Default, Debug)]
pub enum PredictionFlag {
    #[default]
    IERS,
    Predication,
}

impl FromStr for PredictionFlag {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "I" => Ok(PredictionFlag::IERS),
            "P" => Ok(PredictionFlag::Predication),
            _ => Err(ParseError::InvalidPredictionFlag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let data = IERS::default();
        let lower = data.get_floor(60481.5).unwrap();
        let upper = data.get_ceil(60481.5).unwrap();
        assert_eq!(lower.mjd, 60481.0);
        assert_eq!(upper.mjd, 60482.0);
    }
    #[test]
    fn test_get_pm() {
        let data = IERS::default();
        let [x, y] = data.get_pm(5.8488e+4).unwrap();
        approx::assert_relative_eq!(x, 0.0384e-5, epsilon = 1e-7);
        approx::assert_relative_eq!(y, 0.1321e-5, epsilon = 1e-7);
    }

    #[test]
    fn test_get_ut1_utc() {
        let data = IERS::default();
        let ut1_utc = data.get_ut1_utc(60481.5).unwrap();
        assert_eq!(ut1_utc, -0.01231255);
    }
}
