//! The FSW's designed ascent trajectory: the recorded webcast profile
//! (WHITEPAPER 11.1). A real booster flies a mission-designed reference; ours
//! IS the recorded flight, which is exactly what "reproduce the recorded
//! mission" means. Loaded from the vendored stage-1 JSON at startup.

use serde::Deserialize;

#[derive(Deserialize)]
struct RawProfile {
    time: Vec<f64>,
    velocity: Vec<f64>,
    altitude: Vec<f64>,
}

pub struct AscentProfile {
    time: Vec<f64>,
    speed: Vec<f64>,
    alt_m: Vec<f64>,
    vspeed: Vec<f64>,
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let mut lo = 0;
    let mut hi = xs.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x { lo = mid } else { hi = mid }
    }
    let f = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + f * (ys[hi] - ys[lo])
}

impl AscentProfile {
    pub fn load(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let raw: RawProfile = serde_json::from_str(&text).ok()?;
        // Resample to a uniform 0.5 s grid with a light moving average
        // (mirrors reference.py's cleaning).
        let t_end = *raw.time.last()?;
        let n = (t_end / 0.5) as usize + 1;
        let time: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let speed_raw: Vec<f64> = time
            .iter()
            .map(|&t| interp(t, &raw.time, &raw.velocity))
            .collect();
        let alt_raw: Vec<f64> = time
            .iter()
            .map(|&t| interp(t, &raw.time, &raw.altitude) * 1000.0)
            .collect();
        let smooth = |v: &[f64]| -> Vec<f64> {
            (0..v.len())
                .map(|i| {
                    let lo = i.saturating_sub(4);
                    let hi = (i + 5).min(v.len());
                    v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
                })
                .collect()
        };
        let speed = smooth(&speed_raw);
        let alt_m = smooth(&alt_raw);
        let vspeed: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(1);
                let hi = (i + 1).min(n - 1);
                if hi == lo {
                    0.0
                } else {
                    (alt_m[hi] - alt_m[lo]) / ((hi - lo) as f64 * 0.5)
                }
            })
            .collect();
        Some(AscentProfile {
            time,
            speed,
            alt_m,
            vspeed,
        })
    }

    pub fn speed(&self, t: f64) -> f64 {
        interp(t, &self.time, &self.speed)
    }

    pub fn altitude(&self, t: f64) -> f64 {
        interp(t, &self.time, &self.alt_m)
    }

    pub fn vspeed(&self, t: f64) -> f64 {
        interp(t, &self.time, &self.vspeed)
    }
}
