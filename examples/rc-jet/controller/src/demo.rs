//! Idle pilot: a slow roll left and right while nobody is on the sticks.
//!
//! Left alone the jet flies straight and the attitude instruments never move,
//! which makes the ADI look broken. This nudges the ailerons until the first
//! real pilot input, then hands the aircraft over for good — an attract mode,
//! not an autopilot.

use std::f64::consts::TAU;
use std::time::Instant;

/// One full right-then-left roll.
const PERIOD_S: f64 = 14.0;
/// Peak aileron, in radians (~0.85°).
///
/// Roll rate at cruise is roughly `(C_lda / -C_lp) * (2V/b) * delta_a`, about
/// [`ROLL_RATE_PER_RAD`] for this airframe, and roll damping settles in tens of
/// milliseconds, so the bank is essentially the integral of this. That puts the
/// peak near 30° of bank — a fingertip input against the 25° of full throw.
const PEAK_AILERON_RAD: f64 = 0.0148;

/// Flies [`roll_aileron`] until a human takes over.
pub struct IdlePilot {
    start: Instant,
    flying: bool,
}

impl Default for IdlePilot {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            flying: true,
        }
    }
}

impl IdlePilot {
    /// Aileron the idle pilot wants now, or `None` once `pilot_input` has been
    /// seen even once.
    pub fn aileron(&mut self, pilot_input: bool) -> Option<f64> {
        self.flying &= !pilot_input;
        self.flying
            .then(|| roll_aileron(self.start.elapsed().as_secs_f64()))
    }
}

/// Aileron `t` seconds in.
///
/// A cosine, not a sine: bank is the integral of roll rate, so this is what
/// makes the bank itself a sine — swinging evenly either side of level, wings
/// passing through flat twice a period, average heading held. Driving with a
/// sine would bank one way only and turn the jet in circles.
fn roll_aileron(t: f64) -> f64 {
    PEAK_AILERON_RAD * (TAU * t / PERIOD_S).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Roll rate per radian of aileron at cruise, from the airframe's
    /// `C_lda = 0.15`, `C_lp = -0.5`, `b = 2.65 m` at 70 m/s.
    const ROLL_RATE_PER_RAD: f64 = 15.85;

    /// Bank angle over one period, integrating `p = ROLL_RATE_PER_RAD * aileron`
    /// from level. Roll damping settles far faster than the 14 s drive, so the
    /// quasi-steady rate is a fair stand-in for the airframe.
    fn bank_samples() -> Vec<f64> {
        let steps = 14_000;
        let dt = PERIOD_S / f64::from(steps);
        let mut bank = 0.0;
        (0..steps)
            .map(|i| {
                bank += ROLL_RATE_PER_RAD * roll_aileron((f64::from(i) + 0.5) * dt) * dt;
                bank
            })
            .collect()
    }

    /// The point of the whole thing: the ADI shows a bank of a few tens of
    /// degrees, and the same amount each way.
    #[test]
    fn banks_about_thirty_degrees_either_side() {
        let bank: Vec<f64> = bank_samples();
        let max = bank.iter().copied().fold(f64::MIN, f64::max).to_degrees();
        let min = bank.iter().copied().fold(f64::MAX, f64::min).to_degrees();
        assert!((25.0..35.0).contains(&max), "peak bank {max}°");
        assert!((-35.0..=-25.0).contains(&min), "opposite bank {min}°");
        assert!((max + min).abs() < 1.0, "lopsided: {max}° vs {min}°");
    }

    /// Wings level at the end of a period, so the jet holds its heading on
    /// average instead of spiralling.
    #[test]
    fn a_period_leaves_the_wings_level() {
        let bank = bank_samples();
        assert!(
            bank.last().unwrap().abs() < 1e-3,
            "residual {:?}",
            bank.last()
        );
    }

    #[test]
    fn stays_a_fingertip_input() {
        let max = (0..2000)
            .map(|i| roll_aileron(f64::from(i) * 0.05).abs())
            .fold(0.0, f64::max);
        assert!(max <= PEAK_AILERON_RAD + 1e-12, "peak aileron {max}");
        // Never more than a tenth of the 25° full throw.
        assert!(PEAK_AILERON_RAD < 25f64.to_radians() / 10.0);
    }

    #[test]
    fn hands_over_permanently_on_first_pilot_input() {
        let mut pilot = IdlePilot::default();
        assert!(pilot.aileron(false).is_some());
        assert!(pilot.aileron(true).is_none());
        // Releasing the stick must not bring the demo back.
        assert!(pilot.aileron(false).is_none());
    }
}
