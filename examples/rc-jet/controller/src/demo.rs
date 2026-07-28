//! Idle pilot: gentle periodic banks while nobody is on the sticks.
//!
//! Left alone the jet flies straight and the attitude instruments never move,
//! which makes the ADI look broken. This nudges the ailerons until the first
//! real pilot input, then hands the aircraft over for good — an attract mode,
//! not an autopilot.

use std::f64::consts::TAU;
use std::time::Instant;

/// Wings-level lead-in, so the jet settles into trimmed flight before the
/// first bank.
const LEAD_IN_S: f64 = 5.0;
/// Length of one bank-and-recover burst.
const BURST_S: f64 = 6.0;
/// Wings-level pause between bursts.
const PAUSE_S: f64 = 8.0;
/// Peak aileron during a burst, in radians (~0.9°).
///
/// Roll rate at cruise is roughly `(C_lda / -C_lp) * (2V/b) * delta_a`, about
/// 16 rad/s per radian of aileron, and roll damping settles in tens of
/// milliseconds — so half a burst integrates to a bank of a few tens of
/// degrees. Full throw is 25°: this is a fingertip input.
const PEAK_AILERON_RAD: f64 = 0.017;

/// Flies [`bank_burst`] until a human takes over.
pub struct IdlePilot {
    start: Instant,
    flying: bool,
}

impl IdlePilot {
    pub fn new(enabled: bool) -> Self {
        Self {
            start: Instant::now(),
            flying: enabled,
        }
    }

    /// Aileron the idle pilot wants now, or `None` once `pilot_input` has been
    /// seen even once.
    pub fn aileron(&mut self, pilot_input: bool) -> Option<f64> {
        self.flying &= !pilot_input;
        self.flying
            .then(|| bank_burst(self.start.elapsed().as_secs_f64()))
    }
}

/// Aileron of the burst sequence `t` seconds after start: bursts alternating
/// right and left, each a whole sine period so the wings come back level on
/// their own and the jet holds its average heading.
fn bank_burst(t: f64) -> f64 {
    let t = t - LEAD_IN_S;
    if t < 0.0 {
        return 0.0;
    }
    let cycle = BURST_S + PAUSE_S;
    let phase = t % cycle;
    if phase >= BURST_S {
        return 0.0;
    }
    let sign = if ((t / cycle) as u64).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    sign * PEAK_AILERON_RAD * (TAU * phase / BURST_S).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CYCLE_S: f64 = BURST_S + PAUSE_S;

    #[test]
    fn stays_level_during_lead_in_and_pauses() {
        for t in [0.0, 1.0, LEAD_IN_S - 0.01] {
            assert_eq!(bank_burst(t), 0.0, "lead-in at {t}");
        }
        for t in [BURST_S, BURST_S + 1.0, CYCLE_S - 0.01] {
            assert_eq!(bank_burst(LEAD_IN_S + t), 0.0, "pause at {t}");
        }
    }

    #[test]
    fn each_burst_starts_and_ends_neutral() {
        for burst in 0..4 {
            let base = LEAD_IN_S + f64::from(burst) * CYCLE_S;
            assert!(bank_burst(base).abs() < 1e-12);
            assert!(bank_burst(base + BURST_S - 1e-9).abs() < 1e-6);
        }
    }

    #[test]
    fn bursts_alternate_direction_and_stay_gentle() {
        let peak = |burst: u32| bank_burst(LEAD_IN_S + f64::from(burst) * CYCLE_S + BURST_S / 4.0);
        assert!((peak(0) - PEAK_AILERON_RAD).abs() < 1e-12);
        assert!((peak(1) + PEAK_AILERON_RAD).abs() < 1e-12);
        assert!((peak(2) - PEAK_AILERON_RAD).abs() < 1e-12);
        let max = (0..2000)
            .map(|i| bank_burst(f64::from(i) * 0.05).abs())
            .fold(0.0, f64::max);
        assert!(max <= PEAK_AILERON_RAD + 1e-12, "peak {max}");
        // Never more than a tenth of the 25° full throw.
        assert!(PEAK_AILERON_RAD < 25f64.to_radians() / 10.0);
    }

    /// A burst is a full sine period, so the net roll impulse (and hence the
    /// bank it leaves behind) cancels.
    #[test]
    fn a_burst_integrates_to_zero() {
        let steps = 60_000;
        let dt = BURST_S / f64::from(steps);
        let area: f64 = (0..steps)
            .map(|i| bank_burst(LEAD_IN_S + (f64::from(i) + 0.5) * dt) * dt)
            .sum();
        assert!(area.abs() < 1e-6, "net aileron impulse {area}");
    }

    #[test]
    fn hands_over_permanently_on_first_pilot_input() {
        let mut pilot = IdlePilot::new(true);
        assert!(pilot.aileron(false).is_some());
        assert!(pilot.aileron(true).is_none());
        // Releasing the stick must not bring the demo back.
        assert!(pilot.aileron(false).is_none());
    }

    #[test]
    fn disabled_pilot_never_flies() {
        assert!(IdlePilot::new(false).aileron(false).is_none());
    }
}
