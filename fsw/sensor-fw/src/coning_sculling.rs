/// Coning & sculling pre-integration filter.
///
/// Implements a 2-sample Bortz coning / sculling integrator based on
/// Savage's "Strapdown Inertial Navigation Integration Algorithm Design".
/// Accumulates N raw IMU samples and emits one corrected average per window.
///
/// All arithmetic is f32 -- sufficient for the small delta-quantities
/// accumulated over 2-4 sample windows (~92 bytes of state).
pub struct IntegratedSample {
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
}

pub struct ConingScullingIntegrator {
    accum_delta_angle: [f32; 3],
    accum_delta_vel: [f32; 3],
    prev_delta_angle: [f32; 3],
    coning_integral: [f32; 3],
    accum_dt: f32,
    sample_count: u32,
    decimation: u32,
}

impl ConingScullingIntegrator {
    pub fn new(decimation: u32) -> Self {
        Self {
            accum_delta_angle: [0.0; 3],
            accum_delta_vel: [0.0; 3],
            prev_delta_angle: [0.0; 3],
            coning_integral: [0.0; 3],
            accum_dt: 0.0,
            sample_count: 0,
            decimation,
        }
    }

    /// Feed one raw IMU sample. Returns corrected averages every N samples.
    /// Units are pass-through: if gyro is dps and accel is g, output matches.
    pub fn push(&mut self, gyro: [f32; 3], accel: [f32; 3], dt: f32) -> Option<IntegratedSample> {
        if dt <= 0.0 || dt > 0.1 {
            return None;
        }

        let delta_angle = scale(gyro, dt);
        let delta_vel = scale(accel, dt);

        // Coning correction (2-sample Bortz equation approximation)
        if self.sample_count > 0 {
            let coning_term = scale(cross(self.prev_delta_angle, delta_angle), 2.0 / 3.0);
            self.coning_integral = add(self.coning_integral, coning_term);
        }

        // Sculling correction (rotation-acceleration coupling)
        let sculling_term = scale(cross(self.accum_delta_angle, delta_vel), 0.5);
        let corrected_delta_vel = add(delta_vel, sculling_term);

        self.accum_delta_angle = add(self.accum_delta_angle, delta_angle);
        self.accum_delta_vel = add(self.accum_delta_vel, corrected_delta_vel);
        self.prev_delta_angle = delta_angle;
        self.accum_dt += dt;
        self.sample_count += 1;

        if self.sample_count >= self.decimation {
            let inv_dt = 1.0 / self.accum_dt;
            let corrected_angle = add(self.accum_delta_angle, self.coning_integral);
            let output = IntegratedSample {
                gyro: scale(corrected_angle, inv_dt),
                accel: scale(self.accum_delta_vel, inv_dt),
            };
            self.reset();
            Some(output)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.accum_delta_angle = [0.0; 3];
        self.accum_delta_vel = [0.0; 3];
        self.prev_delta_angle = [0.0; 3];
        self.coning_integral = [0.0; 3];
        self.accum_dt = 0.0;
        self.sample_count = 0;
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
