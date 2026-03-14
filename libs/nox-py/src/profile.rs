use std::{
    collections::HashMap,
    fmt,
    time::{Duration, Instant},
};

#[derive(Default, Clone, Copy, Debug)]
pub struct TickTimings {
    pub h2d_upload_ms: f64,
    pub call_setup_ms: f64,
    pub kernel_invoke_ms: f64,
    pub d2h_download_ms: f64,
}

#[derive(Default, Clone, Debug)]
pub struct Profiler {
    pub build: RollingMean,
    pub compile: RollingMean,
    pub copy_to_client: RollingMean,
    pub execute_buffers: RollingMean,
    pub copy_to_host: RollingMean,
    pub h2d_upload: RollingMean,
    pub call_setup: RollingMean,
    pub kernel_invoke: RollingMean,
    pub d2h_download: RollingMean,
    pub add_to_history: RollingMean,
}

impl Profiler {
    pub fn tick_mean(&self) -> f64 {
        self.copy_to_client.mean()
            + self.execute_buffers.mean()
            + self.copy_to_host.mean()
            + self.add_to_history.mean()
    }

    pub fn profile(&self, time_step: Duration) -> HashMap<&'static str, f64> {
        let tick_mean = self.tick_mean();
        let time_step = time_step.as_secs_f64() * 1000.0;
        let profile = [
            ("build", self.build.mean()),
            ("compile", self.compile.mean()),
            ("copy_to_client", self.copy_to_client.mean()),
            ("execute_buffers", self.execute_buffers.mean()),
            ("copy_to_host", self.copy_to_host.mean()),
            ("h2d_upload", self.h2d_upload.mean()),
            ("call_setup", self.call_setup.mean()),
            ("kernel_invoke", self.kernel_invoke.mean()),
            ("d2h_download", self.d2h_download.mean()),
            ("add_to_history", self.add_to_history.mean()),
            ("tick", tick_mean),
            ("time_step", time_step),
            ("real_time_factor", time_step / tick_mean),
        ];
        profile.into_iter().collect()
    }
}

#[derive(Default, Clone, Debug)]
pub struct RollingMean {
    sum: Duration,
    count: u32,
}

impl RollingMean {
    pub fn observe(&mut self, start: &mut Instant) {
        let sample = start.elapsed();
        self.sum += sample;
        self.count += 1;
        *start = Instant::now();
    }

    pub fn mean(&self) -> f64 {
        self.mean_duration().as_secs_f64() * 1000.0
    }

    pub fn observe_duration(&mut self, sample: Duration) {
        self.sum += sample;
        self.count += 1;
    }

    pub fn mean_duration(&self) -> Duration {
        self.sum / self.count.max(1)
    }
}

impl fmt::Display for RollingMean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.mean_duration())
    }
}
