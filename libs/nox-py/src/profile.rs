use std::{
    collections::{HashMap, VecDeque},
    fmt,
    time::{Duration, Instant},
};

const MEDIAN_WINDOW: usize = 1024;

#[derive(Default, Clone, Copy, Debug)]
pub struct TickTimings {
    pub h2d_upload_ms: f64,
    pub kernel_invoke_ms: f64,
    pub d2h_download_ms: f64,
}

/// Wall times of one `run()` batch; `total` covers the whole batch.
pub struct BatchTimings {
    pub total: Duration,
    pub exec: Duration,
    pub history: Duration,
    pub detail: Option<TickTimings>,
}

#[derive(Default, Clone, Debug)]
pub struct Profiler {
    pub detailed_timing: bool,
    pub build: RollingMean,
    /// First batch (lazy JIT, first allocations); excluded from every other statistic.
    pub warmup: Option<Duration>,
    pub tick: RollingMean,
    tick_recent: VecDeque<Duration>,
    pub copy_to_client: RollingMean,
    pub execute_buffers: RollingMean,
    pub copy_to_host: RollingMean,
    pub h2d_upload: RollingMean,
    pub kernel_invoke: RollingMean,
    pub d2h_download: RollingMean,
    pub add_to_history: RollingMean,
}

impl Profiler {
    pub fn record_batch(&mut self, batch: BatchTimings) {
        if self.warmup.is_none() {
            self.warmup = Some(batch.total);
            return;
        }
        self.tick.observe_duration(batch.total);
        if self.tick_recent.len() == MEDIAN_WINDOW {
            self.tick_recent.pop_front();
        }
        self.tick_recent.push_back(batch.total);
        if let Some(t) = batch.detail {
            let from_ms = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
            let h2d = from_ms(t.h2d_upload_ms);
            let kernel = from_ms(t.kernel_invoke_ms);
            let d2h = from_ms(t.d2h_download_ms);
            self.copy_to_client.observe_duration(h2d);
            self.execute_buffers.observe_duration(kernel);
            self.copy_to_host.observe_duration(d2h);
            self.h2d_upload.observe_duration(h2d);
            self.kernel_invoke.observe_duration(kernel);
            self.d2h_download.observe_duration(d2h);
        } else {
            self.execute_buffers.observe_duration(batch.exec);
        }
        self.add_to_history.observe_duration(batch.history);
    }

    pub fn profile(
        &self,
        time_step: Duration,
        ticks_per_telemetry: u64,
    ) -> HashMap<&'static str, f64> {
        let warmup = self.warmup.unwrap_or_default();
        // A single-batch run has no steady state; report that batch rather than nothing.
        let (tick, tick_median) = if self.tick_recent.is_empty() {
            (warmup, warmup)
        } else {
            let mut sorted = Vec::from(self.tick_recent.clone());
            sorted.sort_unstable();
            (self.tick.mean_duration(), sorted[sorted.len() / 2])
        };
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let sim_time_step_ms = time_step.as_secs_f64() * 1000.0;
        let batch_sim_time_ms = sim_time_step_ms * ticks_per_telemetry.max(1) as f64;
        let profile = [
            ("build", self.build.mean()),
            ("warmup", ms(warmup)),
            ("copy_to_client", self.copy_to_client.mean()),
            ("execute_buffers", self.execute_buffers.mean()),
            ("copy_to_host", self.copy_to_host.mean()),
            ("h2d_upload", self.h2d_upload.mean()),
            ("kernel_invoke", self.kernel_invoke.mean()),
            ("d2h_download", self.d2h_download.mean()),
            ("add_to_history", self.add_to_history.mean()),
            ("tick", ms(tick)),
            ("tick_median", ms(tick_median)),
            ("time_step", sim_time_step_ms),
            ("ticks_per_telemetry", ticks_per_telemetry as f64),
            ("real_time_factor", batch_sim_time_ms / ms(tick)),
            (
                "real_time_factor_median",
                batch_sim_time_ms / ms(tick_median),
            ),
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
