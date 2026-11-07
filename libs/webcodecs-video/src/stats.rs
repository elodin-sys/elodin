#[derive(Clone, Debug)]
pub struct Snapshot {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub shown: Option<usize>,
    pub playhead_us: i64,
    pub playing: bool,
    pub renderer: String,
    pub rendered_fps: f64,
    pub decoded_fps: f64,
    pub decode_queue: u32,
    pub open_frames: usize,
    pub last_render_ms: f64,
    pub avg_render_ms: f64,
    pub seek_ms: Option<f64>,
    pub dropped: u64,
    pub status: String,
}

#[derive(Debug)]
pub struct Stats {
    pub rendered: u64,
    pub dropped: u64,
    pub last_render_ms: f64,
    render_ms_sum: f64,
    render_ms_n: u64,
    rendered_fps: f64,
    decoded_fps: f64,
    window_start_ms: f64,
    window_rendered: u64,
    window_decoded: u64,
    pub seek_started_ms: Option<f64>,
    pub last_seek_ms: Option<f64>,
    pub status: String,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            rendered: 0,
            dropped: 0,
            last_render_ms: 0.0,
            render_ms_sum: 0.0,
            render_ms_n: 0,
            rendered_fps: 0.0,
            decoded_fps: 0.0,
            window_start_ms: 0.0,
            window_rendered: 0,
            window_decoded: 0,
            seek_started_ms: None,
            last_seek_ms: None,
            status: "ready".to_string(),
        }
    }

    pub fn note_render(&mut self, ms: f64) {
        self.rendered += 1;
        self.last_render_ms = ms;
        self.render_ms_sum += ms;
        self.render_ms_n += 1;
    }

    pub fn begin_seek(&mut self, now: f64) {
        self.seek_started_ms = Some(now);
    }

    pub fn finish_seek(&mut self, now: f64) {
        if let Some(started) = self.seek_started_ms.take() {
            self.last_seek_ms = Some(now - started);
        }
    }

    pub fn observe(&mut self, now: f64, decoded_total: u64) {
        if self.window_start_ms == 0.0 {
            self.window_start_ms = now;
            self.window_rendered = self.rendered;
            self.window_decoded = decoded_total;
            return;
        }
        let elapsed = now - self.window_start_ms;
        if elapsed < 500.0 {
            return;
        }
        let rendered = self.rendered - self.window_rendered;
        let decoded = decoded_total - self.window_decoded;
        self.rendered_fps = rendered as f64 * 1000.0 / elapsed;
        self.decoded_fps = decoded as f64 * 1000.0 / elapsed;
        self.window_start_ms = now;
        self.window_rendered = self.rendered;
        self.window_decoded = decoded_total;
    }

    pub fn snapshot(
        &self,
        codec: &str,
        width: u32,
        height: u32,
        frame_count: usize,
        shown: Option<usize>,
        playhead_us: i64,
        playing: bool,
        renderer: &str,
        decode_queue: u32,
        open_frames: usize,
        dropped: u64,
    ) -> Snapshot {
        Snapshot {
            codec: codec.to_string(),
            width,
            height,
            frame_count,
            shown,
            playhead_us,
            playing,
            renderer: renderer.to_string(),
            rendered_fps: self.rendered_fps,
            decoded_fps: self.decoded_fps,
            decode_queue,
            open_frames,
            last_render_ms: self.last_render_ms,
            avg_render_ms: if self.render_ms_n == 0 {
                0.0
            } else {
                self.render_ms_sum / self.render_ms_n as f64
            },
            seek_ms: self.last_seek_ms,
            dropped: self.dropped + dropped,
            status: self.status.clone(),
        }
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            codec: String::new(),
            width: 0,
            height: 0,
            frame_count: 0,
            shown: None,
            playhead_us: 0,
            playing: false,
            renderer: String::new(),
            rendered_fps: 0.0,
            decoded_fps: 0.0,
            decode_queue: 0,
            open_frames: 0,
            last_render_ms: 0.0,
            avg_render_ms: 0.0,
            seek_ms: None,
            dropped: 0,
            status: "loading".to_string(),
        }
    }
}

impl Snapshot {
    pub fn text(&self) -> String {
        let frame = self
            .shown
            .map(|index| index.to_string())
            .unwrap_or_else(|| "-".to_string());
        let last = self.frame_count.saturating_sub(1);
        let seek = self
            .seek_ms
            .map(|ms| format!("{ms:.1} ms"))
            .unwrap_or_else(|| "-".to_string());
        let state = if self.playing { "playing" } else { "paused" };
        format!(
            "{width}x{height}  {codec}  {count} frames\n\
             playhead {secs:.2}s  frame {frame} / {last}\n\
             renderer {renderer}\n\
             rendered fps {rendered:.1}\n\
             decoder fps {decoded:.1}\n\
             decodeQueueSize {queue}\n\
             open VideoFrames {open}\n\
             render {last_ms:.2} ms (avg {avg:.2})\n\
             seek latency {seek}\n\
             dropped {dropped}\n\
             status {state}  {status}",
            width = self.width,
            height = self.height,
            codec = self.codec,
            count = self.frame_count,
            secs = self.playhead_us as f64 / 1_000_000.0,
            frame = frame,
            last = last,
            renderer = self.renderer,
            rendered = self.rendered_fps,
            decoded = self.decoded_fps,
            queue = self.decode_queue,
            open = self.open_frames,
            last_ms = self.last_render_ms,
            avg = self.avg_render_ms,
            seek = seek,
            dropped = self.dropped,
            state = state,
            status = self.status,
        )
    }
}
