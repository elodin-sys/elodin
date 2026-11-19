//! Recording a witness: the simulation thread copies each tick's records into a block, and a
//! witness thread hashes it, checks it against a reference and appends it to the file.

use std::collections::VecDeque;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError};
use std::time::{Duration, Instant};

use impeller::types::{ComponentId, Timestamp};
use serde::Serialize;

use super::compare::{Mismatch, Side, first_difference};
use super::format::{Header, HookLabel, Labels, Reader, Tick, Writer};
use super::{Block, Channels, Layout, Raw, WINDOW};
use crate::World;
use crate::prepared::HookWrites;
use crate::tick_metrics::{PhaseStats, PhaseStatsSnapshot};

/// Ticks the witness thread can fall behind before the run fails.
const QUEUE: usize = 1024;
/// Blocks allocated before the run, enough to ride out writeback stalls of a few hundred ticks.
const POOL: usize = 256;

/// A witness ready to record, made before the run starts.
pub struct Plan {
    pub(super) file: File,
    pub(super) path: PathBuf,
    pub(super) layout: Layout,
    pub(super) labels: Vec<HookLabel>,
    pub(super) channels: Channels,
    pub(super) raw: Raw,
    pub(super) reference: Option<Reader>,
    pub(super) stop_after_first_mismatch: bool,
}

impl Plan {
    /// Starts the witness thread. `slot_of` gives the DB slot of a component's row, whose
    /// recording the commit reports.
    pub fn start(
        self,
        start: Timestamp,
        slot_of: impl Fn(ComponentId, usize) -> Option<usize>,
    ) -> Recorder {
        let labels = Labels {
            build: concat!(env!("CARGO_PKG_VERSION"), "+", env!("GIT_HASH")).to_string(),
            start_timestamp: start.0,
            hooks: self.labels,
        };
        let layout = Arc::new(self.layout);
        let header = Header::new(layout.schema.clone(), labels);
        let slots: Vec<_> = layout
            .commit_rows()
            .map(|(id, row)| slot_of(id, row))
            .collect();
        let (queue, queued) = std::sync::mpsc::sync_channel(QUEUE);
        let (spare, spares) = std::sync::mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let layout = layout.clone();
            let stop = stop.clone();
            let (file, raw, reference) = (self.file, self.raw, self.reference);
            let stop_at_first = self.stop_after_first_mismatch;
            std::thread::Builder::new()
                .name("elodin-witness".into())
                .spawn(move || {
                    let thread = WitnessThread {
                        layout,
                        header,
                        raw,
                        stop_at_first,
                        stop: stop.clone(),
                    };
                    let result = thread.run(file, reference, queued, spare);
                    if result.is_err() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    result
                })
                .expect("failed to spawn the witness thread")
        };
        let block_bytes = layout.block_bytes();
        let mut free = (0..POOL)
            .map(|_| {
                // Touches every page now, so no tick takes a page fault for it.
                let mut bytes = vec![1u8; block_bytes];
                bytes.clear();
                Block {
                    bytes,
                    ends: Vec::with_capacity(layout.schema.records.len()),
                    ..Block::default()
                }
            })
            .collect::<Vec<_>>();
        Recorder {
            mask: vec![false; slots.iter().flatten().max().map_or(0, |max| max + 1)],
            slots,
            layout,
            channels: self.channels,
            start: start.0,
            block: free.pop().unwrap_or_default(),
            free,
            queue,
            spares,
            thread,
            stop,
            in_flight: 0,
            queue_high_water: 0,
            capture: Duration::ZERO,
            stats: PhaseStats::default(),
            samples: Vec::new(),
            failure: None,
            path: self.path,
        }
    }
}

enum Message {
    Tick(Block),
    End(Option<String>),
}

/// The simulation thread's side of a witness.
pub struct Recorder {
    layout: Arc<Layout>,
    /// Passed to the hooks, which fill them.
    pub channels: Channels,
    start: i64,
    block: Block,
    free: Vec<Block>,
    queue: SyncSender<Message>,
    spares: Receiver<Block>,
    thread: std::thread::JoinHandle<Result<Written, String>>,
    /// Set by the witness thread when it fails, or finds the first mismatch of a run asked to
    /// stop there.
    stop: Arc<AtomicBool>,
    /// The DB slot of each row of the `after_commit` records.
    slots: Vec<Option<usize>>,
    /// Per DB slot, whether the cycle's commit recorded it.
    mask: Vec<bool>,
    in_flight: u64,
    queue_high_water: u64,
    /// Copying this cycle so far.
    capture: Duration,
    stats: PhaseStats,
    samples: Vec<u32>,
    failure: Option<String>,
    path: PathBuf,
}

impl Recorder {
    /// Starts the cycle that ends at `tick` and takes the systems' rows.
    pub fn after_systems(&mut self, tick: u64, timestamp: Timestamp, world: &World) {
        let started = Instant::now();
        self.layout.begin(&mut self.block, tick, timestamp.0);
        self.layout.capture_systems(&mut self.block, world);
        self.channels.clear_filled();
        self.capture += started.elapsed();
    }

    /// Takes what hook `hook` filled and wrote.
    pub fn after_hook(&mut self, hook: usize, world: &World, writes: &HookWrites) {
        let started = Instant::now();
        let (block, channels) = (&mut self.block, &self.channels);
        let deferred = &writes.deferred;
        self.layout
            .capture_hook(hook, block, world, channels, deferred, self.start);
        self.channels.clear_filled();
        self.capture += started.elapsed();
    }

    /// Takes which witnessed rows the commit recorded, by DB slot, and hands the cycle to the
    /// witness thread. An error ends the run; `finish` reports it.
    pub fn after_commit(&mut self, recorded: &[usize]) -> Result<(), String> {
        let started = Instant::now();
        for &slot in recorded {
            if let Some(seen) = self.mask.get_mut(slot) {
                *seen = true;
            }
        }
        let (mask, slots) = (&self.mask, &self.slots);
        let was_recorded = |row: usize| slots[row].is_some_and(|slot| mask[slot]);
        self.layout.capture_commit(&mut self.block, was_recorded);
        for &slot in recorded {
            if let Some(seen) = self.mask.get_mut(slot) {
                *seen = false;
            }
        }
        while let Ok(block) = self.spares.try_recv() {
            self.free.push(block);
            self.in_flight -= 1;
        }
        let tick = self.block.tick;
        let next = self.free.pop().unwrap_or_default();
        let block = std::mem::replace(&mut self.block, next);
        let sent = match self.queue.try_send(Message::Tick(block)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(format!(
                "the witness thread fell {QUEUE} ticks behind the simulation"
            )),
            Err(TrySendError::Disconnected(_)) => Err("the witness thread stopped".to_string()),
        };
        self.in_flight += 1;
        self.queue_high_water = self.queue_high_water.max(self.in_flight);
        self.capture += started.elapsed();
        self.stats.observe(self.capture);
        self.samples
            .push(self.capture.as_nanos().min(u32::MAX as u128) as u32);
        self.capture = Duration::ZERO;
        sent.inspect_err(|err| self.failure = Some(format!("{err} at tick {tick}")))
    }

    /// Whether the run should stop: the witness failed, or found the first mismatch of a run
    /// asked to stop there.
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Forgets the capture times so far, as the loop does when warmup ends.
    pub fn reset_report(&mut self) {
        self.stats = PhaseStats::default();
        self.samples.clear();
    }

    /// Writes the trailer, with `failure` when the run failed, and waits for the witness thread.
    pub fn finish(self, failure: Option<String>) -> WitnessReport {
        let _ = self.queue.send(Message::End(failure));
        drop(self.queue);
        let result = match self.thread.join() {
            Ok(result) => result,
            Err(_) => Err("the witness thread panicked".to_string()),
        };
        let mut report = WitnessReport {
            path: self.path.display().to_string(),
            root: None,
            ticks: 0,
            raw_ticks: 0,
            bytes_written: 0,
            capture: PhaseStatsSnapshot::new(&self.stats, Some(self.samples)),
            writer_ns: 0,
            max_writer_ns: 0,
            queue_high_water: self.queue_high_water,
            first_mismatch: None,
            failure: self.failure,
        };
        match result {
            Ok(written) => {
                report.root = Some(written.root);
                report.ticks = written.ticks;
                report.raw_ticks = written.raw_ticks;
                report.bytes_written = written.bytes;
                report.writer_ns = written.busy.as_nanos() as u64;
                report.max_writer_ns = written.max_busy.as_nanos() as u64;
                report.first_mismatch = written.first_mismatch;
            }
            Err(err) => report.failure = Some(err),
        }
        report
    }
}

/// The receipt's `witness` section.
#[derive(Debug, Clone, Serialize)]
pub struct WitnessReport {
    pub path: String,
    pub root: Option<String>,
    pub ticks: u64,
    /// Ticks written with their exact bytes.
    pub raw_ticks: u64,
    pub bytes_written: u64,
    /// Per measured cycle, the simulation thread's copying, inside the phases above.
    pub capture: PhaseStatsSnapshot,
    /// The witness thread's hashing, checking and writing, over the whole run.
    pub writer_ns: u64,
    pub max_writer_ns: u64,
    /// Most cycles handed to the witness thread and not yet written.
    pub queue_high_water: u64,
    /// The first difference from the reference, which is `left`.
    pub first_mismatch: Option<Mismatch>,
    pub failure: Option<String>,
}

struct Written {
    root: String,
    ticks: u64,
    raw_ticks: u64,
    bytes: u64,
    busy: Duration,
    max_busy: Duration,
    first_mismatch: Option<Mismatch>,
}

struct WitnessThread {
    layout: Arc<Layout>,
    header: Header,
    raw: Raw,
    stop_at_first: bool,
    stop: Arc<AtomicBool>,
}

impl WitnessThread {
    fn run(
        self,
        file: File,
        mut reference: Option<Reader>,
        queued: Receiver<Message>,
        spare: Sender<Block>,
    ) -> Result<Written, String> {
        let error = |err: std::io::Error| format!("cannot write the witness: {err}");
        let mut writer = Writer::new(file, &self.header).map_err(error)?;
        // Ticks wait here before being written, so a first mismatch can keep the ones before it.
        let hold = if reference.is_some() {
            WINDOW as usize
        } else {
            0
        };
        let mut pending: VecDeque<Block> = VecDeque::new();
        let mut kept = 0..0;
        let mut first_mismatch = None;
        let mut reference_tick = Tick::default();
        let (mut busy, mut max_busy) = (Duration::ZERO, Duration::ZERO);
        let mut failure = Some("the run ended without finishing the witness".to_string());
        for message in queued {
            let mut block = match message {
                Message::Tick(block) => block,
                Message::End(end) => {
                    failure = end;
                    break;
                }
            };
            let started = Instant::now();
            self.layout.digest(&mut block);
            if first_mismatch.is_none()
                && let Some(reference) = &mut reference
                && let Some(mismatch) = self.check(reference, &mut reference_tick, &block)?
            {
                kept = mismatch.tick.saturating_sub(WINDOW)..mismatch.tick + WINDOW + 1;
                first_mismatch = Some(mismatch);
                if self.stop_at_first {
                    self.stop.store(true, Ordering::Relaxed);
                }
            }
            pending.push_back(block);
            while pending.len() > hold {
                let block = pending.pop_front().expect("pending is not empty");
                let raw = self.raw.keeps(block.tick) || kept.contains(&block.tick);
                writer.tick(&block, raw).map_err(error)?;
                let _ = spare.send(block);
            }
            let took = started.elapsed();
            busy += took;
            max_busy = max_busy.max(took);
        }
        for block in pending {
            let raw = self.raw.keeps(block.tick) || kept.contains(&block.tick);
            writer.tick(&block, raw).map_err(error)?;
        }
        let trailer = writer.finish(failure).map_err(error)?;
        Ok(Written {
            root: trailer.root,
            ticks: writer.ticks,
            raw_ticks: writer.raw_ticks,
            bytes: writer.bytes,
            busy,
            max_busy,
            first_mismatch,
        })
    }

    /// The first difference between `block` and the reference's next tick.
    fn check(
        &self,
        reference: &mut Reader,
        tick: &mut Tick,
        block: &Block,
    ) -> Result<Option<Mismatch>, String> {
        if !reference.next(tick)? {
            let note = format!("the reference ends before tick {}", block.tick);
            return Ok(Some(Mismatch::at_tick(block.tick, note)));
        }
        let left = Side {
            block: &tick.block,
            raw: tick.raw,
            start: reference.header.labels.start_timestamp,
        };
        let right = Side {
            block,
            raw: true,
            start: self.header.labels.start_timestamp,
        };
        let schema = &self.layout.schema;
        Ok(first_difference(schema, &self.header.labels, left, right))
    }
}
