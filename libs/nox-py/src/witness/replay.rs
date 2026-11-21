//! Replaying `pre_commit` hooks from a witness: each tick with exact bytes loads the recorded
//! rows into the world, runs the hooks, and compares what they write with the recording.

use impeller::types::Timestamp;
use serde::Serialize;

use super::compare::{Mismatch, locate};
use super::format::{Reader, Tick};
use super::{Block, Channels, Layout};
use crate::World;
use crate::prepared::PreCommit;

#[derive(Debug, Serialize)]
pub struct ReplayReport {
    pub trace: String,
    /// Ticks replayed: those the trace kept exact bytes of.
    pub ticks: u64,
    /// Ticks the trace kept only digests of.
    pub skipped_ticks: u64,
    pub hooks: Vec<String>,
    /// Records compared, and their bytes.
    pub records: u64,
    pub bytes: u64,
    /// The first difference from the recording, which is `left`.
    pub first_mismatch: Option<Mismatch>,
}

/// Replays `trace` through `hooks`, one entry per recorded hook. Hook `n` sees the recorded rows
/// it saw in the run, and the external channels the run recorded unless `channels` is live. Hooks
/// whose stage the witness left out are not run. The replay stops at the first mismatch.
pub fn replay(
    trace: &mut Reader,
    layout: &Layout,
    world: &mut World,
    mut hooks: Vec<Option<Box<dyn PreCommit + Send>>>,
    mut channels: Channels,
) -> Result<ReplayReport, String> {
    if layout.after_systems.is_none() {
        return Err(format!(
            "{} has no after_systems stage, which a replay starts from",
            trace.path.display()
        ));
    }
    let labels = trace.header.labels.clone();
    let mut report = ReplayReport {
        trace: trace.path.display().to_string(),
        ticks: 0,
        skipped_ticks: 0,
        hooks: hooks
            .iter()
            .flatten()
            .map(|hook| hook.name().to_string())
            .collect(),
        records: 0,
        bytes: 0,
        first_mismatch: None,
    };
    let mut tick = Tick::default();
    let mut replayed = Block::default();
    while trace.next(&mut tick)? {
        if !tick.raw {
            report.skipped_ticks += 1;
            continue;
        }
        let recorded = &tick.block;
        layout.load_systems(world, recorded);
        for (n, hook) in hooks.iter_mut().enumerate() {
            let (Some(hook), Some(stage)) = (hook, layout.after_hook[n]) else {
                continue;
            };
            channels.clear_filled();
            if channels.replay {
                layout.load_external(stage, &mut channels, recorded);
            }
            let timestamp = Timestamp(recorded.timestamp);
            let writes = hook
                .step(recorded.tick, timestamp, world, &mut channels)
                .map_err(|cause| {
                    let name = hook.name();
                    format!(
                        "pre_commit hook {n} ({name}) failed at tick {}: {cause}",
                        recorded.tick
                    )
                })?;
            replayed.clear();
            let start = labels.start_timestamp;
            layout.capture_hook(n, &mut replayed, world, &channels, &writes.deferred, start);
            let records = layout.records(stage).filter(|(_, record)| record.hashed());
            for (index, _) in records {
                let local = index - layout.stages[stage].start;
                let (left, right) = (recorded.record(index), replayed.record(local));
                report.records += 1;
                report.bytes += right.len() as u64;
                if left != right {
                    let schema = &layout.schema;
                    let mismatch = locate(
                        schema,
                        &labels,
                        recorded.tick,
                        index,
                        Some(left),
                        Some(right),
                    );
                    report.first_mismatch = Some(mismatch);
                    report.ticks += 1;
                    return Ok(report);
                }
            }
            layout.load_writes(stage, world, recorded);
        }
        report.ticks += 1;
    }
    if report.ticks == 0 {
        return Err(format!(
            "{} kept no exact bytes to replay; record it with raw=True, a raw window, or a reference",
            trace.path.display()
        ));
    }
    Ok(report)
}
