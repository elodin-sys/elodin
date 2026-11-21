//! Exact comparison of witnesses: digests first, then the bytes below the first that differs.

use std::path::Path;

use serde::Serialize;

use super::Block;
use super::format::{Kind, Labels, Reader, Schema, Tick};

/// Where two witnesses first differ. `left` and `right` are the differing byte of each, when both
/// kept that tick's exact bytes.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Mismatch {
    pub tick: u64,
    pub stage: Option<String>,
    pub hook: Option<String>,
    /// `field`, `channel`, `deferred` or `recorded`.
    pub kind: Option<&'static str>,
    pub record: Option<String>,
    pub entity: Option<String>,
    pub row: Option<u64>,
    pub byte: Option<u64>,
    pub left: Option<u8>,
    pub right: Option<u8>,
    pub note: Option<String>,
}

impl Mismatch {
    pub fn at_tick(tick: u64, note: String) -> Self {
        Self {
            tick,
            note: Some(note),
            ..Self::default()
        }
    }
}

/// One side of a tick comparison.
pub struct Side<'a> {
    pub block: &'a Block,
    /// Whether `block` has the records' bytes.
    pub raw: bool,
    pub start: i64,
}

/// The first difference between two ticks of witnesses with the same schema.
pub fn first_difference(
    schema: &Schema,
    labels: &Labels,
    left: Side<'_>,
    right: Side<'_>,
) -> Option<Mismatch> {
    let tick = right.block.tick;
    if left.block.tick != tick {
        let note = format!(
            "left has tick {} where right has tick {tick}",
            left.block.tick
        );
        return Some(Mismatch::at_tick(left.block.tick.min(tick), note));
    }
    let sim_time = |side: &Side<'_>| side.block.timestamp - side.start;
    if sim_time(&left) != sim_time(&right) {
        let note = format!(
            "simulation time is {} in left and {} in right",
            sim_time(&left),
            sim_time(&right)
        );
        return Some(Mismatch::at_tick(tick, note));
    }
    let mut digests = left.block.digests.iter().zip(&right.block.digests);
    let digest = digests.position(|(l, r)| l != r)?;
    let record = schema.hashed_records()[digest];
    let left_bytes = left.raw.then(|| left.block.record(record));
    let right_bytes = right.raw.then(|| right.block.record(record));
    Some(locate(
        schema,
        labels,
        tick,
        record,
        left_bytes,
        right_bytes,
    ))
}

/// Names record `record` of `tick`, and below it the first differing row and byte when both
/// sides have its bytes.
pub fn locate(
    schema: &Schema,
    labels: &Labels,
    tick: u64,
    record: usize,
    left: Option<&[u8]>,
    right: Option<&[u8]>,
) -> Mismatch {
    let schema_record = &schema.records[record];
    let stage = &schema.stages[schema_record.stage];
    let hook = stage
        .strip_prefix("after_hook:")
        .and_then(|n| labels.hooks.get(n.parse::<usize>().ok()?))
        .map(|hook| hook.name.clone());
    let mut mismatch = Mismatch {
        tick,
        stage: Some(stage.clone()),
        hook,
        kind: Some(match schema_record.kind {
            Kind::Field => "field",
            Kind::Channel | Kind::External => "channel",
            Kind::Deferred => "deferred",
            Kind::Recorded => "recorded",
        }),
        record: Some(schema_record.name.clone()),
        ..Mismatch::default()
    };
    let (Some(left), Some(right)) = (left, right) else {
        mismatch.note = Some("exact bytes are not in both witnesses at this tick".into());
        return mismatch;
    };
    if left.len() != right.len() {
        let filled = |bytes: &[u8]| match bytes.len() {
            0 if schema_record.kind == Kind::Channel => "not filled".to_string(),
            len => format!("{len} bytes"),
        };
        mismatch.note = Some(format!(
            "{} in left, {} in right",
            filled(left),
            filled(right)
        ));
        return mismatch;
    }
    let Some(offset) = left.iter().zip(right).position(|(l, r)| l != r) else {
        mismatch.note = Some("the digests differ but the bytes are equal".into());
        return mismatch;
    };
    mismatch.left = Some(left[offset]);
    mismatch.right = Some(right[offset]);
    let Some(batch) = schema_record.batch else {
        mismatch.byte = Some(offset as u64);
        return mismatch;
    };
    let row = offset / schema_record.row_bytes as usize;
    mismatch.row = Some(row as u64);
    mismatch.byte = Some((offset % schema_record.row_bytes as usize) as u64);
    // A channel row takes the entity of its batch's first field.
    let fields = &schema.batches[batch].fields;
    let field = match schema_record.kind {
        Kind::Field | Kind::Recorded => fields.get(schema_record.index),
        _ => fields.first(),
    };
    mismatch.entity = field
        .and_then(|field| field.entities.get(row))
        .map(|&id| schema.entity_name(id));
    mismatch
}

/// Why two schemas differ, in words.
pub fn schema_difference(left: &Schema, right: &Schema) -> String {
    if left.stages != right.stages {
        return format!("the stages are {:?} and {:?}", left.stages, right.stages);
    }
    let names = |schema: &Schema| {
        schema
            .batches
            .iter()
            .map(|b| b.name.clone())
            .collect::<Vec<_>>()
    };
    if names(left) != names(right) {
        return format!("the batches are {:?} and {:?}", names(left), names(right));
    }
    for (l, r) in left.batches.iter().zip(&right.batches) {
        let fields = |b: &super::format::BatchSchema| {
            b.fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>()
        };
        if fields(l) != fields(r) {
            return format!(
                "batch {:?} has fields {:?} and {:?}",
                l.name,
                fields(l),
                fields(r)
            );
        }
        for (lf, rf) in l.fields.iter().zip(&r.fields) {
            let name = format!("{}.{}", l.name, lf.name);
            if lf.entities.len() != rf.entities.len() {
                return format!(
                    "{name} has {} rows and {}",
                    lf.entities.len(),
                    rf.entities.len()
                );
            }
            if let Some(row) = lf
                .entities
                .iter()
                .zip(&rf.entities)
                .position(|(a, b)| a != b)
            {
                return format!(
                    "{name} has a different entity order: row {row} is {} and {}",
                    left.entity_name(lf.entities[row]),
                    right.entity_name(rf.entities[row])
                );
            }
            if (&lf.component, &lf.dtype, &lf.shape, &lf.access)
                != (&rf.component, &rf.dtype, &rf.shape, &rf.access)
            {
                return format!(
                    "{name} is {} {} {:?} ({}) and {} {} {:?} ({})",
                    lf.component,
                    lf.dtype,
                    lf.shape,
                    lf.access,
                    rf.component,
                    rf.dtype,
                    rf.shape,
                    rf.access
                );
            }
        }
    }
    if left.channels != right.channels {
        let channels = |schema: &Schema| {
            let channels = schema.channels.iter();
            channels
                .map(|c| format!("{} ({} bytes)", c.name, c.row_bytes))
                .collect::<Vec<_>>()
        };
        return format!(
            "the channels are {:?} and {:?}",
            channels(left),
            channels(right)
        );
    }
    if left.entities != right.entities {
        return "the entities have different names".into();
    }
    "the records differ".into()
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub path: String,
    pub build: String,
    pub ticks: u64,
    pub raw_ticks: u64,
    pub root: Option<String>,
    pub failure: Option<String>,
}

impl Summary {
    fn of(reader: &Reader) -> Self {
        Self {
            path: reader.path.display().to_string(),
            build: reader.header.labels.build.clone(),
            ticks: reader.ticks,
            raw_ticks: reader.raw_ticks,
            root: reader.trailer.as_ref().map(|t| t.root.clone()),
            failure: reader.trailer.as_ref().and_then(|t| t.failure.clone()),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub identical: bool,
    pub ticks: u64,
    /// Ticks that differ, up to the first one when the comparison stopped there.
    pub mismatched_ticks: u64,
    pub first_mismatch: Option<Mismatch>,
    pub left: Summary,
    pub right: Summary,
}

/// Compares two witness files tick by tick, stopping at the first mismatch when asked. Ticks
/// missing from one side are mismatches; different schemas are an error.
pub fn compare(left: &Path, right: &Path, stop_at_first: bool) -> Result<Comparison, String> {
    let mut readers = [Reader::open(left)?, Reader::open(right)?];
    let [l, r] = &readers;
    if l.header.schema != r.header.schema {
        let difference = schema_difference(&l.header.schema, &r.header.schema);
        return Err(format!(
            "the witnesses have different schemas: {difference}"
        ));
    }
    let schema = l.header.schema.clone();
    let labels = l.header.labels.clone();
    let starts = [
        l.header.labels.start_timestamp,
        r.header.labels.start_timestamp,
    ];
    let mut ticks = [Tick::default(), Tick::default()];
    let (mut compared, mut mismatched, mut first) = (0, 0, None);
    loop {
        let [lt, rt] = &mut ticks;
        let more = [readers[0].next(lt)?, readers[1].next(rt)?];
        let mismatch = match more {
            [false, false] => break,
            [true, false] | [false, true] => {
                let (tick, side) = if more[0] {
                    (lt.block.tick, "right")
                } else {
                    (rt.block.tick, "left")
                };
                let note = format!("tick {tick} and those after it are missing from {side}");
                first.get_or_insert(Mismatch::at_tick(tick, note));
                mismatched += 1;
                break;
            }
            [true, true] => {
                let left = Side {
                    block: &lt.block,
                    raw: lt.raw,
                    start: starts[0],
                };
                let right = Side {
                    block: &rt.block,
                    raw: rt.raw,
                    start: starts[1],
                };
                first_difference(&schema, &labels, left, right)
            }
        };
        compared += 1;
        if let Some(mismatch) = mismatch {
            let inventory = mismatch.stage.is_none() && lt.block.tick != rt.block.tick;
            first.get_or_insert(mismatch);
            mismatched += 1;
            if stop_at_first || inventory {
                break;
            }
        }
    }
    let [l, r] = &readers;
    Ok(Comparison {
        identical: first.is_none(),
        ticks: compared,
        mismatched_ticks: mismatched,
        first_mismatch: first,
        left: Summary::of(l),
        right: Summary::of(r),
    })
}
