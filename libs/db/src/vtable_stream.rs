use std::{
    mem::size_of,
    ops::Range,
    sync::{
        Arc,
        atomic::{self, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use impeller2::{
    types::{ComponentView, LenPacket, Msg, PACKET_HEADER_LEN, PacketId, PrimType, RequestId},
    vtable::{Op, RealizedOp, RealizedPair},
};
use impeller2_stellar::PacketSink;
use impeller2_wkt::{
    ComponentValue, FixedRateBehavior, FixedRateOp, InitialTimestamp, MeanOp, VTableStream,
};
use stellarator::{
    io::AsyncWrite,
    rent,
    sync::{Mutex, WaitCell, WaitQueue},
};
use tracing::{trace, warn};

use crate::{Component, DB, Error, FixedRateStreamState};

pub async fn handle_vtable_stream<A: AsyncWrite + 'static>(
    vtable_stream: VTableStream,
    db: Arc<DB>,
    tx: Arc<Mutex<PacketSink<A>>>,
    req_id: RequestId,
) -> Result<(), Error> {
    let VTableStream { vtable, id } = vtable_stream;
    trace!("spawning vtable stream");
    let table_len = vtable
        .fields
        .iter()
        .map(|f| f.offset.to_index() + f.len as usize)
        .chain(vtable.ops.iter().map(|op| match op {
            Op::Table { offset, len } => offset.to_index() + *len as usize,
            _ => 0,
        }))
        .max();
    let table_len = table_len.unwrap_or(0);
    let table = FieldTable::new(vtable.fields.len(), table_len, id);
    for (i, field) in vtable.fields.iter().enumerate() {
        let mut realized_op = vtable.realize(field.arg, None)?;
        let mut plan = vec![];
        let mut timestamp: Option<Range<usize>> = None;
        let mut schema: Option<(&[u64], PrimType)> = None;
        // loops until a pair is found, or the maximum number of iterations is reached
        'find: for _ in 0..u16::MAX {
            match realized_op {
                RealizedOp::Pair(RealizedPair {
                    entity_id,
                    component_id,
                }) => {
                    let component = db
                        .with_state(|s| s.get_component(entity_id, component_id).cloned())
                        .ok_or(Error::ComponentNotFound(component_id))?
                        .clone();
                    plan.insert(0, StreamStage::RealTime(RealTimeStage { component }));
                    break 'find;
                }
                RealizedOp::Schema(s) => {
                    schema = Some((s.dim, s.ty));
                    realized_op = vtable.realize(s.arg, None)?;
                }
                RealizedOp::Timestamp(t) => {
                    if let Some(range) = t.range {
                        timestamp = Some(range);
                    }
                    realized_op = vtable.realize(t.arg, None)?;
                }
                RealizedOp::Ext(ext) if ext.id == MeanOp::ID => {
                    let MeanOp { window } = postcard::from_bytes(ext.data)?;
                    trace!(?window, "pushed mean stage");
                    plan.push(StreamStage::Mean(MeanStage::new(window)));
                    realized_op = vtable.realize(ext.arg, None)?;
                }
                RealizedOp::Ext(ext) if ext.id == FixedRateBehavior::ID => {
                    let FixedRateOp {
                        stream_id,
                        behavior,
                    } = postcard::from_bytes(ext.data)?;
                    let RealizedOp::Pair(RealizedPair {
                        entity_id,
                        component_id,
                    }) = vtable.realize(ext.arg, None)?
                    else {
                        return Err(Error::Impeller(impeller2::error::Error::InvalidOp));
                    };
                    let component = db
                        .with_state(|s| s.get_component(entity_id, component_id).cloned())
                        .ok_or(Error::ComponentNotFound(component_id))?
                        .clone();

                    let state = FixedRateStreamState::from_state(
                        stream_id,
                        req_id,
                        Duration::from_nanos(behavior.timestep.unwrap_or_else(|| {
                            db.default_stream_time_step.load(atomic::Ordering::Relaxed)
                        })),
                        match behavior.initial_timestamp {
                            InitialTimestamp::Earliest => db.earliest_timestamp,
                            InitialTimestamp::Latest => db.last_updated.latest(),
                            InitialTimestamp::Manual(timestamp) => timestamp,
                        },
                        Default::default(),
                        behavior.frequency.unwrap_or(60),
                    );
                    plan.insert(
                        0,
                        StreamStage::FixedRate(FixedRateStage {
                            component,
                            state,
                            last_tick: Instant::now(),
                        }),
                    );
                    break 'find;
                }
                _ => return Err(Error::Impeller(impeller2::error::Error::InvalidOp)),
            }
        }
        let field_range = field.offset.to_index()..field.offset.to_index() + field.len as usize;
        let shard = table.field(field_range, i);
        let timestamp = timestamp.map(|r| table.field(r, i));
        let component = plan
            .first()
            .and_then(|s| s.as_component())
            .ok_or(Error::Impeller(impeller2::error::Error::InvalidOp))
            .inspect_err(|_| warn!("component not found"))?;
        if let Some((shape, ty)) = schema {
            let component_shape = component.schema.shape();
            if &component_shape[..] != shape || component.schema.prim_type != ty {
                warn!(
                    "invalid schema: expected {:?}, got {:?}",
                    shape, component_shape
                );
                return Err(Error::Impeller(impeller2::error::Error::InvalidOp));
            }
        }
        if component.schema.size() != field.len as usize {
            warn!(
                "invalid len: expected {:?}, got {:?}",
                component.schema.size(),
                field.len
            );
            return Err(Error::Impeller(impeller2::error::Error::InvalidOp));
        }
        let prim_type = component.schema.prim_type;
        stellarator::spawn(handle_plan(plan, shard, timestamp, prim_type));
    }
    loop {
        table.wait_ready().await;
        let mut pkt = table.take().await;
        let tx = tx.lock().await;
        rent!(tx.send(pkt.with_request_id(req_id)).await, pkt)?;
        table.replace_pkt(pkt).await;
        table.notify_writers();
    }
}

struct FieldTableInner {
    ready_wait_cell: WaitCell,
    writeable_wait_cell: WaitQueue,
    filled_fields: AtomicBitVec,
    table: Mutex<Option<LenPacket>>,
}

struct FieldTable {
    inner: Arc<FieldTableInner>,
}

impl FieldTable {
    pub fn new(total_field_count: usize, len: usize, id: PacketId) -> Self {
        let mut table = LenPacket::table(id, len);
        for _ in 0..len {
            table.push(0);
        }
        Self {
            inner: Arc::new(FieldTableInner {
                table: Mutex::new(Some(table)),
                ready_wait_cell: WaitCell::new(),
                writeable_wait_cell: WaitQueue::new(),
                filled_fields: AtomicBitVec::new(total_field_count),
            }),
        }
    }

    pub fn field(&self, range: Range<usize>, index: usize) -> Field {
        const HEADER_OFFSET: usize = PACKET_HEADER_LEN + size_of::<u32>();
        let range = range.start + HEADER_OFFSET..range.end + HEADER_OFFSET;
        Field {
            table: self.inner.clone(),
            index,
            range,
        }
    }

    pub async fn wait_ready(&self) {
        let _ = self
            .inner
            .ready_wait_cell
            .wait_for(|| self.inner.filled_fields.all_set())
            .await;
    }

    pub async fn take(&self) -> LenPacket {
        let mut table = self.inner.table.lock().await;
        table.take().expect("missing inner table")
    }

    pub async fn replace_pkt(&self, pkt: LenPacket) {
        let mut table = self.inner.table.lock().await;
        *table = Some(pkt);
    }

    pub fn notify_writers(&self) {
        self.inner.filled_fields.set_all(false);
        self.inner.writeable_wait_cell.wake_all();
    }
}

struct Field {
    table: Arc<FieldTableInner>,
    index: usize,
    range: Range<usize>,
}

impl Field {
    pub async fn wait_writeable(&self) {
        let _ = self
            .table
            .writeable_wait_cell
            .wait_for(|| !self.table.filled_fields.get(self.index).unwrap_or(true))
            .await;
    }

    pub fn set_written(&self) {
        self.table.filled_fields.set(self.index, true);
        self.table.ready_wait_cell.wake();
    }

    pub async fn with_buf(&self, mut f: impl FnMut(&mut [u8])) {
        let mut buf = self.table.table.lock().await;
        let buf = buf.as_mut().expect("missing inner table");
        f(&mut buf.inner[self.range.clone()])
    }
}

struct FixedRateStage {
    component: Component,
    state: FixedRateStreamState,
    last_tick: Instant,
}

impl FixedRateStage {
    pub async fn next(
        &mut self,
        shard: &Field,
        timestamp_shard: Option<&Field>,
    ) -> Result<bool, Error> {
        let sleep_time = self
            .state
            .sleep_time()
            .saturating_sub(self.last_tick.elapsed());
        futures_lite::future::race(
            async {
                self.state.playing_cell.wait_for_change().await;
            },
            stellarator::sleep(sleep_time),
        )
        .await;
        if self
            .state
            .playing_cell
            .wait_cell
            .wait_for(|| self.state.is_playing() || self.state.is_scrubbed())
            .await
            .is_err()
        {
            return Ok(true);
        }
        let current_timestamp = self.state.current_timestamp();
        let Some((timestamp, buf)) = self.component.time_series.get_nearest(current_timestamp)
        else {
            return Ok(true);
        };
        shard
            .with_buf(|shard| {
                shard.copy_from_slice(buf);
            })
            .await;
        if let Some(timestamp_shard) = timestamp_shard {
            timestamp_shard
                .with_buf(|shard| {
                    shard.copy_from_slice(&timestamp.to_le_bytes());
                })
                .await;
        }
        self.state.try_increment_tick(current_timestamp);
        self.last_tick = Instant::now();
        Ok(true)
    }
}

struct RealTimeStage {
    component: Component,
}

impl RealTimeStage {
    pub async fn next(
        &self,
        shard: &Field,
        timestamp_shard: Option<&Field>,
    ) -> Result<bool, Error> {
        trace!(
            component.id = ?self.component.component_id,
            "real time stage waiting"
        );
        self.component.time_series.wait().await;
        let Some((&timestamp, buf)) = self.component.time_series.latest() else {
            return Ok(true);
        };
        shard
            .with_buf(|shard| {
                shard.copy_from_slice(buf);
            })
            .await;
        if let Some(timestamp_shard) = timestamp_shard {
            timestamp_shard
                .with_buf(|shard| {
                    shard.copy_from_slice(&timestamp.to_le_bytes());
                })
                .await;
        }
        trace!("real time stage completed");
        Ok(true)
    }
}

pub struct MeanStage {
    window: u16,
    acc: Option<ComponentValue>,
    count: u16,
}

impl MeanStage {
    fn new(window: u16) -> Self {
        Self {
            window,
            acc: None,
            count: 0,
        }
    }

    async fn next(
        &mut self,
        field: &Field,
        _timestamp_shard: Option<&Field>, // TODO: maybe mean this as well?
        ty: PrimType,
    ) -> Result<bool, Error> {
        trace!("handling mean stage");
        field
            .with_buf(|f| {
                let len = f.len() / ty.size();
                let shape = [len];
                let view = ComponentView::try_from_bytes_shape(f, &shape, ty).unwrap();
                let acc = self
                    .acc
                    .get_or_insert_with(|| ComponentValue::zeros(&[len], ty));
                acc.add_view(view);
                self.count += 1;
                if self.count >= self.window {
                    acc.div(self.count as f64);
                    f.copy_from_slice(acc.as_bytes());
                    acc.fill_zeros();
                    self.count = 0;
                }
            })
            .await;
        Ok(self.count == 0)
    }
}

enum StreamStage {
    RealTime(RealTimeStage),
    FixedRate(FixedRateStage),
    Mean(MeanStage),
}

impl StreamStage {
    pub fn as_component(&self) -> Option<&Component> {
        // returns optional because future stages may not have a component
        match self {
            StreamStage::RealTime(real_time_stage) => Some(&real_time_stage.component),
            StreamStage::FixedRate(fixed_rate_stage) => Some(&fixed_rate_stage.component),
            StreamStage::Mean(_) => None,
        }
    }

    async fn next(
        &mut self,
        shard: &Field,
        timestamp_shard: Option<&Field>,
        prim_type: PrimType,
    ) -> Result<bool, Error> {
        match self {
            StreamStage::RealTime(real_time_stage) => {
                real_time_stage.next(shard, timestamp_shard).await
            }
            StreamStage::FixedRate(fixed_rate_stage) => {
                fixed_rate_stage.next(shard, timestamp_shard).await
            }
            StreamStage::Mean(mean_stage) => {
                mean_stage.next(shard, timestamp_shard, prim_type).await
            }
        }
    }
}

async fn handle_plan(
    mut plan: Vec<StreamStage>,
    shard: Field,
    timestamp_shard: Option<Field>,
    prim_type: PrimType,
) {
    trace!("spawning shard plan");
    loop {
        shard.wait_writeable().await;
        let mut finished = true;
        for stage in plan.iter_mut() {
            match stage
                .next(&shard, timestamp_shard.as_ref(), prim_type)
                .await
            {
                Ok(stage_finished) => {
                    finished &= stage_finished;
                }
                Err(err) => {
                    warn!(?err, "error in stream stage");
                    return;
                }
            }
        }
        if finished {
            shard.set_written();
        }
        trace!("shard plan complete")
    }
}

struct AtomicBitVec {
    store: Vec<AtomicUsize>,
    size: usize,
}

impl AtomicBitVec {
    pub fn new(size: usize) -> Self {
        let words = size.div_ceil(64);
        let store = (0..words).map(|_| AtomicUsize::new(0)).collect();
        AtomicBitVec { store, size }
    }

    pub fn set(&self, index: usize, value: bool) -> Option<()> {
        if index >= self.size {
            return None;
        }
        let bit = index % 64;
        let word = index / 64;
        let value = (value as usize) << bit;
        self.store.get(word)?.fetch_or(value, Ordering::SeqCst);
        Some(())
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.size {
            return None;
        }
        let bit = index % 64;
        let word = index / 64;
        let value = self.store.get(word)?.load(Ordering::SeqCst);
        Some(value & (1 << bit) != 0)
    }

    pub fn all_set(&self) -> bool {
        self.store
            .iter()
            .map(|word| word.load(Ordering::SeqCst).count_ones() as usize)
            .sum::<usize>()
            >= self.size
    }

    pub fn set_all(&self, value: bool) {
        let value = if value { usize::MAX } else { 0 };
        self.store.iter().for_each(|word| {
            word.store(value, Ordering::SeqCst);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_bitvec_set_get() {
        let bitvec = AtomicBitVec::new(100);
        assert!(!bitvec.get(0).unwrap());
        assert!(!bitvec.get(1).unwrap());
        bitvec.set(0, false).unwrap();
        assert!(!bitvec.get(0).unwrap());
        assert!(!bitvec.get(1).unwrap());
        bitvec.set(1, true).unwrap();
        assert!(!bitvec.get(0).unwrap());
        assert!(bitvec.get(1).unwrap());
        bitvec.set(50, true);
        assert!(bitvec.get(50).unwrap())
    }

    #[test]
    fn test_atomic_bitvec_set_all() {
        let bitvec = AtomicBitVec::new(100);
        bitvec.set_all(true);
        assert!(bitvec.all_set());
        bitvec.set_all(false);
        assert!(!bitvec.all_set());
    }
    #[test]
    fn test_atomic_bitvec_out_of_range() {
        let bitvec = AtomicBitVec::new(10);
        assert!(bitvec.set(100, true).is_none());
        assert!(bitvec.get(100).is_none());
        assert!(bitvec.get(200).is_none());
    }
}
