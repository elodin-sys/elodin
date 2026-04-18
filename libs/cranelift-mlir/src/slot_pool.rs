//! Per-function stack-slot pool for the ptr-ABI lowering path.
//!
//! Cranelift's `create_sized_stack_slot(StackSlotKind::ExplicitSlot,
//! ...)` allocates a fresh frame offset on every call; there's no
//! downstream coalescing pass (see [wasmtime
//! #6661](https://github.com/bytecodealliance/wasmtime/issues/6661)).
//! For a ptr-ABI body with N F64 elementwise ops that all write into
//! their own 24-byte slot, the frame grows linearly with N and each
//! `stack_addr` targets a distinct cache line.
//!
//! The pool reuses slots whose owner value has reached its
//! `last_use_pos` (see [`crate::useinfo::UseInfo::last_use_pos`]).
//! A per-(bytes, align) free-list grants a dead slot to the next
//! allocation of the same shape, keeping the physical frame offset
//! hot in cache across "virtual" slots.
//!
//! The pool is scoped to one `lower_body_mem` invocation; when that
//! returns, the pool is dropped and all `StackSlot` handles remain
//! live in the `FunctionBuilder`'s frame layout exactly as if we had
//! never pooled (the reuse is purely about *fewer* slots, not freeing
//! frame memory mid-function).
//!
//! Reference pattern: the GC-safepoint free-list in
//! [`cranelift-frontend`](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/frontend/src/frontend/safepoints.rs)
//! (`free_stack_slots: SlotSizeMap<SmallVec<[StackSlot; 4]>>`) —
//! a narrower version of the same idea, keyed on size only and
//! applied only to GC-tracked values.

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::{InstBuilder, StackSlot, StackSlotData, StackSlotKind, Value};
use cranelift_frontend::FunctionBuilder;

use crate::ir::ValueId;

/// Per-function stack-slot pool. Lives for one `lower_body_mem` call.
///
/// Supports two kinds of ownership:
///
/// 1. **Primary ownership** (`record_owner`): a vid whose match arm
///    called `alloc` is the slot's primary holder.
/// 2. **Shared ownership** (`share_owner`): aliasing ops like
///    `Reshape` and `BitcastConvert` forward an operand's ptr to a
///    new result vid. Both the operand and the result are added to
///    the slot's holder set. The slot returns to the free-list only
///    when EVERY holder has been released (i.e., hit its
///    `last_use_pos`).
///
/// Data layout:
///
/// - `free: HashMap<(bytes, align), Vec<StackSlot>>`
///   FIFO per-shape free-list; `Vec::pop()` returns the most-recently-
///   released slot first so the hottest one wins reuse and stays in
///   cache.
/// - `vid_to_slot: HashMap<ValueId, StackSlot>`
///   Which slot does each vid hold (0 or 1 per vid).
/// - `slot_holders: HashMap<StackSlot, (bytes, align, HashSet<ValueId>)>`
///   Inverse map: who currently holds this slot. Needed so
///   `release_for_vid` can tell whether releasing one vid finally
///   drops the last holder.
///
/// Aliasing correctness:
/// a Reshape at position P reads operand vid V (owning slot S) and
/// produces result vid W forwarding ptr(S). If `use_info.last_use_pos[V]
/// == P` we'd release S too early — W's future uses would see a slot
/// that's been handed to a later allocator. The fix is to call
/// `share_owner(V, W)` in the aliasing match arm: now S has
/// `{V, W}` as holders. Releasing V at end-of-P leaves `{W}` — the
/// slot stays live until W is also released.
#[derive(Default)]
pub(crate) struct SlotPool {
    free: HashMap<(u32, u8), Vec<StackSlot>>,
    vid_to_slot: HashMap<ValueId, StackSlot>,
    slot_holders: HashMap<StackSlot, (u32, u8, HashSet<ValueId>)>,
    /// Total number of `alloc` calls that reused a pooled slot.
    pub hits: u64,
    /// Total number of `alloc` calls that fell back to
    /// `create_sized_stack_slot`. `hits + misses` = total pooled-path
    /// allocations; the ratio is a useful "is the pool working?"
    /// signal in the stderr InstrReport.
    pub misses: u64,
}

impl SlotPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve a slot of `bytes` × `2^align_shift` alignment. Prefers
    /// a pooled slot if one with matching size+align is available;
    /// otherwise creates a fresh `ExplicitSlot`. Returns both the
    /// `StackSlot` handle (for ownership tracking) and a `stack_addr`
    /// `Value` pointing at offset 0 (ready to use as a tensor base).
    pub fn alloc(
        &mut self,
        builder: &mut FunctionBuilder,
        bytes: u32,
        align_shift: u8,
    ) -> (StackSlot, Value) {
        let key = (bytes, align_shift);
        let ss = if let Some(list) = self.free.get_mut(&key)
            && let Some(ss) = list.pop()
        {
            self.hits = self.hits.saturating_add(1);
            ss
        } else {
            self.misses = self.misses.saturating_add(1);
            builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                bytes,
                align_shift,
            ))
        };
        let ptr = builder.ins().stack_addr(crate::lower::ptr_type(), ss, 0);
        (ss, ptr)
    }

    /// Associate a just-allocated slot with the vid that owns it.
    /// Called immediately after `alloc` at the match-arm site so
    /// `release_for_vid` can later return the slot to the free-list
    /// when the vid's `last_use_pos` is reached.
    pub fn record_owner(&mut self, vid: ValueId, ss: StackSlot, bytes: u32, align_shift: u8) {
        self.vid_to_slot.insert(vid, ss);
        let entry = self
            .slot_holders
            .entry(ss)
            .or_insert_with(|| (bytes, align_shift, HashSet::new()));
        entry.2.insert(vid);
    }

    /// Mark `alias_vid` as an additional holder of whatever slot
    /// `source_vid` currently holds. Call this from aliasing ops
    /// (`Reshape`, `BitcastConvert`, any arm that returns an operand's
    /// ptr verbatim) so the slot stays live until BOTH vids are
    /// released. No-op if `source_vid` is not a slot holder (e.g.
    /// the source is a function parameter whose ptr came from a
    /// block parameter, not from `alloc`).
    pub fn share_owner(&mut self, source_vid: ValueId, alias_vid: ValueId) {
        let Some(&ss) = self.vid_to_slot.get(&source_vid) else {
            return;
        };
        self.vid_to_slot.insert(alias_vid, ss);
        if let Some(entry) = self.slot_holders.get_mut(&ss) {
            entry.2.insert(alias_vid);
        }
    }

    /// Drop `vid` from its slot's holder set. If the set becomes
    /// empty, push the `StackSlot` onto the matching `(bytes, align)`
    /// free-list so a later `alloc` can reuse it. Safe to call on a
    /// vid that doesn't hold anything — useful from a generic
    /// per-instruction cleanup loop that iterates all operand vids.
    pub fn release_for_vid(&mut self, vid: ValueId) {
        let Some(ss) = self.vid_to_slot.remove(&vid) else {
            return;
        };
        let now_empty = match self.slot_holders.get_mut(&ss) {
            Some((_, _, holders)) => {
                holders.remove(&vid);
                holders.is_empty()
            }
            None => false,
        };
        if now_empty && let Some((bytes, align_shift, _)) = self.slot_holders.remove(&ss) {
            self.free.entry((bytes, align_shift)).or_default().push(ss);
        }
    }

    /// Stats readers for the InstrReport output.
    pub fn total_allocs(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }

    pub fn hit_rate_pct(&self) -> f64 {
        let total = self.total_allocs();
        if total == 0 {
            return 0.0;
        }
        100.0 * (self.hits as f64) / (total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Behavior-only unit tests: the numeric pool logic without a
    /// live `FunctionBuilder` (we don't need real StackSlots to
    /// validate that `release_for_vid` feeds the free-list correctly).
    ///
    /// We pun a StackSlot via `::default()` because the production
    /// API goes through `create_sized_stack_slot`; tests only need
    /// to check the pool's internal bookkeeping.

    fn synth_ss(n: u32) -> StackSlot {
        // StackSlot is a Cranelift entity (newtype around u32).
        // `from_u32` constructs a synthetic handle — sufficient for
        // pool-bookkeeping tests that don't emit real IR.
        StackSlot::from_u32(n)
    }

    #[test]
    fn record_then_release_moves_to_free_list() {
        let mut pool = SlotPool::new();
        let ss = synth_ss(0);
        pool.record_owner(ValueId(42), ss, 24, 4);
        assert_eq!(pool.vid_to_slot.len(), 1);
        pool.release_for_vid(ValueId(42));
        assert_eq!(pool.vid_to_slot.len(), 0);
        assert_eq!(pool.free.get(&(24, 4)).map(|v| v.len()), Some(1));
    }

    #[test]
    fn release_for_unknown_vid_is_noop() {
        let mut pool = SlotPool::new();
        pool.release_for_vid(ValueId(999)); // never recorded
        assert!(pool.vid_to_slot.is_empty());
        assert!(pool.free.is_empty());
    }

    #[test]
    fn free_list_buckets_by_size_align() {
        let mut pool = SlotPool::new();
        pool.record_owner(ValueId(1), synth_ss(1), 24, 4);
        pool.record_owner(ValueId(2), synth_ss(2), 48, 4);
        pool.record_owner(ValueId(3), synth_ss(3), 24, 3);
        pool.release_for_vid(ValueId(1));
        pool.release_for_vid(ValueId(2));
        pool.release_for_vid(ValueId(3));
        assert_eq!(pool.free.get(&(24, 4)).map(|v| v.len()), Some(1));
        assert_eq!(pool.free.get(&(48, 4)).map(|v| v.len()), Some(1));
        assert_eq!(pool.free.get(&(24, 3)).map(|v| v.len()), Some(1));
    }

    #[test]
    fn shared_owner_keeps_slot_live_until_all_released() {
        // Reshape-style scenario: %10 owns slot S; Reshape forwards
        // ptr(S) to %11, which we mark as a shared holder. Releasing
        // %10 must NOT yet release S; releasing %11 then does.
        let mut pool = SlotPool::new();
        let ss = synth_ss(7);
        pool.record_owner(ValueId(10), ss, 24, 4);
        pool.share_owner(ValueId(10), ValueId(11));
        pool.release_for_vid(ValueId(10));
        assert!(
            pool.free.get(&(24, 4)).map(|v| v.len()).unwrap_or(0) == 0,
            "slot returned to free-list while %11 still holds it"
        );
        assert!(pool.vid_to_slot.contains_key(&ValueId(11)));
        pool.release_for_vid(ValueId(11));
        assert_eq!(pool.free.get(&(24, 4)).map(|v| v.len()), Some(1));
    }

    #[test]
    fn share_owner_on_unknown_source_is_noop() {
        let mut pool = SlotPool::new();
        pool.share_owner(ValueId(10), ValueId(11));
        assert!(pool.vid_to_slot.is_empty());
    }

    #[test]
    fn release_order_independent_for_shared_slot() {
        // Same as shared_owner test but release %11 first. The slot
        // must stay live until both are released.
        let mut pool = SlotPool::new();
        let ss = synth_ss(8);
        pool.record_owner(ValueId(20), ss, 16, 3);
        pool.share_owner(ValueId(20), ValueId(21));
        pool.release_for_vid(ValueId(21));
        assert!(pool.free.get(&(16, 3)).map(|v| v.len()).unwrap_or(0) == 0);
        pool.release_for_vid(ValueId(20));
        assert_eq!(pool.free.get(&(16, 3)).map(|v| v.len()), Some(1));
    }

    #[test]
    fn hit_rate_and_counters() {
        let mut pool = SlotPool::new();
        assert_eq!(pool.hit_rate_pct(), 0.0);
        pool.hits = 3;
        pool.misses = 7;
        assert_eq!(pool.total_allocs(), 10);
        assert_eq!(pool.hit_rate_pct(), 30.0);
    }
}
