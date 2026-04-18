//! Per-op wall-time sampling for the cranelift-mlir profiler.
//!
//! Compile-time-sampled instrumentation that brackets every N-th
//! emission of a small set of interesting ops (F64X2 loads/stores,
//! fadd/fmul, stack_addr, iconst) with `__cranelift_op_begin /
//! __cranelift_op_end` probes. Active whenever
//! `ELODIN_CRANELIFT_DEBUG_DIR` is set; silent otherwise, and the
//! emitted machine code is bit-identical to a plain build when off.
//!
//! The sampling rate is a hardcoded constant chosen so probe cost
//! stays under ~1% of a typical tick at steady state; see
//! `OP_SAMPLE_RATE`.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;

/// Category IDs used by `__cranelift_op_begin / _end`. Must stay in
/// sync with the match arm in `profile.rs` that decodes them.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// F64 / F64X2 `fadd`.
    Fadd = 0,
    /// F64 / F64X2 `fsub`.
    Fsub = 1,
    /// F64 / F64X2 `fmul`.
    Fmul = 2,
    /// F64 / F64X2 `fdiv`.
    Fdiv = 3,
    /// F64 / F64X2 memory load from a stack slot.
    Load = 4,
    /// F64 / F64X2 memory store to a stack slot.
    Store = 5,
    /// `stack_addr` — address-gen for a slot-backed tensor.
    StackAddr = 6,
    /// Integer-constant materialization.
    Iconst = 7,
    /// F64-constant materialization.
    F64Const = 8,
}

impl OpCategory {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Fadd,
            1 => Self::Fsub,
            2 => Self::Fmul,
            3 => Self::Fdiv,
            4 => Self::Load,
            5 => Self::Store,
            6 => Self::StackAddr,
            7 => Self::Iconst,
            8 => Self::F64Const,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Fadd => "fadd",
            Self::Fsub => "fsub",
            Self::Fmul => "fmul",
            Self::Fdiv => "fdiv",
            Self::Load => "load",
            Self::Store => "store",
            Self::StackAddr => "stack_addr",
            Self::Iconst => "iconst",
            Self::F64Const => "f64const",
        }
    }
    pub fn all() -> &'static [OpCategory] {
        &[
            Self::Fadd,
            Self::Fsub,
            Self::Fmul,
            Self::Fdiv,
            Self::Load,
            Self::Store,
            Self::StackAddr,
            Self::Iconst,
            Self::F64Const,
        ]
    }
}

/// Hard sample rate for the *helper-level* sampler. The plan called out
/// N=256 under the assumption that we'd sample at runtime (~32k ops/tick),
/// but the actual instrumentation site is at COMPILE time — the count is
/// per-source-op across the whole module — so the typical count per
/// category is in the dozens, not the tens of thousands.
///
/// `OP_SAMPLE_RATE = 1` therefore brackets every emission. This keeps
/// the measurement complete and the overhead manageable:
///   cube-sat: ~44 F64 arith emissions × 2 probes × ~50 ns = 4.4 us
///   per tick of overhead, ~0.5% at the 825 us baseline.
///
/// Since every emission is bracketed, the sampled `total_ns` equals
/// the true measured ns — `sample_rate_for` returns 1 for these ops so
/// no report-time scaling is applied.
pub const OP_SAMPLE_RATE: u64 = 1;

/// Sample rate for the inner-loop sampler used inside
/// `lower_ptr_*_simd_f64` chunk loops. Load / Store / StackAddr fire
/// many times per helper invocation (one per chunk + one per tail),
/// so the rate is higher to bound probe overhead. One-in-sixteen
/// keeps overhead under ~1% while still giving enough samples to
/// attribute per-category wall time to within a few percent.
pub const INNER_OP_SAMPLE_RATE: u64 = 16;

/// Return the sample rate applied by the compile-time `OpSampler` for
/// the given category. Used at report time to multiply the measured
/// `total_ns` back up to a full-sim estimate:
/// `estimated_ns = measured_sample_ns × sample_rate_for(cat)`.
pub fn sample_rate_for(cat: OpCategory) -> u64 {
    match cat {
        // Helper-level sampler (bracket every emission).
        OpCategory::Fadd | OpCategory::Fsub | OpCategory::Fmul | OpCategory::Fdiv => OP_SAMPLE_RATE,
        // Inner-loop sampler (bracket every 16th chunk).
        OpCategory::Load
        | OpCategory::Store
        | OpCategory::StackAddr
        | OpCategory::Iconst
        | OpCategory::F64Const => INNER_OP_SAMPLE_RATE,
    }
}

/// Per-category emission counter used by the compile-time sampler.
/// Incremented every time an instrumented op is emitted; when the
/// counter hits a multiple of `rate`, `should_sample_at` returns
/// `true` and the caller wraps that emission in the probe pair.
///
/// The rate is passed in per-call so one sampler can serve both the
/// helper-level (rate=1) and inner-loop (rate=16) callsites.
#[derive(Debug, Default, Clone)]
pub struct OpSampler {
    counters: HashMap<OpCategory, u64>,
}

impl OpSampler {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }

    /// Helper-level sample check (rate = `OP_SAMPLE_RATE`).
    pub fn should_sample(&mut self, category: OpCategory) -> bool {
        self.should_sample_at(category, OP_SAMPLE_RATE)
    }

    /// Inner-loop sample check (rate = `INNER_OP_SAMPLE_RATE`).
    pub fn should_sample_inner(&mut self, category: OpCategory) -> bool {
        self.should_sample_at(category, INNER_OP_SAMPLE_RATE)
    }

    /// Bump the emission counter for `category`; return `true` iff the
    /// new count is a multiple of `rate`. Uses modulo so we don't
    /// skip the very first op.
    pub fn should_sample_at(&mut self, category: OpCategory, rate: u64) -> bool {
        let counter = self.counters.entry(category).or_insert(0);
        *counter += 1;
        (*counter).is_multiple_of(rate)
    }
}

/// Global "is PROFILE_OP_TIMES enabled?" flag. Set once from
/// `CompileConfig::from_env` when constructing per-function
/// configuration; cached via an atomic so probe emission can check
/// cheaply in the hot JIT-time dispatch code.
static OP_TIMES_ENABLED: AtomicU32 = AtomicU32::new(0);

pub fn set_op_times_enabled(enabled: bool) {
    OP_TIMES_ENABLED.store(
        if enabled { 1 } else { 0 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub fn op_times_enabled() -> bool {
    OP_TIMES_ENABLED.load(std::sync::atomic::Ordering::Relaxed) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_every_nth() {
        let mut s = OpSampler::new();
        let mut hits = 0;
        // With OP_SAMPLE_RATE=1 this samples every emission, with
        // larger rates it samples every Nth. Either way, 3 × rate
        // emissions should produce 3 hits.
        for _ in 0..(OP_SAMPLE_RATE as usize * 3) {
            if s.should_sample(OpCategory::Fmul) {
                hits += 1;
            }
        }
        assert_eq!(hits, 3);
    }

    #[test]
    fn independent_categories() {
        let mut s = OpSampler::new();
        // Fmul and Fadd counters advance independently: hitting
        // sample-worthy multiples of one shouldn't affect the other.
        for _ in 0..10 {
            let _ = s.should_sample(OpCategory::Fmul);
        }
        // Fadd starts at 0 — first call should sample when rate=1,
        // or should NOT sample when rate>1. Assert only the
        // non-interference property.
        let before = OP_SAMPLE_RATE > 1;
        let after = s.should_sample(OpCategory::Fadd);
        assert_ne!(after, before);
    }

    #[test]
    fn category_round_trip() {
        for &c in OpCategory::all() {
            assert_eq!(OpCategory::from_u32(c.as_u32()), Some(c));
        }
    }
}
