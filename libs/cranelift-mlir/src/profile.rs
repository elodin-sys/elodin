//! Live runtime profiling for JIT-compiled cranelift-mlir functions.
//!
//! ## Overview
//!
//! When `ELODIN_CRANELIFT_DEBUG_DIR=/path` is set at simulation start,
//! `lower::define_function` injects `__cranelift_profile_enter` /
//! `__cranelift_profile_exit` calls into the prologue/epilogue of every
//! JIT-compiled function's Cranelift IR. These extern "C" probes accumulate
//! per-function wall time, call counts, and min/max into a thread-local
//! `HashMap<u32, FunctionStats>` keyed by `FuncId`.
//!
//! At `CompiledModule` drop time, `dump_report()` formats the aggregated
//! stats to stderr and writes `profile.json` into
//! `ELODIN_CRANELIFT_DEBUG_DIR`, then resets the thread-local store.
//!
//! ## Zero overhead when disabled
//!
//! The probes are **compile-time gated** in `define_function` on
//! `CompileConfig::profile_enabled`. When that flag is false the Cranelift
//! IR (and therefore the emitted machine code) has no trace of profiling:
//! no extra calls, no branches, no thread-local access. The default path
//! is bit-identical to the pre-profile codebase.
//!
//! See [`libs/cranelift-mlir/PERFORMANCE.md`](../PERFORMANCE.md) for the
//! full end-user guide.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::lower::{FuncAbi, INSTR_COUNTS, InstrCountEntry};

/// Per-function runtime statistics, accumulated by the profile probes.
/// One entry per `FuncId`. `total_ns` is inclusive (body + callees);
/// `exclusive_ns` subtracts children time; `time_in_calls_ns` is the
/// subset of inclusive time spent inside emitted `call` ops
/// (tensor_rt, libm, other JIT functions).
#[derive(Default, Clone, Debug)]
pub struct FunctionStats {
    pub calls: u64,
    /// Inclusive wall time (body + callees).
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    /// Exclusive wall time (`total_ns − children_ns`).
    pub exclusive_ns: u64,
    /// Portion of `total_ns` spent inside emitted `call` ops.
    pub time_in_calls_ns: u64,
    /// Log-scale latency histogram, 32 buckets from 100 ns to 100 ms.
    /// Bucket `i` covers `[10^(2 + i·6/32), 10^(2 + (i+1)·6/32)) ns`;
    /// every exit bumps one bucket.
    pub histogram: [u64; HISTOGRAM_BUCKETS],
}

/// Width of the latency histogram. 32 log-scale buckets spanning 6
/// decades (100 ns to 100 ms) gives ~5.33 buckets/decade — enough to
/// distinguish libm-scalar (tens of ns) from tensor_rt (µs) from
/// LAPACK (tens of µs) from long ticks (ms).
pub(crate) const HISTOGRAM_BUCKETS: usize = 32;

/// Lower bound of the histogram in nanoseconds (bucket 0 lo).
const HISTOGRAM_MIN_NS: f64 = 100.0;
/// Upper bound of the histogram in nanoseconds (bucket N-1 hi).
const HISTOGRAM_MAX_NS: f64 = 1.0e8;
/// Cached `log10(max/min)` once at compile time.
const HISTOGRAM_LOG_RANGE: f64 = 6.0; // log10(1e8/1e2)

/// Map an elapsed-ns value to a histogram bucket index.
#[inline]
fn histogram_bucket(ns: u64) -> usize {
    if ns <= HISTOGRAM_MIN_NS as u64 {
        return 0;
    }
    if ns >= HISTOGRAM_MAX_NS as u64 {
        return HISTOGRAM_BUCKETS - 1;
    }
    let log_pos = (ns as f64).log10() - HISTOGRAM_MIN_NS.log10();
    let bucket = (log_pos / HISTOGRAM_LOG_RANGE * HISTOGRAM_BUCKETS as f64) as usize;
    bucket.min(HISTOGRAM_BUCKETS - 1)
}

/// Return the low edge (ns) of a histogram bucket.
#[inline]
fn histogram_bucket_lo_ns(i: usize) -> f64 {
    HISTOGRAM_MIN_NS * 10f64.powf(HISTOGRAM_LOG_RANGE * i as f64 / HISTOGRAM_BUCKETS as f64)
}

/// Stack frame for the runtime call stack. `children_ns` accumulates
/// time spent inside callees so the exit probe can compute exclusive
/// time. Held per-thread in `TL_STACK`.
#[derive(Clone, Copy)]
struct StackFrame {
    fid: u32,
    start: Instant,
    children_ns: u64,
}

/// Parent/callee edge statistics, accumulated in `TL_EDGES` keyed by
/// `(parent_fid, callee_fid)` and populated at child exit when the
/// parent frame is still on the stack.
#[derive(Default, Clone, Debug)]
pub struct EdgeStats {
    pub calls: u64,
    pub total_ns: u64,
}

/// Dummy FuncId reserved for probe-overhead calibration. Picked to
/// be well above any plausible real FuncId; stats recorded against it
/// are discarded after calibration.
const CALIBRATION_FID: u32 = u32::MAX - 1;

/// Number of probe pairs to run during calibration. Calibration runtime
/// at 80 ns/pair: 100k × 80 ns = 8 ms — negligible at sim shutdown.
const CALIBRATION_ITERS: u64 = 100_000;

/// Median probe overhead in nanoseconds, measured once per `dump_report`
/// invocation. Zero until calibration runs. Used to compute
/// `corrected_total_ns` for the report.
static PROBE_OVERHEAD_NS: AtomicU64 = AtomicU64::new(0);

/// Per-family probe-overhead measurements. Each entry is ns-per-probe
/// (either a single probe or a begin/end pair, as noted on the
/// accessor). All start at 0 and are populated the first time
/// `calibrate_probe_overhead` runs. When correcting `main_wall_ns`,
/// we multiply each family's per-probe cost by the number of times
/// that probe fired in the sim and subtract the grand total from the
/// observed wall time.
static PROBE_OVERHEAD_CALL_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_OVERHEAD_MARSHAL_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_OVERHEAD_XCEND_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_OVERHEAD_LOOP_ITER_NS: AtomicU64 = AtomicU64::new(0);

// Set to `true` while `calibrate_probe_overhead` is running. The main
// enter/exit probes check this flag to skip histogram / edge / timeline
// side-effects during calibration (they still need the timing path to
// be exercised so the overhead measurement is representative). Using a
// thread-local since calibration runs on the same thread that dumps.
thread_local! {
    static CALIBRATING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Process-global accumulator for stats flushed by exiting threads.
/// Each thread owns a `ThreadStatsSentinel` held in a thread-local;
/// when the thread exits, the sentinel's `Drop` moves its TL_STATS
/// here. `dump_report` drains this accumulator alongside the caller
/// thread's live TL_STATS and merges both. The JIT is currently
/// single-threaded so the global is always empty at dump time.
static GLOBAL_STATS: OnceLock<Mutex<HashMap<u32, FunctionStats>>> = OnceLock::new();

fn global_stats() -> &'static Mutex<HashMap<u32, FunctionStats>> {
    GLOBAL_STATS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Explicit flush of the current thread's `TL_STATS` into
/// `GLOBAL_STATS`. Call this before a worker thread exits so its
/// stats contribute to the eventual `dump_report`. Thread-local
/// `Drop` is unreliable here: Rust's drop order between thread-locals
/// is implementation-defined, so a sentinel-based flush could fire
/// after `TL_STATS` is already destroyed.
pub fn flush_current_thread() {
    let stats: Vec<(u32, FunctionStats)> = TL_STATS
        .try_with(|m| m.borrow_mut().drain().collect())
        .unwrap_or_default();
    if stats.is_empty() {
        return;
    }
    let mut g = match global_stats().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    for (fid, s) in stats {
        let entry = g.entry(fid).or_default();
        merge_stats_into(entry, &s);
    }
}

/// Sum `src` into `dst`. Used both by the thread-exit flush and the
/// per-dump merge across all threads.
fn merge_stats_into(dst: &mut FunctionStats, src: &FunctionStats) {
    dst.calls = dst.calls.saturating_add(src.calls);
    dst.total_ns = dst.total_ns.saturating_add(src.total_ns);
    dst.exclusive_ns = dst.exclusive_ns.saturating_add(src.exclusive_ns);
    dst.time_in_calls_ns = dst.time_in_calls_ns.saturating_add(src.time_in_calls_ns);
    if dst.min_ns == 0 || (src.min_ns != 0 && src.min_ns < dst.min_ns) {
        dst.min_ns = src.min_ns;
    }
    if src.max_ns > dst.max_ns {
        dst.max_ns = src.max_ns;
    }
    for (i, &c) in src.histogram.iter().enumerate() {
        dst.histogram[i] = dst.histogram[i].saturating_add(c);
    }
}

thread_local! {
    /// Per-function stats. Keyed by `FuncId::as_u32()`. Uses a regular
    /// (non-const) initializer because `HashMap::new` isn't const.
    static TL_STATS: RefCell<HashMap<u32, FunctionStats>> =
        RefCell::new(HashMap::new());


    /// Call stack of open probe frames. Each entry is a `StackFrame`
    /// that tracks `fid`, `start` (Instant), and `children_ns` for
    /// exclusive-time accounting.
    static TL_STACK: RefCell<Vec<StackFrame>> =
        const { RefCell::new(Vec::new()) };

    /// Parent/callee edges, keyed by `(parent_fid, callee_fid)`.
    /// Populated in `profile_exit` when a parent frame is still on
    /// the stack.
    static TL_EDGES: RefCell<HashMap<(u32, u32), EdgeStats>> =
        RefCell::new(HashMap::new());

    /// Per-main-tick waveform. When debug mode is enabled, every
    /// `main` exit appends its elapsed ns here. Written to JSON at
    /// dump time.
    static TL_TIMELINE: RefCell<Vec<u64>> =
        const { RefCell::new(Vec::new()) };

    /// Bracketing stacks for marshal and transcendental begin/end
    /// probes. Single-entry deep in practice.
    static TL_MARSHAL_STACK: RefCell<Vec<(u32, u64, Instant)>> =
        const { RefCell::new(Vec::new()) };
    static TL_XCEND_STACK: RefCell<Vec<(u32, Instant)>> =
        const { RefCell::new(Vec::new()) };

    /// Call-op timing stack. Each `__cranelift_call_begin` pushes an
    /// `Instant`; `__cranelift_call_end` pops and attributes the
    /// elapsed time to the enclosing function (top of `TL_STACK`).
    static TL_CALL_STACK: RefCell<Vec<Instant>> =
        const { RefCell::new(Vec::new()) };

    /// When the `tracy` cargo feature is enabled, `profile_enter` pushes a
    /// `tracy_client::Span` onto this parallel stack; `profile_exit` pops
    /// and drops it, closing the Tracy zone. The stack exists only when
    /// the feature is built; otherwise the probe functions are no-ops on
    /// the Tracy side.
    #[cfg(feature = "tracy")]
    static TL_TRACY_STACK: RefCell<Vec<tracy_client::Span>> =
        const { RefCell::new(Vec::new()) };
}

/// Static metadata registered at JIT compile time. Indexed by `FuncId`.
/// The Mutex-guarded store is a one-time fill at the end of
/// `compile_module_with_config` and is never updated again for the lifetime
/// of a `CompiledModule`.
struct StaticRegistry {
    names: HashMap<u32, String>,
    abis: HashMap<u32, FuncAbi>,
    /// StableHLO source line per function (populated by the parser)
    /// for attaching source locations to Tracy zones.
    source_lines: HashMap<u32, u32>,
    /// Static IR instruction counts per `FuncId` and opcode name,
    /// sourced from `INSTR_COUNTS` at compile time. Feeds the
    /// per-op-category estimate and runtime-weighted SIMD utilization.
    instr_stats: HashMap<u32, StaticInstrStats>,
    /// Per-loop static metadata, populated by `register_loop` /
    /// `lower_while`. Keyed by `loop_id`.
    loops: HashMap<u32, LoopStatic>,
}

#[derive(Default, Clone, Debug)]
struct StaticInstrStats {
    scalar: usize,
    vector: usize,
    #[allow(dead_code)]
    op_kind_counts: HashMap<&'static str, usize>,
}

#[derive(Default, Clone, Debug)]
struct LoopStatic {
    parent_fid: u32,
    body_ops: HashMap<&'static str, usize>,
}

static REGISTRY: OnceLock<Mutex<StaticRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<StaticRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(StaticRegistry {
            names: HashMap::new(),
            abis: HashMap::new(),
            source_lines: HashMap::new(),
            instr_stats: HashMap::new(),
            loops: HashMap::new(),
        })
    })
}

/// Record per-loop static metadata at compile time. Called from
/// `lower_while` when profiling is enabled.
pub(crate) fn register_loop(loop_id: u32, parent_fid: u32, body_ops: HashMap<&'static str, usize>) {
    let Ok(mut reg) = registry().lock() else {
        return;
    };
    reg.loops.insert(
        loop_id,
        LoopStatic {
            parent_fid,
            body_ops,
        },
    );
}

/// Record a StableHLO source line for a function name. Called while
/// `register_static_data` is already under lock; internal.
fn register_source_line(
    reg: &mut StaticRegistry,
    name: &str,
    line: u32,
    func_ids: &HashMap<String, cranelift_module::FuncId>,
) {
    if let Some(fid) = func_ids.get(name) {
        reg.source_lines.insert(fid.as_u32(), line);
    }
}

/// Called once per `CompiledModule` construction. Registers the
/// `FuncId → name + abi + static instr counts + optional source line`
/// mapping so `dump_report` can render names and compute weighted
/// metrics. `source_lines` maps function name → StableHLO source line
/// for Tracy source navigation. Multiple `CompiledModule`s in the
/// same process would clobber each other's names; that's acceptable
/// since simulations run one module at a time.
pub(crate) fn register_static_data(
    func_ids: &HashMap<String, cranelift_module::FuncId>,
    source_lines: &HashMap<String, u32>,
) {
    // Ensure Tracy is running. Safe to call repeatedly — `Client::start`
    // is idempotent. Only runs when the `tracy` feature is built in;
    // without the feature the call is compiled out entirely.
    #[cfg(feature = "tracy")]
    {
        let _ = tracy_client::Client::start();
    }
    let mut reg = registry().lock().expect("profile registry poisoned");
    reg.names.clear();
    reg.abis.clear();
    reg.source_lines.clear();
    reg.instr_stats.clear();
    reg.loops.clear();
    for (name, fid) in func_ids {
        reg.names.insert(fid.as_u32(), name.clone());
    }
    for (name, line) in source_lines {
        register_source_line(&mut reg, name, *line, func_ids);
    }
    INSTR_COUNTS.with(|c| {
        let counts = c.borrow();
        for e in counts.iter() {
            if let Some(fid) = func_ids.get(&e.name) {
                reg.abis.insert(fid.as_u32(), e.abi);
                reg.instr_stats.insert(
                    fid.as_u32(),
                    StaticInstrStats {
                        scalar: e.scalar,
                        vector: e.vector,
                        op_kind_counts: e.op_kind_counts.clone(),
                    },
                );
            }
        }
    });
}

/// Debug-mode probe-emission gate. Delegated to the single
/// `ELODIN_CRANELIFT_DEBUG_DIR` env var via [`crate::debug::enabled`].
pub fn enabled() -> bool {
    crate::debug::enabled()
}

/// Tick waveform capture is always on whenever debug mode is enabled.
/// Previously gated by a separate env var; folded into debug mode.
pub fn timeline_enabled() -> bool {
    crate::debug::enabled()
}

/// Entry probe. Called at the start of every instrumented JIT function.
///
/// # Safety
///
/// This is the callee of a JIT-emitted `call` instruction; the JIT only
/// emits it when profiling was enabled at compile time. Always safe to
/// call from Rust (no unsafe internals), but exposed as `extern "C"` for
/// ABI stability.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_profile_enter(fid: u32) {
    let now = Instant::now();
    TL_STACK.with(|s| {
        s.borrow_mut().push(StackFrame {
            fid,
            start: now,
            children_ns: 0,
        })
    });
    #[cfg(feature = "tracy")]
    tracy_zone_push(fid);
}

/// Push a Tracy span keyed on `fid`. Called inside the enter probe when
/// the `tracy` feature is built. No-op when Tracy isn't running (no
/// profiler connected yet). Passes the StableHLO source line when
/// available so Tracy's "Go to source" lands on the actual function.
#[cfg(feature = "tracy")]
fn tracy_zone_push(fid: u32) {
    let Some(client) = tracy_client::Client::running() else {
        return;
    };
    // Look the name + source line up under a quick lock — reads only, no
    // contention in the single-threaded JIT.
    let (name, source_line) = {
        let Ok(reg) = registry().lock() else {
            return;
        };
        let name = reg
            .names
            .get(&fid)
            .cloned()
            .unwrap_or_else(|| format!("cranelift_fid_{fid}"));
        let line = reg.source_lines.get(&fid).copied().unwrap_or(0);
        (name, line)
    };
    // `span_alloc` heap-allocates the SpanLocation per call. Acceptable
    // for a Tracy-only deep-dive path; we avoid this cost on the default
    // profile path because `tracy_zone_push` itself is `#[cfg]`-gated out.
    // Signature: span_alloc(name, function, file, line, callstack_depth).
    let span = client.span_alloc(
        Some(&name),
        "stablehlo_fn",
        "stablehlo.mlir",
        source_line,
        0,
    );
    TL_TRACY_STACK.with(|s| s.borrow_mut().push(span));
}

#[cfg(feature = "tracy")]
fn tracy_zone_pop() {
    TL_TRACY_STACK.with(|s| {
        s.borrow_mut().pop();
    });
}

/// Cross-ABI marshal counters. Updated by
/// `__cranelift_marshal_begin/end`. Using atomics keeps
/// the hot path lock-free; the counters are additive so relaxed ordering
/// is sufficient.
static MARSHAL_S2P_CALLS: AtomicU64 = AtomicU64::new(0);
static MARSHAL_S2P_BYTES: AtomicU64 = AtomicU64::new(0);
static MARSHAL_S2P_NS: AtomicU64 = AtomicU64::new(0);
static MARSHAL_P2S_CALLS: AtomicU64 = AtomicU64::new(0);
static MARSHAL_P2S_BYTES: AtomicU64 = AtomicU64::new(0);
static MARSHAL_P2S_NS: AtomicU64 = AtomicU64::new(0);

/// Cross-ABI marshal begin probe. Called at the start of a
/// scalar↔pointer marshal region. `direction` 0 = scalar→pointer,
/// 1 = pointer→scalar. `bytes` is the payload size. Counts and bytes
/// are accumulated immediately; timing is recorded on the matching
/// `__cranelift_marshal_end`.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_marshal_begin(direction: u32, bytes: u64) {
    if CALIBRATING.with(|c| c.get()) {
        // Still exercise the timing path for representative overhead,
        // but skip the global counter updates to avoid contaminating
        // the report.
        let now = Instant::now();
        TL_MARSHAL_STACK.with(|s| s.borrow_mut().push((direction, bytes, now)));
        return;
    }
    let now = Instant::now();
    match direction {
        0 => {
            MARSHAL_S2P_CALLS.fetch_add(1, Ordering::Relaxed);
            MARSHAL_S2P_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
        1 => {
            MARSHAL_P2S_CALLS.fetch_add(1, Ordering::Relaxed);
            MARSHAL_P2S_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
        _ => return,
    }
    TL_MARSHAL_STACK.with(|s| s.borrow_mut().push((direction, bytes, now)));
}

/// Cross-ABI marshal end probe. Pops the matching begin frame and
/// accumulates elapsed time into the per-direction ns counter.
/// Tolerates stack-empty and direction-mismatch cases defensively so
/// a probe bug can never crash the sim.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_marshal_end() {
    let now = Instant::now();
    let frame = TL_MARSHAL_STACK.with(|s| s.borrow_mut().pop());
    let Some((direction, _bytes, start)) = frame else {
        return;
    };
    if CALIBRATING.with(|c| c.get()) {
        // Exercise the duration-compute path but skip the atomic.
        let _ = now.duration_since(start).as_nanos() as u64;
        return;
    }
    let elapsed = now.duration_since(start).as_nanos() as u64;
    match direction {
        0 => {
            MARSHAL_S2P_NS.fetch_add(elapsed, Ordering::Relaxed);
        }
        1 => {
            MARSHAL_P2S_NS.fetch_add(elapsed, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Transcendental call-site counters. `mode` 0 = libm scalar fallback,
/// 1 = wide-SIMD batch. Updated by `__cranelift_xcend_begin/end`.
static XCEND_LIBM_CALLS: AtomicU64 = AtomicU64::new(0);
static XCEND_LIBM_NS: AtomicU64 = AtomicU64::new(0);
static XCEND_SIMD_CALLS: AtomicU64 = AtomicU64::new(0);
static XCEND_SIMD_NS: AtomicU64 = AtomicU64::new(0);

/// Transcendental begin probe. Increments the call count for the
/// given mode and pushes a timing frame.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_xcend_begin(mode: u32) {
    if CALIBRATING.with(|c| c.get()) {
        let now = Instant::now();
        TL_XCEND_STACK.with(|s| s.borrow_mut().push((mode, now)));
        return;
    }
    let now = Instant::now();
    match mode {
        0 => {
            XCEND_LIBM_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        1 => {
            XCEND_SIMD_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        _ => return,
    }
    TL_XCEND_STACK.with(|s| s.borrow_mut().push((mode, now)));
}

/// Transcendental end probe.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_xcend_end() {
    let now = Instant::now();
    let frame = TL_XCEND_STACK.with(|s| s.borrow_mut().pop());
    let Some((mode, start)) = frame else { return };
    if CALIBRATING.with(|c| c.get()) {
        let _ = now.duration_since(start).as_nanos() as u64;
        return;
    }
    let elapsed = now.duration_since(start).as_nanos() as u64;
    match mode {
        0 => {
            XCEND_LIBM_NS.fetch_add(elapsed, Ordering::Relaxed);
        }
        1 => {
            XCEND_SIMD_NS.fetch_add(elapsed, Ordering::Relaxed);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Call-op timing probes
// ---------------------------------------------------------------------------

/// Call-op begin probe. Pushes a timing frame so the end probe can
/// attribute elapsed time to the enclosing function's
/// `time_in_calls_ns`, separating inline IR from callee time.
///
/// Total observed call-begin/end pairs — used to weight the `call`
/// family in the probe-overhead correction.
static CALL_PROBE_PAIRS: AtomicU64 = AtomicU64::new(0);

#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_call_begin() {
    if CALIBRATING.with(|c| c.get()) {
        let now = Instant::now();
        TL_CALL_STACK.with(|s| s.borrow_mut().push(now));
        return;
    }
    CALL_PROBE_PAIRS.fetch_add(1, Ordering::Relaxed);
    let now = Instant::now();
    TL_CALL_STACK.with(|s| s.borrow_mut().push(now));
}

/// Call-op end probe. Pops the matching begin frame and accumulates
/// the elapsed time into the enclosing function's `time_in_calls_ns`
/// stat.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_call_end() {
    let now = Instant::now();
    let start = TL_CALL_STACK.with(|s| s.borrow_mut().pop());
    let Some(start) = start else { return };
    if CALIBRATING.with(|c| c.get()) {
        let _ = now.duration_since(start).as_nanos() as u64;
        return;
    }
    let elapsed = now.duration_since(start).as_nanos() as u64;
    // Attribute the elapsed time to the function currently on top of
    // TL_STACK. If there's no current function (unexpected), bail.
    let current_fid = TL_STACK.with(|s| s.borrow().last().map(|f| f.fid));
    let Some(fid) = current_fid else { return };
    TL_STATS.with(|m| {
        let mut m = m.borrow_mut();
        let entry = m.entry(fid).or_default();
        entry.time_in_calls_ns = entry.time_in_calls_ns.saturating_add(elapsed);
    });
}

// ---------------------------------------------------------------------------
// Loop iteration probes
// ---------------------------------------------------------------------------

/// Per-loop iteration counters, keyed by `loop_id`.
static LOOP_ITER_COUNTERS: OnceLock<Mutex<HashMap<u32, u64>>> = OnceLock::new();

fn loop_iter_counters() -> &'static Mutex<HashMap<u32, u64>> {
    LOOP_ITER_COUNTERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Loop-iter probe. Called at the top of every `while`-loop body.
/// `loop_id` is monotonic and assigned at compile time. Mutex is OK
/// because `loop_id` space is sparse (Vec would be wasteful) and
/// contention is negligible at one bump per iteration.
///
/// Total loop-iter invocations across all bodies, used to weight the
/// `loop_iter` family in the probe-overhead correction.
static LOOP_ITER_TOTAL: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Per-op wall-time sampling
// ---------------------------------------------------------------------------

/// Number of op-timing categories; must match `op_sampler::OpCategory` count.
const OP_CATEGORY_COUNT: usize = 9;

/// Per-category sampled ns accumulator. Index = `OpCategory as u32`.
static OP_CATEGORY_NS: [AtomicU64; OP_CATEGORY_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static OP_CATEGORY_SAMPLES: [AtomicU64; OP_CATEGORY_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

thread_local! {
    /// Sampled op begin/end timing stack. `(category_id, start_instant)`.
    static TL_OP_STACK: std::cell::RefCell<Vec<(u32, Instant)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Op-begin probe. Records the start timestamp for a sampled op
/// emission. No-op if the category is out of range.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_op_begin(category: u32) {
    if (category as usize) >= OP_CATEGORY_COUNT {
        return;
    }
    let now = Instant::now();
    TL_OP_STACK.with(|s| s.borrow_mut().push((category, now)));
}

/// Op-end probe. Pops the matching begin frame and accumulates
/// `elapsed × OP_SAMPLE_RATE` into the per-category ns counter so
/// the atomic represents the estimated full-sim total.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_op_end() {
    let now = Instant::now();
    let frame = TL_OP_STACK.with(|s| s.borrow_mut().pop());
    let Some((category, start)) = frame else {
        return;
    };
    if (category as usize) >= OP_CATEGORY_COUNT {
        return;
    }
    let elapsed = now.duration_since(start).as_nanos() as u64;
    // Store raw measured ns; the report-time scale depends on which
    // sampler fired (helper-level or inner-loop rate), so it's
    // applied in `dump_report` via `op_sampler::sample_rate_for`.
    OP_CATEGORY_NS[category as usize].fetch_add(elapsed, Ordering::Relaxed);
    OP_CATEGORY_SAMPLES[category as usize].fetch_add(1, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_loop_iter(loop_id: u32) {
    if CALIBRATING.with(|c| c.get()) {
        return;
    }
    LOOP_ITER_TOTAL.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut map) = loop_iter_counters().lock() {
        *map.entry(loop_id).or_insert(0) += 1;
    }
}

/// Exit probe. Called immediately before every `return` in every
/// instrumented JIT function. Pops the stack, records elapsed time
/// in `TL_STATS[fid]`, updates the parent's `children_ns` for
/// exclusive-time accounting, records the `(parent, callee)` edge,
/// bumps the latency histogram, and — on top-level `main` exits —
/// appends to the per-tick waveform.
#[unsafe(no_mangle)]
pub extern "C" fn __cranelift_profile_exit(fid: u32) {
    let now = Instant::now();
    let (frame, parent_fid) = TL_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        let frame = stack.pop();
        let parent_fid = stack.last().map(|f| f.fid);
        (frame, parent_fid)
    });
    let Some(frame) = frame else {
        // Mismatched exit (no matching enter). Ignore rather than
        // panic so a probe bug can never crash the sim.
        return;
    };
    if frame.fid != fid {
        // Stack inversion (shouldn't happen with our compile-time emission,
        // but defend against it). Put it back and bail.
        TL_STACK.with(|s| s.borrow_mut().push(frame));
        return;
    }
    let elapsed_ns = now.duration_since(frame.start).as_nanos() as u64;
    let exclusive_ns = elapsed_ns.saturating_sub(frame.children_ns);

    // Propagate inclusive elapsed to the parent's children_ns accumulator.
    if parent_fid.is_some() {
        TL_STACK.with(|s| {
            if let Some(top) = s.borrow_mut().last_mut() {
                top.children_ns = top.children_ns.saturating_add(elapsed_ns);
            }
        });
    }

    // Skip all side-effects during calibration so the calibration loop
    // doesn't pollute the real stats, histograms, edges, or timeline.
    // We still pay the Instant::now() + TL_STACK cost so the measured
    // overhead is representative.
    let calibrating = CALIBRATING.with(|c| c.get());
    if calibrating {
        #[cfg(feature = "tracy")]
        tracy_zone_pop();
        return;
    }

    TL_STATS.with(|map| {
        let mut m = map.borrow_mut();
        let entry = m.entry(fid).or_default();
        entry.calls += 1;
        entry.total_ns += elapsed_ns;
        entry.exclusive_ns = entry.exclusive_ns.saturating_add(exclusive_ns);
        if entry.min_ns == 0 || elapsed_ns < entry.min_ns {
            entry.min_ns = elapsed_ns;
        }
        if elapsed_ns > entry.max_ns {
            entry.max_ns = elapsed_ns;
        }
        entry.histogram[histogram_bucket(elapsed_ns)] += 1;
    });

    // Record the parent→callee edge.
    if let Some(p) = parent_fid {
        TL_EDGES.with(|e| {
            let mut m = e.borrow_mut();
            let edge = m.entry((p, fid)).or_default();
            edge.calls += 1;
            edge.total_ns = edge.total_ns.saturating_add(elapsed_ns);
        });
    }

    // Per-main-tick waveform. Only accumulate when the exiting frame
    // has no parent (top-level main) and timeline capture is enabled.
    if parent_fid.is_none() && timeline_enabled() {
        TL_TIMELINE.with(|t| t.borrow_mut().push(elapsed_ns));
    }

    #[cfg(feature = "tracy")]
    tracy_zone_pop();
}

/// Measure the mean wall time of each probe family by running a
/// tight calibration loop against a reserved `CALIBRATION_FID`.
/// Called once at the start of `dump_report`; the measured ns-per-
/// probe values feed `main_wall_ns_corrected`. The calibration
/// deliberately exercises the real probe code path so HashMap / Vec
/// / Instant behaviour is representative; dummy-fid stats are
/// subtracted afterwards so calibration never leaks into the report.
/// Covers all five families: enter/exit, call, marshal, xcend, and
/// loop_iter.
pub(crate) fn calibrate_probe_overhead() -> u64 {
    // Flip the calibration flag so enter/exit skips stats / histogram /
    // edges / timeline side-effects during the loop. The timing path
    // (Instant::now + TL_STACK) is still exercised so the measurement
    // reflects the real probe cost.
    CALIBRATING.with(|c| c.set(true));

    // Family 1: function enter/exit.
    let start = Instant::now();
    for _ in 0..CALIBRATION_ITERS {
        __cranelift_profile_enter(CALIBRATION_FID);
        __cranelift_profile_exit(CALIBRATION_FID);
    }
    let enter_exit_ns = start.elapsed().as_nanos() as u64;
    let per_enter_exit_pair = enter_exit_ns / CALIBRATION_ITERS.max(1);
    PROBE_OVERHEAD_NS.store(per_enter_exit_pair, Ordering::Relaxed);

    // Family 2: call_begin / call_end (trt_call + libm dispatch).
    let start = Instant::now();
    for _ in 0..CALIBRATION_ITERS {
        __cranelift_call_begin();
        __cranelift_call_end();
    }
    let call_ns = start.elapsed().as_nanos() as u64;
    PROBE_OVERHEAD_CALL_NS.store(call_ns / CALIBRATION_ITERS.max(1), Ordering::Relaxed);

    // Family 3: marshal_begin / marshal_end (scalar↔pointer conv).
    let start = Instant::now();
    for _ in 0..CALIBRATION_ITERS {
        __cranelift_marshal_begin(0, 0);
        __cranelift_marshal_end();
    }
    let marshal_ns = start.elapsed().as_nanos() as u64;
    PROBE_OVERHEAD_MARSHAL_NS.store(marshal_ns / CALIBRATION_ITERS.max(1), Ordering::Relaxed);

    // Family 4: xcend_begin / xcend_end (transcendental dispatch).
    let start = Instant::now();
    for _ in 0..CALIBRATION_ITERS {
        __cranelift_xcend_begin(0);
        __cranelift_xcend_end();
    }
    let xcend_ns = start.elapsed().as_nanos() as u64;
    PROBE_OVERHEAD_XCEND_NS.store(xcend_ns / CALIBRATION_ITERS.max(1), Ordering::Relaxed);

    // Family 5: loop_iter (While body iteration).
    let start = Instant::now();
    for _ in 0..CALIBRATION_ITERS {
        __cranelift_loop_iter(u32::MAX);
    }
    let loop_iter_ns = start.elapsed().as_nanos() as u64;
    PROBE_OVERHEAD_LOOP_ITER_NS.store(loop_iter_ns / CALIBRATION_ITERS.max(1), Ordering::Relaxed);

    CALIBRATING.with(|c| c.set(false));
    per_enter_exit_pair
}

/// Serde-serializable function entry for the JSON dump.
#[derive(serde::Serialize)]
struct JsonFunctionEntry<'a> {
    fid: u32,
    name: &'a str,
    abi: &'static str,
    calls: u64,
    total_ns: u64,
    /// `total_ns` minus estimated probe overhead for this function.
    corrected_total_ns: u64,
    /// Time inside this function's body, excluding recorded time
    /// spent in instrumented callees.
    exclusive_ns: u64,
    /// Time spent inside emitted `call` ops (tensor_rt, libm, JIT
    /// callees). Populated when call-bracket probes are active.
    time_in_calls_ns: u64,
    /// Per-function minimum elapsed wall time.
    min_ns: u64,
    /// Per-function maximum elapsed wall time.
    max_ns: u64,
    /// Per-function mean elapsed wall time (inclusive, raw).
    mean_ns: u64,
    /// Latency percentiles interpolated from the 32-bucket histogram.
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    p99_9_ns: u64,
    scalar_instrs: usize,
    vector_instrs: usize,
    vec_pct: f64,
    /// Percentage of total recorded time across all functions (including
    /// main). If main itself is in the set, this will show main near 100%.
    total_pct: f64,
    /// Raw per-bucket counts for downstream tooling. Emitted inline so
    /// `diff_profile.py` can reconstruct percentiles without re-running
    /// the sim.
    histogram: [u64; HISTOGRAM_BUCKETS],
}

#[derive(serde::Serialize)]
struct JsonMarshal {
    scalar_to_pointer_calls: u64,
    scalar_to_pointer_bytes: u64,
    scalar_to_pointer_total_ns: u64,
    pointer_to_scalar_calls: u64,
    pointer_to_scalar_bytes: u64,
    pointer_to_scalar_total_ns: u64,
}

#[derive(serde::Serialize)]
struct JsonTranscendental {
    libm_scalar_calls: u64,
    libm_total_ns: u64,
    wide_simd_calls: u64,
    wide_simd_total_ns: u64,
}

#[derive(serde::Serialize)]
struct JsonSimd {
    static_vector_pct: f64,
    runtime_weighted_vector_pct: f64,
}

#[derive(serde::Serialize)]
struct JsonEdge {
    parent_fid: u32,
    parent_name: String,
    callee_fid: u32,
    callee_name: String,
    calls: u64,
    total_ns: u64,
}

#[derive(serde::Serialize)]
struct JsonLoopStat {
    loop_id: u32,
    parent_fid: u32,
    parent_name: String,
    iters: u64,
    body_ops: HashMap<String, u64>,
}

#[derive(serde::Serialize)]
struct JsonProbeOverhead {
    measured_ns_per_probe: u64,
    /// Total overhead attributed across all reported functions
    /// (function-enter/exit pairs only; see `by_family.total_ns` for
    /// the all-probe-families total used by `main_wall_ns_corrected`).
    total_overhead_ns: u64,
    calibration_iters: u64,
    /// Per-family per-probe ns + total attributed ns. Each family's
    /// `total_ns` is `per_probe_ns × observed_probe_count`. The sum
    /// of `by_family.total_ns` is what `main_wall_ns_corrected`
    /// subtracts from `main_wall_ns`.
    by_family: JsonProbeOverheadByFamily,
}

#[derive(serde::Serialize)]
struct JsonProbeOverheadByFamily {
    enter_exit: JsonProbeFamily,
    call: JsonProbeFamily,
    marshal: JsonProbeFamily,
    xcend: JsonProbeFamily,
    loop_iter: JsonProbeFamily,
    total_ns: u64,
}

#[derive(serde::Serialize)]
struct JsonProbeFamily {
    per_probe_ns: u64,
    probe_count: u64,
    total_ns: u64,
}

#[derive(serde::Serialize)]
struct JsonReport<'a> {
    main_fid: u32,
    main_ticks: u64,
    main_wall_ns: u64,
    main_wall_ns_corrected: u64,
    profile_overhead: JsonProbeOverhead,
    functions: Vec<JsonFunctionEntry<'a>>,
    op_kind_executed: Vec<(String, u128)>,
    simd: JsonSimd,
    marshal: JsonMarshal,
    transcendental: JsonTranscendental,
    call_graph: Vec<JsonEdge>,
    loop_stats: Vec<JsonLoopStat>,
    /// Per-main-tick wall ns. Populated only when debug mode was
    /// enabled at compile time. Empty otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    main_tick_waveform: Vec<u64>,
    /// Per-op-category sampled wall time. Populated only when debug
    /// mode was enabled at compile time. Each entry has `name`,
    /// `sample_count` (bracketed emissions), and `total_ns`
    /// (estimated full-sim ns = sampled ns × sample rate).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_category_timing: Vec<JsonOpCategoryTiming>,
}

#[derive(serde::Serialize)]
struct JsonOpCategoryTiming {
    name: &'static str,
    sample_count: u64,
    total_ns: u64,
}

/// Interpolate the value at percentile `pct` (0.0..1.0) from a
/// histogram. Returns the lower edge of the matching bucket as a
/// conservative estimate; for a log-scale histogram this is accurate
/// within ~20% per-bucket width.
fn percentile_from_histogram(hist: &[u64; HISTOGRAM_BUCKETS], pct: f64) -> u64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = (total as f64 * pct).ceil() as u64;
    let mut cum = 0u64;
    for (i, &c) in hist.iter().enumerate() {
        cum = cum.saturating_add(c);
        if cum >= target {
            return histogram_bucket_lo_ns(i) as u64;
        }
    }
    histogram_bucket_lo_ns(HISTOGRAM_BUCKETS - 1) as u64
}

/// Compute the corrected wall time for a `FunctionStats` given the
/// calibrated probe-pair cost. Saturating-subtracts 2 × calls × overhead
/// (one enter, one exit) from the raw `total_ns`.
#[inline]
fn corrected_total_ns(stats: &FunctionStats, overhead_ns: u64) -> u64 {
    stats
        .total_ns
        .saturating_sub(stats.calls.saturating_mul(2).saturating_mul(overhead_ns))
}

/// Dump the final report to stderr, then reset the thread-local store.
/// Called by `CompiledModule::drop` — no-op when profiling was disabled.
pub(crate) fn dump_report(main_fid: u32) {
    if !enabled() {
        return;
    }

    // Run probe-overhead calibration before snapshotting TL_STATS so
    // the calibration fid doesn't leak into the report.
    let overhead_ns = calibrate_probe_overhead();

    // Merge current-thread TL_STATS with any exited-thread stats in
    // GLOBAL_STATS. For the single-threaded case the global map is
    // empty and the local copy is the whole report.
    let mut merged: HashMap<u32, FunctionStats> = HashMap::new();
    TL_STATS.with(|m| {
        for (k, v) in m.borrow().iter() {
            merged.insert(*k, v.clone());
        }
    });
    if let Ok(g) = global_stats().lock() {
        for (fid, s) in g.iter() {
            let entry = merged.entry(*fid).or_default();
            merge_stats_into(entry, s);
        }
    }
    let snapshot: Vec<(u32, FunctionStats)> = merged.into_iter().collect();

    if snapshot.is_empty() {
        // Profiling was enabled but no probes fired. Still emit a header
        // so the user knows they asked for a profile and didn't get one.
        eprintln!("[elodin-cranelift] profile: ELODIN_CRANELIFT_DEBUG_DIR set but no probes fired");
        return;
    }

    let reg = registry().lock().expect("profile registry poisoned");

    // Find the `main` entry if it's in the snapshot — it's our "total
    // tick time" reference.
    let main_stats = snapshot.iter().find(|(fid, _)| *fid == main_fid).cloned();

    // Aggregate probe overhead across every recorded function.
    let total_calls: u64 = snapshot.iter().map(|(_, s)| s.calls).sum();
    let total_overhead_ns = 2u64.saturating_mul(total_calls).saturating_mul(overhead_ns);

    eprintln!("[elodin-cranelift] profile report");
    eprintln!(
        "  probe overhead: enter/exit={}ns call={}ns marshal={}ns xcend={}ns loop_iter={}ns  (calibrated via {} iterations per family)",
        overhead_ns,
        PROBE_OVERHEAD_CALL_NS.load(Ordering::Relaxed),
        PROBE_OVERHEAD_MARSHAL_NS.load(Ordering::Relaxed),
        PROBE_OVERHEAD_XCEND_NS.load(Ordering::Relaxed),
        PROBE_OVERHEAD_LOOP_ITER_NS.load(Ordering::Relaxed),
        CALIBRATION_ITERS
    );
    if let Some((_, ref m)) = main_stats {
        let total_sec = m.total_ns as f64 / 1e9;
        let corrected_ns = corrected_total_ns(m, overhead_ns);
        let corrected_sec = corrected_ns as f64 / 1e9;
        let mean_us = if m.calls > 0 {
            m.total_ns as f64 / m.calls as f64 / 1e3
        } else {
            0.0
        };
        let corrected_mean_us = if m.calls > 0 {
            corrected_ns as f64 / m.calls as f64 / 1e3
        } else {
            0.0
        };
        let min_us = m.min_ns as f64 / 1e3;
        let max_us = m.max_ns as f64 / 1e3;
        let p50_us = percentile_from_histogram(&m.histogram, 0.50) as f64 / 1e3;
        let p95_us = percentile_from_histogram(&m.histogram, 0.95) as f64 / 1e3;
        let p99_us = percentile_from_histogram(&m.histogram, 0.99) as f64 / 1e3;
        let p999_us = percentile_from_histogram(&m.histogram, 0.999) as f64 / 1e3;
        eprintln!(
            "  wall (raw): {:.2} s   wall (corrected): {:.2} s   ticks: {}",
            total_sec, corrected_sec, m.calls
        );
        eprintln!(
            "  tick_time: min={:.1}us mean={:.1}us (corrected={:.1}us) max={:.1}us",
            min_us, mean_us, corrected_mean_us, max_us
        );
        eprintln!(
            "  tick_latency: p50={:.1}us p95={:.1}us p99={:.1}us p99.9={:.1}us",
            p50_us, p95_us, p99_us, p999_us
        );
    } else {
        eprintln!("  (main function stats not captured; only per-callee entries below)");
    }

    // Print per-op-category sampled wall time. Each category's ns is
    // multiplied by its specific sample rate (helper-level rate for
    // arith ops; `INNER_OP_SAMPLE_RATE` for inner-loop Load / Store
    // / StackAddr).
    if crate::op_sampler::op_times_enabled() {
        let mut rows: Vec<(&'static str, u64, u64, u64)> = Vec::new();
        for &cat in crate::op_sampler::OpCategory::all() {
            let samples = OP_CATEGORY_SAMPLES[cat.as_u32() as usize].load(Ordering::Relaxed);
            let raw_ns = OP_CATEGORY_NS[cat.as_u32() as usize].load(Ordering::Relaxed);
            let rate = crate::op_sampler::sample_rate_for(cat);
            let scaled_ns = raw_ns.saturating_mul(rate);
            if samples > 0 {
                rows.push((cat.name(), samples, rate, scaled_ns));
            }
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.3));
        if !rows.is_empty() {
            eprintln!("  per-op wall time (sampled):");
            for (name, samples, rate, total_ns) in rows {
                let total_ms = total_ns as f64 / 1e6;
                eprintln!(
                    "    {:10} {:>6} samples × rate={:>3} → est. {:>8.2} ms",
                    name, samples, rate, total_ms
                );
            }
        }
    }

    // Sort functions by cumulative total_ns descending.
    let mut sorted: Vec<(u32, FunctionStats)> = snapshot;
    sorted.sort_by_key(|(_, s)| std::cmp::Reverse(s.total_ns));

    // Total time excluding `main` (to compute % share of the work inside
    // `main`). If main wasn't instrumented, fall back to sum across all.
    let total_ns = main_stats
        .as_ref()
        .map(|(_, s)| s.total_ns)
        .unwrap_or_else(|| sorted.iter().map(|(_, s)| s.total_ns).sum::<u64>().max(1));

    eprintln!("  top functions by cumulative time:");
    for (fid, stats) in sorted.iter().take(15) {
        if Some(*fid) == main_stats.as_ref().map(|(f, _)| *f) {
            // Already shown as the header; skip from the top-N list.
            continue;
        }
        let name = reg
            .names
            .get(fid)
            .cloned()
            .unwrap_or_else(|| format!("fid_{fid}"));
        let total_sec = stats.total_ns as f64 / 1e9;
        let pct = 100.0 * stats.total_ns as f64 / total_ns as f64;
        let mean_us = if stats.calls > 0 {
            stats.total_ns as f64 / stats.calls as f64 / 1e3
        } else {
            0.0
        };
        let p99_us = percentile_from_histogram(&stats.histogram, 0.99) as f64 / 1e3;
        let excl_us = stats.exclusive_ns as f64 / 1e3;
        let in_calls_us = stats.time_in_calls_ns as f64 / 1e3;
        let total_us = stats.total_ns as f64 / 1e3;
        let inline_us = (total_us - in_calls_us).max(0.0);
        let vec_pct = reg
            .instr_stats
            .get(fid)
            .map(|s| {
                let tot = (s.scalar + s.vector).max(1);
                100.0 * s.vector as f64 / tot as f64
            })
            .unwrap_or(0.0);
        let abi_tag = match reg.abis.get(fid) {
            Some(FuncAbi::Scalar) => "",
            Some(FuncAbi::Pointer) => " (ptr-ABI)",
            None => "",
        };
        // `in_calls_us` > 0 when call-bracket probes fired; otherwise
        // the row shows 0 calls / 100% inline.
        eprintln!(
            "    {:<16}{:>8.2}s ({:>5.1}%)  {:>7} calls  mean={:>6.1}us  p99={:>6.1}us  excl={:>6.1}us  in_calls={:>6.1}us  inline={:>6.1}us  vec%={:>4.1}{}",
            name,
            total_sec,
            pct,
            stats.calls,
            mean_us,
            p99_us,
            excl_us / (stats.calls as f64).max(1.0),
            in_calls_us / (stats.calls as f64).max(1.0),
            inline_us / (stats.calls as f64).max(1.0),
            vec_pct,
            abi_tag
        );
    }

    // Per-op-category estimate. For each function we know the static
    // `op_kind_counts` from `INSTR_REPORT`; multiplying by runtime call
    // count gives an "ops executed by kind" estimate across the sim.
    let mut weighted_kinds: HashMap<&'static str, u128> = HashMap::new();
    let mut total_weighted_ops: u128 = 0;
    for (fid, stats) in &sorted {
        if let Some(s) = reg.instr_stats.get(fid) {
            for (&kind, &count) in &s.op_kind_counts {
                let w = stats.calls as u128 * count as u128;
                *weighted_kinds.entry(kind).or_insert(0) += w;
                total_weighted_ops += w;
            }
        }
    }
    if total_weighted_ops > 0 {
        let mut kinds_sorted: Vec<(&'static str, u128)> =
            weighted_kinds.iter().map(|(k, v)| (*k, *v)).collect();
        kinds_sorted.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        eprintln!("  per-op-category estimate (static × runtime call counts):");
        for (name, count) in kinds_sorted.iter().take(15) {
            let pct = 100.0 * *count as f64 / total_weighted_ops as f64;
            eprintln!("    {:<14} {:>14} executed  ({:>5.2}%)", name, count, pct);
        }
    }

    // Runtime-weighted SIMD utilization. Weighted by call count × static
    // IR instruction count, so a hot vectorized function contributes more
    // than a cold one.
    let (mut w_vec, mut w_tot) = (0u128, 0u128);
    for (fid, stats) in &sorted {
        if let Some(s) = reg.instr_stats.get(fid) {
            w_vec += stats.calls as u128 * s.vector as u128;
            w_tot += stats.calls as u128 * (s.scalar + s.vector) as u128;
        }
    }
    let static_total: (usize, usize) = reg
        .instr_stats
        .values()
        .fold((0, 0), |(s, v), e| (s + e.scalar, v + e.vector));
    let static_pct = if static_total.0 + static_total.1 > 0 {
        100.0 * static_total.1 as f64 / (static_total.0 + static_total.1) as f64
    } else {
        0.0
    };
    let runtime_pct = if w_tot > 0 {
        100.0 * w_vec as f64 / w_tot as f64
    } else {
        0.0
    };
    eprintln!("  simd utilization:");
    eprintln!("    static (unweighted):   {:>5.1}% vector", static_pct);
    eprintln!("    runtime-weighted:      {:>5.1}% vector", runtime_pct);
    if runtime_pct > static_pct {
        let ratio = if static_pct > 0.0 {
            runtime_pct / static_pct
        } else {
            0.0
        };
        eprintln!(
            "    → vectorized functions are {:.2}x more frequently called than average",
            ratio
        );
    }

    // Cross-ABI marshal section. Reports how many scalar↔pointer
    // boundary crossings happened, how many bytes were copied, and
    // the actual wall time spent inside marshal.
    let s2p_calls = MARSHAL_S2P_CALLS.load(Ordering::Relaxed);
    let s2p_bytes = MARSHAL_S2P_BYTES.load(Ordering::Relaxed);
    let s2p_ns = MARSHAL_S2P_NS.load(Ordering::Relaxed);
    let p2s_calls = MARSHAL_P2S_CALLS.load(Ordering::Relaxed);
    let p2s_bytes = MARSHAL_P2S_BYTES.load(Ordering::Relaxed);
    let p2s_ns = MARSHAL_P2S_NS.load(Ordering::Relaxed);
    if s2p_calls + p2s_calls > 0 {
        eprintln!("  cross-ABI marshal:");
        eprintln!(
            "    scalar→pointer: {:>10} calls, {:>12} bytes, {:>10.3} ms total",
            s2p_calls,
            s2p_bytes,
            s2p_ns as f64 / 1e6
        );
        eprintln!(
            "    pointer→scalar: {:>10} calls, {:>12} bytes, {:>10.3} ms total",
            p2s_calls,
            p2s_bytes,
            p2s_ns as f64 / 1e6
        );
    }

    // Transcendental split section. Tracks whether the wide-SIMD
    // path displaced the libm scalar fallback.
    let libm_calls = XCEND_LIBM_CALLS.load(Ordering::Relaxed);
    let libm_ns = XCEND_LIBM_NS.load(Ordering::Relaxed);
    let simd_calls = XCEND_SIMD_CALLS.load(Ordering::Relaxed);
    let simd_ns = XCEND_SIMD_NS.load(Ordering::Relaxed);
    if libm_calls + simd_calls > 0 {
        let total = libm_calls + simd_calls;
        let simd_pct = 100.0 * simd_calls as f64 / total as f64;
        eprintln!("  transcendental calls:");
        eprintln!(
            "    libm scalar fallback:  {:>10} calls, {:>10.3} ms total",
            libm_calls,
            libm_ns as f64 / 1e6
        );
        eprintln!(
            "    wide-SIMD batch:       {:>10} calls, {:>10.3} ms total  ({:>5.1}% of all xcend)",
            simd_calls,
            simd_ns as f64 / 1e6,
            simd_pct
        );
    }

    // Hot edges: top parent→callee pairs by cumulative inclusive time.
    let edges_snapshot: Vec<((u32, u32), EdgeStats)> =
        TL_EDGES.with(|e| e.borrow().iter().map(|(k, v)| (*k, v.clone())).collect());
    if !edges_snapshot.is_empty() {
        let mut edges_sorted = edges_snapshot.clone();
        edges_sorted.sort_by_key(|(_, s)| std::cmp::Reverse(s.total_ns));
        eprintln!("  hot edges (parent → callee, by cumulative time):");
        for ((pfid, cfid), stats) in edges_sorted.iter().take(10) {
            let pname = reg
                .names
                .get(pfid)
                .cloned()
                .unwrap_or_else(|| format!("fid_{pfid}"));
            let cname = reg
                .names
                .get(cfid)
                .cloned()
                .unwrap_or_else(|| format!("fid_{cfid}"));
            eprintln!(
                "    {:<16} → {:<16}  {:>7} calls, {:>8.3} ms total",
                pname,
                cname,
                stats.calls,
                stats.total_ns as f64 / 1e6
            );
        }
    }

    // Loop-aware dynamic op counts. Augments `op_kind_executed` by
    // adding `iters * body_ops` per recorded loop; the section above
    // uses static-only counts, this addendum shows how the loops
    // shift the picture.
    let loop_iters_snapshot: HashMap<u32, u64> = loop_iter_counters()
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    let loop_stats_for_report: Vec<(u32, u64, LoopStatic)> = reg
        .loops
        .iter()
        .map(|(loop_id, static_info)| {
            let iters = loop_iters_snapshot.get(loop_id).copied().unwrap_or(0);
            (*loop_id, iters, static_info.clone())
        })
        .collect();
    if !loop_stats_for_report.is_empty() {
        let total_iters: u64 = loop_stats_for_report.iter().map(|(_, i, _)| *i).sum();
        if total_iters > 0 {
            eprintln!("  loop iterations (dynamic):");
            let mut sorted_loops = loop_stats_for_report.clone();
            sorted_loops.sort_by_key(|(_, i, _)| std::cmp::Reverse(*i));
            for (loop_id, iters, static_info) in sorted_loops.iter().take(10) {
                let parent = reg
                    .names
                    .get(&static_info.parent_fid)
                    .cloned()
                    .unwrap_or_else(|| format!("fid_{}", static_info.parent_fid));
                eprintln!(
                    "    loop#{:<4} in {:<20} iters={:>10}",
                    loop_id, parent, iters
                );
            }
        }
    }

    // Per-tick waveform summary; the full array goes into JSON.
    let timeline_len = TL_TIMELINE.with(|t| t.borrow().len());
    if timeline_len > 0 {
        eprintln!(
            "  tick waveform: {} samples captured (JSON contains full array)",
            timeline_len
        );
    }

    // Structured JSON dump. When `ELODIN_CRANELIFT_DEBUG_DIR=/path` is
    // set, we write `profile.json` into that directory alongside the
    // human-readable stderr output. Downstream analysis tools (including
    // `libs/cranelift-mlir/scripts/diff_profile.py`) consume this format.
    if let Some(path) = crate::debug::dir().map(|d| d.join("profile.json")) {
        let timeline_snapshot: Vec<u64> = TL_TIMELINE.with(|t| t.borrow().clone());
        write_json_report(
            &path,
            main_fid,
            &main_stats,
            &sorted,
            total_ns,
            &reg,
            &weighted_kinds,
            static_pct,
            runtime_pct,
            s2p_calls,
            s2p_bytes,
            s2p_ns,
            p2s_calls,
            p2s_bytes,
            p2s_ns,
            libm_calls,
            libm_ns,
            simd_calls,
            simd_ns,
            overhead_ns,
            total_overhead_ns,
            &edges_snapshot,
            &loop_stats_for_report,
            &timeline_snapshot,
        );
    }

    // Clear the TL store so a subsequent compiled module starts clean.
    // Atomic marshal/xcend counters are process-global and intentionally
    // not reset here: if the same process ever re-enters `dump_report`
    // with a new module (unusual), the counts would carry over —
    // documented in PERFORMANCE.md. For the single-module case (the only
    // one we ship), this is exactly the right behavior.
    reset();
}

#[allow(clippy::too_many_arguments)]
fn write_json_report(
    path: &std::path::Path,
    main_fid: u32,
    main_stats: &Option<(u32, FunctionStats)>,
    sorted: &[(u32, FunctionStats)],
    total_ns: u64,
    reg: &std::sync::MutexGuard<'_, StaticRegistry>,
    weighted_kinds: &HashMap<&'static str, u128>,
    static_pct: f64,
    runtime_pct: f64,
    s2p_calls: u64,
    s2p_bytes: u64,
    s2p_ns: u64,
    p2s_calls: u64,
    p2s_bytes: u64,
    p2s_ns: u64,
    libm_calls: u64,
    libm_ns: u64,
    simd_calls: u64,
    simd_ns: u64,
    overhead_ns: u64,
    total_overhead_ns: u64,
    edges: &[((u32, u32), EdgeStats)],
    loop_stats: &[(u32, u64, LoopStatic)],
    timeline: &[u64],
) {
    let main_ticks = main_stats.as_ref().map(|(_, s)| s.calls).unwrap_or(0);
    let main_wall_ns = main_stats.as_ref().map(|(_, s)| s.total_ns).unwrap_or(0);

    // Compute the full-family probe-overhead attribution. Each
    // family's `total_ns` is per-probe cost × observed probe count.
    // Function enter/exit is 2 probes per call; the other families
    // are already per-pair (call / marshal / xcend) or
    // per-invocation (loop_iter).
    let call_per_pair = PROBE_OVERHEAD_CALL_NS.load(Ordering::Relaxed);
    let marshal_per_pair = PROBE_OVERHEAD_MARSHAL_NS.load(Ordering::Relaxed);
    let xcend_per_pair = PROBE_OVERHEAD_XCEND_NS.load(Ordering::Relaxed);
    let loop_iter_per_probe = PROBE_OVERHEAD_LOOP_ITER_NS.load(Ordering::Relaxed);
    let call_probe_count = CALL_PROBE_PAIRS.load(Ordering::Relaxed);
    let marshal_probe_count = s2p_calls + p2s_calls;
    let xcend_probe_count = libm_calls + simd_calls;
    let loop_iter_probe_count = LOOP_ITER_TOTAL.load(Ordering::Relaxed);
    let enter_exit_per_pair = overhead_ns;
    let enter_exit_probe_count: u64 = sorted.iter().map(|(_, s)| s.calls).sum();
    let enter_exit_total_ns = enter_exit_per_pair.saturating_mul(enter_exit_probe_count);
    let call_total_ns = call_per_pair.saturating_mul(call_probe_count);
    let marshal_total_ns = marshal_per_pair.saturating_mul(marshal_probe_count);
    let xcend_total_ns = xcend_per_pair.saturating_mul(xcend_probe_count);
    let loop_iter_total_ns = loop_iter_per_probe.saturating_mul(loop_iter_probe_count);
    let probe_family_total_ns = enter_exit_total_ns
        .saturating_add(call_total_ns)
        .saturating_add(marshal_total_ns)
        .saturating_add(xcend_total_ns)
        .saturating_add(loop_iter_total_ns);

    let probe_overhead_by_family = JsonProbeOverheadByFamily {
        enter_exit: JsonProbeFamily {
            per_probe_ns: enter_exit_per_pair,
            probe_count: enter_exit_probe_count,
            total_ns: enter_exit_total_ns,
        },
        call: JsonProbeFamily {
            per_probe_ns: call_per_pair,
            probe_count: call_probe_count,
            total_ns: call_total_ns,
        },
        marshal: JsonProbeFamily {
            per_probe_ns: marshal_per_pair,
            probe_count: marshal_probe_count,
            total_ns: marshal_total_ns,
        },
        xcend: JsonProbeFamily {
            per_probe_ns: xcend_per_pair,
            probe_count: xcend_probe_count,
            total_ns: xcend_total_ns,
        },
        loop_iter: JsonProbeFamily {
            per_probe_ns: loop_iter_per_probe,
            probe_count: loop_iter_probe_count,
            total_ns: loop_iter_total_ns,
        },
        total_ns: probe_family_total_ns,
    };

    // v4 corrected wall subtracts the SUM of all probe families, not
    // just enter/exit. This gives a far more realistic "probe-free"
    // number for comparison against pre-instrumentation baselines.
    let main_wall_ns_corrected = main_wall_ns.saturating_sub(probe_family_total_ns);

    let functions: Vec<JsonFunctionEntry> = sorted
        .iter()
        .map(|(fid, s)| {
            let name = reg.names.get(fid).map(|s| s.as_str()).unwrap_or("unknown");
            let abi = match reg.abis.get(fid) {
                Some(FuncAbi::Scalar) => "scalar",
                Some(FuncAbi::Pointer) => "pointer",
                None => "unknown",
            };
            let (scalar_i, vector_i) = reg
                .instr_stats
                .get(fid)
                .map(|s| (s.scalar, s.vector))
                .unwrap_or((0, 0));
            let vec_pct = if scalar_i + vector_i > 0 {
                100.0 * vector_i as f64 / (scalar_i + vector_i) as f64
            } else {
                0.0
            };
            let mean_ns = if s.calls > 0 { s.total_ns / s.calls } else { 0 };
            let total_pct = if total_ns > 0 {
                100.0 * s.total_ns as f64 / total_ns as f64
            } else {
                0.0
            };
            JsonFunctionEntry {
                fid: *fid,
                name,
                abi,
                calls: s.calls,
                total_ns: s.total_ns,
                corrected_total_ns: corrected_total_ns(s, overhead_ns),
                exclusive_ns: s.exclusive_ns,
                time_in_calls_ns: s.time_in_calls_ns,
                min_ns: s.min_ns,
                max_ns: s.max_ns,
                mean_ns,
                p50_ns: percentile_from_histogram(&s.histogram, 0.50),
                p95_ns: percentile_from_histogram(&s.histogram, 0.95),
                p99_ns: percentile_from_histogram(&s.histogram, 0.99),
                p99_9_ns: percentile_from_histogram(&s.histogram, 0.999),
                scalar_instrs: scalar_i,
                vector_instrs: vector_i,
                vec_pct,
                total_pct,
                histogram: s.histogram,
            }
        })
        .collect();

    let mut op_kind_sorted: Vec<(String, u128)> = weighted_kinds
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    op_kind_sorted.sort_by_key(|(_, v)| std::cmp::Reverse(*v));

    // Serialize edges.
    let edge_entries: Vec<JsonEdge> = {
        let mut v: Vec<JsonEdge> = edges
            .iter()
            .map(|((pfid, cfid), stats)| JsonEdge {
                parent_fid: *pfid,
                parent_name: reg
                    .names
                    .get(pfid)
                    .cloned()
                    .unwrap_or_else(|| format!("fid_{pfid}")),
                callee_fid: *cfid,
                callee_name: reg
                    .names
                    .get(cfid)
                    .cloned()
                    .unwrap_or_else(|| format!("fid_{cfid}")),
                calls: stats.calls,
                total_ns: stats.total_ns,
            })
            .collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.total_ns));
        v
    };

    let loop_entries: Vec<JsonLoopStat> = loop_stats
        .iter()
        .map(|(loop_id, iters, static_info)| JsonLoopStat {
            loop_id: *loop_id,
            parent_fid: static_info.parent_fid,
            parent_name: reg
                .names
                .get(&static_info.parent_fid)
                .cloned()
                .unwrap_or_else(|| format!("fid_{}", static_info.parent_fid)),
            iters: *iters,
            body_ops: static_info
                .body_ops
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v as u64))
                .collect(),
        })
        .collect();

    let report = JsonReport {
        main_fid,
        main_ticks,
        main_wall_ns,
        main_wall_ns_corrected,
        profile_overhead: JsonProbeOverhead {
            measured_ns_per_probe: overhead_ns,
            total_overhead_ns: probe_family_total_ns,
            calibration_iters: CALIBRATION_ITERS,
            by_family: probe_overhead_by_family,
        },
        functions,
        op_kind_executed: op_kind_sorted,
        simd: JsonSimd {
            static_vector_pct: static_pct,
            runtime_weighted_vector_pct: runtime_pct,
        },
        marshal: JsonMarshal {
            scalar_to_pointer_calls: s2p_calls,
            scalar_to_pointer_bytes: s2p_bytes,
            scalar_to_pointer_total_ns: s2p_ns,
            pointer_to_scalar_calls: p2s_calls,
            pointer_to_scalar_bytes: p2s_bytes,
            pointer_to_scalar_total_ns: p2s_ns,
        },
        transcendental: JsonTranscendental {
            libm_scalar_calls: libm_calls,
            libm_total_ns: libm_ns,
            wide_simd_calls: simd_calls,
            wide_simd_total_ns: simd_ns,
        },
        call_graph: edge_entries,
        loop_stats: loop_entries,
        main_tick_waveform: timeline.to_vec(),
        op_category_timing: {
            let mut out = Vec::new();
            for &cat in crate::op_sampler::OpCategory::all() {
                let sample_count =
                    OP_CATEGORY_SAMPLES[cat.as_u32() as usize].load(Ordering::Relaxed);
                let raw_ns = OP_CATEGORY_NS[cat.as_u32() as usize].load(Ordering::Relaxed);
                // Multiply by the category-specific sample rate so
                // the JSON `total_ns` reflects an estimated full-sim
                // total and matches the stderr display. Readers can
                // divide by `sample_count × rate` for raw-per-sample.
                let rate = crate::op_sampler::sample_rate_for(cat);
                let total_ns = raw_ns.saturating_mul(rate);
                if sample_count > 0 {
                    out.push(JsonOpCategoryTiming {
                        name: cat.name(),
                        sample_count,
                        total_ns,
                    });
                }
            }
            out.sort_by_key(|e| std::cmp::Reverse(e.total_ns));
            out
        },
    };

    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!(
                    "[elodin-cranelift] profile: failed to write JSON to {}: {}",
                    path.display(),
                    e
                );
            } else {
                eprintln!(
                    "[elodin-cranelift] profile: JSON report written to {}",
                    path.display()
                );
            }
        }
        Err(e) => {
            eprintln!(
                "[elodin-cranelift] profile: failed to serialize JSON: {}",
                e
            );
        }
    }
}

/// Clear the thread-local stats store and the process-global marshal /
/// transcendental / loop counters. Called by `dump_report` after
/// formatting. Exposed for tests.
pub(crate) fn reset() {
    TL_STATS.with(|m| m.borrow_mut().clear());
    TL_STACK.with(|s| s.borrow_mut().clear());
    TL_EDGES.with(|e| e.borrow_mut().clear());
    TL_TIMELINE.with(|t| t.borrow_mut().clear());
    TL_MARSHAL_STACK.with(|s| s.borrow_mut().clear());
    TL_XCEND_STACK.with(|s| s.borrow_mut().clear());
    TL_CALL_STACK.with(|s| s.borrow_mut().clear());
    if let Ok(mut g) = global_stats().lock() {
        g.clear();
    }
    MARSHAL_S2P_CALLS.store(0, Ordering::Relaxed);
    MARSHAL_S2P_BYTES.store(0, Ordering::Relaxed);
    MARSHAL_S2P_NS.store(0, Ordering::Relaxed);
    MARSHAL_P2S_CALLS.store(0, Ordering::Relaxed);
    MARSHAL_P2S_BYTES.store(0, Ordering::Relaxed);
    MARSHAL_P2S_NS.store(0, Ordering::Relaxed);
    XCEND_LIBM_CALLS.store(0, Ordering::Relaxed);
    XCEND_LIBM_NS.store(0, Ordering::Relaxed);
    XCEND_SIMD_CALLS.store(0, Ordering::Relaxed);
    XCEND_SIMD_NS.store(0, Ordering::Relaxed);
    PROBE_OVERHEAD_NS.store(0, Ordering::Relaxed);
    PROBE_OVERHEAD_CALL_NS.store(0, Ordering::Relaxed);
    PROBE_OVERHEAD_MARSHAL_NS.store(0, Ordering::Relaxed);
    PROBE_OVERHEAD_XCEND_NS.store(0, Ordering::Relaxed);
    PROBE_OVERHEAD_LOOP_ITER_NS.store(0, Ordering::Relaxed);
    CALL_PROBE_PAIRS.store(0, Ordering::Relaxed);
    LOOP_ITER_TOTAL.store(0, Ordering::Relaxed);
    for i in 0..OP_CATEGORY_COUNT {
        OP_CATEGORY_NS[i].store(0, Ordering::Relaxed);
        OP_CATEGORY_SAMPLES[i].store(0, Ordering::Relaxed);
    }
    TL_OP_STACK.with(|s| s.borrow_mut().clear());
    if let Ok(mut m) = loop_iter_counters().lock() {
        m.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `flush_current_thread` moves TL_STATS from a
    /// worker thread into GLOBAL_STATS so the main thread's
    /// `dump_report` can merge it in.
    #[test]
    fn multi_thread_merge_flushes_explicit_thread_stats() {
        // Reset any leftover state from other tests.
        reset();

        // Worker: call enter+exit a few times, then flush and exit.
        let handle = std::thread::spawn(|| {
            for _ in 0..5 {
                __cranelift_profile_enter(42);
                std::thread::sleep(std::time::Duration::from_nanos(100));
                __cranelift_profile_exit(42);
            }
            // Explicit flush — the multi-threaded contract.
            flush_current_thread();
        });
        handle.join().unwrap();

        let g = global_stats().lock().unwrap();
        let entry = g
            .get(&42)
            .expect("worker stats must have been flushed via flush_current_thread");
        assert_eq!(entry.calls, 5);
        assert!(entry.total_ns > 0);
    }

    #[test]
    fn histogram_bucket_monotone() {
        // Verify bucket ordering: small values map to low buckets,
        // large values to high buckets. Guards against log-scale
        // arithmetic regressions in future changes.
        let b0 = histogram_bucket(50); // below floor → bucket 0
        let b1 = histogram_bucket(1_000); // 1 µs
        let b2 = histogram_bucket(1_000_000); // 1 ms
        let b3 = histogram_bucket(1_000_000_000); // 1 s → above ceiling → last bucket
        assert_eq!(b0, 0);
        assert!(b1 < b2);
        assert_eq!(b3, HISTOGRAM_BUCKETS - 1);
    }

    #[test]
    fn percentile_from_histogram_basic() {
        let mut h = [0u64; HISTOGRAM_BUCKETS];
        // Put 100 samples at bucket 10 (mid-range).
        h[10] = 100;
        let p50 = percentile_from_histogram(&h, 0.5);
        // p50 should be at the low edge of bucket 10.
        assert_eq!(p50, histogram_bucket_lo_ns(10) as u64);
    }
}
