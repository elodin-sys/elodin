// Lowering helpers thread the Cranelift FunctionBuilder, JITModule,
// IR Module, symbol tables (libm/tensor-rt ids, func_ids, func_abis),
// value/type maps, and the slot pool by hand. Bundling these into a
// LoweringContext struct is future work; see ARCHITECTURE.md.
#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{
    AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind, Type, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{ArenaMemoryProvider, JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use crate::ir::*;
use crate::tensor_rt;

pub struct CompiledModule {
    module: JITModule,
    main_fn_id: FuncId,
    /// Millisecond-level breakdown of the three compile phases, captured
    /// during `compile_module_with_config`. Read by callers like
    /// `nox-py`'s `cranelift_compile.rs` to surface timings in the
    /// startup log so the parallel-codegen speedup is observable.
    pub timings: CompileTimings,
}

/// Per-phase timings for a single call to `compile_module_with_config`.
///
/// Populated during compile and exposed on [`CompiledModule::timings`]. All
/// fields are in milliseconds, captured with `Instant::now`. Phase 2
/// (codegen) measures the wall-clock elapsed on the calling thread while
/// the rayon thread pool is active; on an N-core machine with well-balanced
/// work, this is roughly `sum_of_per_fn_codegen / N`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompileTimings {
    /// Phase 1 — lowering IR functions to Cranelift IR (serial).
    pub ir_build_ms: f64,
    /// Phase 2 — parallel Cranelift `Context::compile` across all functions.
    pub codegen_ms: f64,
    /// Phase 3 — serial `define_function_bytes` + `finalize_definitions`.
    pub link_ms: f64,
}

impl CompiledModule {
    pub fn get_main_fn(&self) -> *const u8 {
        self.module.get_finalized_function(self.main_fn_id)
    }
}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        // When ELODIN_CRANELIFT_DEBUG_DIR is set, runtime profile probes
        // have been accumulating per-function stats into a thread-local
        // store. Drop is the simulation's natural exit point — dump the
        // report there so the customer sees it on stderr without needing
        // to call anything explicitly. Harmless no-op when debug mode is
        // disabled.
        crate::profile::dump_report(self.main_fn_id.as_u32());
    }
}

#[derive(Default, Clone)]
pub struct CompileConfig {
    /// When `true`, `compile_module_with_config` emits
    /// `__cranelift_profile_enter` / `__cranelift_profile_exit` calls into
    /// every instrumented function's Cranelift IR. Driven by the
    /// presence of `ELODIN_CRANELIFT_DEBUG_DIR` via
    /// `CompileConfig::from_env()`. When `false` (default) the emitted
    /// IR is bit-identical to a plain build.
    pub profile_enabled: bool,
    /// Enable sampled per-op wall-time instrumentation. Requires
    /// `profile_enabled`. Adds tiny probe pairs around every Nth
    /// emission of a small set of hot ops (F64X2 load/store, fadd/
    /// fmul/fsub/fdiv, stack_addr, iconst). Driven by the same
    /// `ELODIN_CRANELIFT_DEBUG_DIR` gate as `profile_enabled`.
    pub profile_op_times: bool,
    /// Internal test hook — not gated by any env var. When `true`,
    /// the `main` function is lowered through the pointer-ABI path
    /// instead of the default scalar-ABI path. Used by `run_mlir_mem`
    /// in the in-crate tests to exercise ptr-ABI lowering on small
    /// tensors that would otherwise be classified scalar-ABI.
    /// Always `false` in production configs (including `from_env`).
    pub force_pointer_abi_main: bool,
}

impl CompileConfig {
    /// Read the environment to construct a config. The single
    /// `ELODIN_CRANELIFT_DEBUG_DIR` env var gates every diagnostic
    /// feature at once — no rebuild needed.
    pub fn from_env() -> Self {
        let debug = crate::debug::enabled();
        Self {
            profile_enabled: debug,
            profile_op_times: debug,
            force_pointer_abi_main: false,
        }
    }
}

type TensorVals = Vec<Value>;

// ---------------------------------------------------------------------------
// LaneRepr — per-value representation stored in the lowering value map.
//
// `Scalar` is the traditional one-SSA-per-element form. `PackedF64` is the
// scalar-ABI SIMD form: F64X2 chunks plus an optional f64 tail for odd
// lane counts. `PtrChunksF64` is the ptr-ABI SSA form used by the
// result-write elision path. Producers emit whichever variant is cheapest
// for the consumer; `unpack_in` / `spill_to_slot` convert back when a
// downstream op can only consume a narrower form.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum LaneRepr {
    /// Traditional scalar representation: one SSA `Value` per tensor element.
    /// In pointer-ABI contexts this is `Scalar(vec![ptr])` — a single value
    /// holding the stack-slot pointer for the whole tensor.
    Scalar(Vec<Value>),
    /// Packed F64X2 representation. Used only for `ElementType::F64` tensors
    /// in scalar-ABI functions. `chunks[i]` is an F64X2 holding lanes
    /// `(2i, 2i+1)` of the original tensor. `tail` holds the last lane when
    /// `n` is odd. `n` is the total element count.
    PackedF64 {
        chunks: Vec<Value>,
        tail: Option<Value>,
        n: usize,
    },
    /// Pointer-ABI result that lives entirely in SSA registers.
    /// `chunks[i]` is an F64X2 holding lanes `(2i, 2i+1)`; `tail` holds
    /// the last lane when `n` is odd.
    ///
    /// Emitted when dataflow analysis proves the result is consumed
    /// exactly once by an elision-friendly op (see
    /// `ELISION_FRIENDLY_USER_KINDS`). Elision-aware consumers read
    /// chunks via `get_chunks`; non-elision-aware consumers call
    /// `as_scalar_or_spill`, which lazily writes the chunks to a fresh
    /// stack slot and replaces `self` with `Scalar([ptr])`.
    ///
    /// `chain_depth` counts back-to-back elision-friendly ops (root = 1,
    /// each hop increments) and is bounded by `ELISION_MAX_CHAIN` so long
    /// chains spill before register pressure exceeds the NEON 32-v-reg
    /// or x86 16-xmm budget.
    ///
    /// Never appears in scalar-ABI bodies — those use `PackedF64`.
    PtrChunksF64 {
        chunks: Vec<Value>,
        tail: Option<Value>,
        n: usize,
        chain_depth: u32,
    },
}

impl LaneRepr {
    fn scalar(vals: Vec<Value>) -> Self {
        Self::Scalar(vals)
    }

    /// Construct a `PtrChunksF64` from freshly-emitted F64X2 chunks and
    /// an optional f64 tail. `chain_depth` is the running count of
    /// back-to-back elision-friendly ops, bounded by the caller.
    fn ptr_chunks(chunks: Vec<Value>, tail: Option<Value>, n: usize, chain_depth: u32) -> Self {
        Self::PtrChunksF64 {
            chunks,
            tail,
            n,
            chain_depth,
        }
    }

    /// The chain depth of a pointer-ABI value (see
    /// `LaneRepr::PtrChunksF64::chain_depth`). Scalar and PackedF64 are
    /// depth 0 — they sit in memory or scalar-ABI lanes.
    fn ptr_chain_depth(&self) -> u32 {
        match self {
            Self::PtrChunksF64 { chain_depth, .. } => *chain_depth,
            _ => 0,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Scalar(v) => v.len(),
            Self::PackedF64 { n, .. } | Self::PtrChunksF64 { n, .. } => *n,
        }
    }

    /// Borrow as a scalar slice. Panics if the repr is `PackedF64` or
    /// `PtrChunksF64` — callers that might encounter those must use
    /// `unpack_in` / `as_scalar_or_spill` first.
    fn as_scalar(&self) -> &[Value] {
        match self {
            Self::Scalar(v) => v.as_slice(),
            Self::PackedF64 { .. } => {
                panic!("LaneRepr::as_scalar on PackedF64 — call unpack_in(builder) first")
            }
            Self::PtrChunksF64 { chain_depth, n, .. } => {
                panic!(
                    "LaneRepr::as_scalar on PtrChunksF64 (n={n}, chain_depth={chain_depth}) — call as_scalar_or_spill(builder) first"
                )
            }
        }
    }

    /// If the repr is packed, emit extractlane instructions to materialize the
    /// scalar form and replace `self` with `Scalar(...)`. Idempotent on
    /// already-scalar values. Does not handle `PtrChunksF64` — ptr-ABI values
    /// use `spill_to_slot` instead (scalar-ABI paths never see PtrChunksF64).
    fn unpack_in(&mut self, builder: &mut FunctionBuilder) {
        if let Self::PackedF64 { chunks, tail, n } = self {
            let mut out = Vec::with_capacity(*n);
            for &v in chunks.iter() {
                out.push(builder.ins().extractlane(v, 0u8));
                out.push(builder.ins().extractlane(v, 1u8));
            }
            if let Some(t) = tail {
                out.push(*t);
            }
            out.truncate(*n);
            *self = Self::Scalar(out);
        }
    }

    /// If `self` is `PtrChunksF64`, allocate a fresh stack slot of
    /// `n * 8` bytes, write every chunk + tail into it, and replace
    /// `self` with `Scalar([ptr])`. Idempotent on already-`Scalar`
    /// values (no-op). Panics on `PackedF64` — the ptr and scalar ABIs
    /// are disjoint by design. Returns the slot pointer for callers
    /// that want it without a second map lookup.
    fn spill_to_slot(&mut self, builder: &mut FunctionBuilder) -> Value {
        self.spill_to_slot_pooled(builder, None, None)
    }

    /// Pool-aware variant of `spill_to_slot`. When both `pool` and
    /// `owner_vid` are `Some`, the spilled slot is allocated through
    /// the pool and released to the free-list when `owner_vid`'s
    /// `last_use_pos` is reached. Falls back to raw `alloc_slot`
    /// otherwise.
    fn spill_to_slot_pooled(
        &mut self,
        builder: &mut FunctionBuilder,
        pool: Option<&mut crate::slot_pool::SlotPool>,
        owner_vid: Option<ValueId>,
    ) -> Value {
        match self {
            Self::Scalar(v) => v
                .first()
                .copied()
                .expect("LaneRepr::spill_to_slot on empty Scalar"),
            Self::PackedF64 { .. } => {
                panic!("LaneRepr::spill_to_slot on PackedF64 — ptr and scalar ABIs are disjoint")
            }
            Self::PtrChunksF64 {
                chunks, tail, n, ..
            } => {
                let byte_size = *n * 8;
                let dst = match (pool, owner_vid) {
                    (Some(p), Some(vid)) => alloc_slot_for_vid(builder, p, Some(vid), byte_size),
                    _ => alloc_slot(builder, byte_size),
                };
                let flags = MemFlags::trusted();
                for (i, &chunk) in chunks.iter().enumerate() {
                    let off = (i * 16) as i32;
                    builder.ins().store(flags, chunk, dst, off);
                }
                if let Some(t) = tail {
                    let off = (chunks.len() * 16) as i32;
                    builder.ins().store(flags, *t, dst, off);
                }
                *self = Self::Scalar(vec![dst]);
                dst
            }
        }
    }

    /// Ensure `self` is in the scalar-pointer form expected by
    /// non-elision-aware consumers, spilling `PtrChunksF64` to a fresh
    /// stack slot if needed. Returns the scalar slice (`&[ptr]` for
    /// ptr-ABI values).
    fn as_scalar_or_spill(&mut self, builder: &mut FunctionBuilder) -> &[Value] {
        if matches!(self, Self::PtrChunksF64 { .. }) {
            self.spill_to_slot(builder);
        }
        self.as_scalar()
    }

    /// Expose the F64X2 chunks and optional tail that back a ptr-ABI
    /// value. Emits loads from the stack slot when `self` is
    /// `Scalar([ptr])` and returns the already-loaded chunks when
    /// `self` is `PtrChunksF64`. `n` is the element count — only used
    /// by the load path to know how many chunks to emit. The caller
    /// handles the odd-tail scalar op when `tail.is_some()`.
    fn get_chunks(&self, builder: &mut FunctionBuilder, n: usize) -> (Vec<Value>, Option<Value>) {
        match self {
            Self::PtrChunksF64 { chunks, tail, .. } => (chunks.clone(), *tail),
            Self::Scalar(v) => {
                let ptr = *v.first().expect("LaneRepr::get_chunks on empty Scalar");
                let full_chunks = n / 2;
                let tail_n = n % 2;
                let flags = MemFlags::trusted();
                let mut chunks = Vec::with_capacity(full_chunks);
                for i in 0..full_chunks {
                    let off = (i * 16) as i32;
                    chunks.push(builder.ins().load(types::F64X2, flags, ptr, off));
                }
                let tail = if tail_n == 1 {
                    let off = (full_chunks * 16) as i32;
                    Some(builder.ins().load(types::F64, flags, ptr, off))
                } else {
                    None
                };
                (chunks, tail)
            }
            Self::PackedF64 { .. } => {
                panic!("LaneRepr::get_chunks on PackedF64 — ptr and scalar ABIs are disjoint")
            }
        }
    }
}

/// Value map lookup returning a scalar slice. Auto-unpacks `PackedF64`
/// entries in place (emitting `extractlane` instructions) so consumers that
/// predate the SIMD work continue to operate on scalar `&[Value]`. The
/// unpack is idempotent — once unpacked, repeat lookups are cheap.
fn get_vals<'a>(
    builder: &mut FunctionBuilder,
    value_map: &'a mut HashMap<ValueId, LaneRepr>,
    vid: &ValueId,
) -> Result<&'a [Value], String> {
    let lr = value_map
        .get_mut(vid)
        .ok_or_else(|| format!("missing value {:?}", vid))?;
    lr.unpack_in(builder);
    match lr {
        LaneRepr::Scalar(v) => Ok(v.as_slice()),
        LaneRepr::PackedF64 { .. } => unreachable!("unpack_in left PackedF64"),
        LaneRepr::PtrChunksF64 { .. } => {
            // PtrChunksF64 only exists on the ptr-ABI path. Scalar-ABI
            // code reaches here via get_vals and never produces it.
            unreachable!("PtrChunksF64 leaked into scalar-ABI get_vals")
        }
    }
}

const LARGE_TENSOR_THRESHOLD: usize = 64;

fn max_reg_returns() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        2
    }
    #[cfg(target_arch = "aarch64")]
    {
        8
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        2
    }
}
/// Stack-slot alignment shift: `2^SLOT_ALIGN` bytes. Set to 4 (16
/// bytes) so ptr-ABI F64X2 loads/stores hit the aligned memory path
/// on x86 (`movapd`) and ARM NEON. Costs up to 8 bytes of padding
/// per non-f64 slot — in the noise against an arena-backed frame.
const SLOT_ALIGN: u8 = 4;
const MIN_RETURN_SLOT: usize = 8;
const LU_PIVOT_EPSILON: f64 = 1e-300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FuncAbi {
    Scalar,
    Pointer,
}

fn classify_function(func_def: &FuncDef) -> FuncAbi {
    if func_def.name == "main" {
        return FuncAbi::Scalar;
    }
    let param_max = func_def
        .params
        .iter()
        .map(|(_, t)| t.num_elements())
        .max()
        .unwrap_or(0);
    let ret_max = func_def
        .result_types
        .iter()
        .map(|t| t.num_elements())
        .max()
        .unwrap_or(0);
    let body_max = scan_body_max_elements(&func_def.body);
    if param_max.max(ret_max).max(body_max) > LARGE_TENSOR_THRESHOLD {
        FuncAbi::Pointer
    } else {
        FuncAbi::Scalar
    }
}

fn scan_body_max_elements(body: &[InstrResult]) -> usize {
    let mut max_elem = 0usize;
    for ir in body {
        for (_, ty) in &ir.values {
            max_elem = max_elem.max(ty.num_elements());
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                ..
            } => {
                max_elem = max_elem.max(scan_body_max_elements(cond_body));
                max_elem = max_elem.max(scan_body_max_elements(loop_body));
            }
            Instruction::Case { branches, .. } => {
                for branch in branches {
                    max_elem = max_elem.max(scan_body_max_elements(branch));
                }
            }
            _ => {}
        }
    }
    max_elem
}

fn classify_all_functions(ir_module: &crate::ir::Module) -> HashMap<String, FuncAbi> {
    let abis: HashMap<String, FuncAbi> = ir_module
        .functions
        .iter()
        .map(|f| (f.name.clone(), classify_function(f)))
        .collect();
    if crate::debug::enabled() {
        let (scalar, pointer) = abis
            .values()
            .fold((0usize, 0usize), |(s, p), abi| match abi {
                FuncAbi::Scalar => (s + 1, p),
                FuncAbi::Pointer => (s, p + 1),
            });
        eprintln!("[elodin-cranelift] abi classification: scalar={scalar} pointer={pointer}");
    }
    abis
}

// ---------------------------------------------------------------------------
// Scalar / vector instruction counter (ELODIN_CRANELIFT_DEBUG_DIR)
// ---------------------------------------------------------------------------
//
// Records per-function tallies of Cranelift IR instructions whose result type
// is a SIMD vector vs a scalar. Used as a cheap proxy for "did we actually
// vectorize this function?" as we roll out SIMD packing. Compilation is
// single-threaded so a thread-local accumulator is sufficient.

pub(crate) struct InstrCountEntry {
    pub name: String,
    pub abi: FuncAbi,
    pub scalar: usize,
    pub vector: usize,
    /// Per-opcode histogram keyed by Cranelift's `Opcode` display name
    /// (e.g. `"fadd"`, `"load"`, `"call"`). Consumed at report time to
    /// summarize which ops dominate each function, and weighted by
    /// runtime call counts by the live profiling layer (`profile.rs`).
    pub op_kind_counts: HashMap<&'static str, usize>,
}

thread_local! {
    pub(crate) static INSTR_COUNTS: RefCell<Vec<InstrCountEntry>> = const { RefCell::new(Vec::new()) };
}

/// Drive all the profiling layers: compile-time (`INSTR_REPORT`),
/// runtime (`PROFILE`). We always collect instr counts when either is on
/// because the live profiler uses them to compute runtime-weighted SIMD
/// utilization and the per-op-category breakdown.
fn collect_instr_counts_enabled() -> bool {
    instr_report_enabled() || profile_enabled()
}

fn instr_report_enabled() -> bool {
    crate::debug::enabled()
}

fn profile_enabled() -> bool {
    crate::debug::enabled()
}

fn count_scalar_and_vector_insts(
    func: &cranelift_codegen::ir::Function,
) -> (usize, usize, HashMap<&'static str, usize>) {
    let mut scalar = 0usize;
    let mut vector = 0usize;
    let mut op_kinds: HashMap<&'static str, usize> = HashMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let results = func.dfg.inst_results(inst);
            if results.iter().any(|&v| func.dfg.value_type(v).is_vector()) {
                vector += 1;
            } else {
                scalar += 1;
            }
            // `Opcode::variant_name` / Display gives a stable string like
            // "fadd"; we cast to &'static str via a lookup that returns
            // the variant name.
            let opcode = func.dfg.insts[inst].opcode();
            let name: &'static str = opcode_static_name(opcode);
            *op_kinds.entry(name).or_insert(0) += 1;
        }
    }
    (scalar, vector, op_kinds)
}

/// Map Cranelift's `Opcode` enum to a `&'static str` naming the variant
/// (e.g. `Opcode::Fadd` -> `"fadd"`). `Opcode` implements `Display` which
/// emits exactly that name, but `Display` returns an owned-style `String`
/// via the formatter; we want `&'static` keys for the histogram. We use
/// a sparse match on the common hot opcodes and fall back to a leaked
/// Box for anything unexpected (rare in practice; matched set covers
/// everything the JIT emits today).
fn opcode_static_name(op: cranelift_codegen::ir::Opcode) -> &'static str {
    use cranelift_codegen::ir::Opcode::*;
    match op {
        Fadd => "fadd",
        Fsub => "fsub",
        Fmul => "fmul",
        Fdiv => "fdiv",
        Fmin => "fmin",
        Fmax => "fmax",
        Fneg => "fneg",
        Fabs => "fabs",
        Sqrt => "sqrt",
        Fcmp => "fcmp",
        Iadd => "iadd",
        Isub => "isub",
        Imul => "imul",
        Sdiv => "sdiv",
        Udiv => "udiv",
        Srem => "srem",
        Urem => "urem",
        Band => "band",
        Bor => "bor",
        Bxor => "bxor",
        Bnot => "bnot",
        Icmp => "icmp",
        Ishl => "ishl",
        Ushr => "ushr",
        Sshr => "sshr",
        Select => "select",
        SelectSpectreGuard => "select_spectre_guard",
        Bitselect => "bitselect",
        Load => "load",
        Store => "store",
        StackAddr => "stack_addr",
        StackLoad => "stack_load",
        StackStore => "stack_store",
        Call => "call",
        CallIndirect => "call_indirect",
        Return => "return",
        Jump => "jump",
        Brif => "brif",
        BrTable => "br_table",
        Iconst => "iconst",
        F32const => "f32const",
        F64const => "f64const",
        Vconst => "vconst",
        Splat => "splat",
        Insertlane => "insertlane",
        Extractlane => "extractlane",
        Bitcast => "bitcast",
        Ireduce => "ireduce",
        Uextend => "uextend",
        Sextend => "sextend",
        Fpromote => "fpromote",
        Fdemote => "fdemote",
        FcvtToSint => "fcvt_to_sint",
        FcvtToUint => "fcvt_to_uint",
        FcvtFromSint => "fcvt_from_sint",
        FcvtFromUint => "fcvt_from_uint",
        Floor => "floor",
        Ceil => "ceil",
        Nearest => "nearest",
        Trunc => "trunc",
        _ => {
            // Fall back to leaking the runtime string. Only hit for
            // opcodes the JIT doesn't emit today; cost is a once-per-
            // process allocation per unmapped opcode. Acceptable.
            static FALLBACK: std::sync::OnceLock<std::sync::Mutex<HashMap<String, &'static str>>> =
                std::sync::OnceLock::new();
            let s = format!("{op:?}").to_ascii_lowercase();
            let map = FALLBACK.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
            let mut guard = map.lock().unwrap();
            if let Some(&name) = guard.get(&s) {
                return name;
            }
            let leaked: &'static str = Box::leak(s.clone().into_boxed_str());
            guard.insert(s, leaked);
            leaked
        }
    }
}

fn record_instr_counts(name: &str, abi: FuncAbi, func: &cranelift_codegen::ir::Function) {
    if !collect_instr_counts_enabled() {
        return;
    }
    let (scalar, vector, op_kind_counts) = count_scalar_and_vector_insts(func);
    INSTR_COUNTS.with(|c| {
        c.borrow_mut().push(InstrCountEntry {
            name: name.to_string(),
            abi,
            scalar,
            vector,
            op_kind_counts,
        });
    });
}

fn reset_instr_counts() {
    INSTR_COUNTS.with(|c| c.borrow_mut().clear());
}

fn print_instr_report() {
    if !instr_report_enabled() {
        return;
    }
    INSTR_COUNTS.with(|c| {
        let counts = c.borrow();
        if counts.is_empty() {
            return;
        }
        let (total_scalar, total_vector) = counts
            .iter()
            .fold((0usize, 0usize), |(s, v), e| (s + e.scalar, v + e.vector));
        let total = total_scalar + total_vector;
        let vec_pct = if total > 0 {
            100.0 * total_vector as f64 / total as f64
        } else {
            0.0
        };
        eprintln!(
            "[elodin-cranelift] instr report: {} funcs, scalar={total_scalar} vector={total_vector} total={total} vec%={vec_pct:.1}",
            counts.len()
        );

        // Top-10 scalar-ABI functions by vector instruction count, plus any that have zero
        // vector instructions among the top-10 by scalar count (noisy functions to investigate).
        let mut by_vector: Vec<&InstrCountEntry> = counts
            .iter()
            .filter(|e| e.abi == FuncAbi::Scalar && e.vector > 0)
            .collect();
        by_vector.sort_by_key(|e| std::cmp::Reverse(e.vector));
        for e in by_vector.iter().take(10) {
            eprintln!(
                "  [vec] {}: scalar={} vector={} (vec%={:.1}) {}",
                e.name,
                e.scalar,
                e.vector,
                100.0 * e.vector as f64 / (e.scalar + e.vector).max(1) as f64,
                format_top_opcodes(&e.op_kind_counts, 5),
            );
        }

        let mut scalar_abi_by_size: Vec<&InstrCountEntry> = counts
            .iter()
            .filter(|e| e.abi == FuncAbi::Scalar)
            .collect();
        scalar_abi_by_size.sort_by_key(|e| std::cmp::Reverse(e.scalar + e.vector));
        for e in scalar_abi_by_size.iter().take(10) {
            eprintln!(
                "  [big] {}: scalar={} vector={} size={} {}",
                e.name,
                e.scalar,
                e.vector,
                e.scalar + e.vector,
                format_top_opcodes(&e.op_kind_counts, 5),
            );
        }

        // Aggregate opcode histogram across all functions. Tells us which
        // ops dominate the whole module by static count — pairs naturally
        // with the runtime-weighted view the live profiler produces.
        let mut total_kinds: HashMap<&'static str, usize> = HashMap::new();
        for e in counts.iter() {
            for (k, v) in &e.op_kind_counts {
                *total_kinds.entry(k).or_insert(0) += v;
            }
        }
        let mut kinds_sorted: Vec<(&'static str, usize)> =
            total_kinds.iter().map(|(k, v)| (*k, *v)).collect();
        kinds_sorted.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        eprintln!("  [ops] top opcodes across all functions:");
        for (name, count) in kinds_sorted.iter().take(15) {
            eprintln!("        {:>10} = {}", name, count);
        }
    });
    // Only clear when profiling isn't going to consume them later in
    // `print_profile_report`. When PROFILE=1 the live layer needs the
    // static data still available at sim exit.
    if !profile_enabled() {
        reset_instr_counts();
    }
}

/// Render the top-N opcodes for a function as `op1=N1 op2=N2 op3=N3`.
fn format_top_opcodes(counts: &HashMap<&'static str, usize>, n: usize) -> String {
    let mut pairs: Vec<(&'static str, usize)> = counts.iter().map(|(k, v)| (*k, *v)).collect();
    pairs.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    pairs
        .iter()
        .take(n)
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn cranelift_type_for(et: ElementType) -> Type {
    match et {
        ElementType::F64 => types::F64,
        ElementType::F32 => types::F32,
        ElementType::I1 => types::I8,
        ElementType::I32 => types::I32,
        ElementType::I64 => types::I64,
        ElementType::UI32 => types::I32,
        ElementType::UI64 => types::I64,
    }
}

pub(crate) fn ptr_type() -> Type {
    types::I64
}

fn is_float(et: ElementType) -> bool {
    matches!(et, ElementType::F64 | ElementType::F32)
}

fn is_unsigned(et: ElementType) -> bool {
    matches!(et, ElementType::UI32 | ElementType::UI64)
}

fn total_return_elements(result_types: &[TensorType]) -> usize {
    result_types.iter().map(|t| t.num_elements()).sum()
}

fn needs_sret(result_types: &[TensorType]) -> bool {
    total_return_elements(result_types) > max_reg_returns()
}

fn add_tensor_params(sig: &mut Signature, ty: &TensorType) {
    let ct = cranelift_type_for(ty.element_type);
    for _ in 0..ty.num_elements() {
        sig.params.push(AbiParam::new(ct));
    }
}

fn add_tensor_returns(sig: &mut Signature, ty: &TensorType) {
    let ct = cranelift_type_for(ty.element_type);
    for _ in 0..ty.num_elements() {
        sig.returns.push(AbiParam::new(ct));
    }
}

// ---------------------------------------------------------------------------
// Extern "C" shims for libm functions
// ---------------------------------------------------------------------------

extern "C" fn libc_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn libc_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn libc_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}
extern "C" fn libc_sqrt(x: f64) -> f64 {
    x.sqrt()
}
extern "C" fn libc_fabs(x: f64) -> f64 {
    x.abs()
}
extern "C" fn libc_fmod(x: f64, y: f64) -> f64 {
    x % y
}
extern "C" fn libc_acos(x: f64) -> f64 {
    x.acos()
}
extern "C" fn libc_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn libc_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn libc_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn libc_tanh(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn libc_tan(x: f64) -> f64 {
    x.tan()
}
extern "C" fn libc_log1p(x: f64) -> f64 {
    x.ln_1p()
}

extern "C" fn libc_asin(x: f64) -> f64 {
    x.asin()
}
extern "C" fn libc_atan(x: f64) -> f64 {
    x.atan()
}
extern "C" fn libc_sinh(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn libc_cosh(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn libc_expm1(x: f64) -> f64 {
    x.exp_m1()
}
extern "C" fn libc_cbrt(x: f64) -> f64 {
    x.cbrt()
}
extern "C" fn libc_erfc(x: f64) -> f64 {
    crate::tensor_rt::erfc_impl(x)
}

extern "C" fn erf_inv_scalar(x: f64) -> f64 {
    crate::tensor_rt::erf_inv_impl(x)
}

// ---------------------------------------------------------------------------
// Libm function IDs in the JIT module
// ---------------------------------------------------------------------------

struct LibmIds {
    sin: FuncId,
    cos: FuncId,
    atan2: FuncId,
    fmod: FuncId,
    acos: FuncId,
    erf_inv: FuncId,
    log: FuncId,
    exp: FuncId,
    pow: FuncId,
    tanh: FuncId,
    tan: FuncId,
    log1p: FuncId,
    asin: FuncId,
    atan: FuncId,
    sinh: FuncId,
    cosh: FuncId,
    erfc: FuncId,
    expm1: FuncId,
    cbrt: FuncId,
}

fn declare_libm_functions(
    jit_module: &mut JITModule,
    call_conv: CallConv,
) -> Result<LibmIds, String> {
    let mut mk = |name: &str, n_params: usize, n_rets: usize| -> Result<FuncId, String> {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        for _ in 0..n_params {
            sig.params.push(AbiParam::new(types::F64));
        }
        for _ in 0..n_rets {
            sig.returns.push(AbiParam::new(types::F64));
        }
        jit_module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("declare libm {name}: {e}"))
    };

    Ok(LibmIds {
        sin: mk("sin", 1, 1)?,
        cos: mk("cos", 1, 1)?,
        atan2: mk("atan2", 2, 1)?,
        fmod: mk("fmod", 2, 1)?,
        acos: mk("acos", 1, 1)?,
        erf_inv: mk("erf_inv_impl", 1, 1)?,
        log: mk("log", 1, 1)?,
        exp: mk("exp", 1, 1)?,
        pow: mk("pow", 2, 1)?,
        tanh: mk("tanh", 1, 1)?,
        tan: mk("tan", 1, 1)?,
        log1p: mk("log1p_impl", 1, 1)?,
        asin: mk("asin_impl", 1, 1)?,
        atan: mk("atan_impl", 1, 1)?,
        sinh: mk("sinh_impl", 1, 1)?,
        cosh: mk("cosh_impl", 1, 1)?,
        erfc: mk("erfc_impl_scalar", 1, 1)?,
        expm1: mk("expm1_impl", 1, 1)?,
        cbrt: mk("cbrt_impl", 1, 1)?,
    })
}

// ---------------------------------------------------------------------------
// Live profiling probe IDs
// ---------------------------------------------------------------------------

/// Cranelift FuncIds for the runtime profiling probes. Declared once per
/// module alongside the libm/tensor-rt ids; only emitted into the JIT IR
/// when `CompileConfig::profile_enabled` is true.
#[derive(Copy, Clone)]
pub(crate) struct ProfileIds {
    pub enter: FuncId,
    pub exit: FuncId,
    pub marshal_begin: FuncId,
    pub marshal_end: FuncId,
    pub xcend_begin: FuncId,
    pub xcend_end: FuncId,
    pub call_begin: FuncId,
    pub call_end: FuncId,
    pub loop_iter: FuncId,
    /// Per-op wall-time sampling probes. Emitted only when
    /// `CompileConfig::profile_op_times` is set, and only around
    /// every Nth emission of an instrumented op category (see
    /// `op_sampler`).
    pub op_begin: FuncId,
    pub op_end: FuncId,
}

// Thread-local handle to the currently-active `ProfileIds`, populated at
// the start of `compile_module_with_config` when profiling is enabled.
// Read by emission helpers that need to inject marshal / transcendental
// probes deep in the lowering pipeline without threading `ProfileIds`
// through every signature. Compilation is single-threaded so this is
// safe.
thread_local! {
    static CURRENT_PROFILE_IDS: RefCell<Option<ProfileIds>> =
        const { RefCell::new(None) };

    /// FuncId of the function currently being lowered. Set by
    /// `define_function` and `compile_region_as_function`. Used by
    /// `lower_while` to associate `loop_id` → `parent_fid` for the
    /// loop-iteration probe. `None` between function definitions.
    static CURRENT_FUNCTION_FID: RefCell<Option<u32>> =
        const { RefCell::new(None) };
}

pub(crate) fn with_current_function_fid<R>(f: impl FnOnce(Option<u32>) -> R) -> R {
    CURRENT_FUNCTION_FID.with(|c| f(*c.borrow()))
}

thread_local! {
    /// Per-compile op-emission counter. Shared across every
    /// instrumented op in every function body so the sample rate
    /// applies globally, not per-function.
    static OP_SAMPLER: RefCell<crate::op_sampler::OpSampler> =
        RefCell::new(crate::op_sampler::OpSampler::new());
}

/// Emit a `__cranelift_op_begin(category)` call. No-op when profiling
/// is disabled.
fn emit_op_begin(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    category: crate::op_sampler::OpCategory,
) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.op_begin, builder.func);
    let cat_v = builder.ins().iconst(types::I32, category.as_u32() as i64);
    builder.ins().call(r, &[cat_v]);
}

/// Emit a `__cranelift_op_end()` call.
fn emit_op_end(builder: &mut FunctionBuilder, jit_module: &mut JITModule) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.op_end, builder.func);
    builder.ins().call(r, &[]);
}

/// Inner-loop bracketing for Load / Store / StackAddr / const ops
/// inside the ptr-ABI SIMD helpers. Uses the higher
/// `INNER_OP_SAMPLE_RATE` so probe cost stays bounded even though
/// these ops fire many times per helper call. Report-time scaling in
/// `profile.rs` multiplies the measured ns by the same rate, so
/// `sample_count × rate` is a lower-bound estimate of full-sim
/// emission count.
fn bracket_sampled_inner<R>(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    category: crate::op_sampler::OpCategory,
    f: impl FnOnce(&mut FunctionBuilder) -> R,
) -> R {
    if !crate::op_sampler::op_times_enabled() {
        return f(builder);
    }
    let sample = OP_SAMPLER.with(|s| s.borrow_mut().should_sample_inner(category));
    if sample {
        emit_op_begin(builder, jit_module, category);
        let r = f(builder);
        emit_op_end(builder, jit_module);
        r
    } else {
        f(builder)
    }
}

fn set_current_function_fid(fid: Option<u32>) {
    CURRENT_FUNCTION_FID.with(|c| *c.borrow_mut() = fid);
}

pub(crate) fn with_current_profile_ids<R>(f: impl FnOnce(Option<ProfileIds>) -> R) -> R {
    CURRENT_PROFILE_IDS.with(|c| f(*c.borrow()))
}

fn set_current_profile_ids(ids: Option<ProfileIds>) {
    CURRENT_PROFILE_IDS.with(|c| *c.borrow_mut() = ids);
}

fn declare_profile_functions(
    jit_module: &mut JITModule,
    call_conv: CallConv,
) -> Result<ProfileIds, String> {
    // Helper to declare a signature with N i32 params and M i64 params.
    let mk_sig = |jit: &mut JITModule,
                  name: &str,
                  i32_params: u32,
                  i64_params: u32|
     -> Result<FuncId, String> {
        let mut sig = jit.make_signature();
        sig.call_conv = call_conv;
        for _ in 0..i32_params {
            sig.params.push(AbiParam::new(types::I32));
        }
        for _ in 0..i64_params {
            sig.params.push(AbiParam::new(types::I64));
        }
        jit.declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("declare {name}: {e}"))
    };

    let enter = mk_sig(jit_module, "__cranelift_profile_enter", 1, 0)?;
    let exit = mk_sig(jit_module, "__cranelift_profile_exit", 1, 0)?;

    // marshal_begin(direction: i32, bytes: i64); marshal_end()
    let marshal_begin = mk_sig(jit_module, "__cranelift_marshal_begin", 1, 1)?;
    let marshal_end = mk_sig(jit_module, "__cranelift_marshal_end", 0, 0)?;

    // xcend_begin(mode: i32); xcend_end()
    let xcend_begin = mk_sig(jit_module, "__cranelift_xcend_begin", 1, 0)?;
    let xcend_end = mk_sig(jit_module, "__cranelift_xcend_end", 0, 0)?;

    // call_begin(); call_end()
    let call_begin = mk_sig(jit_module, "__cranelift_call_begin", 0, 0)?;
    let call_end = mk_sig(jit_module, "__cranelift_call_end", 0, 0)?;

    // loop_iter(loop_id: i32)
    let loop_iter = mk_sig(jit_module, "__cranelift_loop_iter", 1, 0)?;

    // op_begin(category: i32); op_end()
    let op_begin = mk_sig(jit_module, "__cranelift_op_begin", 1, 0)?;
    let op_end = mk_sig(jit_module, "__cranelift_op_end", 0, 0)?;

    Ok(ProfileIds {
        enter,
        exit,
        marshal_begin,
        marshal_end,
        xcend_begin,
        xcend_end,
        call_begin,
        call_end,
        loop_iter,
        op_begin,
        op_end,
    })
}

/// Emit a `__cranelift_marshal_begin(direction, bytes)` call. No-op
/// when profiling is disabled.
fn emit_marshal_begin(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    direction: u32,
    bytes: u64,
) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.marshal_begin, builder.func);
    let dir_v = builder.ins().iconst(types::I32, direction as i64);
    let bytes_v = builder.ins().iconst(types::I64, bytes as i64);
    builder.ins().call(r, &[dir_v, bytes_v]);
}

/// Emit a `__cranelift_marshal_end()` call at the builder's current
/// position. Must be paired with a prior `emit_marshal_begin`.
fn emit_marshal_end(builder: &mut FunctionBuilder, jit_module: &mut JITModule) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.marshal_end, builder.func);
    builder.ins().call(r, &[]);
}

/// Emit a `__cranelift_xcend_begin(mode)` call at the builder's
/// current position. `mode`: 0 = scalar libm fallback (f64 → libm →
/// f64 per element), 1 = wide-SIMD batch (f64x2 chunks via the
/// `wide` crate).
fn emit_xcend_begin(builder: &mut FunctionBuilder, jit_module: &mut JITModule, mode: u32) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.xcend_begin, builder.func);
    let mode_v = builder.ins().iconst(types::I32, mode as i64);
    builder.ins().call(r, &[mode_v]);
}

/// Emit a `__cranelift_xcend_end()` call at the builder's current
/// position. Must be paired with a prior `emit_xcend_begin`.
fn emit_xcend_end(builder: &mut FunctionBuilder, jit_module: &mut JITModule) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.xcend_end, builder.func);
    builder.ins().call(r, &[]);
}

/// Emit a `__cranelift_call_begin()` call at the builder's current
/// position. Brackets every emitted `call` op so the profile can
/// split per-function "time in callees" from inline IR.
fn emit_call_begin(builder: &mut FunctionBuilder, jit_module: &mut JITModule) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.call_begin, builder.func);
    builder.ins().call(r, &[]);
}

/// Emit a `__cranelift_call_end()` call at the builder's current
/// position.
fn emit_call_end(builder: &mut FunctionBuilder, jit_module: &mut JITModule) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.call_end, builder.func);
    builder.ins().call(r, &[]);
}

/// Emit a `__cranelift_loop_iter(loop_id)` call at the builder's
/// current position. Called at the top of each `while` body so the
/// profiler can count dynamic loop iterations.
fn emit_loop_iter(builder: &mut FunctionBuilder, jit_module: &mut JITModule, loop_id: u32) {
    let Some(ids) = with_current_profile_ids(|ids| ids) else {
        return;
    };
    let r = jit_module.declare_func_in_func(ids.loop_iter, builder.func);
    let id_v = builder.ins().iconst(types::I32, loop_id as i64);
    builder.ins().call(r, &[id_v]);
}

/// Monotonic counter for assigning `loop_id`s at compile time. Shared
/// across functions so each lowered `while` body gets a unique id.
static LOOP_ID_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn next_loop_id() -> u32 {
    LOOP_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Walk a StableHLO `while` body and tally static op counts by kind.
/// Used to weight dynamic loop iterations when computing the runtime
/// `op_kind_executed` report. Only counts top-level ops; nested
/// `while` / `case` are handled by their own `register_loop` calls.
fn count_body_op_kinds(body: &[crate::ir::InstrResult]) -> HashMap<&'static str, usize> {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for ir in body {
        // Use the Debug impl's discriminant name as the key. Avoids a
        // long fragile match against every variant of `Instruction`
        // and automatically picks up new variants as they're added.
        // Note: the leaked-box retention pattern mirrors
        // `opcode_static_name` elsewhere in this file.
        let dbg = format!("{:?}", ir.instr);
        let first_word = dbg
            .split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or("other");
        let name: &'static str = leak_static_str(first_word);
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

/// Leak a string into `'static` for use as a HashMap key. Backed by a
/// module-level interner so repeated lookups for the same string share
/// allocation. Intended for compile-time IR walking, not hot-path code.
fn leak_static_str(s: &str) -> &'static str {
    use std::sync::Mutex;
    static INTERN: std::sync::OnceLock<Mutex<HashMap<String, &'static str>>> =
        std::sync::OnceLock::new();
    let m = INTERN.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = m.lock().expect("intern poisoned");
    if let Some(&cached) = guard.get(s) {
        return cached;
    }
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    guard.insert(s.to_string(), leaked);
    leaked
}

/// Inject `__cranelift_profile_enter` at the start of the entry block and
/// `__cranelift_profile_exit` before every `return_` instruction in the
/// function. Called after the FunctionBuilder has finalized the IR but
/// before `jit_module.define_function` runs codegen. The subsequent
/// codegen pass will re-lower the IR including our injected calls, so the
/// machine code picks up the probes naturally.
///
/// Both probes take a single `i32 fid` argument which the runtime
/// profiler uses to key its per-function stats map.
fn inject_profile_probes(
    func: &mut cranelift_codegen::ir::Function,
    jit_module: &mut JITModule,
    profile_ids: &ProfileIds,
    fid: FuncId,
) {
    use cranelift_codegen::cursor::{Cursor, FuncCursor};

    let enter_ref = jit_module.declare_func_in_func(profile_ids.enter, func);
    let exit_ref = jit_module.declare_func_in_func(profile_ids.exit, func);
    let fid_const: i64 = fid.as_u32() as i64;

    let entry_block = match func.layout.entry_block() {
        Some(b) => b,
        None => return, // Nothing to instrument.
    };

    // Collect return instructions under an immutable borrow first so we
    // don't invalidate the iterator when we switch to a cursor for
    // mutation below. `Inst` handles are stable across layout inserts.
    let mut returns: Vec<cranelift_codegen::ir::Inst> = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if func.dfg.insts[inst].opcode().is_return() {
                returns.push(inst);
            }
        }
    }

    // Inject enter probe at the top of the entry block.
    {
        let mut cur = FuncCursor::new(func);
        cur.goto_first_insertion_point(entry_block);
        let fid_val = cur.ins().iconst(types::I32, fid_const);
        cur.ins().call(enter_ref, &[fid_val]);
    }

    // Inject exit probe immediately before every return instruction.
    // FuncCursor::goto_inst positions before `inst` such that
    // `cur.ins()` inserts a new instruction before it.
    for ret_inst in returns {
        let mut cur = FuncCursor::new(func);
        cur.goto_inst(ret_inst);
        let fid_val = cur.ins().iconst(types::I32, fid_const);
        cur.ins().call(exit_ref, &[fid_val]);
    }
}

// ---------------------------------------------------------------------------
// Tensor runtime function IDs — generated from a single table via macro
// ---------------------------------------------------------------------------

macro_rules! define_trt {
    ($( $field:ident, $symbol:expr, $fn_path:path, [ $($param:expr),* ];)*) => {
        struct TensorRtIds { $($field: FuncId,)* }

        fn register_tensor_rt_symbols(jit_builder: &mut JITBuilder) {
            $(jit_builder.symbol($symbol, $fn_path as *const u8);)*
        }

        fn declare_tensor_rt_functions(
            jit_module: &mut JITModule,
            call_conv: CallConv,
        ) -> Result<TensorRtIds, String> {
            fn decl(m: &mut JITModule, cc: CallConv, name: &str, params: &[Type]) -> Result<FuncId, String> {
                let mut sig = m.make_signature();
                sig.call_conv = cc;
                for &t in params { sig.params.push(AbiParam::new(t)); }
                m.declare_function(name, Linkage::Import, &sig)
                    .map_err(|e| format!("declare trt {name}: {e}"))
            }
            Ok(TensorRtIds {
                $($field: decl(jit_module, call_conv, $symbol, &[$($param),*])?,)*
            })
        }
    };
}

define_trt! {
    // f64 binary elementwise: fn(dst, a, b, n)
    add_f64,  "__trt_add_f64",  tensor_rt::tensor_add_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    sub_f64,  "__trt_sub_f64",  tensor_rt::tensor_sub_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    mul_f64,  "__trt_mul_f64",  tensor_rt::tensor_mul_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    div_f64,  "__trt_div_f64",  tensor_rt::tensor_div_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    max_f64,  "__trt_max_f64",  tensor_rt::tensor_max_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    min_f64,  "__trt_min_f64",  tensor_rt::tensor_min_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    pow_f64,  "__trt_pow_f64",  tensor_rt::tensor_pow_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    rem_f64,  "__trt_rem_f64",  tensor_rt::tensor_rem_f64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    rem_i64,  "__trt_rem_i64",  tensor_rt::tensor_rem_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    rem_i32,  "__trt_rem_i32",  tensor_rt::tensor_rem_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    rem_ui32, "__trt_rem_ui32", tensor_rt::tensor_rem_ui32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    // f64 unary elementwise: fn(dst, src, n)
    neg_f64,   "__trt_neg_f64",   tensor_rt::tensor_neg_f64,   [ptr_type(), ptr_type(), types::I64];
    sqrt_f64,  "__trt_sqrt_f64",  tensor_rt::tensor_sqrt_f64,  [ptr_type(), ptr_type(), types::I64];
    abs_f64,   "__trt_abs_f64",   tensor_rt::tensor_abs_f64,   [ptr_type(), ptr_type(), types::I64];
    floor_f64, "__trt_floor_f64", tensor_rt::tensor_floor_f64, [ptr_type(), ptr_type(), types::I64];
    sin_f64,   "__trt_sin_f64",   tensor_rt::tensor_sin_f64,   [ptr_type(), ptr_type(), types::I64];
    cos_f64,   "__trt_cos_f64",   tensor_rt::tensor_cos_f64,   [ptr_type(), ptr_type(), types::I64];
    exp_f64,   "__trt_exp_f64",   tensor_rt::tensor_exp_f64,   [ptr_type(), ptr_type(), types::I64];
    log_f64,   "__trt_log_f64",   tensor_rt::tensor_log_f64,   [ptr_type(), ptr_type(), types::I64];
    tanh_f64,  "__trt_tanh_f64",  tensor_rt::tensor_tanh_f64,  [ptr_type(), ptr_type(), types::I64];
    sign_f64,  "__trt_sign_f64",  tensor_rt::tensor_sign_f64,  [ptr_type(), ptr_type(), types::I64];
    tan_f64,   "__trt_tan_f64",   tensor_rt::tensor_tan_f64,   [ptr_type(), ptr_type(), types::I64];
    acos_f64,   "__trt_acos_f64",   tensor_rt::tensor_acos_f64,   [ptr_type(), ptr_type(), types::I64];
    rsqrt_f64,  "__trt_rsqrt_f64",  tensor_rt::tensor_rsqrt_f64,  [ptr_type(), ptr_type(), types::I64];
    log1p_f64,  "__trt_log1p_f64",  tensor_rt::tensor_log1p_f64,  [ptr_type(), ptr_type(), types::I64];
    ceil_f64,   "__trt_ceil_f64",   tensor_rt::tensor_ceil_f64,   [ptr_type(), ptr_type(), types::I64];
    asin_f64,   "__trt_asin_f64",   tensor_rt::tensor_asin_f64,   [ptr_type(), ptr_type(), types::I64];
    atan_f64,   "__trt_atan_f64",   tensor_rt::tensor_atan_f64,   [ptr_type(), ptr_type(), types::I64];
    sinh_f64,   "__trt_sinh_f64",   tensor_rt::tensor_sinh_f64,   [ptr_type(), ptr_type(), types::I64];
    cosh_f64,   "__trt_cosh_f64",   tensor_rt::tensor_cosh_f64,   [ptr_type(), ptr_type(), types::I64];
    erfc_f64,   "__trt_erfc_f64",   tensor_rt::tensor_erfc_f64,   [ptr_type(), ptr_type(), types::I64];
    expm1_f64,  "__trt_expm1_f64",  tensor_rt::tensor_expm1_f64,  [ptr_type(), ptr_type(), types::I64];
    cbrt_f64,   "__trt_cbrt_f64",   tensor_rt::tensor_cbrt_f64,   [ptr_type(), ptr_type(), types::I64];
    not_i64,    "__trt_not_i64",    tensor_rt::tensor_not_i64,    [ptr_type(), ptr_type(), types::I64];
    not_i32,    "__trt_not_i32",    tensor_rt::tensor_not_i32,    [ptr_type(), ptr_type(), types::I64];
    not_i1,     "__trt_not_i1",     tensor_rt::tensor_not_i1,     [ptr_type(), ptr_type(), types::I64];
    sshr_i64,   "__trt_sshr_i64",   tensor_rt::tensor_sshr_i64,   [ptr_type(), ptr_type(), ptr_type(), types::I64];
    sshr_i32,   "__trt_sshr_i32",   tensor_rt::tensor_sshr_i32,   [ptr_type(), ptr_type(), ptr_type(), types::I64];
    xor_i64,    "__trt_xor_i64",    tensor_rt::tensor_xor_i64,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    xor_i32,    "__trt_xor_i32",    tensor_rt::tensor_xor_i32,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    or_i64,     "__trt_or_i64",     tensor_rt::tensor_or_i64,     [ptr_type(), ptr_type(), ptr_type(), types::I64];
    or_i32,     "__trt_or_i32",     tensor_rt::tensor_or_i32,     [ptr_type(), ptr_type(), ptr_type(), types::I64];
    and_i64,    "__trt_and_i64",    tensor_rt::tensor_and_i64,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    and_i32,    "__trt_and_i32",    tensor_rt::tensor_and_i32,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    and_i8,     "__trt_and_i8",     tensor_rt::tensor_and_i8,     [ptr_type(), ptr_type(), ptr_type(), types::I64];
    or_i8,      "__trt_or_i8",      tensor_rt::tensor_or_i8,      [ptr_type(), ptr_type(), ptr_type(), types::I64];
    shl_i64,    "__trt_shl_i64",    tensor_rt::tensor_shl_i64,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    shl_i32,    "__trt_shl_i32",    tensor_rt::tensor_shl_i32,    [ptr_type(), ptr_type(), ptr_type(), types::I64];
    ushr_i64,   "__trt_ushr_i64",   tensor_rt::tensor_ushr_i64,   [ptr_type(), ptr_type(), ptr_type(), types::I64];
    ushr_i32,   "__trt_ushr_i32",   tensor_rt::tensor_ushr_i32,   [ptr_type(), ptr_type(), ptr_type(), types::I64];
    round_f64, "__trt_round_f64", tensor_rt::tensor_round_f64, [ptr_type(), ptr_type(), types::I64];
    erf_inv_f64, "__trt_erf_inv_f64", tensor_rt::tensor_erf_inv_f64, [ptr_type(), ptr_type(), types::I64];
    // f64 binary: atan2
    atan2_f64, "__trt_atan2_f64", tensor_rt::tensor_atan2_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    // f64 ternary: clamp fn(dst, src, min, max, n)
    clamp_f64, "__trt_clamp_f64", tensor_rt::tensor_clamp_f64, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64];
    // reverse: fn(dst, src, n, shape, rank, dims, n_dims)
    reverse_f64, "__trt_reverse_f64", tensor_rt::tensor_reverse_f64, [ptr_type(), ptr_type(), types::I64, ptr_type(), types::I64, ptr_type(), types::I64];
    // comparison: fn(dst_u8, a, b, n)
    cmp_eq_f64, "__trt_cmp_eq_f64", tensor_rt::tensor_cmp_eq_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_lt_f64, "__trt_cmp_lt_f64", tensor_rt::tensor_cmp_lt_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_le_f64, "__trt_cmp_le_f64", tensor_rt::tensor_cmp_le_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_gt_f64, "__trt_cmp_gt_f64", tensor_rt::tensor_cmp_gt_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ge_f64, "__trt_cmp_ge_f64", tensor_rt::tensor_cmp_ge_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ne_f64, "__trt_cmp_ne_f64", tensor_rt::tensor_cmp_ne_f64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_eq_i64, "__trt_cmp_eq_i64", tensor_rt::tensor_cmp_eq_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ne_i64, "__trt_cmp_ne_i64", tensor_rt::tensor_cmp_ne_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_lt_i64, "__trt_cmp_lt_i64", tensor_rt::tensor_cmp_lt_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_le_i64, "__trt_cmp_le_i64", tensor_rt::tensor_cmp_le_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_gt_i64, "__trt_cmp_gt_i64", tensor_rt::tensor_cmp_gt_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ge_i64, "__trt_cmp_ge_i64", tensor_rt::tensor_cmp_ge_i64, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_eq_i32, "__trt_cmp_eq_i32", tensor_rt::tensor_cmp_eq_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ne_i32, "__trt_cmp_ne_i32", tensor_rt::tensor_cmp_ne_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_lt_i32, "__trt_cmp_lt_i32", tensor_rt::tensor_cmp_lt_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_le_i32, "__trt_cmp_le_i32", tensor_rt::tensor_cmp_le_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_gt_i32, "__trt_cmp_gt_i32", tensor_rt::tensor_cmp_gt_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    cmp_ge_i32, "__trt_cmp_ge_i32", tensor_rt::tensor_cmp_ge_i32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    // select: fn(dst, cond, on_true, on_false, n)
    select_f64, "__trt_select_f64", tensor_rt::tensor_select_f64, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64];
    select_i64, "__trt_select_i64", tensor_rt::tensor_select_i64, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64];
    select_i32, "__trt_select_i32", tensor_rt::tensor_select_i32, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64];
    select_i8,  "__trt_select_i8",  tensor_rt::tensor_select_i8,  [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64];
    // conversion: fn(dst, src, n)
    convert_i64_to_f64, "__trt_cvt_i64_f64", tensor_rt::tensor_convert_i64_to_f64, [ptr_type(), ptr_type(), types::I64];
    convert_f64_to_i64, "__trt_cvt_f64_i64", tensor_rt::tensor_convert_f64_to_i64, [ptr_type(), ptr_type(), types::I64];
    convert_i1_to_f64,  "__trt_cvt_i1_f64",  tensor_rt::tensor_convert_i1_to_f64,  [ptr_type(), ptr_type(), types::I64];
    convert_f64_to_i32, "__trt_cvt_f64_i32",  tensor_rt::tensor_convert_f64_to_i32, [ptr_type(), ptr_type(), types::I64];
    convert_i1_to_i32,  "__trt_cvt_i1_i32",   tensor_rt::tensor_convert_i1_to_i32,  [ptr_type(), ptr_type(), types::I64];
    convert_i64_to_i32, "__trt_cvt_i64_i32",  tensor_rt::tensor_convert_i64_to_i32, [ptr_type(), ptr_type(), types::I64];
    convert_i32_to_f64, "__trt_cvt_i32_f64",  tensor_rt::tensor_convert_i32_to_f64, [ptr_type(), ptr_type(), types::I64];
    convert_f64_to_f32, "__trt_cvt_f64_f32",  tensor_rt::tensor_convert_f64_to_f32, [ptr_type(), ptr_type(), types::I64];
    convert_f32_to_f64, "__trt_cvt_f32_f64",  tensor_rt::tensor_convert_f32_to_f64, [ptr_type(), ptr_type(), types::I64];
    widen_i32_to_i64,   "__trt_widen_i32_i64", tensor_rt::tensor_widen_i32_to_i64,  [ptr_type(), ptr_type(), types::I64];
    convert_ui32_to_i64, "__trt_cvt_ui32_i64", tensor_rt::tensor_convert_ui32_to_i64, [ptr_type(), ptr_type(), types::I64];
    convert_ui32_to_f64, "__trt_cvt_ui32_f64", tensor_rt::tensor_convert_ui32_to_f64, [ptr_type(), ptr_type(), types::I64];
    convert_f64_to_i1,   "__trt_cvt_f64_i1",  tensor_rt::tensor_convert_f64_to_i1,   [ptr_type(), ptr_type(), types::I64];
    convert_i64_to_i1,   "__trt_cvt_i64_i1",  tensor_rt::tensor_convert_i64_to_i1,   [ptr_type(), ptr_type(), types::I64];
    convert_ui64_to_f64, "__trt_cvt_ui64_f64", tensor_rt::tensor_convert_ui64_to_f64, [ptr_type(), ptr_type(), types::I64];
    convert_i32_to_f32,  "__trt_cvt_i32_f32",  tensor_rt::tensor_convert_i32_to_f32,  [ptr_type(), ptr_type(), types::I64];
    convert_f32_to_i32,  "__trt_cvt_f32_i32",  tensor_rt::tensor_convert_f32_to_i32,  [ptr_type(), ptr_type(), types::I64];
    // broadcast: fn(dst, val, n)
    broadcast_f64, "__trt_bcast_f64", tensor_rt::tensor_broadcast_f64, [ptr_type(), types::F64, types::I64];
    broadcast_i64, "__trt_bcast_i64", tensor_rt::tensor_broadcast_i64, [ptr_type(), types::I64, types::I64];
    broadcast_i32, "__trt_bcast_i32", tensor_rt::tensor_broadcast_i32, [ptr_type(), types::I32, types::I64];
    broadcast_i8,  "__trt_bcast_i8",  tensor_rt::tensor_broadcast_i8,  [ptr_type(), types::I8,  types::I64];
    // i64 binary elementwise
    add_i64,  "__trt_add_i64",  tensor_rt::tensor_add_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    sub_i64,  "__trt_sub_i64",  tensor_rt::tensor_sub_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    mul_i64,  "__trt_mul_i64",  tensor_rt::tensor_mul_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    div_i64,  "__trt_div_i64",  tensor_rt::tensor_div_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    max_i64,  "__trt_max_i64",  tensor_rt::tensor_max_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    min_i64,  "__trt_min_i64",  tensor_rt::tensor_min_i64,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    neg_i64,  "__trt_neg_i64",  tensor_rt::tensor_neg_i64,  [ptr_type(), ptr_type(), types::I64];
    abs_i64,  "__trt_abs_i64",  tensor_rt::tensor_abs_i64,  [ptr_type(), ptr_type(), types::I64];
    // i32 binary elementwise
    add_i32,  "__trt_add_i32",  tensor_rt::tensor_add_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    sub_i32,  "__trt_sub_i32",  tensor_rt::tensor_sub_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    mul_i32,  "__trt_mul_i32",  tensor_rt::tensor_mul_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    div_i32,  "__trt_div_i32",  tensor_rt::tensor_div_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    div_ui32, "__trt_div_ui32", tensor_rt::tensor_div_ui32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    max_i32,  "__trt_max_i32",  tensor_rt::tensor_max_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    min_i32,  "__trt_min_i32",  tensor_rt::tensor_min_i32,  [ptr_type(), ptr_type(), ptr_type(), types::I64];
    max_ui32, "__trt_max_ui32", tensor_rt::tensor_max_ui32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    min_ui32, "__trt_min_ui32", tensor_rt::tensor_min_ui32, [ptr_type(), ptr_type(), ptr_type(), types::I64];
    neg_i32,  "__trt_neg_i32",  tensor_rt::tensor_neg_i32,  [ptr_type(), ptr_type(), types::I64];
    abs_i32,  "__trt_abs_i32",  tensor_rt::tensor_abs_i32,  [ptr_type(), ptr_type(), types::I64];
    // integer reduce
    reduce_sum_i64, "__trt_reduce_sum_i64", tensor_rt::tensor_reduce_sum_i64, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_max_i64, "__trt_reduce_max_i64", tensor_rt::tensor_reduce_max_i64, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_min_i64, "__trt_reduce_min_i64", tensor_rt::tensor_reduce_min_i64, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_and_i1,  "__trt_reduce_and_i1", tensor_rt::tensor_reduce_and_i1, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_or_i1,   "__trt_reduce_or_i1",  tensor_rt::tensor_reduce_or_i1,  [ptr_type(), ptr_type(), types::I64, types::I64];
    argmin_f64, "__trt_argmin_f64", tensor_rt::tensor_argmin_f64, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64, types::I64];
    argmax_f64, "__trt_argmax_f64", tensor_rt::tensor_argmax_f64, [ptr_type(), ptr_type(), ptr_type(), ptr_type(), types::I64, types::I64];
    // memory
    memcpy, "__trt_memcpy", tensor_rt::tensor_memcpy, [ptr_type(), ptr_type(), types::I64];
    // layout / shape
    transpose_f64,    "__trt_transpose_f64",    tensor_rt::tensor_transpose_f64,    [ptr_type(), ptr_type(), types::I64, types::I64];
    // (transpose_nd, broadcast_nd, slice -- replaced by generic variants below)
    concat_nd_f64,    "__trt_concat_nd_f64",    tensor_rt::tensor_concat_nd_f64,    [ptr_type(), types::I64, ptr_type(), types::I64, ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64, types::I64, types::I64];
    pad_f64,          "__trt_pad_f64",          tensor_rt::tensor_pad_f64,          [ptr_type(), ptr_type(), types::I64, types::I64, types::F64, ptr_type(), ptr_type(), types::I64, ptr_type()];
    // reduce
    reduce_sum_f64, "__trt_reduce_sum_f64", tensor_rt::tensor_reduce_sum_f64, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_max_f64, "__trt_reduce_max_f64", tensor_rt::tensor_reduce_max_f64, [ptr_type(), ptr_type(), types::I64, types::I64];
    reduce_min_f64, "__trt_reduce_min_f64", tensor_rt::tensor_reduce_min_f64, [ptr_type(), ptr_type(), types::I64, types::I64];
    // indexing
    // byte-generic gather/scatter/layout ops with elem_sz
    gather_generic,    "__trt_gather_generic",    tensor_rt::tensor_gather_generic,    [ptr_type(), ptr_type(), types::I64, ptr_type(), types::I64, types::I64, types::I64];
    gather_nd_generic, "__trt_gather_nd_generic", tensor_rt::tensor_gather_nd_generic, [ptr_type(), ptr_type(), types::I64, ptr_type(), types::I64, types::I64, ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64, types::I64];
    scatter_generic,   "__trt_scatter_generic",   tensor_rt::tensor_scatter_generic,   [ptr_type(), ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64, types::I64, types::I64];
    matmul_f64,    "__trt_matmul_f64",    tensor_rt::tensor_matmul_f64,    [ptr_type(), ptr_type(), ptr_type(), types::I64, types::I64, types::I64];
    // byte-generic layout ops with elem_sz
    broadcast_nd_generic,         "__trt_bcast_nd_generic",     tensor_rt::tensor_broadcast_nd_generic,         [ptr_type(), ptr_type(), types::I64, types::I64, ptr_type(), types::I64, ptr_type(), types::I64, ptr_type(), types::I64];
    slice_generic,                "__trt_slice_generic",        tensor_rt::tensor_slice_generic,                [ptr_type(), ptr_type(), types::I64, types::I64, ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64];
    transpose_nd_generic,         "__trt_transpose_nd_generic", tensor_rt::tensor_transpose_nd_generic,         [ptr_type(), ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64, types::I64];
    dynamic_slice_generic,        "__trt_dyn_slice_generic",    tensor_rt::tensor_dynamic_slice_generic,        [ptr_type(), ptr_type(), types::I64, types::I64, ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64];
    dynamic_update_slice_generic, "__trt_dyn_upd_generic",      tensor_rt::tensor_dynamic_update_slice_generic, [ptr_type(), ptr_type(), ptr_type(), types::I64, types::I64, ptr_type(), types::I64, ptr_type(), ptr_type(), types::I64];
    // iota
    iota_nd_i64, "__trt_iota_nd_i64", tensor_rt::tensor_iota_nd_i64, [ptr_type(), types::I64, ptr_type(), types::I64, types::I64];
    iota_nd_f64, "__trt_iota_nd_f64", tensor_rt::tensor_iota_nd_f64, [ptr_type(), types::I64, ptr_type(), types::I64, types::I64];
}

// ---------------------------------------------------------------------------
// Module compilation entry point
// ---------------------------------------------------------------------------

pub fn compile_module(ir_module: &crate::ir::Module) -> Result<CompiledModule, String> {
    // Pull flags from the env so tests/simple callers automatically pick up
    // `ELODIN_CRANELIFT_DEBUG_DIR` without needing to thread a config
    // through every call site.
    compile_module_with_config(ir_module, CompileConfig::from_env())
}

pub fn compile_module_with_config(
    ir_module: &crate::ir::Module,
    config: CompileConfig,
) -> Result<CompiledModule, String> {
    reset_instr_counts();
    // Defensive clear: a prior compile that errored partway through
    // Phase 1 may have left `(FuncId, Context)` entries behind on this
    // thread. Those FuncIds belong to a now-dead JITModule, so carrying
    // them into the next compile's Phase 2 would corrupt symbol
    // resolution. Start every compile with an empty buffer.
    let _ = drain_pending_functions();

    // Single-caller inliner. Runs once per compile, at the IR level
    // before any Cranelift codegen. Replaces `Instruction::Call`
    // sites whose callee has exactly one caller and a small body
    // with the inlined body, eliminating the call overhead. Clone
    // the module because the public API takes `&Module`; one extra
    // Vec clone is negligible next to Cranelift codegen.
    let ir_module_storage;
    let ir_module = {
        let mut mut_module = ir_module.clone();
        let inlined = crate::inliner::inline_single_caller_callees(&mut mut_module);
        if inlined > 0 && crate::debug::enabled() {
            eprintln!(
                "[elodin-cranelift] inliner: inlined {} single-caller callee(s)",
                inlined
            );
        }
        ir_module_storage = mut_module;
        &ir_module_storage
    };

    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| format!("native ISA: {e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish: {e}"))?;

    let call_conv = isa.default_call_conv();
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    const JIT_ARENA_SIZE: usize = 2 * 1024 * 1024 * 1024;
    let arena = ArenaMemoryProvider::new_with_size(JIT_ARENA_SIZE)
        .map_err(|e| format!("JIT arena allocation: {e}"))?;
    jit_builder.memory_provider(Box::new(arena));

    jit_builder.symbol("sin", libc_sin as *const u8);
    jit_builder.symbol("cos", libc_cos as *const u8);
    jit_builder.symbol("atan2", libc_atan2 as *const u8);
    jit_builder.symbol("sqrt", libc_sqrt as *const u8);
    jit_builder.symbol("fabs", libc_fabs as *const u8);
    jit_builder.symbol("fmod", libc_fmod as *const u8);
    jit_builder.symbol("acos", libc_acos as *const u8);
    jit_builder.symbol("log", libc_log as *const u8);
    jit_builder.symbol("exp", libc_exp as *const u8);
    jit_builder.symbol("pow", libc_pow as *const u8);
    jit_builder.symbol("tanh", libc_tanh as *const u8);
    jit_builder.symbol("tan", libc_tan as *const u8);
    jit_builder.symbol("erf_inv_impl", erf_inv_scalar as *const u8);
    jit_builder.symbol("log1p_impl", libc_log1p as *const u8);
    jit_builder.symbol("asin_impl", libc_asin as *const u8);
    jit_builder.symbol("atan_impl", libc_atan as *const u8);
    jit_builder.symbol("sinh_impl", libc_sinh as *const u8);
    jit_builder.symbol("cosh_impl", libc_cosh as *const u8);
    jit_builder.symbol("erfc_impl_scalar", libc_erfc as *const u8);
    jit_builder.symbol("expm1_impl", libc_expm1 as *const u8);
    jit_builder.symbol("cbrt_impl", libc_cbrt as *const u8);
    jit_builder.symbol("__cranelift_svd", cranelift_svd as *const u8);
    jit_builder.symbol("__cranelift_lu", cranelift_lu as *const u8);
    jit_builder.symbol("__cranelift_trsm", cranelift_trsm as *const u8);
    jit_builder.symbol("__cranelift_cholesky", cranelift_cholesky as *const u8);
    jit_builder.symbol("__cranelift_qr", cranelift_qr as *const u8);
    jit_builder.symbol("__cranelift_orgqr", cranelift_orgqr as *const u8);
    jit_builder.symbol("__cranelift_syevd", cranelift_syevd as *const u8);
    jit_builder.symbol("__cranelift_gesv", cranelift_gesv as *const u8);
    jit_builder.symbol("__cranelift_potrs", cranelift_potrs as *const u8);
    jit_builder.symbol("__cranelift_gelsd", cranelift_gelsd as *const u8);
    jit_builder.symbol("__cranelift_geev", cranelift_geev as *const u8);
    jit_builder.symbol("__cranelift_gesvd", cranelift_gesvd as *const u8);
    jit_builder.symbol(
        "__trt_sort_f64",
        crate::tensor_rt::tensor_sort_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_argsort_f64",
        crate::tensor_rt::tensor_argsort_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_batch_norm_inference_f64",
        crate::tensor_rt::tensor_batch_norm_inference_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_real_dynamic_slice",
        crate::tensor_rt::tensor_real_dynamic_slice as *const u8,
    );
    jit_builder.symbol(
        "__trt_map_f64",
        crate::tensor_rt::tensor_map_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_reduce_window_f64",
        crate::tensor_rt::tensor_reduce_window_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_select_and_scatter_f64",
        crate::tensor_rt::tensor_select_and_scatter_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_conv_f64",
        crate::tensor_rt::tensor_conv_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_fft_f64",
        crate::tensor_rt::tensor_fft_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_rng_f64",
        crate::tensor_rt::tensor_rng_f64 as *const u8,
    );
    // Live profiling probes (gated by ELODIN_CRANELIFT_DEBUG_DIR). Registered
    // unconditionally so calls from JIT IR can resolve; only emitted into
    // the IR when `CompileConfig::profile_enabled` is true.
    jit_builder.symbol(
        "__cranelift_profile_enter",
        crate::profile::__cranelift_profile_enter as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_profile_exit",
        crate::profile::__cranelift_profile_exit as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_marshal_begin",
        crate::profile::__cranelift_marshal_begin as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_marshal_end",
        crate::profile::__cranelift_marshal_end as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_xcend_begin",
        crate::profile::__cranelift_xcend_begin as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_xcend_end",
        crate::profile::__cranelift_xcend_end as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_call_begin",
        crate::profile::__cranelift_call_begin as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_call_end",
        crate::profile::__cranelift_call_end as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_loop_iter",
        crate::profile::__cranelift_loop_iter as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_op_begin",
        crate::profile::__cranelift_op_begin as *const u8,
    );
    jit_builder.symbol(
        "__cranelift_op_end",
        crate::profile::__cranelift_op_end as *const u8,
    );
    register_tensor_rt_symbols(&mut jit_builder);

    let mut jit_module = JITModule::new(jit_builder);
    let func_abis = classify_all_functions(ir_module);
    let func_ids = declare_all_functions(ir_module, &mut jit_module, call_conv, &func_abis)?;
    let libm_ids = declare_libm_functions(&mut jit_module, call_conv)?;
    let trt_ids = declare_tensor_rt_functions(&mut jit_module, call_conv)?;
    let profile_ids = declare_profile_functions(&mut jit_module, call_conv)?;

    // Expose ProfileIds via a thread-local so deep-in-the-stack emission
    // helpers (`emit_marshal_probe`, `emit_xcend_probe`) can look them up
    // without threading `ProfileIds` through every signature. Set only
    // when profiling is enabled; cleared at the bottom of this function.
    if config.profile_enabled {
        set_current_profile_ids(Some(profile_ids));
    }
    // Propagate `profile_op_times` to the global flag read by the
    // op-sampler emission path.
    crate::op_sampler::set_op_times_enabled(config.profile_enabled && config.profile_op_times);

    // Three-phase compile pipeline.
    //
    //   Phase 1 (serial): lower each IR function to Cranelift IR. Uses
    //       `&mut jit_module` for signature/data/func-ref declarations,
    //       so it must stay single-threaded. `build_function_ir` returns
    //       the populated `Context` without invoking codegen.
    //
    //   Phase 2 (parallel): `Context::compile` on a rayon thread pool —
    //       this is the Cranelift legalization + egraph + regalloc +
    //       machine-code emission step, which dominates compile wall
    //       time on large modules. Only needs `&dyn TargetIsa` (immutable,
    //       `Send + Sync` by Cranelift's design).
    //
    //   Phase 3 (serial): link each function's bytes + relocs into the
    //       `JITModule` via `define_function_bytes`. Cheap; just registers
    //       the compiled blob and queues it for `finalize_definitions`.
    //
    // Nested region bodies (map / reduce_window / sort comparators, etc.)
    // are still compiled synchronously during Phase 1 via
    // `compile_region_as_function`. Those bodies are small and not the
    // hot path; parallelizing them is out of scope here.

    // Phase 1: serial IR build.
    let phase1_start = std::time::Instant::now();
    let mut pending: Vec<(FuncId, cranelift_codegen::Context)> =
        Vec::with_capacity(ir_module.functions.len());
    for func_def in &ir_module.functions {
        let fid = func_ids[&func_def.name];
        let abi = func_abis
            .get(&func_def.name)
            .copied()
            .unwrap_or(FuncAbi::Scalar);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_function_ir(
                &mut jit_module,
                func_def,
                ir_module,
                &func_ids,
                &func_abis,
                &libm_ids,
                &trt_ids,
                &profile_ids,
                abi,
                fid,
                &config,
            )
        }));
        match result {
            Ok(Ok((fid, ctx))) => pending.push((fid, ctx)),
            Ok(Err(e)) => return Err(format!("build {}: {e}", func_def.name)),
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                return Err(format!("panic compiling {}: {msg}", func_def.name));
            }
        }
    }

    // Drain any case-branch (or future Phase-1 splitter) functions that
    // were queued as a side effect of lowering the top-level functions
    // above. They're fully-lowered `(FuncId, Context)` pairs; Phase 2
    // treats them identically to a top-level function and schedules
    // their codegen on the same rayon thread pool. Names are held
    // aside to feed into `register_static_data` below so the runtime
    // profile report can resolve split-branch fids back to a symbol.
    let mut extra_names: HashMap<String, FuncId> = HashMap::new();
    for extra in drain_pending_functions() {
        extra_names.insert(extra.name, extra.fid);
        pending.push((extra.fid, extra.ctx));
    }
    let ir_build_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

    // Phase 2: parallel codegen. The returned compiled blob is extracted
    // from each `Context` into owned `Vec<u8>` + `Vec<ModuleReloc>` so the
    // Context can be dropped before Phase 3 reacquires `&mut jit_module`.
    //
    // Amdahl's law gate: a single oversized function (e.g. cube-sat's
    // EGM08 gravity model, ~2.5 MB of machine code after lowering)
    // caps the Phase-2 wall clock at its own serial codegen time even
    // on a 20-core box. The per-function breakdown emitted under
    // `ELODIN_CRANELIFT_DEBUG_DIR` surfaces exactly which `FuncId` is
    // dominant so it can be attacked directly (IR-level shrinking,
    // hand-splitting, or a `opt_level` knob) in follow-up work.
    let phase2_start = std::time::Instant::now();
    let debug_codegen = crate::debug::enabled();
    let compiled: Vec<(FuncId, u64, Vec<u8>, Vec<cranelift_module::ModuleReloc>)> = {
        use rayon::prelude::*;
        let isa = jit_module.isa();
        pending
            .into_par_iter()
            .map(|(fid, mut ctx)| -> Result<_, String> {
                let per_fn_start = debug_codegen.then(std::time::Instant::now);
                let mut ctrl_plane = cranelift_codegen::control::ControlPlane::default();
                ctx.compile(isa, &mut ctrl_plane)
                    .map_err(|e| format!("codegen {}: {:?}", fid.as_u32(), e))?;
                let code = ctx
                    .compiled_code()
                    .ok_or_else(|| format!("codegen {}: no compiled code", fid.as_u32()))?;
                let alignment = code.buffer.alignment as u64;
                let bytes: Vec<u8> = code.code_buffer().to_vec();
                let relocs: Vec<cranelift_module::ModuleReloc> = code
                    .buffer
                    .relocs()
                    .iter()
                    .map(|r| cranelift_module::ModuleReloc::from_mach_reloc(r, &ctx.func, fid))
                    .collect();
                if let Some(t0) = per_fn_start {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[elodin-cranelift] codegen fid={:>3} {:>8.2}ms bytes={}",
                        fid.as_u32(),
                        ms,
                        bytes.len(),
                    );
                }
                Ok((fid, alignment, bytes, relocs))
            })
            .collect::<Result<_, _>>()?
    };
    let codegen_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;

    // Phase 3: serial link into the JIT module.
    let phase3_start = std::time::Instant::now();
    for (fid, alignment, bytes, relocs) in compiled {
        jit_module
            .define_function_bytes(fid, alignment, &bytes, &relocs)
            .map_err(|e| format!("link {}: {e}", fid.as_u32()))?;
    }

    jit_module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {e}"))?;
    let link_ms = phase3_start.elapsed().as_secs_f64() * 1000.0;

    let main_fn_id = *func_ids.get("main").ok_or("no main function")?;

    // Capture the FuncId ↔ name / ABI / static-instr-count mapping so
    // the runtime profiler can resolve `fid` back to function names
    // when rendering the exit report. Always runs (cheap); probes only
    // hit the registry when debug mode is on. Source-line map lets
    // Tracy zones link back to the StableHLO MLIR. Split case-branch
    // functions are folded into the name map so their `fid`s resolve
    // in the report alongside the top-level ones.
    let source_lines: HashMap<String, u32> = ir_module
        .functions
        .iter()
        .filter_map(|f| f.source_line.map(|l| (f.name.clone(), l)))
        .collect();
    let mut all_func_ids = func_ids.clone();
    for (name, fid) in extra_names {
        all_func_ids.insert(name, fid);
    }
    crate::profile::register_static_data(&all_func_ids, &source_lines);

    print_instr_report();

    // Clear the thread-local ProfileIds handle now that compilation is
    // done. The JIT still holds the registered symbols, and subsequent
    // runtime calls from JIT code resolve by name not FuncId.
    set_current_profile_ids(None);

    Ok(CompiledModule {
        module: jit_module,
        main_fn_id,
        timings: CompileTimings {
            ir_build_ms,
            codegen_ms,
            link_ms,
        },
    })
}

fn declare_all_functions(
    ir_module: &crate::ir::Module,
    jit_module: &mut JITModule,
    call_conv: CallConv,
    func_abis: &HashMap<String, FuncAbi>,
) -> Result<HashMap<String, FuncId>, String> {
    let mut ids = HashMap::new();
    for func_def in &ir_module.functions {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        let abi = func_abis
            .get(&func_def.name)
            .copied()
            .unwrap_or(FuncAbi::Scalar);

        if func_def.name == "main" {
            sig.params.push(AbiParam::new(ptr_type()));
            sig.params.push(AbiParam::new(ptr_type()));
        } else if abi == FuncAbi::Pointer {
            for _ in &func_def.params {
                sig.params.push(AbiParam::new(ptr_type()));
            }
            sig.params.push(AbiParam::new(ptr_type()));
        } else {
            for (_vid, ty) in &func_def.params {
                add_tensor_params(&mut sig, ty);
            }
            if needs_sret(&func_def.result_types) {
                sig.params.push(AbiParam::new(ptr_type()));
            } else {
                for ty in &func_def.result_types {
                    add_tensor_returns(&mut sig, ty);
                }
            }
        }

        let linkage = if func_def.is_public {
            Linkage::Export
        } else {
            Linkage::Local
        };
        let fid = jit_module
            .declare_function(&func_def.name, linkage, &sig)
            .map_err(|e| format!("declare {}: {e}", func_def.name))?;
        ids.insert(func_def.name.clone(), fid);
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Region compilation: compile an embedded StableHLO region body into a
// standalone callable JIT function (used by map, reduce_window,
// select_and_scatter, and sort comparators).
// ---------------------------------------------------------------------------

static REGION_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// StableHLO-instruction-count threshold above which a `stablehlo.case`
/// branch is lifted into its own Cranelift function instead of being
/// inlined into the parent. Keeps short `if`-style cases inline (the
/// call-site overhead isn't worth it) while preventing giant branch
/// bodies from piling ~100k F64X2 SSA chunks into a single monolithic
/// caller (see [`ARCHITECTURE.md`](../ARCHITECTURE.md) Amdahl discussion
/// and the customer `inner_536` reproduction). 64 matches
/// `LARGE_TENSOR_THRESHOLD` used by the scalar/pointer ABI classifier.
const CASE_BRANCH_SPLIT_INSTRS: usize = 64;

/// One entry in `PENDING_EXTRA_FNS`: the fully-lowered `Context`, its
/// assigned `FuncId`, and the symbolic name under which it was
/// `declare_function`'d. The name is needed by
/// `crate::profile::register_static_data` so the runtime profile
/// report can resolve split-branch `fid`s back to human-readable
/// entries instead of leaving them unattributed.
struct PendingExtraFn {
    fid: FuncId,
    name: String,
    ctx: cranelift_codegen::Context,
}

thread_local! {
    /// Case-branch functions (and any future Phase-1 splitters) built
    /// as a side effect of lowering the caller. Each entry is a fully-
    /// lowered `(FuncId, Context)` ready for Cranelift codegen. The
    /// three-phase compile driver drains this vec at the end of Phase 1
    /// and folds it into the `pending` set that Phase 2 parallel-codegen
    /// consumes, so these bodies automatically pick up the same rayon
    /// thread-pool parallelism as the top-level functions.
    static PENDING_EXTRA_FNS: RefCell<Vec<PendingExtraFn>> =
        const { RefCell::new(Vec::new()) };
}

/// Queue a Phase-1-built `(FuncId, Context)` for Phase-2 parallel
/// codegen. Called by the case-branch splitter once it has finished
/// building a branch-as-function. `name` is the symbolic name the
/// function was declared under; it flows into
/// [`crate::profile::register_static_data`] so the runtime profile
/// can map `fid` -> name.
fn push_pending_function(fid: FuncId, name: String, ctx: cranelift_codegen::Context) {
    PENDING_EXTRA_FNS.with(|cell| cell.borrow_mut().push(PendingExtraFn { fid, name, ctx }));
}

/// Drain every queued extra function from the thread-local buffer.
/// Invoked once by `compile_module_with_config` after Phase 1
/// completes. Clears the buffer so a subsequent compile on the same
/// thread starts clean.
fn drain_pending_functions() -> Vec<PendingExtraFn> {
    PENDING_EXTRA_FNS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

/// Walk a nested body and collect `ValueId`s that are *referenced* but
/// not *defined* anywhere in the body (including transitively-nested
/// While/Case/Map/ReduceWindow/Sort/SelectAndScatter bodies). These are
/// the free variables the body captures from the enclosing scope; the
/// splitter threads them through as function parameters so the lifted
/// body can still resolve them.
///
/// Returned `Vec` is sorted by `ValueId::0` so call-site argument order
/// is deterministic between caller and callee.
fn collect_captured_vids(body: &[InstrResult]) -> Vec<ValueId> {
    use std::collections::HashSet;
    let mut defined: HashSet<ValueId> = HashSet::new();
    let mut referenced: HashSet<ValueId> = HashSet::new();
    collect_body_defs_uses(body, &mut defined, &mut referenced);
    let mut free: Vec<ValueId> = referenced.difference(&defined).copied().collect();
    free.sort_by_key(|v| v.0);
    free
}

fn collect_body_defs_uses(
    body: &[InstrResult],
    defined: &mut std::collections::HashSet<ValueId>,
    referenced: &mut std::collections::HashSet<ValueId>,
) {
    for ir in body {
        for (vid, _) in &ir.values {
            defined.insert(*vid);
        }
        for op in crate::const_fold::operand_ids(&ir.instr) {
            referenced.insert(op);
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                iter_arg_ids,
                ..
            } => {
                for v in iter_arg_ids {
                    defined.insert(*v);
                }
                collect_body_defs_uses(cond_body, defined, referenced);
                collect_body_defs_uses(loop_body, defined, referenced);
            }
            Instruction::Case { branches, .. } => {
                for b in branches {
                    collect_body_defs_uses(b, defined, referenced);
                }
            }
            Instruction::Map {
                body, body_params, ..
            } => {
                for v in body_params {
                    defined.insert(*v);
                }
                collect_body_defs_uses(body, defined, referenced);
            }
            Instruction::ReduceWindow {
                body, body_params, ..
            } => {
                for v in body_params {
                    defined.insert(*v);
                }
                collect_body_defs_uses(body, defined, referenced);
            }
            Instruction::SelectAndScatter {
                select_body,
                select_params,
                scatter_body,
                scatter_params,
                ..
            } => {
                for v in select_params {
                    defined.insert(*v);
                }
                for v in scatter_params {
                    defined.insert(*v);
                }
                collect_body_defs_uses(select_body, defined, referenced);
                collect_body_defs_uses(scatter_body, defined, referenced);
            }
            Instruction::Sort {
                comparator,
                comparator_params,
                ..
            } => {
                for v in comparator_params {
                    defined.insert(*v);
                }
                collect_body_defs_uses(comparator, defined, referenced);
            }
            _ => {}
        }
    }
}

/// Build a standalone pointer-ABI Cranelift function from a single
/// `stablehlo.case` branch body and queue it for Phase-2 parallel
/// codegen. The returned `FuncId` is a local, void-returning function
/// with signature:
///
/// ```text
/// (out_0: ptr, ..., out_{N-1}: ptr, capt_0: ptr, ..., capt_{M-1}: ptr) -> ()
/// ```
///
/// The first `N` parameters are output-slot pointers (one per
/// `result_types` entry); the branch's `Return` operands are memcpy'd
/// into them on exit. The remaining `M` parameters are the captured
/// outer-scope values, in the order given by `captured`.
///
/// Callee does not call `jit_module.define_function` — the populated
/// `Context` is stashed via [`push_pending_function`] so the same
/// rayon codegen wave as the top-level functions picks it up.
#[allow(clippy::too_many_arguments)]
fn compile_case_branch_as_function_mem(
    jit_module: &mut JITModule,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    branch_body: &[InstrResult],
    result_types: &[TensorType],
    captured: &[(ValueId, TensorType)],
) -> Result<FuncId, String> {
    let id = REGION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = format!("__case_branch_{id}");

    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    for _ in 0..result_types.len() {
        sig.params.push(AbiParam::new(ptr_type()));
    }
    for _ in captured {
        sig.params.push(AbiParam::new(ptr_type()));
    }

    let fid = jit_module
        .declare_function(&name, Linkage::Local, &sig)
        .map_err(|e| format!("declare case-branch {name}: {e}"))?;

    let mut ctx = jit_module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();
    ctx.func.signature = sig;
    ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, fid.as_u32());

    let parent_fid = with_current_function_fid(|f| f);
    set_current_function_fid(Some(fid.as_u32()));

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let block_params: Vec<Value> = builder.block_params(entry).to_vec();
        let n_results = result_types.len();

        let mut value_map: HashMap<ValueId, LaneRepr> = HashMap::new();
        let mut type_map: HashMap<ValueId, TensorType> = HashMap::new();
        for (i, (vid, ty)) in captured.iter().enumerate() {
            value_map.insert(*vid, LaneRepr::scalar(vec![block_params[n_results + i]]));
            type_map.insert(*vid, ty.clone());
        }

        lower_body_mem(
            &mut builder,
            branch_body,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            &mut value_map,
            &mut type_map,
        )?;

        // Memcpy each Return operand's slot into the caller-provided
        // output slot. Mirrors the inline-Case handler at
        // `Instruction::Case` in `lower_instruction_mem`.
        if let Some(ret_ir) = branch_body
            .iter()
            .rev()
            .find(|ir| matches!(ir.instr, Instruction::Return { .. }))
            && let Instruction::Return { operands } = &ret_ir.instr
        {
            let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            for (i, vid) in operands.iter().enumerate() {
                if let (Some(rty), Some(lr)) = (result_types.get(i), value_map.get_mut(vid)) {
                    lr.unpack_in(&mut builder);
                    let vals = lr.as_scalar().to_vec();
                    let nb = builder.ins().iconst(types::I64, rty.byte_size() as i64);
                    builder
                        .ins()
                        .call(memcpy_ref, &[block_params[i], vals[0], nb]);
                }
            }
        }

        builder.ins().return_(&[]);
        builder.finalize();
    }

    record_instr_counts(&name, FuncAbi::Pointer, &ctx.func);

    // Inject profile probes when runtime profiling is active so the
    // split branches show up in the per-function report alongside
    // their callers.
    if let Some(profile_ids) = with_current_profile_ids(|ids| ids) {
        inject_profile_probes(&mut ctx.func, jit_module, &profile_ids, fid);
    }

    push_pending_function(fid, name, ctx);

    set_current_function_fid(parent_fid);
    Ok(fid)
}

fn compile_region_as_function(
    jit_module: &mut JITModule,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    param_types: &[Type],
    return_type: Type,
    body: &[InstrResult],
    body_params: &[ValueId],
    param_tensor_types: &[TensorType],
) -> Result<FuncId, String> {
    let id = REGION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = format!("__region_{id}");

    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    for &pt in param_types {
        sig.params.push(AbiParam::new(pt));
    }
    sig.returns.push(AbiParam::new(return_type));
    let fid = jit_module
        .declare_function(&name, Linkage::Local, &sig)
        .map_err(|e| format!("declare region {name}: {e}"))?;

    let mut ctx = jit_module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();
    ctx.func.signature = sig;
    ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, fid.as_u32());

    // Track parent fid so nested `while` lowering can associate its
    // loop_id with this region.
    let parent_fid = with_current_function_fid(|f| f);
    set_current_function_fid(Some(fid.as_u32()));

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let block_params: Vec<Value> = builder.block_params(entry).to_vec();
        let mut value_map: HashMap<ValueId, LaneRepr> = HashMap::new();
        let mut type_map: HashMap<ValueId, TensorType> = HashMap::new();

        for (i, &pid) in body_params.iter().enumerate() {
            if i < block_params.len() {
                value_map.insert(pid, LaneRepr::scalar(vec![block_params[i]]));
            }
            if i < param_tensor_types.len() {
                type_map.insert(pid, param_tensor_types[i].clone());
            }
        }

        lower_body(
            &mut builder,
            body,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            &mut value_map,
            &mut type_map,
        )?;

        // Extract return value from the region's Return instruction
        let ret_val = body
            .iter()
            .find_map(|ir| {
                if let Instruction::Return { operands } = &ir.instr {
                    operands
                        .first()
                        .and_then(|v| value_map.get(v))
                        .and_then(|vs| vs.as_scalar().first().copied())
                } else {
                    None
                }
            })
            .or_else(|| {
                body.last()
                    .and_then(|ir| ir.values.first())
                    .and_then(|(vid, _)| value_map.get(vid))
                    .and_then(|vs| vs.as_scalar().first().copied())
            })
            .ok_or_else(|| format!("region {name}: no return value found"))?;

        builder.ins().return_(&[ret_val]);
        builder.finalize();
    }

    jit_module
        .define_function(fid, &mut ctx)
        .map_err(|e| format!("define region {name}: {e:?}"))?;

    // Restore the caller's CURRENT_FUNCTION_FID so ongoing lowering
    // upstream sees its own context again.
    set_current_function_fid(parent_fid);

    Ok(fid)
}

// ---------------------------------------------------------------------------
// Function IR building (body lowering + ABI handling)
// ---------------------------------------------------------------------------

/// Build Cranelift IR for a single function and return the populated
/// `Context` to the caller. Intentionally does NOT invoke
/// `jit_module.define_function` (the Cranelift codegen step) so the
/// caller can batch codegen across a rayon thread pool; see the
/// three-phase pipeline in `compile_module_with_config`.
fn build_function_ir(
    jit_module: &mut JITModule,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    profile_ids: &ProfileIds,
    abi: FuncAbi,
    fid: FuncId,
    config: &CompileConfig,
) -> Result<(FuncId, cranelift_codegen::Context), String> {
    let mut ctx = jit_module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    ctx.func.signature = jit_module
        .declarations()
        .get_function_decl(fid)
        .signature
        .clone();
    ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, fid.as_u32());

    // Track the current function for `lower_while` loop registration.
    // Restored to `None` on scope exit.
    set_current_function_fid(Some(fid.as_u32()));

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();
        let mut value_map: HashMap<ValueId, LaneRepr> = HashMap::new();
        let mut type_map: HashMap<ValueId, TensorType> = HashMap::new();

        if func_def.name == "main" && config.force_pointer_abi_main {
            lower_main_body_mem(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else if func_def.name == "main" {
            lower_main_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else if abi == FuncAbi::Pointer {
            lower_pointer_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else {
            lower_callee_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        }

        builder.finalize();
    }

    record_instr_counts(&func_def.name, abi, &ctx.func);

    // Inject live-profiling probes when enabled. Every JIT-compiled
    // function gets `profile_enter` at entry and `profile_exit` at
    // every return site; the FuncId-keyed stats map lets the report
    // break down hot functions by name. Region-as-function bodies
    // (reduce / map / sort comparators, etc.) are instrumented by
    // `compile_region_as_function`'s own profile hook. When
    // `profile_enabled` is false the IR is bit-identical — no extra
    // calls, no branches.
    if config.profile_enabled {
        inject_profile_probes(&mut ctx.func, jit_module, profile_ids, fid);
    }

    set_current_function_fid(None);
    Ok((fid, ctx))
}

fn lower_main_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let inputs_ptr = block_params[0];
    let outputs_ptr = block_params[1];

    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let n = ty.num_elements();
        let ct = cranelift_type_for(ty.element_type);
        let buf_ptr =
            builder
                .ins()
                .load(ptr_type(), MemFlags::trusted(), inputs_ptr, (i * 8) as i32);
        let mut vals = Vec::with_capacity(n);
        for j in 0..n {
            let offset = (j * ty.element_type.byte_size()) as i32;
            let v = builder.ins().load(ct, MemFlags::trusted(), buf_ptr, offset);
            vals.push(v);
        }
        value_map.insert(*vid, LaneRepr::scalar(vals));
        type_map.insert(*vid, ty.clone());
    }

    lower_body(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        for (i, (vid, ty)) in operands
            .iter()
            .zip(func_def.result_types.iter())
            .enumerate()
        {
            let vals = get_vals(builder, value_map, vid)?;
            let buf_ptr =
                builder
                    .ins()
                    .load(ptr_type(), MemFlags::trusted(), outputs_ptr, (i * 8) as i32);
            for (j, &v) in vals.iter().enumerate() {
                let offset = (j * ty.element_type.byte_size()) as i32;
                builder.ins().store(MemFlags::trusted(), v, buf_ptr, offset);
            }
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-backed (pointer ABI) body lowering
// All values in value_map are vec![ptr] -- a single i64 pointer to a stack buffer.
// All ops dispatch to tensor_rt functions.
// ---------------------------------------------------------------------------

fn alloc_slot(builder: &mut FunctionBuilder, byte_size: usize) -> Value {
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        byte_size as u32,
        SLOT_ALIGN,
    ));
    builder.ins().stack_addr(ptr_type(), ss, 0)
}

/// Pool-managed allocation. Reuses a previously-released slot of the
/// same `(byte_size, SLOT_ALIGN)` shape when one is available; else
/// creates a fresh `ExplicitSlot`. The pool records `result_vid` as
/// owner so the slot returns to the free-list once `last_use_pos`
/// passes. When `result_vid` is `None`, falls back to raw
/// `alloc_slot` — used for scratch buffers whose lifetime doesn't
/// map to a single vid.
fn alloc_slot_for_vid(
    builder: &mut FunctionBuilder,
    pool: &mut crate::slot_pool::SlotPool,
    result_vid: Option<ValueId>,
    byte_size: usize,
) -> Value {
    let Some(vid) = result_vid else {
        return alloc_slot(builder, byte_size);
    };
    let (ss, ptr) = pool.alloc(builder, byte_size as u32, SLOT_ALIGN);
    pool.record_owner(vid, ss, byte_size as u32, SLOT_ALIGN);
    ptr
}

// ---------------------------------------------------------------------------
// Result-write elision.
//
// When a ptr-ABI F64 elementwise op's result is consumed exactly once by
// another elision-friendly op, we keep the F64X2 chunks in SSA (as
// `LaneRepr::PtrChunksF64`) instead of storing to a fresh stack slot and
// reloading. The chain is capped at `ELISION_MAX_CHAIN` to stop register
// pressure from erasing the win.
// ---------------------------------------------------------------------------

/// Hard cap on back-to-back elision-friendly ops kept in SSA before a
/// forced spill. Derivation:
///
/// - ARM NEON: 32 v-registers. Three live tensors × ~5 chunks each = 15
///   live F64X2 values — fits with room for temporaries.
/// - x86 AVX2: 16 xmm registers. Tighter but OK at typical chunk counts.
///
/// Depth 6 × 7-chunk p95 = 42 live F64X2 in the worst case — the
/// crossover where NEON avoids spills while x86-16-xmm starts spilling.
const ELISION_MAX_CHAIN: u32 = 6;

/// Consumer instruction kinds (from `useinfo::instr_kind`) that accept
/// `LaneRepr::PtrChunksF64` directly via `get_chunks`. Every other
/// consumer (call, return, gather, ...) forces a spill via
/// `as_scalar_or_spill`.
const ELISION_FRIENDLY_USER_KINDS: &[&str] = &[
    "add", "subtract", "multiply", "divide", "maximum", "minimum", "negate", "sqrt", "abs",
    "floor", "ceil", "nearest", "rsqrt", "sign",
];

/// True when `instr` is one of the ptr-ABI match arms that routes
/// operands through `emit_ptr_*_f64` (and thus handles `PtrChunksF64`
/// natively). The defensive pre-spill skips these arms and fires only
/// for the other arms, where `get()` would otherwise panic.
fn is_elision_aware_mem(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::Add { .. }
            | Instruction::Subtract { .. }
            | Instruction::Multiply { .. }
            | Instruction::Divide { .. }
            | Instruction::Maximum { .. }
            | Instruction::Minimum { .. }
            | Instruction::Negate { .. }
            | Instruction::Sqrt { .. }
            | Instruction::Abs { .. }
            | Instruction::Floor { .. }
            | Instruction::Ceil { .. }
            | Instruction::RoundNearestEven { .. }
            | Instruction::Rsqrt { .. }
            | Instruction::Sign { .. }
    )
}

/// Returns `Some(new_depth)` when the current op can emit `PtrChunksF64`
/// (keep chunks in SSA) without tripping the single-use / user-kind /
/// chain-depth gates. Returns `None` when the op must spill to a stack
/// slot (the legacy `lower_ptr_binop_simd_f64` path).
fn should_elide(
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
    operand_depths: &[u32],
) -> Option<u32> {
    let ui = use_info?;
    let vid = result_vid?;
    let count = ui.use_counts.get(&vid).copied().unwrap_or(0);
    if count != 1 {
        return None;
    }
    let kind = ui.user_kind.get(&vid).copied()?;
    if !ELISION_FRIENDLY_USER_KINDS.contains(&kind) {
        return None;
    }
    let max_op_depth = operand_depths.iter().copied().max().unwrap_or(0);
    let new_depth = max_op_depth + 1;
    if new_depth > ELISION_MAX_CHAIN {
        return None;
    }
    Some(new_depth)
}

/// Elision-or-spill dispatcher for ptr-ABI F64 binary elementwise
/// ops (Add / Sub / Mul / Div). Produces either
/// `LaneRepr::PtrChunksF64` (operands' chunks consumed directly,
/// result kept in SSA) or `LaneRepr::Scalar([ptr])` (fall back to
/// `lower_ptr_binop_simd_f64`, which stores to a fresh stack slot).
///
/// On the elision path, `dst_slot` is never allocated — no `stack_addr`,
/// no chunk stores, no intermediate memory traffic. The next
/// elision-friendly consumer (proven by `should_elide`) will read the
/// chunks directly via `LaneRepr::get_chunks`.
fn lower_ptr_binop_elision_or_spill(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    lhs_repr: &mut LaneRepr,
    rhs_repr: &mut LaneRepr,
    lhs_vid: Option<ValueId>,
    rhs_vid: Option<ValueId>,
    n: usize,
    elem_sz: usize,
    emit_op: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> LaneRepr {
    let l_depth = lhs_repr.ptr_chain_depth();
    let r_depth = rhs_repr.ptr_chain_depth();
    if let Some(new_depth) = should_elide(use_info, result_vid, &[l_depth, r_depth]) {
        // Green: keep the result in SSA. Pull operand chunks (from
        // register if already PtrChunksF64, otherwise load from slot)
        // and emit the op chunk-by-chunk without storing.
        let (l_chunks, l_tail) = lhs_repr.get_chunks(builder, n);
        let (r_chunks, r_tail) = rhs_repr.get_chunks(builder, n);
        let chunk_count = n / 2;
        let mut out_chunks = Vec::with_capacity(chunk_count);
        for i in 0..chunk_count {
            let v = emit_op(builder, l_chunks[i], r_chunks[i]);
            out_chunks.push(v);
        }
        let out_tail = if n & 1 == 1 {
            Some(emit_op(
                builder,
                l_tail.expect("odd-n lhs missing tail"),
                r_tail.expect("odd-n rhs missing tail"),
            ))
        } else {
            None
        };
        return LaneRepr::ptr_chunks(out_chunks, out_tail, n, new_depth);
    }

    // Red: spill to a fresh stack slot via the legacy helper. Operands
    // may themselves be PtrChunksF64 — spill_to_slot_pooled makes sure
    // we have a ptr, and (when the operand vid is known) hands the
    // spilled slot to the pool so it returns after the operand's last
    // use.
    let lhs_ptr = lhs_repr.spill_to_slot_pooled(builder, Some(pool), lhs_vid);
    let rhs_ptr = rhs_repr.spill_to_slot_pooled(builder, Some(pool), rhs_vid);
    let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
    lower_ptr_binop_simd_f64(builder, jit_module, dst, lhs_ptr, rhs_ptr, n, emit_op);
    LaneRepr::scalar(vec![dst])
}

/// Elision-or-spill dispatcher for ptr-ABI F64 Max/Min. Emits the
/// XLA `fcmp + bitselect`/`select` per chunk so NaN handling matches
/// `lower_ptr_cmp_select_simd_f64` bit-for-bit. Green path: operand
/// chunks come from `get_chunks`, result stays in SSA; no stack slot
/// allocated. Red path: operands spill (no-op if already `Scalar`)
/// and `lower_ptr_cmp_select_simd_f64` writes into a fresh slot.
fn lower_ptr_cmp_select_elision_or_spill(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    lhs_repr: &mut LaneRepr,
    rhs_repr: &mut LaneRepr,
    lhs_vid: Option<ValueId>,
    rhs_vid: Option<ValueId>,
    n: usize,
    elem_sz: usize,
    cc: FloatCC,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> LaneRepr {
    let l_depth = lhs_repr.ptr_chain_depth();
    let r_depth = rhs_repr.ptr_chain_depth();
    if let Some(new_depth) = should_elide(use_info, result_vid, &[l_depth, r_depth]) {
        let (l_chunks, l_tail) = lhs_repr.get_chunks(builder, n);
        let (r_chunks, r_tail) = rhs_repr.get_chunks(builder, n);
        let chunk_count = n / 2;
        let mut out_chunks = Vec::with_capacity(chunk_count);
        for i in 0..chunk_count {
            let cmp_i = builder.ins().fcmp(cc, l_chunks[i], r_chunks[i]);
            let cmp_f = builder.ins().bitcast(types::F64X2, MemFlags::new(), cmp_i);
            let r = builder.ins().bitselect(cmp_f, l_chunks[i], r_chunks[i]);
            out_chunks.push(r);
        }
        let out_tail = if n & 1 == 1 {
            let lt = l_tail.expect("odd-n lhs missing tail");
            let rt = r_tail.expect("odd-n rhs missing tail");
            let cmp = builder.ins().fcmp(cc, lt, rt);
            Some(builder.ins().select(cmp, lt, rt))
        } else {
            None
        };
        return LaneRepr::ptr_chunks(out_chunks, out_tail, n, new_depth);
    }

    let lhs_ptr = lhs_repr.spill_to_slot_pooled(builder, Some(pool), lhs_vid);
    let rhs_ptr = rhs_repr.spill_to_slot_pooled(builder, Some(pool), rhs_vid);
    let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
    lower_ptr_cmp_select_simd_f64(builder, jit_module, dst, lhs_ptr, rhs_ptr, n, cc);
    LaneRepr::scalar(vec![dst])
}

/// Clone operands out of `value_map`, invoke
/// `lower_ptr_binop_elision_or_spill`, and write the (possibly
/// post-spill) operand LaneReprs back. This is the call-site-facing
/// thin wrapper the Add/Sub/Mul/Div match arms use.
///
/// Writing operands back matters only when the helper chose the spill
/// path AND the operand was itself a PtrChunksF64: in that case the
/// helper mutates the local clone to `Scalar([ptr])` and the write-back
/// keeps `value_map` consistent for any subsequent reader of the same
/// vid. Elision-path operands are untouched (`get_chunks` is `&self`),
/// so the write-back is a no-op copy in the hot case.
fn emit_ptr_binop_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
    n: usize,
    elem_sz: usize,
    category: crate::op_sampler::OpCategory,
    emit_op: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> Result<LaneRepr, String> {
    // Bracket the chunk-loop with op_begin/end probes when the
    // op-timing sample counter hits this emission. The recorded ns
    // is scaled by `OP_SAMPLE_RATE` at report time so the
    // accumulator represents the full-sim total for this category.
    let sample = crate::op_sampler::op_times_enabled()
        && OP_SAMPLER.with(|s| s.borrow_mut().should_sample(category));
    if sample {
        emit_op_begin(builder, jit_module, category);
    }
    let mut lhs_repr = value_map
        .get(lhs)
        .cloned()
        .ok_or_else(|| format!("mem: missing value {:?}", lhs))?;
    let mut rhs_repr = value_map
        .get(rhs)
        .cloned()
        .ok_or_else(|| format!("mem: missing value {:?}", rhs))?;
    let out = lower_ptr_binop_elision_or_spill(
        builder,
        jit_module,
        pool,
        &mut lhs_repr,
        &mut rhs_repr,
        Some(*lhs),
        Some(*rhs),
        n,
        elem_sz,
        emit_op,
        use_info,
        result_vid,
    );
    value_map.insert(*lhs, lhs_repr);
    value_map.insert(*rhs, rhs_repr);
    if sample {
        emit_op_end(builder, jit_module);
    }
    Ok(out)
}

/// Elision-or-spill dispatcher for ptr-ABI F64 unary elementwise
/// ops (Negate / Sqrt / Abs / Floor / Ceil / Nearest). Cranelift's
/// `fneg`, `sqrt`, `fabs`, `floor`, `ceil`, `nearest` IR ops are
/// polymorphic over F64 and F64X2, so the same closure serves
/// both the chunk loop and the tail.
fn lower_ptr_unop_elision_or_spill(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    src_repr: &mut LaneRepr,
    src_vid: Option<ValueId>,
    n: usize,
    elem_sz: usize,
    emit_op: impl Fn(&mut FunctionBuilder, Value) -> Value,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> LaneRepr {
    let src_depth = src_repr.ptr_chain_depth();
    if let Some(new_depth) = should_elide(use_info, result_vid, &[src_depth]) {
        let (s_chunks, s_tail) = src_repr.get_chunks(builder, n);
        let out_chunks: Vec<Value> = s_chunks.iter().map(|&c| emit_op(builder, c)).collect();
        let out_tail = s_tail.map(|t| emit_op(builder, t));
        return LaneRepr::ptr_chunks(out_chunks, out_tail, n, new_depth);
    }

    let src_ptr = src_repr.spill_to_slot_pooled(builder, Some(pool), src_vid);
    let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
    lower_ptr_unop_simd_f64(builder, jit_module, dst, src_ptr, n, emit_op);
    LaneRepr::scalar(vec![dst])
}

/// Unary call-site wrapper — clone operand out, invoke
/// `lower_ptr_unop_elision_or_spill`, write back post-spill.
fn emit_ptr_unop_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    operand: &ValueId,
    n: usize,
    elem_sz: usize,
    emit_op: impl Fn(&mut FunctionBuilder, Value) -> Value,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> Result<LaneRepr, String> {
    let mut src_repr = value_map
        .get(operand)
        .cloned()
        .ok_or_else(|| format!("mem: missing value {:?}", operand))?;
    let out = lower_ptr_unop_elision_or_spill(
        builder,
        jit_module,
        pool,
        &mut src_repr,
        Some(*operand),
        n,
        elem_sz,
        emit_op,
        use_info,
        result_vid,
    );
    value_map.insert(*operand, src_repr);
    Ok(out)
}

/// Max/Min call-site wrapper — same clone/invoke/write-back pattern
/// as `emit_ptr_binop_f64`, but dispatches to
/// `lower_ptr_cmp_select_elision_or_spill`.
fn emit_ptr_cmp_select_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    pool: &mut crate::slot_pool::SlotPool,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
    n: usize,
    elem_sz: usize,
    cc: FloatCC,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
) -> Result<LaneRepr, String> {
    let mut lhs_repr = value_map
        .get(lhs)
        .cloned()
        .ok_or_else(|| format!("mem: missing value {:?}", lhs))?;
    let mut rhs_repr = value_map
        .get(rhs)
        .cloned()
        .ok_or_else(|| format!("mem: missing value {:?}", rhs))?;
    let out = lower_ptr_cmp_select_elision_or_spill(
        builder,
        jit_module,
        pool,
        &mut lhs_repr,
        &mut rhs_repr,
        Some(*lhs),
        Some(*rhs),
        n,
        elem_sz,
        cc,
        use_info,
        result_vid,
    );
    value_map.insert(*lhs, lhs_repr);
    value_map.insert(*rhs, rhs_repr);
    Ok(out)
}

/// Inline ptr-ABI SIMD for F64 binary elementwise ops. Replaces a
/// `trt_call` into tensor_rt with an unrolled
/// `load.F64X2 / fop.F64X2 / store.F64X2` chunk loop plus a scalar
/// tail for odd `n`. `emit_op` produces the binary op; Cranelift's
/// arith IR is polymorphic over F64 and F64X2 so the same closure
/// serves both paths.
///
/// `MemFlags::trusted()` is safe: stack slots are 16-byte-aligned
/// via `SLOT_ALIGN = 4`, chunk and tail offsets are multiples of 16,
/// and dst/lhs/rhs pointers come from `stack_addr` (never traps).
/// This unlocks `movapd` on x86 and aligned NEON on AArch64.
fn lower_ptr_binop_simd_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    dst: Value,
    lhs: Value,
    rhs: Value,
    n: usize,
    emit_op: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) {
    let chunks = n / 2;
    let flags = MemFlags::trusted();
    for i in 0..chunks {
        let off = (i * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64X2, flags, lhs, off),
        );
        let b_v = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64X2, flags, rhs, off),
        );
        let r = emit_op(builder, a, b_v);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
    if n & 1 == 1 {
        let off = (chunks * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64, flags, lhs, off),
        );
        let b_v = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64, flags, rhs, off),
        );
        let r = emit_op(builder, a, b_v);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
}

/// Inline ptr-ABI SIMD for F64 unary elementwise ops (Negate, Sqrt,
/// Abs, Floor, Ceil, Nearest). Same chunk-loop pattern as
/// `lower_ptr_binop_simd_f64`; Cranelift IR ops `fneg`, `sqrt`,
/// `fabs`, `floor`, `ceil`, `nearest` are polymorphic over F64 and
/// F64X2 so the same closure serves both chunks and tail.
fn lower_ptr_unop_simd_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    dst: Value,
    src: Value,
    n: usize,
    emit_op: impl Fn(&mut FunctionBuilder, Value) -> Value,
) {
    let chunks = n / 2;
    let flags = MemFlags::trusted();
    for i in 0..chunks {
        let off = (i * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64X2, flags, src, off),
        );
        let r = emit_op(builder, a);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
    if n & 1 == 1 {
        let off = (chunks * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64, flags, src, off),
        );
        let r = emit_op(builder, a);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
}

/// Dispatch a ptr-ABI transcendental to the wide-SIMD `tensor_*_f64`
/// helper in `tensor_rt`, bracketed with `xcend_begin(1) / xcend_end`
/// instead of the generic `call_begin / call_end` pair that
/// `trt_call` emits. Pure probe-attribution: the tensor_rt function
/// is already batched, so there's no arithmetic change — the profile
/// just reports transcendental time under the `xcend` family rather
/// than the `call` family. Returns the first arg (dst) for
/// consistency with `trt_call`.
fn trt_call_xcend_unary(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    func_id: FuncId,
    dst: Value,
    src: Value,
    n: usize,
) -> Value {
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    emit_xcend_begin(builder, jit_module, 1);
    builder.ins().call(func_ref, &[dst, src, n_val]);
    emit_xcend_end(builder, jit_module);
    dst
}

/// Binary variant of `trt_call_xcend_unary` for two-operand transcendentals
/// like Atan2 / Power. `dst, a, b, n` arg order.
fn trt_call_xcend_binary(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    func_id: FuncId,
    dst: Value,
    a: Value,
    b: Value,
    n: usize,
) -> Value {
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    emit_xcend_begin(builder, jit_module, 1);
    builder.ins().call(func_ref, &[dst, a, b, n_val]);
    emit_xcend_end(builder, jit_module);
    dst
}

/// Inline ptr-ABI splat of a single F64 scalar into a dst buffer of
/// `n` f64 elements. Used by the `DenseScalar` / `DenseSplat`
/// constant path and the 1-element `BroadcastInDim` fast path to
/// avoid a `trt_call` round-trip. Emits one `splat.F64X2`, unrolls
/// aligned stores into `dst`, plus a scalar tail when `n` is odd.
fn lower_ptr_splat_simd_f64(builder: &mut FunctionBuilder, dst: Value, scalar: Value, n: usize) {
    let chunks = n / 2;
    let flags = MemFlags::trusted();
    let splat = builder.ins().splat(types::F64X2, scalar);
    for i in 0..chunks {
        let off = (i * 16) as i32;
        builder.ins().store(flags, splat, dst, off);
    }
    if n & 1 == 1 {
        let off = (chunks * 16) as i32;
        builder.ins().store(flags, scalar, dst, off);
    }
}

/// Inline ptr-ABI SIMD for F64 Max/Min. Uses the same
/// fcmp + bitcast + bitselect pattern as the scalar-ABI packed path.
/// NaN semantics match XLA: `max(NaN, x) = NaN` — the fcmp mask is
/// false for NaN, so bitselect picks the right operand (x or NaN
/// depending on which side is NaN), matching tensor_rt's scalar loop.
fn lower_ptr_cmp_select_simd_f64(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    dst: Value,
    lhs: Value,
    rhs: Value,
    n: usize,
    cc: FloatCC,
) {
    let chunks = n / 2;
    let flags = MemFlags::trusted();
    for i in 0..chunks {
        let off = (i * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64X2, flags, lhs, off),
        );
        let b_v = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64X2, flags, rhs, off),
        );
        let cmp_i = builder.ins().fcmp(cc, a, b_v);
        // `bitcast` is a value-on-value op; the Cranelift verifier
        // only accepts `big`/`little` byte-order flags, not the
        // memory-access flags we use for loads/stores. Default flags
        // are correct here.
        let cmp_f = builder.ins().bitcast(types::F64X2, MemFlags::new(), cmp_i);
        let r = builder.ins().bitselect(cmp_f, a, b_v);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
    if n & 1 == 1 {
        let off = (chunks * 16) as i32;
        let a = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64, flags, lhs, off),
        );
        let b_v = bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Load,
            |b| b.ins().load(types::F64, flags, rhs, off),
        );
        let cmp = builder.ins().fcmp(cc, a, b_v);
        let r = builder.ins().select(cmp, a, b_v);
        bracket_sampled_inner(
            builder,
            jit_module,
            crate::op_sampler::OpCategory::Store,
            |b| {
                b.ins().store(flags, r, dst, off);
                r
            },
        );
    }
}

fn store_i64_array(builder: &mut FunctionBuilder, vals: &[i64]) -> Value {
    let ptr = alloc_slot(builder, vals.len() * 8);
    for (i, &v) in vals.iter().enumerate() {
        let cv = builder.ins().iconst(types::I64, v);
        builder
            .ins()
            .store(MemFlags::trusted(), cv, ptr, (i * 8) as i32);
    }
    ptr
}

fn lower_pointer_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let param_ptr = block_params[i];
        value_map.insert(*vid, LaneRepr::scalar(vec![param_ptr]));
        type_map.insert(*vid, ty.clone());
    }

    let out_ptr = block_params[func_def.params.len()];

    lower_body_mem(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
        let mut byte_offset = 0i64;
        for (vid, ty) in operands.iter().zip(func_def.result_types.iter()) {
            let lr = value_map
                .get_mut(vid)
                .ok_or_else(|| format!("ptr body: missing return {:?}", vid))?;
            lr.unpack_in(builder);
            let vals = lr.as_scalar().to_vec();
            let dst = builder.ins().iadd_imm(out_ptr, byte_offset);
            let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
            builder.ins().call(memcpy_ref, &[dst, vals[0], nb]);
            byte_offset += ty.byte_size() as i64;
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

/// Test-only: lower `main` through the pointer-ABI path. Activated
/// by `CompileConfig::force_pointer_abi_main = true` which is set
/// only from in-crate integration tests. Never gated by any env var.
fn lower_main_body_mem(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let inputs_ptr = block_params[0];
    let outputs_ptr = block_params[1];

    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let buf_ptr =
            builder
                .ins()
                .load(ptr_type(), MemFlags::trusted(), inputs_ptr, (i * 8) as i32);
        value_map.insert(*vid, LaneRepr::scalar(vec![buf_ptr]));
        type_map.insert(*vid, ty.clone());
    }

    lower_body_mem(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
        for (i, (vid, ty)) in operands
            .iter()
            .zip(func_def.result_types.iter())
            .enumerate()
        {
            let lr = value_map
                .get_mut(vid)
                .ok_or_else(|| format!("main_mem: missing return {:?}", vid))?;
            lr.unpack_in(builder);
            let vals = lr.as_scalar().to_vec();
            let buf_ptr =
                builder
                    .ins()
                    .load(ptr_type(), MemFlags::trusted(), outputs_ptr, (i * 8) as i32);
            let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
            builder.ins().call(memcpy_ref, &[buf_ptr, vals[0], nb]);
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

fn lower_body_mem(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    // Use-info pass so producers can query single-use + consumer-kind
    // to decide between elision (`PtrChunksF64`) and spill
    // (`Scalar([ptr])`).
    let use_info = crate::useinfo::build_use_info(body);
    // Per-body slot pool. Scoped to this `lower_body_mem` call;
    // nested While/Case recursions get their own pool because the
    // recursive call allocates a fresh `SlotPool::default()`.
    let mut pool = crate::slot_pool::SlotPool::new();
    for (pos, ir) in body.iter().enumerate() {
        if matches!(ir.instr, Instruction::Return { .. }) {
            break;
        }
        let result_types: Vec<TensorType> = ir.values.iter().map(|(_, t)| t.clone()).collect();
        let result_vid = ir.values.first().map(|(vid, _)| *vid);
        let result_vals = lower_instruction_mem(
            builder,
            &ir.instr,
            &result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
            Some(&use_info),
            result_vid,
            &mut pool,
        )?;
        for (i, (vid, ty)) in ir.values.iter().enumerate() {
            if i < result_vals.len() {
                value_map.insert(*vid, result_vals[i].clone());
                type_map.insert(*vid, ty.clone());
            }
        }
        // Release pool-owned slots whose last operand use was this
        // instruction. `release_for_vid` is a no-op for vids that
        // don't hold a pool slot (function params, aliased vids,
        // scratch allocations).
        for op in crate::const_fold::operand_ids(&ir.instr) {
            if use_info.last_use_pos.get(&op).copied() == Some(pos) {
                pool.release_for_vid(op);
            }
        }
    }
    // Log pool hit rate in debug mode.
    if pool.total_allocs() > 0 && crate::debug::enabled() {
        eprintln!(
            "[elodin-cranelift] slot_pool: {} allocs, {} hits, {} misses ({:.1}% hit rate)",
            pool.total_allocs(),
            pool.hits,
            pool.misses,
            pool.hit_rate_pct(),
        );
    }
    Ok(())
}

fn lower_instruction_mem(
    builder: &mut FunctionBuilder,
    instr: &Instruction,
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
    use_info: Option<&crate::useinfo::UseInfo>,
    result_vid: Option<ValueId>,
    pool: &mut crate::slot_pool::SlotPool,
) -> Result<Vec<LaneRepr>, String> {
    let rt = result_types
        .first()
        .cloned()
        .unwrap_or(TensorType::scalar(ElementType::F64));
    let n = rt.num_elements();
    let elem_sz = rt.element_type.byte_size();

    // Defensive consumer-side spill. Elision-aware match arms below
    // go through `emit_ptr_*_f64`, which handles `PtrChunksF64`
    // natively. Every other arm uses `get`, which calls `as_scalar`
    // and panics on `PtrChunksF64`. Use-info analysis guarantees no
    // `PtrChunksF64` should land on a non-elision-aware consumer,
    // but edge cases (multi-result ops, newly-wired op kinds,
    // classifier mismatches) could slip through — spill silently
    // rather than panic.
    if !is_elision_aware_mem(instr) {
        let op_ids = crate::const_fold::operand_ids(instr);
        for vid in &op_ids {
            if matches!(value_map.get(vid), Some(LaneRepr::PtrChunksF64 { .. }))
                && let Some(lr) = value_map.get_mut(vid)
            {
                lr.as_scalar_or_spill(builder);
            }
        }
    }

    let get = |vid: &ValueId| -> Result<Value, String> {
        value_map
            .get(vid)
            .and_then(|v| v.as_scalar().first().copied())
            .ok_or_else(|| format!("mem: missing value {:?}", vid))
    };

    let trt_call = |builder: &mut FunctionBuilder,
                    jit_module: &mut JITModule,
                    func_id: FuncId,
                    args: &[Value]|
     -> Value {
        let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
        // Bracket every tensor_rt call site with call_begin/end so
        // per-function attribution can split inline-IR time from
        // time-in-callees. No-op when profiling is disabled.
        emit_call_begin(builder, jit_module);
        builder.ins().call(func_ref, args);
        emit_call_end(builder, jit_module);
        args[0]
    };

    match instr {
        Instruction::Constant { value } => match value {
            ConstantValue::DenseScalar(sv) | ConstantValue::DenseSplat(sv, _) => {
                let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
                let scalar = scalar_to_cranelift(builder, sv, rt.element_type);
                // Inline splat for F64 instead of a trt_call
                // round-trip. F32 / integer types still use tensor_rt.
                if rt.element_type == ElementType::F64 && n >= 2 {
                    lower_ptr_splat_simd_f64(builder, dst, scalar, n);
                    return Ok(vec![LaneRepr::scalar(vec![dst])]);
                }
                let fid = match rt.element_type {
                    ElementType::F64 | ElementType::F32 => trt_ids.broadcast_f64,
                    ElementType::I32 | ElementType::UI32 => trt_ids.broadcast_i32,
                    ElementType::I1 => trt_ids.broadcast_i8,
                    _ => trt_ids.broadcast_i64,
                };
                let n_val = builder.ins().iconst(types::I64, n as i64);
                trt_call(builder, jit_module, fid, &[dst, scalar, n_val]);
                Ok(vec![LaneRepr::scalar(vec![dst])])
            }
            ConstantValue::DenseArray(arr) => {
                let mut bytes = Vec::new();
                for sv in arr {
                    match rt.element_type {
                        ElementType::F64 => bytes.extend_from_slice(&sv.as_f64().to_ne_bytes()),
                        ElementType::F32 => {
                            bytes.extend_from_slice(&(sv.as_f64() as f32).to_ne_bytes())
                        }
                        ElementType::I64 | ElementType::UI64 => {
                            bytes.extend_from_slice(&sv.as_i64().to_ne_bytes())
                        }
                        ElementType::I32 | ElementType::UI32 => {
                            bytes.extend_from_slice(&(sv.as_i64() as i32).to_ne_bytes())
                        }
                        ElementType::I1 => bytes.push(if sv.as_i64() != 0 { 1 } else { 0 }),
                    }
                }
                let actual_bytes = bytes.len();
                let data_id = jit_module
                    .declare_anonymous_data(false, false)
                    .map_err(|e| format!("declare data: {e}"))?;
                let mut desc = DataDescription::new();
                desc.define(bytes.into_boxed_slice());
                desc.set_align(8);
                jit_module
                    .define_data(data_id, &desc)
                    .map_err(|e| format!("define data: {e}"))?;
                let gv = jit_module.declare_data_in_func(data_id, builder.func);
                let data_ptr = builder.ins().global_value(ptr_type(), gv);

                if actual_bytes > 1_000_000 {
                    Ok(vec![LaneRepr::scalar(vec![data_ptr])])
                } else {
                    let dst = alloc_slot_for_vid(builder, pool, result_vid, actual_bytes);
                    let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                    let nb = builder.ins().iconst(types::I64, actual_bytes as i64);
                    builder.ins().call(memcpy_ref, &[dst, data_ptr, nb]);
                    Ok(vec![LaneRepr::scalar(vec![dst])])
                }
            }
        },

        Instruction::Add { lhs, rhs } => {
            // Elision-aware SIMD for F64 tensors. Helper picks
            // between keeping result chunks in SSA (PtrChunksF64) and
            // storing to a fresh stack slot. Fall back to tensor_rt for
            // F32 / ints — those still go through the scalar-pointer path.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_binop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    crate::op_sampler::OpCategory::Fadd,
                    |b, l, r| b.ins().fadd(l, r),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.add_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.add_i32,
                _ => trt_ids.add_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Subtract { lhs, rhs } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_binop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    crate::op_sampler::OpCategory::Fsub,
                    |b, l, r| b.ins().fsub(l, r),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.sub_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.sub_i32,
                _ => trt_ids.sub_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Multiply { lhs, rhs } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_binop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    crate::op_sampler::OpCategory::Fmul,
                    |b, l, r| b.ins().fmul(l, r),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.mul_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.mul_i32,
                _ => trt_ids.mul_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Divide { lhs, rhs } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_binop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    crate::op_sampler::OpCategory::Fdiv,
                    |b, l, r| b.ins().fdiv(l, r),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.div_f64,
                ElementType::UI32 => trt_ids.div_ui32,
                ElementType::I32 => trt_ids.div_i32,
                _ => trt_ids.div_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Maximum { lhs, rhs } => {
            // Elision-aware SIMD Max for F64 on ptr-ABI.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_cmp_select_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    FloatCC::GreaterThan,
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 => trt_ids.max_i32,
                ElementType::UI32 => trt_ids.max_ui32,
                ElementType::I64 | ElementType::UI64 => trt_ids.max_i64,
                _ => trt_ids.max_f64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Minimum { lhs, rhs } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_cmp_select_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    lhs,
                    rhs,
                    n,
                    elem_sz,
                    FloatCC::LessThan,
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 => trt_ids.min_i32,
                ElementType::UI32 => trt_ids.min_ui32,
                ElementType::I64 | ElementType::UI64 => trt_ids.min_i64,
                _ => trt_ids.min_f64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Power { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_binary(
                builder,
                jit_module,
                trt_ids.pow_f64,
                dst,
                get(lhs)?,
                get(rhs)?,
                n,
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Remainder { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I64 => trt_ids.rem_i64,
                ElementType::I32 => trt_ids.rem_i32,
                ElementType::UI32 => trt_ids.rem_ui32,
                _ => trt_ids.rem_f64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Negate { operand } => {
            // Elision-aware SIMD unary for F64 on ptr-ABI.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().fneg(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.neg_i32,
                ElementType::I64 | ElementType::UI64 => trt_ids.neg_i64,
                _ => trt_ids.neg_f64,
            };
            trt_call(builder, jit_module, fid, &[dst, get(operand)?, n_val]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Sqrt { operand } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().sqrt(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.sqrt_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Abs { operand } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().fabs(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.abs_i32,
                ElementType::I64 | ElementType::UI64 => trt_ids.abs_i64,
                _ => trt_ids.abs_f64,
            };
            trt_call(builder, jit_module, fid, &[dst, get(operand)?, n_val]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Floor { operand } => {
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().floor(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.floor_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        // Ptr-ABI transcendentals dispatch to the wide-SIMD
        // `tensor_*_f64` runtime helpers via `trt_call_xcend_unary`
        // so the profile attributes their time to the `xcend` probe
        // family rather than the generic `call` family.
        Instruction::Sine { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.sin_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Cosine { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.cos_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Exponential { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.exp_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Log { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.log_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Tanh { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.tanh_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Reshape { operand } => {
            // Reshape forwards the operand's ptr verbatim; the result
            // vid aliases the operand's slot. Register it as a shared
            // pool holder so the slot stays live until both vids are
            // released.
            if let Some(rv) = result_vid {
                pool.share_owner(*operand, rv);
            }
            Ok(vec![LaneRepr::scalar(vec![get(operand)?])])
        }

        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => {
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let use_float = is_float(l_ty.element_type)
                || matches!(compare_type, CompareType::Float | CompareType::TotalOrder);
            // Inline F64 compare: per-lane `fcmp` + u8 store, no
            // trt_call FFI hop. Output is u8 (0 or 1).
            if use_float && l_ty.element_type == ElementType::F64 {
                let dst = alloc_slot_for_vid(builder, pool, result_vid, n);
                let lhs_ptr = get(lhs)?;
                let rhs_ptr = get(rhs)?;
                let cc = match direction {
                    CompareDirection::Eq => FloatCC::Equal,
                    CompareDirection::Ne => FloatCC::NotEqual,
                    CompareDirection::Lt => FloatCC::LessThan,
                    CompareDirection::Le => FloatCC::LessThanOrEqual,
                    CompareDirection::Gt => FloatCC::GreaterThan,
                    CompareDirection::Ge => FloatCC::GreaterThanOrEqual,
                };
                let flags = MemFlags::trusted();
                for i in 0..n {
                    let off_f = (i * 8) as i32;
                    let off_u = i as i32;
                    let a = builder.ins().load(types::F64, flags, lhs_ptr, off_f);
                    let b = builder.ins().load(types::F64, flags, rhs_ptr, off_f);
                    let cmp = builder.ins().fcmp(cc, a, b);
                    builder.ins().store(flags, cmp, dst, off_u);
                }
                return Ok(vec![LaneRepr::scalar(vec![dst])]);
            }
            let func_id = if use_float {
                match direction {
                    CompareDirection::Eq => trt_ids.cmp_eq_f64,
                    CompareDirection::Ne => trt_ids.cmp_ne_f64,
                    CompareDirection::Lt => trt_ids.cmp_lt_f64,
                    CompareDirection::Le => trt_ids.cmp_le_f64,
                    CompareDirection::Gt => trt_ids.cmp_gt_f64,
                    CompareDirection::Ge => trt_ids.cmp_ge_f64,
                }
            } else if matches!(l_ty.element_type, ElementType::I32 | ElementType::UI32) {
                match direction {
                    CompareDirection::Eq => trt_ids.cmp_eq_i32,
                    CompareDirection::Ne => trt_ids.cmp_ne_i32,
                    CompareDirection::Lt => trt_ids.cmp_lt_i32,
                    CompareDirection::Le => trt_ids.cmp_le_i32,
                    CompareDirection::Gt => trt_ids.cmp_gt_i32,
                    CompareDirection::Ge => trt_ids.cmp_ge_i32,
                }
            } else {
                match direction {
                    CompareDirection::Eq => trt_ids.cmp_eq_i64,
                    CompareDirection::Ne => trt_ids.cmp_ne_i64,
                    CompareDirection::Lt => trt_ids.cmp_lt_i64,
                    CompareDirection::Le => trt_ids.cmp_le_i64,
                    CompareDirection::Gt => trt_ids.cmp_gt_i64,
                    CompareDirection::Ge => trt_ids.cmp_ge_i64,
                }
            };
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            builder
                .ins()
                .call(func_ref, &[dst, get(lhs)?, get(rhs)?, n_val]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            // Inline F64 select: per-lane `select(cond != 0, on_true,
            // on_false)`. No SIMD benefit (cond is u8, mask
            // materialization costs too much), but the tight scalar
            // loop still beats the FFI hop.
            if rt.element_type == ElementType::F64 {
                let cond_ptr = get(cond)?;
                let t_ptr = get(on_true)?;
                let f_ptr = get(on_false)?;
                let flags = MemFlags::trusted();
                for i in 0..n {
                    let off_u = i as i32;
                    let off_f = (i * 8) as i32;
                    let c = builder.ins().load(types::I8, flags, cond_ptr, off_u);
                    let mask = builder.ins().icmp_imm(IntCC::NotEqual, c, 0);
                    let t = builder.ins().load(types::F64, flags, t_ptr, off_f);
                    let f = builder.ins().load(types::F64, flags, f_ptr, off_f);
                    let r = builder.ins().select(mask, t, f);
                    builder.ins().store(flags, r, dst, off_f);
                }
                return Ok(vec![LaneRepr::scalar(vec![dst])]);
            }
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.select_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.select_i32,
                ElementType::I1 => trt_ids.select_i8,
                _ => trt_ids.select_i64,
            };
            let func_ref = jit_module.declare_func_in_func(fid, builder.func);
            builder.ins().call(
                func_ref,
                &[dst, get(cond)?, get(on_true)?, get(on_false)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Convert { operand } => {
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            if src_ty.element_type == rt.element_type
                || (matches!(src_ty.element_type, ElementType::I32 | ElementType::UI32)
                    && matches!(rt.element_type, ElementType::I32 | ElementType::UI32))
                || (matches!(src_ty.element_type, ElementType::I64 | ElementType::UI64)
                    && matches!(rt.element_type, ElementType::I64 | ElementType::UI64))
            {
                // No-op Convert forwards the operand ptr; share pool
                // ownership so the slot stays live for both vids.
                if let Some(rv) = result_vid {
                    pool.share_owner(*operand, rv);
                }
                return Ok(vec![LaneRepr::scalar(vec![get(operand)?])]);
            }
            let func_id = match (src_ty.element_type, rt.element_type) {
                (ElementType::I64, ElementType::F64) => trt_ids.convert_i64_to_f64,
                (ElementType::UI64, ElementType::F64) => trt_ids.convert_ui64_to_f64,
                (ElementType::F64, ElementType::I64) => trt_ids.convert_f64_to_i64,
                (ElementType::I1, ElementType::F64) => trt_ids.convert_i1_to_f64,
                (ElementType::F64, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_f64_to_i32
                }
                (ElementType::I64 | ElementType::UI64, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_i64_to_i32
                }
                (ElementType::I1, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_i1_to_i32
                }
                (ElementType::I32, ElementType::F64) => trt_ids.convert_i32_to_f64,
                (ElementType::UI32, ElementType::F64) => trt_ids.convert_ui32_to_f64,
                (ElementType::F64, ElementType::F32) => trt_ids.convert_f64_to_f32,
                (ElementType::F32, ElementType::F64) => trt_ids.convert_f32_to_f64,
                (ElementType::I32, ElementType::I64 | ElementType::UI64) => {
                    trt_ids.widen_i32_to_i64
                }
                (ElementType::UI32, ElementType::I64 | ElementType::UI64) => {
                    trt_ids.convert_ui32_to_i64
                }
                (ElementType::F64, ElementType::I1) => trt_ids.convert_f64_to_i1,
                (ElementType::I64, ElementType::I1) => trt_ids.convert_i64_to_i1,
                (ElementType::I32, ElementType::F32) => trt_ids.convert_i32_to_f32,
                (ElementType::F32, ElementType::I32) => trt_ids.convert_f32_to_i32,
                _ => {
                    return Err(format!(
                        "mem: unsupported convert {:?} -> {:?}",
                        src_ty.element_type, rt.element_type
                    ));
                }
            };
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            builder.ins().call(func_ref, &[dst, get(operand)?, n_val]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::BroadcastInDim {
            operand,
            broadcast_dims,
        } => {
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let src_n = src_ty.num_elements();
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);

            if src_n == 1 {
                let src_ptr = get(operand)?;
                let src_et = src_ty.element_type;
                let ct = cranelift_type_for(src_et);
                let scalar = builder.ins().load(ct, MemFlags::trusted(), src_ptr, 0);
                // Inline F64 scalar splat, skipping the trt_call
                // round-trip. F32 / integer types still dispatch.
                if src_et == ElementType::F64 && n >= 2 && rt.element_type == ElementType::F64 {
                    lower_ptr_splat_simd_f64(builder, dst, scalar, n);
                } else {
                    let fid = match src_et {
                        ElementType::F64 | ElementType::F32 => trt_ids.broadcast_f64,
                        ElementType::I32 | ElementType::UI32 => trt_ids.broadcast_i32,
                        ElementType::I1 => trt_ids.broadcast_i8,
                        _ => trt_ids.broadcast_i64,
                    };
                    let func_ref = jit_module.declare_func_in_func(fid, builder.func);
                    let n_val = builder.ins().iconst(types::I64, n as i64);
                    builder.ins().call(func_ref, &[dst, scalar, n_val]);
                }
            } else if src_n == n {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let nb = builder.ins().iconst(types::I64, (n * elem_sz) as i64);
                builder.ins().call(memcpy_ref, &[dst, get(operand)?, nb]);
            } else {
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                let n_src_v = builder.ins().iconst(types::I64, src_n as i64);
                let dst_shape_ptr = store_i64_array(builder, &rt.shape);
                let dst_rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
                let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
                let src_rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                let bd_ptr = store_i64_array(builder, broadcast_dims);
                let esz = builder.ins().iconst(types::I64, elem_sz as i64);
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.broadcast_nd_generic, builder.func);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        get(operand)?,
                        n_dst_v,
                        n_src_v,
                        dst_shape_ptr,
                        dst_rank_v,
                        src_shape_ptr,
                        src_rank_v,
                        bd_ptr,
                        esz,
                    ],
                );
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Transpose {
            operand,
            permutation,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            if matches!(rt.element_type, ElementType::F64) && rt.rank() == 2 {
                let func_ref = jit_module.declare_func_in_func(trt_ids.transpose_f64, builder.func);
                let rows_v = builder.ins().iconst(types::I64, src_ty.shape[0]);
                let cols_v = builder.ins().iconst(types::I64, src_ty.shape[1]);
                builder
                    .ins()
                    .call(func_ref, &[dst, get(operand)?, rows_v, cols_v]);
            } else {
                let n_val = builder.ins().iconst(types::I64, n as i64);
                let shape_ptr = store_i64_array(builder, &src_ty.shape);
                let perm_ptr = store_i64_array(builder, permutation);
                let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                let esz = builder.ins().iconst(types::I64, elem_sz as i64);
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.transpose_nd_generic, builder.func);
                builder.ins().call(
                    func_ref,
                    &[dst, get(operand)?, n_val, shape_ptr, perm_ptr, rank_v, esz],
                );
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let starts_ptr = store_i64_array(builder, start_indices);
            let limits_ptr = store_i64_array(builder, limit_indices);
            let esz = builder.ins().iconst(types::I64, elem_sz as i64);
            let func_ref = jit_module.declare_func_in_func(trt_ids.slice_generic, builder.func);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    shape_ptr,
                    rank_v,
                    starts_ptr,
                    limits_ptr,
                    esz,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Concatenate {
            operands,
            dimension,
        } => {
            let dim = *dimension as usize;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);

            if dim == 0 || rt.rank() <= 1 {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let mut byte_off = 0i64;
                for vid in operands {
                    let src_ty = type_map.get(vid).cloned().unwrap_or(rt.clone());
                    let src_bytes = src_ty.byte_size();
                    let d = builder.ins().iadd_imm(dst, byte_off);
                    let nb = builder.ins().iconst(types::I64, src_bytes as i64);
                    builder.ins().call(memcpy_ref, &[d, get(vid)?, nb]);
                    byte_off += src_bytes as i64;
                }
            } else if operands.len() == 2 {
                let a_ty = type_map.get(&operands[0]).cloned().unwrap_or(rt.clone());
                let b_ty = type_map.get(&operands[1]).cloned().unwrap_or(rt.clone());
                let func_ref = jit_module.declare_func_in_func(trt_ids.concat_nd_f64, builder.func);
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                let n_a_v = builder.ins().iconst(types::I64, a_ty.num_elements() as i64);
                let n_b_v = builder.ins().iconst(types::I64, b_ty.num_elements() as i64);
                let dst_shape_ptr = store_i64_array(builder, &rt.shape);
                let a_shape_ptr = store_i64_array(builder, &a_ty.shape);
                let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
                let dim_v = builder.ins().iconst(types::I64, dim as i64);
                let esz_v = builder.ins().iconst(types::I64, elem_sz as i64);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        n_dst_v,
                        get(&operands[0])?,
                        n_a_v,
                        get(&operands[1])?,
                        n_b_v,
                        dst_shape_ptr,
                        a_shape_ptr,
                        rank_v,
                        dim_v,
                        esz_v,
                    ],
                );
            } else {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let mut byte_off = 0i64;
                for vid in operands {
                    let src_ty = type_map.get(vid).cloned().unwrap_or(rt.clone());
                    let src_bytes = src_ty.byte_size();
                    let d = builder.ins().iadd_imm(dst, byte_off);
                    let nb = builder.ins().iconst(types::I64, src_bytes as i64);
                    builder.ins().call(memcpy_ref, &[d, get(vid)?, nb]);
                    byte_off += src_bytes as i64;
                }
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Pad {
            operand,
            padding_value,
            low,
            ..
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let pad_ptr = get(padding_value)?;
            let pad_scalar = builder
                .ins()
                .load(types::F64, MemFlags::trusted(), pad_ptr, 0);
            let func_ref = jit_module.declare_func_in_func(trt_ids.pad_f64, builder.func);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let dst_shape_ptr = store_i64_array(builder, &rt.shape);
            let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
            let low_ptr = store_i64_array(builder, low);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    pad_scalar,
                    dst_shape_ptr,
                    src_shape_ptr,
                    rank_v,
                    low_ptr,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let sizes_ptr = store_i64_array(builder, slice_sizes);

            let starts_ss = alloc_slot(builder, start_indices.len() * 8);
            for (i, idx_vid) in start_indices.iter().enumerate() {
                let idx_ptr = get(idx_vid)?;
                let idx_et = type_map
                    .get(idx_vid)
                    .map(|t| t.element_type)
                    .unwrap_or(ElementType::I64);
                let ct = cranelift_type_for(idx_et);
                let raw = builder.ins().load(ct, MemFlags::trusted(), idx_ptr, 0);
                let idx_val = if ct == types::I32 {
                    builder.ins().sextend(types::I64, raw)
                } else {
                    raw
                };
                builder
                    .ins()
                    .store(MemFlags::trusted(), idx_val, starts_ss, (i * 8) as i32);
            }

            let esz = builder.ins().iconst(types::I64, elem_sz as i64);
            let func_ref =
                jit_module.declare_func_in_func(trt_ids.dynamic_slice_generic, builder.func);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    shape_ptr,
                    rank_v,
                    starts_ss,
                    sizes_ptr,
                    esz,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let upd_ty = type_map.get(update).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let n_upd = upd_ty.num_elements();
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let n_upd_v = builder.ins().iconst(types::I64, n_upd as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let upd_shape_ptr = store_i64_array(builder, &upd_ty.shape);

            let starts_ss = alloc_slot(builder, start_indices.len() * 8);
            for (i, idx_vid) in start_indices.iter().enumerate() {
                let idx_ptr = get(idx_vid)?;
                let idx_et = type_map
                    .get(idx_vid)
                    .map(|t| t.element_type)
                    .unwrap_or(ElementType::I64);
                let ct = cranelift_type_for(idx_et);
                let raw = builder.ins().load(ct, MemFlags::trusted(), idx_ptr, 0);
                let idx_val = if ct == types::I32 {
                    builder.ins().sextend(types::I64, raw)
                } else {
                    raw
                };
                builder
                    .ins()
                    .store(MemFlags::trusted(), idx_val, starts_ss, (i * 8) as i32);
            }

            let esz = builder.ins().iconst(types::I64, elem_sz as i64);
            let func_ref =
                jit_module.declare_func_in_func(trt_ids.dynamic_update_slice_generic, builder.func);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    get(update)?,
                    n_src_v,
                    n_upd_v,
                    shape_ptr,
                    rank_v,
                    starts_ss,
                    upd_shape_ptr,
                    esz,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Gather {
            operand,
            indices,
            dims,
            slice_sizes,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I64));

            let ivd = dims.index_vector_dim as usize;
            let idx_rank = idx_ty.rank();
            let n_index_dims = if ivd < idx_rank {
                idx_ty.shape[ivd] as usize
            } else {
                1
            };
            let use_nd = n_index_dims > 1 && !dims.start_index_map.is_empty();

            let n_total_idx = idx_ty.num_elements();
            let idx_ptr = get(indices)?;
            let widened_idx = if matches!(idx_ty.element_type, ElementType::I32 | ElementType::UI32)
            {
                let wide_buf = alloc_slot(builder, n_total_idx * 8);
                let widen_ref =
                    jit_module.declare_func_in_func(trt_ids.widen_i32_to_i64, builder.func);
                let n_v = builder.ins().iconst(types::I64, n_total_idx as i64);
                builder.ins().call(widen_ref, &[wide_buf, idx_ptr, n_v]);
                wide_buf
            } else {
                idx_ptr
            };

            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);

            let n_src_v = builder
                .ins()
                .iconst(types::I64, src_ty.num_elements() as i64);
            let esz = builder.ins().iconst(types::I64, elem_sz as i64);

            if use_nd {
                let n_batch = if idx_rank > 1 {
                    n_total_idx / n_index_dims
                } else {
                    1
                };
                let n_batch_v = builder.ins().iconst(types::I64, n_batch as i64);
                let n_idx_dims_v = builder.ins().iconst(types::I64, n_index_dims as i64);
                let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
                let src_rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                let sim_ptr = store_i64_array(builder, &dims.start_index_map);
                let ss_ptr = store_i64_array(builder, slice_sizes);
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.gather_nd_generic, builder.func);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        get(operand)?,
                        n_src_v,
                        widened_idx,
                        n_batch_v,
                        n_idx_dims_v,
                        src_shape_ptr,
                        src_rank_v,
                        sim_ptr,
                        ss_ptr,
                        n_dst_v,
                        esz,
                    ],
                );
            } else {
                let n_idx = if !dims.collapsed_slice_dims.is_empty() {
                    idx_ty.shape.first().copied().unwrap_or(1) as usize
                } else {
                    n_total_idx
                };
                let row_size = if n_idx > 0 { n / n_idx } else { 1 };
                let n_idx_v = builder.ins().iconst(types::I64, n_idx as i64);
                let row_v = builder.ins().iconst(types::I64, row_size as i64);
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.gather_generic, builder.func);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        get(operand)?,
                        n_src_v,
                        widened_idx,
                        n_idx_v,
                        row_v,
                        esz,
                    ],
                );
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I64));
            let upd_ty = type_map.get(updates).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let n_updates = idx_ty.num_elements();
            let inner_size = if n_updates > 0 {
                upd_ty.num_elements() / n_updates
            } else {
                1
            };
            let idx_ptr = get(indices)?;
            let widened_idx = if matches!(idx_ty.element_type, ElementType::I32 | ElementType::UI32)
            {
                let wide_buf = alloc_slot(builder, n_updates * 8);
                let widen_ref =
                    jit_module.declare_func_in_func(trt_ids.widen_i32_to_i64, builder.func);
                let n_upd_v2 = builder.ins().iconst(types::I64, n_updates as i64);
                builder
                    .ins()
                    .call(widen_ref, &[wide_buf, idx_ptr, n_upd_v2]);
                wide_buf
            } else {
                idx_ptr
            };
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let n_upd_v = builder.ins().iconst(types::I64, n_updates as i64);
            let inner_v = builder.ins().iconst(types::I64, inner_size as i64);
            let esz = builder.ins().iconst(types::I64, elem_sz as i64);
            let func_ref = jit_module.declare_func_in_func(trt_ids.scatter_generic, builder.func);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_src_v,
                    widened_idx,
                    get(updates)?,
                    n_upd_v,
                    inner_v,
                    esz,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::DotGeneral { lhs, rhs, dims } => {
            let l_ty = type_map.get(lhs).cloned().unwrap_or(rt.clone());
            let r_ty = type_map.get(rhs).cloned().unwrap_or(rt.clone());
            let (m, k, nn) = if l_ty.rank() == 1 && r_ty.rank() == 1 {
                let k = l_ty.shape[0] as usize;
                (1usize, k, 1usize)
            } else if l_ty.rank() >= 2 && r_ty.rank() >= 2 {
                let m = l_ty.shape[0] as usize;
                let nn = r_ty.shape[r_ty.rank() - 1] as usize;
                let k = if !dims.lhs_contracting.is_empty() {
                    l_ty.shape[dims.lhs_contracting[0] as usize] as usize
                } else {
                    1
                };
                (m, k, nn)
            } else if l_ty.rank() == 1 && r_ty.rank() >= 2 {
                let k = l_ty.shape[0] as usize;
                let nn = r_ty.shape[r_ty.rank() - 1] as usize;
                (1usize, k, nn)
            } else {
                let m = l_ty.shape[0] as usize;
                let k = r_ty.shape[0] as usize;
                (m, k, 1usize)
            };
            let out_size = m * nn * elem_sz;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, out_size);
            let func_ref = jit_module.declare_func_in_func(trt_ids.matmul_f64, builder.func);
            let m_v = builder.ins().iconst(types::I64, m as i64);
            let k_v = builder.ins().iconst(types::I64, k as i64);
            let n_v = builder.ins().iconst(types::I64, nn as i64);
            builder
                .ins()
                .call(func_ref, &[dst, get(lhs)?, get(rhs)?, m_v, k_v, n_v]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Reduce {
            operand,
            init: _,
            op,
            dimensions: _,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let use_int = !is_float(src_ty.element_type);
            let func_id = match (op, use_int) {
                (ReduceOp::Add, false) => trt_ids.reduce_sum_f64,
                (ReduceOp::Add, true) => trt_ids.reduce_sum_i64,
                (ReduceOp::Maximum, false) => trt_ids.reduce_max_f64,
                (ReduceOp::Maximum, true) => trt_ids.reduce_max_i64,
                (ReduceOp::Minimum, false) => trt_ids.reduce_min_f64,
                (ReduceOp::Minimum, true) => trt_ids.reduce_min_i64,
                (ReduceOp::And, _) => trt_ids.reduce_and_i1,
                (ReduceOp::Or, _) => trt_ids.reduce_or_i1,
            };
            let n_in = src_ty.num_elements();
            let n_out = n;
            let inner = if n_out > 0 { n_in / n_out } else { n_in };
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n_out * elem_sz);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            let outer_v = builder.ins().iconst(types::I64, n_out as i64);
            let inner_v = builder.ins().iconst(types::I64, inner as i64);
            builder
                .ins()
                .call(func_ref, &[dst, get(operand)?, outer_v, inner_v]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::ReduceArgminmax {
            values,
            indices,
            dimensions: _,
            is_min,
        } => {
            let val_ty = type_map.get(values).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map.get(indices).cloned().unwrap_or(TensorType {
                shape: val_ty.shape.clone(),
                element_type: ElementType::I64,
            });
            let n_in = val_ty.num_elements();
            let n_out = rt.num_elements().max(1);
            let inner = if n_out > 0 { n_in / n_out } else { n_in };
            let val_sz = val_ty.element_type.byte_size();
            let idx_sz = idx_ty.element_type.byte_size();
            let dst_v = alloc_slot(builder, n_out * val_sz);
            let dst_i = alloc_slot(builder, n_out * idx_sz);
            let fid = if *is_min {
                trt_ids.argmin_f64
            } else {
                trt_ids.argmax_f64
            };
            let func_ref = jit_module.declare_func_in_func(fid, builder.func);
            let outer_v = builder.ins().iconst(types::I64, n_out as i64);
            let inner_v = builder.ins().iconst(types::I64, inner as i64);
            builder.ins().call(
                func_ref,
                &[dst_v, dst_i, get(values)?, get(indices)?, outer_v, inner_v],
            );
            Ok(vec![
                LaneRepr::scalar(vec![dst_v]),
                LaneRepr::scalar(vec![dst_i]),
            ])
        }

        Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
        } => {
            let mut slots: Vec<(cranelift_codegen::ir::StackSlot, TensorType)> = Vec::new();
            for vid in init_values {
                let ty = type_map
                    .get(vid)
                    .cloned()
                    .unwrap_or(TensorType::scalar(ElementType::F64));
                let byte_sz = ty.byte_size();
                let ss = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    byte_sz as u32,
                    SLOT_ALIGN,
                ));
                let addr = builder.ins().stack_addr(ptr_type(), ss, 0);
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let nb = builder.ins().iconst(types::I64, byte_sz as i64);
                builder.ins().call(memcpy_ref, &[addr, get(vid)?, nb]);
                slots.push((ss, ty));
            }

            let header = builder.create_block();
            let body_blk = builder.create_block();
            let exit = builder.create_block();
            builder.ins().jump(header, &[]);

            builder.switch_to_block(header);
            let mut cond_vm = value_map.clone();
            let mut cond_tm = type_map.clone();
            for (i, (ss, ty)) in slots.iter().enumerate() {
                let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
                cond_vm.insert(vid, LaneRepr::scalar(vec![addr]));
                cond_tm.insert(vid, ty.clone());
            }
            lower_body_mem(
                builder,
                cond_body,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &mut cond_vm,
                &mut cond_tm,
            )?;

            let cond_val = {
                let mut cv = None;
                for ir in cond_body.iter().rev() {
                    if let Instruction::Return { operands } = &ir.instr {
                        if let Some(vid) = operands.first() {
                            let ptr = cond_vm
                                .get(vid)
                                .and_then(|v| v.as_scalar().first().copied());
                            if let Some(p) = ptr {
                                cv = Some(builder.ins().load(types::I8, MemFlags::trusted(), p, 0));
                            }
                        }
                        break;
                    }
                }
                cv.ok_or("while: no condition value")?
            };
            builder.ins().brif(cond_val, body_blk, &[], exit, &[]);

            builder.switch_to_block(body_blk);
            builder.seal_block(body_blk);
            let mut body_vm = value_map.clone();
            let mut body_tm = type_map.clone();
            for (i, (ss, ty)) in slots.iter().enumerate() {
                let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
                body_vm.insert(vid, LaneRepr::scalar(vec![addr]));
                body_tm.insert(vid, ty.clone());
            }
            lower_body_mem(
                builder,
                loop_body,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &mut body_vm,
                &mut body_tm,
            )?;

            if let Some(ret_ir) = loop_body
                .iter()
                .rev()
                .find(|ir| matches!(ir.instr, Instruction::Return { .. }))
                && let Instruction::Return { operands } = &ret_ir.instr
            {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                for (vid, (ss, ty)) in operands.iter().zip(slots.iter()) {
                    if let Some(lr) = body_vm.get_mut(vid) {
                        lr.unpack_in(builder);
                        let vals = lr.as_scalar().to_vec();
                        let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                        let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
                        builder.ins().call(memcpy_ref, &[addr, vals[0], nb]);
                    }
                }
            }
            builder.ins().jump(header, &[]);
            builder.seal_block(header);

            builder.switch_to_block(exit);
            builder.seal_block(exit);

            let mut result_groups = Vec::new();
            for i in 0..result_types.len() {
                if i < slots.len() {
                    let (ss, _) = &slots[i];
                    let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                    result_groups.push(LaneRepr::scalar(vec![addr]));
                }
            }
            Ok(result_groups)
        }

        Instruction::Iota { dimension } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let fid = if is_float(rt.element_type) {
                trt_ids.iota_nd_f64
            } else {
                trt_ids.iota_nd_i64
            };
            let func_ref = jit_module.declare_func_in_func(fid, builder.func);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let shape_ptr = store_i64_array(builder, &rt.shape);
            let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
            let dim_v = builder.ins().iconst(types::I64, *dimension);
            builder
                .ins()
                .call(func_ref, &[dst, n_val, shape_ptr, rank_v, dim_v]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Call { callee, args } => {
            let fid = func_ids
                .get(callee)
                .ok_or_else(|| format!("mem: unknown callee: {callee}"))?;
            let callee_def = ir_module
                .get_func(callee)
                .ok_or_else(|| format!("mem: no func def for {callee}"))?;
            let callee_abi = func_abis.get(callee).copied().unwrap_or(FuncAbi::Scalar);
            let func_ref = jit_module.declare_func_in_func(*fid, builder.func);

            if callee_abi == FuncAbi::Pointer {
                let mut call_args: Vec<Value> = args.iter().map(get).collect::<Result<_, _>>()?;
                let total_ret_bytes: usize =
                    callee_def.result_types.iter().map(|t| t.byte_size()).sum();
                let ret_buf = alloc_slot(builder, total_ret_bytes.max(MIN_RETURN_SLOT));
                call_args.push(ret_buf);
                builder.ins().call(func_ref, &call_args);

                let mut result_groups = Vec::new();
                let mut off = 0i64;
                for rty in &callee_def.result_types {
                    let addr = builder.ins().iadd_imm(ret_buf, off);
                    result_groups.push(LaneRepr::scalar(vec![addr]));
                    off += rty.byte_size() as i64;
                }
                Ok(result_groups)
            } else {
                let callee_sret = needs_sret(&callee_def.result_types);
                let mut call_args = Vec::new();
                for (vid, (_pv, pty)) in args.iter().zip(callee_def.params.iter()) {
                    let ptr = get(vid)?;
                    let n_elem = pty.num_elements();
                    let ct = cranelift_type_for(pty.element_type);
                    let esz = pty.element_type.byte_size();
                    for j in 0..n_elem {
                        let v = builder
                            .ins()
                            .load(ct, MemFlags::trusted(), ptr, (j * esz) as i32);
                        call_args.push(v);
                    }
                }

                if callee_sret {
                    let total_bytes: usize =
                        callee_def.result_types.iter().map(|t| t.byte_size()).sum();
                    let ret_buf = alloc_slot(builder, total_bytes);
                    call_args.push(ret_buf);
                    builder.ins().call(func_ref, &call_args);

                    let mut result_groups = Vec::new();
                    let mut byte_off = 0i64;
                    for rty in &callee_def.result_types {
                        let addr = builder.ins().iadd_imm(ret_buf, byte_off);
                        result_groups.push(LaneRepr::scalar(vec![addr]));
                        byte_off += rty.byte_size() as i64;
                    }
                    Ok(result_groups)
                } else {
                    let call = builder.ins().call(func_ref, &call_args);
                    let results: Vec<Value> = builder.inst_results(call).to_vec();

                    let mut result_groups = Vec::new();
                    let mut off = 0;
                    for rty in &callee_def.result_types {
                        let n_elem = rty.num_elements();
                        let esz = rty.element_type.byte_size();
                        let buf = alloc_slot(builder, n_elem * esz);
                        for j in 0..n_elem {
                            if off + j < results.len() {
                                builder.ins().store(
                                    MemFlags::trusted(),
                                    results[off + j],
                                    buf,
                                    (j * esz) as i32,
                                );
                            }
                        }
                        off += n_elem;
                        result_groups.push(LaneRepr::scalar(vec![buf]));
                    }
                    Ok(result_groups)
                }
            }
        }

        Instruction::Atan2 { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_binary(
                builder,
                jit_module,
                trt_ids.atan2_f64,
                dst,
                get(lhs)?,
                get(rhs)?,
                n,
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Acos { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.acos_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::ErfInv { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            // ErfInv falls back through the regular call path — its
            // wide-SIMD variant is on the critical path only for cbrt
            // and a few others not yet here; keep as trt_call.
            trt_call(
                builder,
                jit_module,
                trt_ids.erf_inv_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Tan { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.tan_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Sign { operand } => {
            // Elision-aware SIMD Sign for F64 on ptr-ABI.
            // Emits `fcmp gt 0` + `fcmp lt 0` + two selects per lane.
            // The F64X2 path uses bitcast + bitselect because `select`
            // requires a scalar i8 condition; the scalar-tail path uses
            // the simpler `select` form directly.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| {
                        let ty = b.func.dfg.value_type(v);
                        let zero_s = b.ins().f64const(0.0);
                        let one_s = b.ins().f64const(1.0);
                        let neg_one_s = b.ins().f64const(-1.0);
                        if ty == types::F64X2 {
                            let zero = b.ins().splat(types::F64X2, zero_s);
                            let one = b.ins().splat(types::F64X2, one_s);
                            let neg_one = b.ins().splat(types::F64X2, neg_one_s);
                            let is_pos_i = b.ins().fcmp(FloatCC::GreaterThan, v, zero);
                            let is_pos_f = b.ins().bitcast(types::F64X2, MemFlags::new(), is_pos_i);
                            let is_neg_i = b.ins().fcmp(FloatCC::LessThan, v, zero);
                            let is_neg_f = b.ins().bitcast(types::F64X2, MemFlags::new(), is_neg_i);
                            let step1 = b.ins().bitselect(is_pos_f, one, zero);
                            b.ins().bitselect(is_neg_f, neg_one, step1)
                        } else {
                            let is_pos = b.ins().fcmp(FloatCC::GreaterThan, v, zero_s);
                            let is_neg = b.ins().fcmp(FloatCC::LessThan, v, zero_s);
                            let step1 = b.ins().select(is_pos, one_s, zero_s);
                            b.ins().select(is_neg, neg_one_s, step1)
                        }
                    },
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.sign_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::BitcastConvert { operand } => {
            // Aliasing op: result ptr == operand ptr; share pool
            // ownership so the slot stays live for both vids.
            if let Some(rv) = result_vid {
                pool.share_owner(*operand, rv);
            }
            Ok(vec![LaneRepr::scalar(vec![get(operand)?])])
        }
        Instruction::RoundNearestEven { operand } => {
            // Elision-aware SIMD nearest for F64 on ptr-ABI.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().nearest(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.round_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Reverse {
            operand,
            dimensions,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_val = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let dims_ptr = store_i64_array(builder, dimensions);
            let n_dims_val = builder.ins().iconst(types::I64, dimensions.len() as i64);
            let func_ref = jit_module.declare_func_in_func(trt_ids.reverse_f64, builder.func);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_val,
                    shape_ptr,
                    rank_val,
                    dims_ptr,
                    n_dims_val,
                ],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Clamp { operand, min, max } => {
            // Inline F64 clamp: `if src<min { min } else if src>max
            // { max } else { src }`. SIMD path uses fcmp + bitselect
            // per chunk; scalar tail uses fcmp + select.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
                let src_ptr = get(operand)?;
                let min_ptr = get(min)?;
                let max_ptr = get(max)?;
                let flags = MemFlags::trusted();
                let chunks = n / 2;
                for i in 0..chunks {
                    let off = (i * 16) as i32;
                    let s = builder.ins().load(types::F64X2, flags, src_ptr, off);
                    let mn = builder.ins().load(types::F64X2, flags, min_ptr, off);
                    let mx = builder.ins().load(types::F64X2, flags, max_ptr, off);
                    let gt_max = builder.ins().fcmp(FloatCC::GreaterThan, s, mx);
                    let gt_max_f = builder.ins().bitcast(types::F64X2, MemFlags::new(), gt_max);
                    let inner = builder.ins().bitselect(gt_max_f, mx, s);
                    let lt_min = builder.ins().fcmp(FloatCC::LessThan, s, mn);
                    let lt_min_f = builder.ins().bitcast(types::F64X2, MemFlags::new(), lt_min);
                    let r = builder.ins().bitselect(lt_min_f, mn, inner);
                    builder.ins().store(flags, r, dst, off);
                }
                if n & 1 == 1 {
                    let off = (chunks * 16) as i32;
                    let s = builder.ins().load(types::F64, flags, src_ptr, off);
                    let mn = builder.ins().load(types::F64, flags, min_ptr, off);
                    let mx = builder.ins().load(types::F64, flags, max_ptr, off);
                    let gt_max = builder.ins().fcmp(FloatCC::GreaterThan, s, mx);
                    let inner = builder.ins().select(gt_max, mx, s);
                    let lt_min = builder.ins().fcmp(FloatCC::LessThan, s, mn);
                    let r = builder.ins().select(lt_min, mn, inner);
                    builder.ins().store(flags, r, dst, off);
                }
                return Ok(vec![LaneRepr::scalar(vec![dst])]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.clamp_f64,
                &[dst, get(operand)?, get(min)?, get(max)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Case { index, branches } => {
            let idx_ptr = get(index)?;
            let idx_et = type_map
                .get(index)
                .map(|t| t.element_type)
                .unwrap_or(ElementType::I64);
            let idx_ct = cranelift_type_for(idx_et);
            let raw_idx = builder.ins().load(idx_ct, MemFlags::trusted(), idx_ptr, 0);
            let idx = if idx_ct.bytes() < 8 {
                builder.ins().sextend(types::I64, raw_idx)
            } else {
                raw_idx
            };

            let result_slots: Vec<_> = result_types
                .iter()
                .map(|rty| alloc_slot(builder, rty.byte_size()))
                .collect();

            let branch_blocks: Vec<Block> = (0..branches.len())
                .map(|_| builder.create_block())
                .collect();
            let merge_block = builder.create_block();

            let idx_ty = builder.func.dfg.value_type(idx);
            let empty_args: &[BlockArg] = &[];
            if branches.len() == 1 {
                builder.ins().jump(branch_blocks[0], empty_args);
            } else if branches.len() == 2 {
                let zero = builder.ins().iconst(idx_ty, 0);
                let cmp = builder.ins().icmp(IntCC::Equal, idx, zero);
                builder.ins().brif(
                    cmp,
                    branch_blocks[0],
                    empty_args,
                    branch_blocks[1],
                    empty_args,
                );
            } else {
                for i in 0..branches.len() - 1 {
                    let cmp_val = builder.ins().iconst(idx_ty, i as i64);
                    let cmp = builder.ins().icmp(IntCC::Equal, idx, cmp_val);
                    let next = if i == branches.len() - 2 {
                        branch_blocks[branches.len() - 1]
                    } else {
                        builder.create_block()
                    };
                    builder
                        .ins()
                        .brif(cmp, branch_blocks[i], empty_args, next, empty_args);
                    if i < branches.len() - 2 {
                        builder.switch_to_block(next);
                        builder.seal_block(next);
                    }
                }
            }

            let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            for (bi, branch) in branches.iter().enumerate() {
                builder.switch_to_block(branch_blocks[bi]);
                builder.seal_block(branch_blocks[bi]);

                // Big-branch fast path: split the branch into its own
                // pointer-ABI function so the giant Cranelift IR it
                // would otherwise produce (elision chains, shape-array
                // stores, dot_general helper calls per op) doesn't
                // stack up inside the caller. Callee returns via the
                // same `result_slots` the inline path would have
                // written to, keeping the merge handoff identical.
                if crate::const_fold::count_body_instructions(branch) > CASE_BRANCH_SPLIT_INSTRS {
                    let captured_vids = collect_captured_vids(branch);
                    let mut captured: Vec<(ValueId, TensorType)> =
                        Vec::with_capacity(captured_vids.len());
                    for vid in &captured_vids {
                        let Some(ty) = type_map.get(vid).cloned() else {
                            continue;
                        };
                        captured.push((*vid, ty));
                    }

                    let branch_fid = compile_case_branch_as_function_mem(
                        jit_module,
                        ir_module,
                        func_ids,
                        func_abis,
                        libm_ids,
                        trt_ids,
                        branch,
                        result_types,
                        &captured,
                    )?;

                    let func_ref = jit_module.declare_func_in_func(branch_fid, builder.func);
                    let mut call_args: Vec<Value> =
                        Vec::with_capacity(result_slots.len() + captured.len());
                    call_args.extend_from_slice(&result_slots);
                    for (vid, _) in &captured {
                        let ptr = value_map
                            .get_mut(vid)
                            .ok_or_else(|| format!("mem case: missing captured {vid:?}"))?;
                        ptr.unpack_in(builder);
                        let vals = ptr.as_scalar().to_vec();
                        call_args.push(vals[0]);
                    }
                    builder.ins().call(func_ref, &call_args);
                    builder.ins().jump(merge_block, empty_args);
                    continue;
                }

                let mut br_vm = value_map.clone();
                let mut br_tm = type_map.clone();
                lower_body_mem(
                    builder, branch, ir_module, func_ids, libm_ids, trt_ids, func_abis, jit_module,
                    &mut br_vm, &mut br_tm,
                )?;

                if let Some(ret_ir) = branch
                    .iter()
                    .rev()
                    .find(|ir| matches!(ir.instr, Instruction::Return { .. }))
                    && let Instruction::Return { operands } = &ret_ir.instr
                {
                    for (i, vid) in operands.iter().enumerate() {
                        if let (Some(rty), Some(lr)) = (result_types.get(i), br_vm.get_mut(vid)) {
                            lr.unpack_in(builder);
                            let vals = lr.as_scalar().to_vec();
                            let nb = builder.ins().iconst(types::I64, rty.byte_size() as i64);
                            builder
                                .ins()
                                .call(memcpy_ref, &[result_slots[i], vals[0], nb]);
                        }
                    }
                }
                builder.ins().jump(merge_block, empty_args);
            }

            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            Ok(result_slots
                .into_iter()
                .map(|s| LaneRepr::scalar(vec![s]))
                .collect())
        }
        Instruction::CustomCall { call_target, .. } => {
            Err(format!("mem: custom_call not yet supported: {call_target}"))
        }

        Instruction::Return { .. } => Ok(vec![]),

        Instruction::Rsqrt { operand } => {
            // Elision-aware SIMD Rsqrt for F64 on ptr-ABI.
            // `1.0 / sqrt(x)` — two Cranelift IR ops per lane,
            // polymorphic over F64 and F64X2.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| {
                        let ty = b.func.dfg.value_type(v);
                        let one_scalar = b.ins().f64const(1.0);
                        let one = if ty == types::F64X2 {
                            b.ins().splat(types::F64X2, one_scalar)
                        } else {
                            one_scalar
                        };
                        let s = b.ins().sqrt(v);
                        b.ins().fdiv(one, s)
                    },
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.rsqrt_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Log1p { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(
                builder,
                jit_module,
                trt_ids.log1p_f64,
                dst,
                get(operand)?,
                n,
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Ceil { operand } => {
            // Elision-aware SIMD ceil for F64 on ptr-ABI.
            if rt.element_type == ElementType::F64 && n >= 2 {
                let out = emit_ptr_unop_f64(
                    builder,
                    jit_module,
                    pool,
                    value_map,
                    operand,
                    n,
                    elem_sz,
                    |b, v| b.ins().ceil(v),
                    use_info,
                    result_vid,
                )?;
                return Ok(vec![out]);
            }
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.ceil_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Asin { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.asin_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Atan { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.atan_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Sinh { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.sinh_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Cosh { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.cosh_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Erfc { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.erfc_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Expm1 { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(
                builder,
                jit_module,
                trt_ids.expm1_f64,
                dst,
                get(operand)?,
                n,
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Cbrt { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            trt_call_xcend_unary(builder, jit_module, trt_ids.cbrt_f64, dst, get(operand)?, n);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Sort {
            inputs,
            dimension,
            comparator,
            ..
        } => {
            let src_ty = type_map.get(&inputs[0]).cloned().unwrap_or(rt.clone());
            let dim = if *dimension < 0 {
                src_ty.rank() as i64 + dimension
            } else {
                *dimension
            } as usize;
            let shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
            let sort_len = shape.get(dim).copied().unwrap_or(n);
            let ascending = !comparator.iter().any(|ir| {
                matches!(
                    &ir.instr,
                    Instruction::Compare {
                        direction: CompareDirection::Gt,
                        ..
                    }
                )
            });

            if inputs.len() == 2 {
                let idx_ty = type_map
                    .get(&inputs[1])
                    .cloned()
                    .unwrap_or(TensorType::scalar(ElementType::I64));
                let val_esz = src_ty.element_type.byte_size();
                let idx_esz = idx_ty.element_type.byte_size();
                let val_n = src_ty.num_elements();
                let idx_n = idx_ty.num_elements();
                let dst_vals = alloc_slot(builder, val_n * val_esz);
                let dst_idxs = alloc_slot(builder, idx_n * idx_esz);
                let val_nb = builder.ins().iconst(types::I64, (val_n * val_esz) as i64);
                let idx_nb = builder.ins().iconst(types::I64, (idx_n * idx_esz) as i64);
                trt_call(
                    builder,
                    jit_module,
                    trt_ids.memcpy,
                    &[dst_vals, get(&inputs[0])?, val_nb],
                );
                trt_call(
                    builder,
                    jit_module,
                    trt_ids.memcpy,
                    &[dst_idxs, get(&inputs[1])?, idx_nb],
                );
                let sort_v = builder.ins().iconst(types::I64, sort_len as i64);
                let asc_v = builder.ins().iconst(types::I8, ascending as i64);
                let pt = ptr_type();
                lapack_call(
                    builder,
                    jit_module,
                    "__trt_argsort_f64",
                    &[pt, pt, types::I64, types::I8],
                    &[dst_vals, dst_idxs, sort_v, asc_v],
                )?;
                return Ok(vec![
                    LaneRepr::scalar(vec![dst_vals]),
                    LaneRepr::scalar(vec![dst_idxs]),
                ]);
            }

            if inputs.len() != 1 {
                return Err(format!("mem sort: unsupported {} operands", inputs.len()));
            }
            let mut n_outer = 1usize;
            for s in shape.iter().take(dim) {
                n_outer *= *s;
            }
            let mut n_inner = 1usize;
            for s in shape.iter().skip(dim + 1) {
                n_inner *= *s;
            }
            let total = n;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, total * elem_sz);
            let total_v = builder.ins().iconst(types::I64, (total * elem_sz) as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.memcpy,
                &[dst, get(&inputs[0])?, total_v],
            );
            let outer_v = builder.ins().iconst(types::I64, n_outer as i64);
            let sort_v = builder.ins().iconst(types::I64, sort_len as i64);
            let inner_v = builder.ins().iconst(types::I64, n_inner as i64);
            let ascending_v = builder.ins().iconst(types::I8, ascending as i64);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_sort_f64",
                &[pt, types::I64, types::I64, types::I64, types::I8],
                &[dst, outer_v, sort_v, inner_v, ascending_v],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::IsFinite { operand } => {
            // Inline `is_finite`: matches `f64::is_finite()` via
            // `abs(x) < INFINITY` — NaN and ±Inf both compare false
            // (fcmp on NaN returns false; abs(Inf) is not less-than
            // Inf). Output is u8.
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n);
            let src_ptr = get(operand)?;
            let flags = MemFlags::trusted();
            let inf = builder.ins().f64const(f64::INFINITY);
            for i in 0..n {
                let off_f = (i * 8) as i32;
                let off_u = i as i32;
                let x = builder.ins().load(types::F64, flags, src_ptr, off_f);
                let ax = builder.ins().fabs(x);
                let ok = builder.ins().fcmp(FloatCC::LessThan, ax, inf);
                builder.ins().store(flags, ok, dst, off_u);
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Not { operand } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I1 => trt_ids.not_i1,
                ElementType::I32 | ElementType::UI32 => trt_ids.not_i32,
                _ => trt_ids.not_i64,
            };
            trt_call(builder, jit_module, fid, &[dst, get(operand)?, n_val]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::ShiftRightArithmetic { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.sshr_i32,
                _ => trt_ids.sshr_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Xor { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.xor_i32,
                _ => trt_ids.xor_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::Or { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I1 => trt_ids.or_i8,
                ElementType::I32 | ElementType::UI32 => trt_ids.or_i32,
                _ => trt_ids.or_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::And { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I1 => trt_ids.and_i8,
                ElementType::I32 | ElementType::UI32 => trt_ids.and_i32,
                _ => trt_ids.and_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::ShiftLeft { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.shl_i32,
                _ => trt_ids.shl_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
        Instruction::ShiftRightLogical { lhs, rhs } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => trt_ids.ushr_i32,
                _ => trt_ids.ushr_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::BatchNormInference {
            operand,
            scale,
            offset,
            mean,
            variance,
            epsilon,
            feature_index,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let eps_val = builder.ins().f64const(*epsilon);
            let fi_val = builder.ins().iconst(types::I64, *feature_index);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_val = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_batch_norm_inference_f64",
                &[
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    types::F64,
                    types::I64,
                    pt,
                    types::I64,
                    types::I64,
                ],
                &[
                    dst,
                    get(operand)?,
                    get(scale)?,
                    get(offset)?,
                    get(mean)?,
                    get(variance)?,
                    eps_val,
                    fi_val,
                    shape_ptr,
                    rank_val,
                    n_val,
                ],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::RealDynamicSlice {
            operand,
            start_indices,
            limit_indices,
            strides,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_val = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let esz = builder.ins().iconst(types::I64, elem_sz as i64);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_real_dynamic_slice",
                &[pt, pt, pt, pt, pt, pt, types::I64, types::I64, types::I64],
                &[
                    dst,
                    get(operand)?,
                    get(start_indices)?,
                    get(limit_indices)?,
                    get(strides)?,
                    shape_ptr,
                    rank_val,
                    esz,
                    n_dst_v,
                ],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Map {
            inputs,
            body,
            body_params,
            ..
        } => {
            let n_inputs = inputs.len();
            let scalar_ty = TensorType::scalar(ElementType::F64);
            let param_types_cr: Vec<Type> = (0..n_inputs).map(|_| types::F64).collect();
            let param_tt: Vec<TensorType> = (0..n_inputs).map(|_| scalar_ty.clone()).collect();
            let region_fid = compile_region_as_function(
                jit_module,
                ir_module,
                func_ids,
                func_abis,
                libm_ids,
                trt_ids,
                &param_types_cr,
                types::F64,
                body,
                body_params,
                &param_tt,
            )?;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let fn_ptr_val = jit_module.declare_func_in_func(region_fid, builder.func);
            let fn_addr = builder.ins().func_addr(ptr_type(), fn_ptr_val);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let n_in_val = builder.ins().iconst(types::I64, n_inputs as i64);
            let input_ptrs_slot = alloc_slot(builder, n_inputs * 8);
            for (i, inp) in inputs.iter().enumerate() {
                let p = get(inp)?;
                builder
                    .ins()
                    .store(MemFlags::trusted(), p, input_ptrs_slot, (i * 8) as i32);
            }
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_map_f64",
                &[pt, pt, types::I64, types::I64, pt],
                &[dst, input_ptrs_slot, n_in_val, n_val, fn_addr],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::ReduceWindow {
            operands,
            init_values,
            body,
            body_params,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
        } => {
            if operands.is_empty() {
                return Err("reduce_window: no operands".into());
            }
            let src_ty = type_map.get(&operands[0]).cloned().unwrap_or(rt.clone());
            let scalar_ty = TensorType::scalar(ElementType::F64);
            let param_tt = vec![scalar_ty.clone(), scalar_ty.clone()];
            let region_fid = compile_region_as_function(
                jit_module,
                ir_module,
                func_ids,
                func_abis,
                libm_ids,
                trt_ids,
                &[types::F64, types::F64],
                types::F64,
                body,
                body_params,
                &param_tt,
            )?;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let fn_ptr_val = jit_module.declare_func_in_func(region_fid, builder.func);
            let fn_addr = builder.ins().func_addr(ptr_type(), fn_ptr_val);
            let init_ptr = get(&init_values[0])?;
            let init_scalar = builder
                .ins()
                .load(types::F64, MemFlags::trusted(), init_ptr, 0);
            let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_val = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let n_out_val = builder.ins().iconst(types::I64, n as i64);
            let wd_ptr = store_i64_array(builder, window_dimensions);
            let ws_ptr = store_i64_array(builder, window_strides);
            let bd_ptr = store_i64_array(builder, base_dilations);
            let wdil_ptr = store_i64_array(builder, window_dilations);
            let pad_flat: Vec<i64> = padding.iter().flat_map(|&(l, h)| [l, h]).collect();
            let pad_ptr = store_i64_array(builder, &pad_flat);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_reduce_window_f64",
                &[
                    pt,
                    pt,
                    types::F64,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    types::I64,
                    types::I64,
                    pt,
                ],
                &[
                    dst,
                    get(&operands[0])?,
                    init_scalar,
                    wd_ptr,
                    ws_ptr,
                    bd_ptr,
                    wdil_ptr,
                    pad_ptr,
                    src_shape_ptr,
                    rank_val,
                    n_out_val,
                    fn_addr,
                ],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::SelectAndScatter {
            operand,
            source,
            init_value,
            select_body,
            select_params,
            scatter_body,
            scatter_params,
            window_dimensions,
            window_strides,
            padding,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let scalar_ty = TensorType::scalar(ElementType::F64);
            let param_tt2 = vec![scalar_ty.clone(), scalar_ty.clone()];
            let select_fid = compile_region_as_function(
                jit_module,
                ir_module,
                func_ids,
                func_abis,
                libm_ids,
                trt_ids,
                &[types::F64, types::F64],
                types::I8,
                select_body,
                select_params,
                &param_tt2,
            )?;
            let scatter_fid = compile_region_as_function(
                jit_module,
                ir_module,
                func_ids,
                func_abis,
                libm_ids,
                trt_ids,
                &[types::F64, types::F64],
                types::F64,
                scatter_body,
                scatter_params,
                &param_tt2,
            )?;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let sel_ref = jit_module.declare_func_in_func(select_fid, builder.func);
            let sel_addr = builder.ins().func_addr(ptr_type(), sel_ref);
            let scat_ref = jit_module.declare_func_in_func(scatter_fid, builder.func);
            let scat_addr = builder.ins().func_addr(ptr_type(), scat_ref);
            let init_ptr = get(init_value)?;
            let init_scalar = builder
                .ins()
                .load(types::F64, MemFlags::trusted(), init_ptr, 0);
            let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_val = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let n_op_val = builder
                .ins()
                .iconst(types::I64, src_ty.num_elements() as i64);
            let source_ty = type_map.get(source).cloned().unwrap_or(rt.clone());
            let n_src_val = builder
                .ins()
                .iconst(types::I64, source_ty.num_elements() as i64);
            let wd_ptr = store_i64_array(builder, window_dimensions);
            let ws_ptr = store_i64_array(builder, window_strides);
            let pad_flat: Vec<i64> = padding.iter().flat_map(|&(l, h)| [l, h]).collect();
            let pad_ptr = store_i64_array(builder, &pad_flat);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_select_and_scatter_f64",
                &[
                    pt,
                    pt,
                    pt,
                    types::F64,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    types::I64,
                    types::I64,
                    types::I64,
                ],
                &[
                    dst,
                    get(operand)?,
                    get(source)?,
                    init_scalar,
                    sel_addr,
                    scat_addr,
                    wd_ptr,
                    ws_ptr,
                    pad_ptr,
                    src_shape_ptr,
                    rank_val,
                    n_op_val,
                    n_src_val,
                ],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Convolution {
            lhs,
            rhs,
            dimension_numbers,
            window_strides,
            padding,
            lhs_dilation,
            rhs_dilation,
            feature_group_count,
            batch_group_count,
        } => {
            let lhs_ty = type_map.get(lhs).cloned().unwrap_or(rt.clone());
            let rhs_ty = type_map.get(rhs).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let dn_vals: Vec<i64> = vec![
                dimension_numbers.input_batch_dimension,
                dimension_numbers.input_feature_dimension,
                dimension_numbers.kernel_input_feature_dimension,
                dimension_numbers.kernel_output_feature_dimension,
                dimension_numbers.output_batch_dimension,
                dimension_numbers.output_feature_dimension,
            ];
            let dn_ptr = store_i64_array(builder, &dn_vals);
            let in_sp_ptr = store_i64_array(builder, &dimension_numbers.input_spatial_dimensions);
            let k_sp_ptr = store_i64_array(builder, &dimension_numbers.kernel_spatial_dimensions);
            let o_sp_ptr = store_i64_array(builder, &dimension_numbers.output_spatial_dimensions);
            let n_spatial = builder.ins().iconst(
                types::I64,
                dimension_numbers.input_spatial_dimensions.len() as i64,
            );
            let lhs_shape_ptr = store_i64_array(builder, &lhs_ty.shape);
            let rhs_shape_ptr = store_i64_array(builder, &rhs_ty.shape);
            let lhs_rank_val = builder.ins().iconst(types::I64, lhs_ty.rank() as i64);
            let ws_ptr = store_i64_array(builder, window_strides);
            let pad_flat: Vec<i64> = padding.iter().flat_map(|&(l, h)| [l, h]).collect();
            let pad_ptr = store_i64_array(builder, &pad_flat);
            let ld_ptr = store_i64_array(builder, lhs_dilation);
            let rd_ptr = store_i64_array(builder, rhs_dilation);
            let fgc = builder.ins().iconst(types::I64, *feature_group_count);
            let bgc = builder.ins().iconst(types::I64, *batch_group_count);
            let n_out_val = builder.ins().iconst(types::I64, n as i64);
            let out_shape_ptr = store_i64_array(builder, &rt.shape);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_conv_f64",
                &[
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    types::I64,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    pt,
                    types::I64,
                    types::I64,
                    types::I64,
                    pt,
                    types::I64,
                ],
                &[
                    dst,
                    get(lhs)?,
                    get(rhs)?,
                    dn_ptr,
                    in_sp_ptr,
                    k_sp_ptr,
                    o_sp_ptr,
                    n_spatial,
                    lhs_shape_ptr,
                    rhs_shape_ptr,
                    ws_ptr,
                    pad_ptr,
                    ld_ptr,
                    rd_ptr,
                    lhs_rank_val,
                    fgc,
                    bgc,
                    out_shape_ptr,
                    n_out_val,
                ],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::CholeskyOp { operand, lower } => {
            // StableHLO spec permits batched input: the last two dims are the
            // matrix dims, any leading dims are batch.  We loop B times,
            // calling the runtime per matrix slice.
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let shape = &src_ty.shape;
            if shape.len() < 2 {
                return Err("mem: Cholesky requires rank >= 2 input".to_string());
            }
            let nn = shape[shape.len() - 1] as usize;
            if shape[shape.len() - 2] as usize != nn {
                return Err(format!(
                    "mem: Cholesky last two dims must be square, got {}x{}",
                    shape[shape.len() - 2],
                    nn,
                ));
            }
            let batch_size: usize = shape[..shape.len() - 2]
                .iter()
                .map(|&d| d as usize)
                .product();
            let f8 = 8;
            let mat_b = nn * nn * f8;
            let info_b = 4;
            let (_, scratch) = lapack_slot(builder, mat_b + mat_b + info_b);
            let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            let dst = alloc_slot_for_vid(builder, pool, result_vid, batch_size * mat_b);
            let operand_ptr = get(operand)?;
            let n_val = builder.ins().iconst(types::I64, nn as i64);
            let lower_val = builder.ins().iconst(types::I32, i64::from(*lower));
            let mat_b_val = builder.ins().iconst(types::I64, mat_b as i64);
            let pt = ptr_type();
            let out_off = mat_b as i32;
            let info_off = (mat_b + mat_b) as i32;
            let out_ptr = builder.ins().iadd_imm(scratch, out_off as i64);
            let info_ptr = builder.ins().iadd_imm(scratch, info_off as i64);
            for b in 0..batch_size {
                let slice_off = (b * mat_b) as i64;
                let src = builder.ins().iadd_imm(operand_ptr, slice_off);
                builder.ins().call(memcpy_ref, &[scratch, src, mat_b_val]);
                lapack_call(
                    builder,
                    jit_module,
                    "__cranelift_cholesky",
                    &[pt, types::I64, types::I32, pt, pt],
                    &[scratch, n_val, lower_val, out_ptr, info_ptr],
                )?;
                let dst_slice = builder.ins().iadd_imm(dst, slice_off);
                builder
                    .ins()
                    .call(memcpy_ref, &[dst_slice, out_ptr, mat_b_val]);
            }
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::TriangularSolve {
            a,
            b,
            left_side: _,
            lower,
            unit_diagonal,
            transpose_a: _,
        } => {
            let a_ty = type_map.get(a).cloned().unwrap_or(rt.clone());
            let b_ty = type_map.get(b).cloned().unwrap_or(rt.clone());
            let m = a_ty.shape[0] as usize;
            let n = a_ty.shape.get(1).copied().unwrap_or(a_ty.shape[0]) as usize;
            let nrhs = if b_ty.num_elements() / m > 0 {
                b_ty.num_elements() / m
            } else {
                1
            };
            let f8 = 8;
            let a_b = m * n * f8;
            let b_b = m * nrhs * f8;
            let out_b = m * nrhs * f8;
            let (_, base) = lapack_slot(builder, a_b + b_b + out_b);
            let memcpy_a = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            let a_nb = builder.ins().iconst(types::I64, a_b as i64);
            builder.ins().call(memcpy_a, &[base, get(a)?, a_nb]);
            let b_base = builder.ins().iadd_imm(base, a_b as i64);
            let memcpy_b = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            let b_nb = builder.ins().iconst(types::I64, b_b as i64);
            builder.ins().call(memcpy_b, &[b_base, get(b)?, b_nb]);
            let out_ptr = builder.ins().iadd_imm(base, (a_b + b_b) as i64);
            let m_val = builder.ins().iconst(types::I64, m as i64);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
            let uplo_val = builder
                .ins()
                .iconst(types::I8, if *lower { b'L' as i64 } else { b'U' as i64 });
            let diag_val = builder.ins().iconst(
                types::I8,
                if *unit_diagonal {
                    b'U' as i64
                } else {
                    b'N' as i64
                },
            );
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__cranelift_trsm",
                &[
                    pt,
                    pt,
                    types::I64,
                    types::I64,
                    types::I64,
                    types::I8,
                    types::I8,
                    pt,
                ],
                &[
                    base, b_base, m_val, n_val, nrhs_val, uplo_val, diag_val, out_ptr,
                ],
            )?;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, out_b);
            let memcpy_out = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            let out_nb = builder.ins().iconst(types::I64, out_b as i64);
            builder.ins().call(memcpy_out, &[dst, out_ptr, out_nb]);
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Fft {
            operand,
            fft_type,
            fft_length,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let fft_n = fft_length
                .first()
                .copied()
                .unwrap_or(src_ty.num_elements() as i64) as usize;
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            let nb = builder
                .ins()
                .iconst(types::I64, (src_ty.num_elements() * elem_sz) as i64);
            builder.ins().call(memcpy_ref, &[dst, get(operand)?, nb]);
            let n_val = builder.ins().iconst(types::I64, fft_n as i64);
            let fft_type_val = builder.ins().iconst(
                types::I8,
                match fft_type {
                    FftType::Fft => 0,
                    FftType::Ifft => 1,
                    FftType::Rfft => 2,
                    FftType::Irfft => 3,
                },
            );
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_fft_f64",
                &[pt, types::I64, types::I8],
                &[dst, n_val, fft_type_val],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }

        Instruction::Rng {
            operands,
            rng_distribution,
        } => {
            let dst = alloc_slot_for_vid(builder, pool, result_vid, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let dist_val = builder.ins().iconst(
                types::I8,
                match rng_distribution {
                    RngDistribution::Uniform => 0,
                    RngDistribution::Normal => 1,
                },
            );
            let min_ptr = if !operands.is_empty() {
                get(&operands[0])?
            } else {
                let z = builder.ins().f64const(0.0);
                let ss = alloc_slot(builder, 8);
                builder.ins().store(MemFlags::trusted(), z, ss, 0);
                ss
            };
            let max_ptr = if operands.len() > 1 {
                get(&operands[1])?
            } else {
                let o = builder.ins().f64const(1.0);
                let ss = alloc_slot(builder, 8);
                builder.ins().store(MemFlags::trusted(), o, ss, 0);
                ss
            };
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_rng_f64",
                &[pt, types::I64, types::I8, pt, pt],
                &[dst, n_val, dist_val, min_ptr, max_ptr],
            )?;
            Ok(vec![LaneRepr::scalar(vec![dst])])
        }
    }
}

fn lower_callee_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let use_sret = needs_sret(&func_def.result_types);
    let mut param_offset = 0;

    for (vid, ty) in &func_def.params {
        let n = ty.num_elements();
        let vals: Vec<Value> = block_params[param_offset..param_offset + n].to_vec();
        value_map.insert(*vid, LaneRepr::scalar(vals));
        type_map.insert(*vid, ty.clone());
        param_offset += n;
    }

    let sret_ptr = if use_sret {
        Some(block_params[param_offset])
    } else {
        None
    };

    lower_body(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last() {
        if let Instruction::Return { operands } = &ret_instr.instr {
            if let Some(out_ptr) = sret_ptr {
                let mut offset = 0i32;
                for vid in operands {
                    let vals = match value_map.get_mut(vid) {
                        Some(lr) => {
                            lr.unpack_in(builder);
                            lr.as_scalar().to_vec()
                        }
                        None => continue,
                    };
                    for &v in &vals {
                        let sz = builder.func.dfg.value_type(v).bytes() as i32;
                        builder.ins().store(MemFlags::trusted(), v, out_ptr, offset);
                        offset += sz;
                    }
                }
                builder.ins().return_(&[]);
            } else {
                let mut ret_vals = Vec::new();
                for vid in operands {
                    if let Some(lr) = value_map.get_mut(vid) {
                        lr.unpack_in(builder);
                        ret_vals.extend_from_slice(lr.as_scalar());
                    }
                }
                builder.ins().return_(&ret_vals);
            }
        } else {
            builder.ins().return_(&[]);
        }
    } else {
        builder.ins().return_(&[]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Body and instruction lowering
// ---------------------------------------------------------------------------

fn lower_body(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    for ir in body {
        if matches!(ir.instr, Instruction::Return { .. }) {
            break;
        }

        let result_types: Vec<TensorType> = ir.values.iter().map(|(_, t)| t.clone()).collect();

        let result_vals = lower_instruction(
            builder,
            &ir.instr,
            &result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        )?;

        for (i, (vid, ty)) in ir.values.iter().enumerate() {
            if i < result_vals.len() {
                value_map.insert(*vid, result_vals[i].clone());
                type_map.insert(*vid, ty.clone());
            }
        }
    }
    Ok(())
}

fn to_block_args(vals: &[Value]) -> Vec<BlockArg> {
    vals.iter().map(|&v| BlockArg::Value(v)).collect()
}

fn make_zero(builder: &mut FunctionBuilder, et: ElementType) -> Value {
    if is_float(et) {
        builder.ins().f64const(0.0)
    } else {
        builder.ins().iconst(cranelift_type_for(et), 0)
    }
}

fn lower_instruction(
    builder: &mut FunctionBuilder,
    instr: &Instruction,
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<LaneRepr>, String> {
    let rt = result_types
        .first()
        .cloned()
        .unwrap_or(TensorType::scalar(ElementType::F64));

    match instr {
        // ----- Constants -----
        Instruction::Constant { value } => Ok(vec![lower_constant(builder, value, &rt)]),

        // ----- Arithmetic -----
        Instruction::Add { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out = elementwise_binop_packed_f64(builder, &l, &r, |b, a, c| {
                    b.ins().fadd(a, c)
                });
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fadd(a, c)
                } else {
                    b.ins().iadd(a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Subtract { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out = elementwise_binop_packed_f64(builder, &l, &r, |b, a, c| {
                    b.ins().fsub(a, c)
                });
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fsub(a, c)
                } else {
                    b.ins().isub(a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Multiply { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out = elementwise_binop_packed_f64(builder, &l, &r, |b, a, c| {
                    b.ins().fmul(a, c)
                });
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fmul(a, c)
                } else {
                    b.ins().imul(a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Divide { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out = elementwise_binop_packed_f64(builder, &l, &r, |b, a, c| {
                    b.ins().fdiv(a, c)
                });
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fdiv(a, c)
                } else if is_unsigned(et) {
                    b.ins().udiv(a, c)
                } else {
                    b.ins().sdiv(a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Maximum { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out =
                    elementwise_cmp_select_packed_f64(builder, &l, &r, FloatCC::GreaterThan);
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let out = elementwise_binop(builder, &l, &r, |b, a, c| {
                if is_float(et) {
                    let cmp = b.ins().fcmp(FloatCC::GreaterThan, a, c);
                    b.ins().select(cmp, a, c)
                } else {
                    let cmp = b.ins().icmp(IntCC::SignedGreaterThan, a, c);
                    b.ins().select(cmp, a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Unary -----
        Instruction::Negate { operand } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().fneg(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| {
                    if is_float(et) {
                        builder.ins().fneg(v)
                    } else {
                        let zero = builder.ins().iconst(cranelift_type_for(et), 0);
                        builder.ins().isub(zero, v)
                    }
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Sqrt { operand } => {
            if rt.element_type == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().sqrt(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().sqrt(v)).collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Transcendental (SIMD via wide crate / libm fallback) -----
        //
        // With 2+ f64 lanes, marshal through a stack buffer and call
        // the SIMD `tensor_rt::tensor_*_f64` helper (f64x2 via the
        // `wide` crate). For scalar tensors, per-element libm wins.
        Instruction::Sine { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.sin_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.sin, jit_module)
        }
        Instruction::Cosine { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.cos_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.cos, jit_module)
        }
        Instruction::Atan2 { lhs, rhs } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_binary_f64(
                    builder,
                    value_map,
                    lhs,
                    rhs,
                    trt_ids.atan2_f64,
                    jit_module,
                );
            }
            lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.atan2, jit_module)
        }
        Instruction::Acos { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.acos_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.acos, jit_module)
        }
        Instruction::Exponential { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.exp_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.exp, jit_module)
        }
        Instruction::Log { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.log_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.log, jit_module)
        }
        Instruction::Power { lhs, rhs } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_binary_f64(
                    builder,
                    value_map,
                    lhs,
                    rhs,
                    trt_ids.pow_f64,
                    jit_module,
                );
            }
            lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.pow, jit_module)
        }
        Instruction::Tan { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.tan_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.tan, jit_module)
        }
        Instruction::Tanh { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.tanh_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.tanh, jit_module)
        }
        Instruction::ErfInv { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.erf_inv, jit_module)
        }

        Instruction::Abs { operand } => {
            let et = rt.element_type;
            // For f64, use Cranelift's native `fabs` (polymorphic over scalar
            // and vector), which produces a single `fabs.F64X2` per chunk on
            // AArch64/x86-64 — no libm call needed.
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().fabs(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = if is_float(et) {
                vals.iter().map(|&v| builder.ins().fabs(v)).collect()
            } else {
                vals.iter()
                    .map(|&v| {
                        let zero = builder.ins().iconst(cranelift_type_for(et), 0);
                        let neg = builder.ins().isub(zero, v);
                        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, v, zero);
                        builder.ins().select(is_neg, neg, v)
                    })
                    .collect()
            };
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Sign { operand } => {
            let vals = get_vals(builder, value_map, operand)?;
            let et = rt.element_type;
            let out: Vec<Value> = if is_float(et) {
                vals.iter()
                    .map(|&v| {
                        let (zero, one, neg_one) = match et {
                            ElementType::F32 => (
                                builder.ins().f32const(0.0),
                                builder.ins().f32const(1.0),
                                builder.ins().f32const(-1.0),
                            ),
                            _ => (
                                builder.ins().f64const(0.0),
                                builder.ins().f64const(1.0),
                                builder.ins().f64const(-1.0),
                            ),
                        };
                        let is_pos = builder.ins().fcmp(FloatCC::GreaterThan, v, zero);
                        let is_neg = builder.ins().fcmp(FloatCC::LessThan, v, zero);
                        let step1 = builder.ins().select(is_pos, one, zero);
                        builder.ins().select(is_neg, neg_one, step1)
                    })
                    .collect()
            } else {
                let ct = cranelift_type_for(et);
                vals.iter()
                    .map(|&v| {
                        let zero = builder.ins().iconst(ct, 0);
                        let one = builder.ins().iconst(ct, 1);
                        let neg_one = builder.ins().iconst(ct, -1i64);
                        let is_pos = builder.ins().icmp(IntCC::SignedGreaterThan, v, zero);
                        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, v, zero);
                        let step1 = builder.ins().select(is_pos, one, zero);
                        builder.ins().select(is_neg, neg_one, step1)
                    })
                    .collect()
            };
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Minimum { lhs, rhs } => {
            let et = rt.element_type;
            if et == ElementType::F64 && rt.num_elements() >= 2 {
                let (l, r) = align_packed_f64(builder, value_map, lhs, rhs)?;
                let out = elementwise_cmp_select_packed_f64(builder, &l, &r, FloatCC::LessThan);
                return Ok(vec![out]);
            }
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let out = elementwise_binop(builder, &l, &r, |b, a, c| {
                if is_float(et) {
                    let cmp = b.ins().fcmp(FloatCC::LessThan, a, c);
                    b.ins().select(cmp, a, c)
                } else {
                    let cmp = b.ins().icmp(IntCC::SignedLessThan, a, c);
                    b.ins().select(cmp, a, c)
                }
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Remainder { lhs, rhs } => {
            if is_float(rt.element_type) {
                lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.fmod, jit_module)
            } else {
                let l = get_vals(builder, value_map, lhs)?.to_vec();
                let r = get_vals(builder, value_map, rhs)?.to_vec();
                let n = l.len().max(r.len());
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let lv = if i < l.len() { l[i] } else { l[0] };
                        let rv = if i < r.len() { r[i] } else { r[0] };
                        builder.ins().srem(lv, rv)
                    })
                    .collect();
                Ok(vec![LaneRepr::scalar(out)])
            }
        }

        Instruction::Clamp { operand, min, max } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let mins = get_vals(builder, value_map, min)?.to_vec();
            let maxs = get_vals(builder, value_map, max)?.to_vec();
            let et = rt.element_type;
            let n = vals.len();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    let v = vals[i];
                    let lo = if i < mins.len() { mins[i] } else { mins[0] };
                    let hi = if i < maxs.len() { maxs[i] } else { maxs[0] };
                    if is_float(et) {
                        let clamped_lo = builder.ins().fmax(v, lo);
                        builder.ins().fmin(clamped_lo, hi)
                    } else {
                        let gt_lo = builder.ins().icmp(IntCC::SignedGreaterThan, v, lo);
                        let step1 = builder.ins().select(gt_lo, v, lo);
                        let lt_hi = builder.ins().icmp(IntCC::SignedLessThan, step1, hi);
                        builder.ins().select(lt_hi, step1, hi)
                    }
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Reverse {
            operand,
            dimensions,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let shape = &src_ty.shape;
            let n = vals.len();

            if shape.is_empty() || n <= 1 {
                return Ok(vec![LaneRepr::scalar(vals)]);
            }

            let mut result = vals.clone();
            let rank = shape.len();
            let strides: Vec<usize> = {
                let mut s = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    s[i] = s[i + 1] * shape[i + 1] as usize;
                }
                s
            };

            for &dim in dimensions {
                let d = dim as usize;
                let dim_size = shape[d] as usize;
                if dim_size <= 1 {
                    continue;
                }
                let mut next = result.clone();
                for (flat_idx, slot) in next.iter_mut().enumerate().take(n) {
                    let coord_d = (flat_idx / strides[d]) % dim_size;
                    let reversed_coord = dim_size - 1 - coord_d;
                    let src_idx = flat_idx - coord_d * strides[d] + reversed_coord * strides[d];
                    *slot = result[src_idx];
                }
                result = next;
            }
            Ok(vec![LaneRepr::scalar(result)])
        }

        Instruction::Floor { operand } => {
            if rt.element_type == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().floor(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().floor(v)).collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::RoundNearestEven { operand } => {
            if rt.element_type == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().nearest(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().nearest(v)).collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Comparison and select -----
        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => {
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let use_float = is_float(l_ty.element_type)
                || matches!(compare_type, CompareType::Float | CompareType::TotalOrder);
            let out: Vec<Value> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| {
                    if use_float {
                        let cc = match direction {
                            CompareDirection::Eq => FloatCC::Equal,
                            CompareDirection::Ne => FloatCC::NotEqual,
                            CompareDirection::Lt => FloatCC::LessThan,
                            CompareDirection::Le => FloatCC::LessThanOrEqual,
                            CompareDirection::Gt => FloatCC::GreaterThan,
                            CompareDirection::Ge => FloatCC::GreaterThanOrEqual,
                        };
                        builder.ins().fcmp(cc, a, b)
                    } else {
                        let uns = matches!(compare_type, CompareType::Unsigned)
                            || is_unsigned(l_ty.element_type);
                        let cc = match (direction, uns) {
                            (CompareDirection::Eq, _) => IntCC::Equal,
                            (CompareDirection::Ne, _) => IntCC::NotEqual,
                            (CompareDirection::Lt, false) => IntCC::SignedLessThan,
                            (CompareDirection::Le, false) => IntCC::SignedLessThanOrEqual,
                            (CompareDirection::Gt, false) => IntCC::SignedGreaterThan,
                            (CompareDirection::Ge, false) => IntCC::SignedGreaterThanOrEqual,
                            (CompareDirection::Lt, true) => IntCC::UnsignedLessThan,
                            (CompareDirection::Le, true) => IntCC::UnsignedLessThanOrEqual,
                            (CompareDirection::Gt, true) => IntCC::UnsignedGreaterThan,
                            (CompareDirection::Ge, true) => IntCC::UnsignedGreaterThanOrEqual,
                        };
                        builder.ins().icmp(cc, a, b)
                    }
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let c = get_vals(builder, value_map, cond)?.to_vec();
            let t = get_vals(builder, value_map, on_true)?.to_vec();
            let f = get_vals(builder, value_map, on_false)?.to_vec();
            let out: Vec<Value> = t
                .iter()
                .zip(f.iter())
                .enumerate()
                .map(|(i, (&tv, &fv))| {
                    let cv = if i < c.len() { c[i] } else { c[0] };
                    builder.ins().select(cv, tv, fv)
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Shape ops -----
        Instruction::Reshape { operand } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let n = rt.num_elements();
            let out = if vals.len() == n {
                vals
            } else if vals.len() > n {
                vals[..n].to_vec()
            } else {
                let mut out = vals;
                let ct = cranelift_type_for(rt.element_type);
                while out.len() < n {
                    out.push(builder.ins().iconst(ct, 0));
                }
                out
            };
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::BroadcastInDim {
            operand,
            broadcast_dims,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n = rt.num_elements();
            if vals.len() == 1 {
                if rt.element_type == ElementType::F64 && n >= 2 {
                    return Ok(vec![splat_f64_to_packed(builder, vals[0], n)]);
                }
                Ok(vec![LaneRepr::scalar(vec![vals[0]; n])])
            } else {
                Ok(vec![LaneRepr::scalar(broadcast_values(
                    &vals,
                    &rt.shape,
                    broadcast_dims,
                    &src_ty.shape,
                ))])
            }
        }

        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![LaneRepr::scalar(slice_tensor(
                &vals,
                &src_ty.shape,
                start_indices,
                limit_indices,
            ))])
        }

        Instruction::Concatenate {
            operands,
            dimension,
        } => {
            let parts: Vec<(Vec<Value>, TensorType)> = operands
                .iter()
                .map(|vid| {
                    let v = get_vals(builder, value_map, vid)?.to_vec();
                    let ty = type_map
                        .get(vid)
                        .cloned()
                        .unwrap_or(TensorType::scalar(ElementType::F64));
                    Ok((v, ty))
                })
                .collect::<Result<_, String>>()?;

            let dim = *dimension as usize;
            if dim == 0 || parts.iter().all(|(_, ty)| ty.rank() <= 1) {
                let mut all_vals = Vec::new();
                for (v, _) in &parts {
                    all_vals.extend_from_slice(v);
                }
                Ok(vec![LaneRepr::scalar(all_vals)])
            } else {
                Ok(vec![LaneRepr::scalar(lower_concatenate_nd(
                    &parts, dim, &rt,
                ))])
            }
        }

        // ----- Type conversions -----
        Instruction::Convert { operand } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| convert_value(builder, v, &src_ty.element_type, &rt.element_type))
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::BitcastConvert { operand } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let dst_ct = cranelift_type_for(rt.element_type);
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| builder.ins().bitcast(dst_ct, MemFlags::new(), v))
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Iota { dimension } => {
            let n = rt.num_elements();
            let ct = cranelift_type_for(rt.element_type);
            let dim = *dimension as usize;
            let shape: Vec<usize> = rt.shape.iter().map(|&d| d as usize).collect();
            let rank = shape.len();
            let mut strides = vec![1usize; rank];
            for d in (0..rank.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out: Vec<Value> = (0..n)
                .map(|flat| {
                    let idx_along_dim = (flat / strides[dim]) % shape[dim];
                    if is_float(rt.element_type) {
                        builder.ins().f64const(idx_along_dim as f64)
                    } else {
                        builder.ins().iconst(ct, idx_along_dim as i64)
                    }
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Integer bitwise ops -----
        Instruction::Xor { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().bxor(a, c))
        }
        Instruction::Or { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().bor(a, c))
        }
        Instruction::And { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().band(a, c))
        }
        Instruction::ShiftLeft { lhs, rhs } => {
            let bit_width = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => 32i64,
                _ => 64i64,
            };
            let ct = cranelift_type_for(rt.element_type);
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                let shifted = b.ins().ishl(a, c);
                let overflow =
                    b.ins()
                        .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, c, bit_width);
                let zero = b.ins().iconst(ct, 0);
                b.ins().select(overflow, zero, shifted)
            });
            Ok(vec![LaneRepr::scalar(out)])
        }
        Instruction::ShiftRightLogical { lhs, rhs } => {
            let bit_width = match rt.element_type {
                ElementType::I32 | ElementType::UI32 => 32i64,
                _ => 64i64,
            };
            let ct = cranelift_type_for(rt.element_type);
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                let shifted = b.ins().ushr(a, c);
                let overflow =
                    b.ins()
                        .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, c, bit_width);
                let zero = b.ins().iconst(ct, 0);
                b.ins().select(overflow, zero, shifted)
            });
            Ok(vec![LaneRepr::scalar(out)])
        }

        // ----- Dot product / matmul -----
        Instruction::DotGeneral { lhs, rhs, dims } => {
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let r_ty = type_map
                .get(rhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));

            // Vectorizable short-dot path: rank-1 · rank-1 with f64 operands
            // and at least 4 lanes, standard contracting dim [0] x [0].
            // Uses `fmul.F64X2` + horizontal reduce.
            let f64_dot =
                l_ty.element_type == ElementType::F64 && r_ty.element_type == ElementType::F64;
            let is_standard_dot_1d = dims.lhs_contracting == [0]
                && dims.rhs_contracting == [0]
                && dims.lhs_batch.is_empty()
                && dims.rhs_batch.is_empty();
            if f64_dot
                && l_ty.rank() == 1
                && r_ty.rank() == 1
                && l_ty.num_elements() >= 4
                && is_standard_dot_1d
                && l_ty.num_elements() == r_ty.num_elements()
            {
                let lp = get_packed_f64(builder, value_map, lhs)?;
                let rp = get_packed_f64(builder, value_map, rhs)?;
                let acc = packed_dot_f64(builder, &lp, &rp);
                return Ok(vec![LaneRepr::scalar(vec![acc])]);
            }

            // Matvec: rank-2 · rank-1 with lhs_contracting = [1],
            // rhs_contracting = [0]. Output is rank-1 with `rows` lanes.
            let is_standard_matvec = dims.lhs_contracting == [1]
                && dims.rhs_contracting == [0]
                && dims.lhs_batch.is_empty()
                && dims.rhs_batch.is_empty();
            if f64_dot
                && l_ty.rank() == 2
                && r_ty.rank() == 1
                && is_standard_matvec
                && (l_ty.shape[1] as usize) >= 4
                && l_ty.shape[1] == r_ty.shape[0]
            {
                let rows = l_ty.shape[0] as usize;
                let cols = l_ty.shape[1] as usize;
                let l = get_vals(builder, value_map, lhs)?.to_vec();
                let rp = get_packed_f64(builder, value_map, rhs)?;
                if l.len() != rows * cols {
                    let l_ty2 = l_ty.clone();
                    let r_ty2 = r_ty.clone();
                    let r = get_vals(builder, value_map, rhs)?.to_vec();
                    return Ok(vec![LaneRepr::scalar(lower_dot_general(
                        builder, &l, &r, &l_ty2, &r_ty2, &rt, dims,
                    ))]);
                }
                let mut out = Vec::with_capacity(rows);
                for row in 0..rows {
                    let row_slice = &l[row * cols..(row + 1) * cols];
                    let lp = pack_f64_slice(builder, row_slice);
                    out.push(packed_dot_f64(builder, &lp, &rp));
                }
                return Ok(vec![LaneRepr::scalar(out)]);
            }

            // Standard matmul: lhs_contracting = [1], rhs_contracting = [0],
            // no batching, both f64, inner dim >= 4. Emits one packed dot
            // product per output cell.
            let is_standard_matmul = dims.lhs_contracting.len() == 1
                && dims.rhs_contracting.len() == 1
                && dims.lhs_contracting[0] == 1
                && dims.rhs_contracting[0] == 0
                && dims.lhs_batch.is_empty()
                && dims.rhs_batch.is_empty();
            if f64_dot
                && l_ty.rank() == 2
                && r_ty.rank() == 2
                && is_standard_matmul
                && (l_ty.shape[1] as usize) >= 4
                && l_ty.shape[1] == r_ty.shape[0]
            {
                let m = l_ty.shape[0] as usize;
                let k = l_ty.shape[1] as usize;
                let n_cols = r_ty.shape[1] as usize;
                let l = get_vals(builder, value_map, lhs)?.to_vec();
                let r = get_vals(builder, value_map, rhs)?.to_vec();
                if l.len() != m * k || r.len() != k * n_cols {
                    let l_ty2 = l_ty.clone();
                    let r_ty2 = r_ty.clone();
                    return Ok(vec![LaneRepr::scalar(lower_dot_general(
                        builder, &l, &r, &l_ty2, &r_ty2, &rt, dims,
                    ))]);
                }
                let mut out = Vec::with_capacity(m * n_cols);
                let mut l_rows = Vec::with_capacity(m);
                for row in 0..m {
                    let row_slice = &l[row * k..(row + 1) * k];
                    l_rows.push(pack_f64_slice(builder, row_slice));
                }
                for lp in l_rows.iter().take(m) {
                    for col in 0..n_cols {
                        let r_col: Vec<Value> =
                            (0..k).map(|i| r[i * n_cols + col]).collect();
                        let rp = pack_f64_slice(builder, &r_col);
                        out.push(packed_dot_f64(builder, lp, &rp));
                    }
                }
                return Ok(vec![LaneRepr::scalar(out)]);
            }

            // Fallback: scalar kernels (handles small k, batched, type mismatches).
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            Ok(vec![LaneRepr::scalar(lower_dot_general(
                builder, &l, &r, &l_ty, &r_ty, &rt, dims,
            ))])
        }

        // ----- Reduce -----
        Instruction::Reduce {
            operand,
            init,
            op,
            dimensions,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let init_vals = get_vals(builder, value_map, init)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![LaneRepr::scalar(lower_reduce(
                builder, &vals, &init_vals, &src_ty, &rt, op, dimensions,
            ))])
        }

        Instruction::ReduceArgminmax {
            values,
            indices,
            is_min,
            ..
        } => {
            let vals = get_vals(builder, value_map, values)?.to_vec();
            let idx_vals = get_vals(builder, value_map, indices)?.to_vec();
            if vals.is_empty() {
                return Err("ReduceArgminmax: empty input".to_string());
            }
            let mut best_v = vals[0];
            let mut best_i = idx_vals[0];
            for k in 1..vals.len() {
                let cmp = if *is_min {
                    builder.ins().fcmp(FloatCC::LessThan, vals[k], best_v)
                } else {
                    builder.ins().fcmp(FloatCC::GreaterThan, vals[k], best_v)
                };
                best_v = builder.ins().select(cmp, vals[k], best_v);
                best_i = builder.ins().select(cmp, idx_vals[k], best_i);
            }
            Ok(vec![LaneRepr::scalar(vec![best_v]), LaneRepr::scalar(vec![best_i])])
        }

        // ----- Gather -----
        Instruction::Gather {
            operand,
            indices,
            dims,
            slice_sizes,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let idx_vals = get_vals(builder, value_map, indices)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I32));
            Ok(vec![LaneRepr::scalar(lower_gather(
                builder,
                &vals,
                &idx_vals,
                &src_ty,
                &idx_ty,
                &rt,
                dims,
                slice_sizes,
            ))])
        }

        // ----- Function call -----
        Instruction::Call { callee, args } => lower_call(
            builder, callee, args, ir_module, func_ids, func_abis, jit_module, value_map,
        ),

        // ----- While loop (real Cranelift loop blocks) -----
        Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
        } => lower_while(
            builder,
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
            result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        ),

        // ----- Case / conditional branching -----
        Instruction::Case { index, branches } => lower_case(
            builder,
            index,
            branches,
            result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        ),

        // ----- Transpose -----
        Instruction::Transpose {
            operand,
            permutation,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![LaneRepr::scalar(lower_transpose(
                &vals,
                &src_ty,
                &rt,
                permutation,
            ))])
        }

        // ----- Dynamic slice -----
        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_vals: Vec<Value> = start_indices
                .iter()
                .map(|v| get_vals(builder, value_map, v).map(|vs| vs[0]))
                .collect::<Result<_, _>>()?;
            Ok(vec![LaneRepr::scalar(lower_dynamic_slice(
                builder,
                &vals,
                &idx_vals,
                &src_ty,
                slice_sizes,
            ))])
        }

        // ----- Dynamic update slice -----
        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let upd = get_vals(builder, value_map, update)?.to_vec();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let upd_ty = type_map.get(update).cloned().unwrap_or(rt.clone());
            let idx_vals: Vec<Value> = start_indices
                .iter()
                .map(|v| get_vals(builder, value_map, v).map(|vs| vs[0]))
                .collect::<Result<_, _>>()?;
            Ok(vec![LaneRepr::scalar(lower_dynamic_update_slice(
                builder, &vals, &upd, &idx_vals, &src_ty, &upd_ty,
            ))])
        }

        Instruction::Pad {
            operand,
            padding_value,
            low,
            high,
            interior,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let pad_val = get_vals(builder, value_map, padding_value)?[0];
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());

            let is_noop = low.iter().all(|&x| x == 0)
                && high.iter().all(|&x| x == 0)
                && interior.iter().all(|&x| x == 0);
            if is_noop {
                return Ok(vec![LaneRepr::scalar(vals)]);
            }

            let n = rt.num_elements();
            let rank = src_ty.shape.len();
            let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
            let out_shape: Vec<usize> = rt.shape.iter().map(|&d| d as usize).collect();
            let mut src_strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
            }
            let mut out_strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }

            let mut result = vec![pad_val; n];
            for (src_flat, &src_val) in vals.iter().enumerate() {
                let mut valid = true;
                let mut out_flat = 0;
                let mut remaining = src_flat;
                for d in 0..rank {
                    let coord = remaining / src_strides[d];
                    remaining %= src_strides[d];
                    let int_step = interior.get(d).copied().unwrap_or(0) as usize;
                    let out_coord =
                        low.get(d).copied().unwrap_or(0) as usize + coord * (1 + int_step);
                    if out_coord >= out_shape[d] {
                        valid = false;
                        break;
                    }
                    out_flat += out_coord * out_strides[d];
                }
                if valid && out_flat < n {
                    result[out_flat] = src_val;
                }
            }
            Ok(vec![LaneRepr::scalar(result)])
        }

        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => {
            let vals = get_vals(builder, value_map, operand)?.to_vec();
            let idx_vals = get_vals(builder, value_map, indices)?.to_vec();
            let upd_vals = get_vals(builder, value_map, updates)?.to_vec();
            let et = rt.element_type;
            let elem_sz = et.byte_size();

            let total_bytes = vals.len() * elem_sz;
            let ss = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                total_bytes as u32,
                SLOT_ALIGN,
            ));
            let base = builder.ins().stack_addr(ptr_type(), ss, 0);
            for (i, &v) in vals.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
            }

            let n_indices = idx_vals.len();
            let inner_size = if n_indices > 0 && upd_vals.len() >= n_indices {
                upd_vals.len() / n_indices
            } else {
                0
            };

            if inner_size > 1 {
                for (u_idx, &upd_v) in upd_vals.iter().enumerate() {
                    let batch = u_idx / inner_size;
                    let inner = u_idx % inner_size;
                    let raw_idx = idx_vals[batch.min(n_indices - 1)];
                    let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
                        raw_idx
                    } else if builder.func.dfg.value_type(raw_idx).bytes() < 8 {
                        builder.ins().sextend(types::I64, raw_idx)
                    } else {
                        raw_idx
                    };
                    let flat_pos = builder.ins().imul_imm(idx_i64, inner_size as i64);
                    let flat_pos = builder.ins().iadd_imm(flat_pos, inner as i64);
                    let byte_offset = builder.ins().imul_imm(flat_pos, elem_sz as i64);
                    let addr = builder.ins().iadd(base, byte_offset);
                    builder.ins().store(MemFlags::trusted(), upd_v, addr, 0);
                }
            } else {
                for (u_idx, &upd_v) in upd_vals.iter().enumerate() {
                    let raw_idx = if u_idx < idx_vals.len() {
                        idx_vals[u_idx]
                    } else {
                        idx_vals[0]
                    };
                    let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
                        raw_idx
                    } else if builder.func.dfg.value_type(raw_idx).bytes() < 8 {
                        builder.ins().sextend(types::I64, raw_idx)
                    } else {
                        raw_idx
                    };
                    let byte_offset = builder.ins().imul_imm(idx_i64, elem_sz as i64);
                    let addr = builder.ins().iadd(base, byte_offset);
                    builder.ins().store(MemFlags::trusted(), upd_v, addr, 0);
                }
            }

            let ct = cranelift_type_for(et);
            let mut result = Vec::with_capacity(vals.len());
            for i in 0..vals.len() {
                let v = builder
                    .ins()
                    .load(ct, MemFlags::trusted(), base, (i * elem_sz) as i32);
                result.push(v);
            }
            Ok(vec![LaneRepr::scalar(result)])
        }

        Instruction::CustomCall {
            call_target,
            operands,
            backend_config,
        } => lower_custom_call(
            builder,
            call_target,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            backend_config,
        ),

        Instruction::Rsqrt { operand } => {
            if rt.element_type == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                let one_x2 = {
                    let one = builder.ins().f64const(1.0);
                    builder.ins().splat(types::F64X2, one)
                };
                let one_scalar = builder.ins().f64const(1.0);
                let mut chunks = Vec::with_capacity(x.chunks.len());
                for c in &x.chunks {
                    let s = builder.ins().sqrt(*c);
                    chunks.push(builder.ins().fdiv(one_x2, s));
                }
                let tail = x.tail.map(|t| {
                    let s = builder.ins().sqrt(t);
                    builder.ins().fdiv(one_scalar, s)
                });
                return Ok(vec![LaneRepr::PackedF64 {
                    chunks,
                    tail,
                    n: x.n,
                }]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| {
                    let s = builder.ins().sqrt(v);
                    let one = builder.ins().f64const(1.0);
                    builder.ins().fdiv(one, s)
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Log1p { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.log1p_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.log1p, jit_module)
        }

        Instruction::IsFinite { operand } => {
            let vals = get_vals(builder, value_map, operand)?;
            let one = builder.ins().iconst(types::I8, 1);
            let zero = builder.ins().iconst(types::I8, 0);
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| {
                    let abs_v = builder.ins().fabs(v);
                    let inf = builder.ins().f64const(f64::INFINITY);
                    let is_fin = builder.ins().fcmp(
                        cranelift_codegen::ir::condcodes::FloatCC::LessThan,
                        abs_v,
                        inf,
                    );
                    builder.ins().select(is_fin, one, zero)
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Not { operand } => {
            let vals = get_vals(builder, value_map, operand)?;
            let et = rt.element_type;
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| {
                    if matches!(et, ElementType::I1) {
                        let one = builder.ins().iconst(cranelift_codegen::ir::types::I8, 1);
                        builder.ins().bxor(v, one)
                    } else {
                        builder.ins().bnot(v)
                    }
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Ceil { operand } => {
            if rt.element_type == ElementType::F64 && rt.num_elements() >= 2 {
                let x = get_packed_f64(builder, value_map, operand)?;
                return Ok(vec![elementwise_unop_packed_f64(builder, &x, |b, v| {
                    b.ins().ceil(v)
                })]);
            }
            let vals = get_vals(builder, value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().ceil(v)).collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::ShiftRightArithmetic { lhs, rhs } => {
            let l = get_vals(builder, value_map, lhs)?.to_vec();
            let r = get_vals(builder, value_map, rhs)?.to_vec();
            let l = l.as_slice();
            let r = r.as_slice();
            let out = elementwise_binop(builder, l, r, |b, a, c| b.ins().sshr(a, c));
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::Asin { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.asin_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.asin, jit_module)
        }
        Instruction::Atan { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.atan_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.atan, jit_module)
        }
        Instruction::Sinh { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.sinh_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.sinh, jit_module)
        }
        Instruction::Cosh { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.cosh_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.cosh, jit_module)
        }
        Instruction::Erfc { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.erfc, jit_module)
        }
        Instruction::Expm1 { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.expm1_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.expm1, jit_module)
        }
        Instruction::Cbrt { operand } => {
            let n = rt.num_elements();
            if rt.element_type == ElementType::F64 && n >= 2 {
                return lower_trt_simd_unary_f64(
                    builder,
                    value_map,
                    operand,
                    trt_ids.cbrt_f64,
                    jit_module,
                );
            }
            lower_libm_unary(builder, value_map, operand, libm_ids.cbrt, jit_module)
        }

        Instruction::Sort {
            inputs,
            dimension,
            comparator,
            ..
        } => {
            if inputs.len() == 2 {
                let vals = get_vals(builder, value_map, &inputs[0])?.to_vec();
                let idxs = get_vals(builder, value_map, &inputs[1])?.to_vec();
                let ascending = !comparator.iter().any(|ir| {
                    matches!(
                        &ir.instr,
                        Instruction::Compare {
                            direction: CompareDirection::Gt,
                            ..
                        }
                    )
                });
                let n_vals = vals.len();
                let val_ss = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (n_vals * 8) as u32,
                    SLOT_ALIGN,
                ));
                let idx_ss = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (n_vals * 8) as u32,
                    SLOT_ALIGN,
                ));
                let val_base = builder.ins().stack_addr(ptr_type(), val_ss, 0);
                let idx_base = builder.ins().stack_addr(ptr_type(), idx_ss, 0);
                for (i, &v) in vals.iter().enumerate() {
                    builder
                        .ins()
                        .store(MemFlags::trusted(), v, val_base, (i * 8) as i32);
                }
                for (i, &v) in idxs.iter().enumerate() {
                    builder
                        .ins()
                        .store(MemFlags::trusted(), v, idx_base, (i * 8) as i32);
                }
                let sort_v = builder.ins().iconst(types::I64, n_vals as i64);
                let asc_v = builder.ins().iconst(types::I8, ascending as i64);
                let pt = ptr_type();
                lapack_call(
                    builder,
                    jit_module,
                    "__trt_argsort_f64",
                    &[pt, pt, types::I64, types::I8],
                    &[val_base, idx_base, sort_v, asc_v],
                )?;
                let out_vals: Vec<Value> = (0..n_vals)
                    .map(|i| {
                        builder.ins().load(
                            types::F64,
                            MemFlags::trusted(),
                            val_base,
                            (i * 8) as i32,
                        )
                    })
                    .collect();
                let out_idxs: Vec<Value> = (0..n_vals)
                    .map(|i| {
                        builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            idx_base,
                            (i * 8) as i32,
                        )
                    })
                    .collect();
                return Ok(vec![
                    LaneRepr::scalar(out_vals),
                    LaneRepr::scalar(out_idxs),
                ]);
            }
            if inputs.len() != 1 {
                return Err(format!(
                    "scalar sort: unsupported {} operands",
                    inputs.len()
                ));
            }
            let vals = get_vals(builder, value_map, &inputs[0])?.to_vec();
            let src_ty = type_map.get(&inputs[0]).cloned().unwrap_or(rt.clone());
            let rank = src_ty.rank();
            let dim = if *dimension < 0 {
                rank as i64 + dimension
            } else {
                *dimension
            } as usize;
            let shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
            let sort_len = shape[dim];
            let mut n_outer = 1usize;
            for s in shape.iter().take(dim) {
                n_outer *= *s;
            }
            let mut n_inner = 1usize;
            for s in shape.iter().skip(dim + 1) {
                n_inner *= *s;
            }

            // Detect ascending vs descending from comparator body
            let ascending = !comparator.iter().any(|ir| {
                matches!(
                    &ir.instr,
                    Instruction::Compare {
                        direction: CompareDirection::Gt,
                        ..
                    }
                )
            });

            // Store all values into a stack buffer, sort at runtime, load back
            let total = vals.len();
            let ss = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                (total * 8) as u32,
                SLOT_ALIGN,
            ));
            let base = builder.ins().stack_addr(ptr_type(), ss, 0);
            for (i, &v) in vals.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), v, base, (i * 8) as i32);
            }

            // Call runtime sort
            let outer_v = builder.ins().iconst(types::I64, n_outer as i64);
            let sort_v = builder.ins().iconst(types::I64, sort_len as i64);
            let inner_v = builder.ins().iconst(types::I64, n_inner as i64);
            let asc_v = builder.ins().iconst(types::I8, ascending as i64);
            let pt = ptr_type();
            lapack_call(
                builder,
                jit_module,
                "__trt_sort_f64",
                &[pt, types::I64, types::I64, types::I64, types::I8],
                &[base, outer_v, sort_v, inner_v, asc_v],
            )?;

            let out: Vec<Value> = (0..total)
                .map(|i| {
                    builder
                        .ins()
                        .load(types::F64, MemFlags::trusted(), base, (i * 8) as i32)
                })
                .collect();
            Ok(vec![LaneRepr::scalar(out)])
        }

        Instruction::BatchNormInference { .. }
        | Instruction::RealDynamicSlice { .. }
        | Instruction::Map { .. }
        | Instruction::ReduceWindow { .. }
        | Instruction::SelectAndScatter { .. }
        | Instruction::Convolution { .. }
        | Instruction::CholeskyOp { .. }
        | Instruction::TriangularSolve { .. }
        | Instruction::Fft { .. }
        | Instruction::Rng { .. } => Err(
            "scalar: op requires pointer-ABI runtime delegation; ensure tensor > 64 elements or set CompileConfig::force_pointer_abi_main (tests only)".to_string()
        ),

        Instruction::Return { .. } => Ok(vec![]),
    }
}

// ---------------------------------------------------------------------------
// While loop — real Cranelift loop with header/body/exit blocks
// ---------------------------------------------------------------------------

fn lower_while(
    builder: &mut FunctionBuilder,
    cond_body: &[InstrResult],
    loop_body: &[InstrResult],
    init_values: &[ValueId],
    iter_arg_ids: &[ValueId],
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<LaneRepr>, String> {
    let mut init_tensors: Vec<(TensorVals, TensorType)> = Vec::new();
    for vid in init_values {
        let vals = get_vals(builder, value_map, vid)?.to_vec();
        let ty = type_map
            .get(vid)
            .cloned()
            .unwrap_or(TensorType::scalar(ElementType::F64));
        init_tensors.push((vals, ty));
    }

    let all_init_flat: Vec<Value> = init_tensors
        .iter()
        .flat_map(|(vals, _)| vals.iter().copied())
        .collect();
    let all_types: Vec<Type> = init_tensors
        .iter()
        .flat_map(|(vals, ty)| std::iter::repeat_n(cranelift_type_for(ty.element_type), vals.len()))
        .collect();

    let header_block = builder.create_block();
    let body_block = builder.create_block();
    let exit_block = builder.create_block();

    for &ct in &all_types {
        builder.append_block_param(header_block, ct);
        builder.append_block_param(body_block, ct);
        builder.append_block_param(exit_block, ct);
    }

    let init_args = to_block_args(&all_init_flat);
    builder.ins().jump(header_block, &init_args);

    // --- Header: evaluate condition ---
    builder.switch_to_block(header_block);
    let header_params = builder.block_params(header_block).to_vec();

    let mut cond_vmap: HashMap<ValueId, LaneRepr> = HashMap::new();
    let mut cond_tmap: HashMap<ValueId, TensorType> = HashMap::new();
    let mut offset = 0;
    for (i, (vals, ty)) in init_tensors.iter().enumerate() {
        let n = vals.len();
        let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
        cond_vmap.insert(
            vid,
            LaneRepr::scalar(header_params[offset..offset + n].to_vec()),
        );
        cond_tmap.insert(vid, ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        cond_body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        &mut cond_vmap,
        &mut cond_tmap,
    )?;

    let cond_val = extract_return_predicate(builder, cond_body, &mut cond_vmap)?;

    let header_args = to_block_args(&header_params);
    builder
        .ins()
        .brif(cond_val, body_block, &header_args, exit_block, &header_args);

    // --- Body: execute loop iteration ---
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    // Emit `__cranelift_loop_iter(loop_id)` at the top of the body
    // block and record per-loop static metadata so the profile can
    // weight runtime iterations by static op counts. No-op when
    // profiling is disabled.
    if with_current_profile_ids(|ids| ids).is_some() {
        let loop_id = next_loop_id();
        emit_loop_iter(builder, jit_module, loop_id);
        let parent_fid = with_current_function_fid(|f| f).unwrap_or(0);
        let body_ops = count_body_op_kinds(loop_body);
        crate::profile::register_loop(loop_id, parent_fid, body_ops);
    }

    let body_params = builder.block_params(body_block).to_vec();
    let mut body_vmap: HashMap<ValueId, LaneRepr> = HashMap::new();
    let mut body_tmap: HashMap<ValueId, TensorType> = HashMap::new();
    offset = 0;
    for (i, (vals, ty)) in init_tensors.iter().enumerate() {
        let n = vals.len();
        let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
        body_vmap.insert(
            vid,
            LaneRepr::scalar(body_params[offset..offset + n].to_vec()),
        );
        body_tmap.insert(vid, ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        loop_body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        &mut body_vmap,
        &mut body_tmap,
    )?;

    let new_vals = extract_return_values(builder, loop_body, &mut body_vmap)?;
    let new_args = to_block_args(&new_vals);
    builder.ins().jump(header_block, &new_args);
    builder.seal_block(header_block);

    // --- Exit: collect results ---
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);

    let exit_params = builder.block_params(exit_block).to_vec();
    let mut result_groups = Vec::new();
    offset = 0;
    for rty in result_types {
        let n = rty.num_elements();
        let end = (offset + n).min(exit_params.len());
        result_groups.push(LaneRepr::scalar(exit_params[offset..end].to_vec()));
        offset = end;
    }
    if result_groups.is_empty() {
        result_groups.push(LaneRepr::scalar(exit_params));
    }

    Ok(result_groups)
}

fn extract_return_predicate(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    value_map: &mut HashMap<ValueId, LaneRepr>,
) -> Result<Value, String> {
    for ir in body.iter().rev() {
        if let Instruction::Return { operands } = &ir.instr
            && let Some(vid) = operands.first()
        {
            let lr = value_map
                .get_mut(vid)
                .ok_or_else(|| format!("cond return value {:?} not found", vid))?;
            lr.unpack_in(builder);
            return Ok(lr.as_scalar()[0]);
        }
    }
    Err("no return instruction in condition body".to_string())
}

fn extract_return_values(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    value_map: &mut HashMap<ValueId, LaneRepr>,
) -> Result<Vec<Value>, String> {
    for ir in body.iter().rev() {
        if let Instruction::Return { operands } = &ir.instr {
            let mut vals = Vec::new();
            for vid in operands {
                let lr = value_map
                    .get_mut(vid)
                    .ok_or_else(|| format!("loop body return value {:?} not found", vid))?;
                lr.unpack_in(builder);
                vals.extend_from_slice(lr.as_scalar());
            }
            return Ok(vals);
        }
    }
    Ok(vec![])
}

// ---------------------------------------------------------------------------
// Case — real branching with dispatch chain and merge block
// ---------------------------------------------------------------------------
//
// Scalar-ABI case branches are intentionally left inline. A scalar-ABI
// function is classified by the 64-element max-tensor rule, so every
// op inside expands to at most 64 Cranelift scalar values; a branch
// body with N StableHLO ops produces O(N * 64) Cranelift IR — bounded
// and cheap for Cranelift's codegen. The large-function compile-time
// explosion comes from the pointer-ABI F64X2 elision chains on big
// tensors, which is handled by the pointer-ABI splitter
// (`compile_case_branch_as_function_mem`) at the `Instruction::Case`
// arm of `lower_instruction_mem`. Splitting scalar-ABI branches
// requires a multi-return sret-style callee and has no measured
// benefit on the current regression suite or the customer workload,
// so it is a documented follow-up.

fn lower_case(
    builder: &mut FunctionBuilder,
    index: &ValueId,
    branches: &[Vec<InstrResult>],
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<LaneRepr>, String> {
    let idx_vals = get_vals(builder, value_map, index)?;
    let idx = idx_vals[0];
    let n_branches = branches.len();

    let merge_block = builder.create_block();
    for rty in result_types {
        let ct = cranelift_type_for(rty.element_type);
        for _ in 0..rty.num_elements() {
            builder.append_block_param(merge_block, ct);
        }
    }

    let branch_blocks: Vec<Block> = (0..n_branches).map(|_| builder.create_block()).collect();

    let idx_ty = builder.func.dfg.value_type(idx);
    let empty_args: &[BlockArg] = &[];
    if n_branches == 1 {
        builder.ins().jump(branch_blocks[0], empty_args);
    } else if n_branches == 2 {
        let zero = builder.ins().iconst(idx_ty, 0);
        let cmp = builder.ins().icmp(IntCC::Equal, idx, zero);
        builder.ins().brif(
            cmp,
            branch_blocks[0],
            empty_args,
            branch_blocks[1],
            empty_args,
        );
    } else {
        for i in 0..n_branches - 1 {
            let cmp_val = builder.ins().iconst(idx_ty, i as i64);
            let cmp = builder.ins().icmp(IntCC::Equal, idx, cmp_val);
            if i == n_branches - 2 {
                builder.ins().brif(
                    cmp,
                    branch_blocks[i],
                    empty_args,
                    branch_blocks[n_branches - 1],
                    empty_args,
                );
            } else {
                let next_dispatch = builder.create_block();
                builder
                    .ins()
                    .brif(cmp, branch_blocks[i], empty_args, next_dispatch, empty_args);
                builder.switch_to_block(next_dispatch);
                builder.seal_block(next_dispatch);
            }
        }
    }

    for (i, branch) in branches.iter().enumerate() {
        builder.switch_to_block(branch_blocks[i]);
        builder.seal_block(branch_blocks[i]);

        // Branch bodies share the parent's value space. Clone the parent maps
        // so that captures of outer-scope values resolve correctly when the
        // parser preserves parent ValueIds for captured names.
        let mut br_vmap = value_map.clone();
        let mut br_tmap = type_map.clone();

        lower_body(
            builder,
            branch,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            &mut br_vmap,
            &mut br_tmap,
        )?;

        let ret_vals = extract_return_values(builder, branch, &mut br_vmap)?;
        let ret_args = to_block_args(&ret_vals);
        builder.ins().jump(merge_block, &ret_args);
    }

    builder.switch_to_block(merge_block);
    builder.seal_block(merge_block);

    let merge_params = builder.block_params(merge_block).to_vec();
    let mut result_groups = Vec::new();
    let mut offset = 0;
    for rty in result_types {
        let n = rty.num_elements();
        let end = (offset + n).min(merge_params.len());
        result_groups.push(LaneRepr::scalar(merge_params[offset..end].to_vec()));
        offset = end;
    }
    if result_groups.is_empty() && !merge_params.is_empty() {
        result_groups.push(LaneRepr::scalar(merge_params));
    }

    Ok(result_groups)
}

// ---------------------------------------------------------------------------
// Function call lowering
// ---------------------------------------------------------------------------

fn lower_call(
    builder: &mut FunctionBuilder,
    callee: &str,
    args: &[ValueId],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, LaneRepr>,
) -> Result<Vec<LaneRepr>, String> {
    let fid = func_ids
        .get(callee)
        .ok_or_else(|| format!("unknown callee: {callee}"))?;
    let callee_def = ir_module
        .get_func(callee)
        .ok_or_else(|| format!("no func def for {callee}"))?;
    let callee_abi = func_abis.get(callee).copied().unwrap_or(FuncAbi::Scalar);

    let func_ref = jit_module.declare_func_in_func(*fid, builder.func);

    if callee_abi == FuncAbi::Pointer {
        // Precompute total bytes so the marshal_begin probe records the
        // size up-front (matching the plan's expected record shape).
        let total_marshal_bytes: u64 = args
            .iter()
            .zip(callee_def.params.iter())
            .map(|(_, (_, t))| t.byte_size() as u64)
            .sum();

        // Bracket the scalar→pointer marshal with begin/end so the
        // profiler attributes its time correctly.
        emit_marshal_begin(builder, jit_module, 0, total_marshal_bytes);

        let mut call_args = Vec::new();
        for (vid, (_param_vid, param_ty)) in args.iter().zip(callee_def.params.iter()) {
            let v = get_vals(builder, value_map, vid)?;
            let byte_sz = param_ty.byte_size();
            let ss = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                byte_sz as u32,
                SLOT_ALIGN,
            ));
            let addr = builder.ins().stack_addr(ptr_type(), ss, 0);
            let elem_sz = param_ty.element_type.byte_size();
            for (j, &val) in v.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), val, addr, (j * elem_sz) as i32);
            }
            call_args.push(addr);
        }

        let total_ret_bytes: usize = callee_def.result_types.iter().map(|t| t.byte_size()).sum();
        let ret_ss = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            total_ret_bytes.max(MIN_RETURN_SLOT) as u32,
            SLOT_ALIGN,
        ));
        let ret_addr = builder.ins().stack_addr(ptr_type(), ret_ss, 0);
        call_args.push(ret_addr);

        emit_marshal_end(builder, jit_module);

        // Wrap the callee invocation with call_begin/end so
        // inline-vs-callee attribution picks up cross-ABI call cost.
        emit_call_begin(builder, jit_module);
        let _call = builder.ins().call(func_ref, &call_args);
        emit_call_end(builder, jit_module);

        // Bracket the pointer→scalar unmarshal separately.
        emit_marshal_begin(builder, jit_module, 1, total_ret_bytes as u64);
        let mut result_groups = Vec::new();
        let mut byte_offset = 0i32;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            let ct = cranelift_type_for(ret_ty.element_type);
            let elem_sz = ret_ty.element_type.byte_size() as i32;
            let mut group = Vec::new();
            for j in 0..n {
                let v = builder.ins().load(
                    ct,
                    MemFlags::trusted(),
                    ret_addr,
                    byte_offset + (j as i32 * elem_sz),
                );
                group.push(v);
            }
            byte_offset += (n as i32) * elem_sz;
            result_groups.push(LaneRepr::scalar(group));
        }
        emit_marshal_end(builder, jit_module);
        return Ok(result_groups);
    }

    let callee_sret = needs_sret(&callee_def.result_types);

    let mut call_args = Vec::new();
    for vid in args {
        let v = get_vals(builder, value_map, vid)?;
        call_args.extend_from_slice(v);
    }

    if callee_sret {
        let total_bytes: usize = callee_def.result_types.iter().map(|t| t.byte_size()).sum();
        let ss = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            total_bytes as u32,
            SLOT_ALIGN,
        ));
        let ss_addr = builder.ins().stack_addr(ptr_type(), ss, 0);
        call_args.push(ss_addr);

        // Bracket the scalar-ABI sret call with begin/end.
        emit_call_begin(builder, jit_module);
        let _call = builder.ins().call(func_ref, &call_args);
        emit_call_end(builder, jit_module);

        let mut result_groups = Vec::new();
        let mut byte_offset = 0i32;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            let ct = cranelift_type_for(ret_ty.element_type);
            let elem_sz = ret_ty.element_type.byte_size() as i32;
            let mut group = Vec::new();
            for j in 0..n {
                let v = builder.ins().load(
                    ct,
                    MemFlags::trusted(),
                    ss_addr,
                    byte_offset + (j as i32 * elem_sz),
                );
                group.push(v);
            }
            byte_offset += (n as i32) * elem_sz;
            result_groups.push(LaneRepr::scalar(group));
        }
        Ok(result_groups)
    } else {
        // Bracket the scalar-ABI direct call with begin/end.
        emit_call_begin(builder, jit_module);
        let call = builder.ins().call(func_ref, &call_args);
        let results: Vec<Value> = builder.inst_results(call).to_vec();
        emit_call_end(builder, jit_module);

        let mut result_groups = Vec::new();
        let mut off = 0;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            if off + n <= results.len() {
                result_groups.push(LaneRepr::scalar(results[off..off + n].to_vec()));
                off += n;
            }
        }
        if result_groups.is_empty() && !results.is_empty() {
            result_groups.push(LaneRepr::scalar(results));
        }
        Ok(result_groups)
    }
}

// ---------------------------------------------------------------------------
// Helpers: constants, conversions, elementwise ops
// ---------------------------------------------------------------------------

fn lower_constant(
    builder: &mut FunctionBuilder,
    value: &ConstantValue,
    ty: &TensorType,
) -> LaneRepr {
    let n = ty.num_elements();
    match value {
        ConstantValue::DenseScalar(sv) => {
            let v = scalar_to_cranelift(builder, sv, ty.element_type);
            if n == 1 {
                LaneRepr::scalar(vec![v])
            } else if ty.element_type == ElementType::F64 && n >= 2 {
                splat_f64_to_packed(builder, v, n)
            } else {
                LaneRepr::scalar(vec![v; n])
            }
        }
        ConstantValue::DenseArray(arr) => LaneRepr::scalar(
            arr.iter()
                .map(|sv| scalar_to_cranelift(builder, sv, ty.element_type))
                .collect(),
        ),
        ConstantValue::DenseSplat(sv, _) => {
            let v = scalar_to_cranelift(builder, sv, ty.element_type);
            if ty.element_type == ElementType::F64 && n >= 2 {
                splat_f64_to_packed(builder, v, n)
            } else {
                LaneRepr::scalar(vec![v; n])
            }
        }
    }
}

/// Splat a single f64 SSA value across an F64X2-packed tensor of `n` lanes.
fn splat_f64_to_packed(builder: &mut FunctionBuilder, v: Value, n: usize) -> LaneRepr {
    let wide = builder.ins().splat(types::F64X2, v);
    let chunks_count = n / 2;
    let chunks = vec![wide; chunks_count];
    let tail = if n % 2 == 1 { Some(v) } else { None };
    LaneRepr::PackedF64 { chunks, tail, n }
}

fn scalar_to_cranelift(builder: &mut FunctionBuilder, sv: &ScalarValue, et: ElementType) -> Value {
    match et {
        ElementType::F64 => builder.ins().f64const(sv.as_f64()),
        ElementType::F32 => builder.ins().f32const(sv.as_f64() as f32),
        ElementType::I64 | ElementType::UI64 => builder.ins().iconst(types::I64, sv.as_i64()),
        ElementType::I32 | ElementType::UI32 => builder.ins().iconst(types::I32, sv.as_i64()),
        ElementType::I1 => builder.ins().iconst(types::I8, sv.as_i64()),
    }
}

fn elementwise_binop(
    builder: &mut FunctionBuilder,
    l: &[Value],
    r: &[Value],
    f: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> TensorVals {
    let n = l.len().max(r.len());
    (0..n)
        .map(|i| {
            let lv = if i < l.len() { l[i] } else { l[0] };
            let rv = if i < r.len() { r[i] } else { r[0] };
            f(builder, lv, rv)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SIMD packed-f64 helpers.
//
// Elementwise f64 ops produce `LaneRepr::PackedF64 { chunks, tail, n }`
// directly: `chunks` are `F64X2` Values, `tail` is a single scalar f64
// for odd element counts. Consumers that need scalars use
// `LaneRepr::unpack_in` to lazily emit `extractlane` instructions.
// ---------------------------------------------------------------------------

/// Pack two scalar f64 values into an F64X2.
fn pack_f64x2(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
    let v0 = builder.ins().splat(types::F64X2, a);
    builder.ins().insertlane(v0, b, 1u8)
}

/// Borrowed view over an operand's packed form. `chunks` is a reference to
/// `F64X2` SSA values, `tail` is an optional scalar f64 for odd tensors.
/// Values are Copy, so this struct is cheap.
#[derive(Clone)]
struct PackedF64Chunks {
    chunks: Vec<Value>,
    tail: Option<Value>,
    n: usize,
}

/// Pack `vid` into `LaneRepr::PackedF64` form in place inside the value_map.
/// If the value is already `PackedF64`, this is a no-op. Caching the packed
/// form here avoids re-packing the same operand across multiple elementwise
/// ops (critical for chains like `mul_add` where the same operand is reused).
fn pack_in_place(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    vid: &ValueId,
) -> Result<(), String> {
    let lr = value_map
        .get_mut(vid)
        .ok_or_else(|| format!("missing value {:?}", vid))?;
    if let LaneRepr::Scalar(scalars) = lr {
        let n = scalars.len();
        let mut chunks = Vec::with_capacity(n / 2);
        let mut i = 0;
        while i + 2 <= n {
            chunks.push(pack_f64x2(builder, scalars[i], scalars[i + 1]));
            i += 2;
        }
        let tail = if i < n { Some(scalars[i]) } else { None };
        *lr = LaneRepr::PackedF64 { chunks, tail, n };
    }
    Ok(())
}

/// Produce a packed F64X2 view of `vid`. The packed form is cached in the
/// value_map so subsequent ops on the same operand skip re-packing.
fn get_packed_f64(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    vid: &ValueId,
) -> Result<PackedF64Chunks, String> {
    pack_in_place(builder, value_map, vid)?;
    match value_map.get(vid) {
        Some(LaneRepr::PackedF64 { chunks, tail, n }) => Ok(PackedF64Chunks {
            chunks: chunks.clone(),
            tail: *tail,
            n: *n,
        }),
        _ => Err(format!("pack_in_place left {:?} non-packed", vid)),
    }
}

/// If one side is a splat scalar (single-element tensor) and the other is a
/// multi-lane tensor, broadcast the scalar into the packed shape of the
/// other operand. Returns the broadcasted packed form.
fn broadcast_scalar_f64_to_packed(
    builder: &mut FunctionBuilder,
    scalar: Value,
    target_len: usize,
) -> PackedF64Chunks {
    let chunks_count = target_len / 2;
    let wide = builder.ins().splat(types::F64X2, scalar);
    let chunks = vec![wide; chunks_count];
    let tail = if target_len % 2 == 1 {
        Some(scalar)
    } else {
        None
    };
    PackedF64Chunks {
        chunks,
        tail,
        n: target_len,
    }
}

/// Resolve two operands into a consistent packed shape, handling StableHLO's
/// scalar-broadcasts-to-shape semantics. Returns `(l, r)` each with `n` lanes.
fn align_packed_f64(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
) -> Result<(PackedF64Chunks, PackedF64Chunks), String> {
    let lhs_l = value_map
        .get(lhs)
        .ok_or_else(|| format!("missing {:?}", lhs))?
        .len();
    let rhs_l = value_map
        .get(rhs)
        .ok_or_else(|| format!("missing {:?}", rhs))?
        .len();
    let target = lhs_l.max(rhs_l);
    let l = if lhs_l == 1 && target > 1 {
        // Get the single scalar, then splat.
        let s = match value_map.get(lhs).unwrap() {
            LaneRepr::Scalar(v) => v[0],
            LaneRepr::PackedF64 { tail: Some(t), .. } => *t,
            LaneRepr::PackedF64 { chunks, .. } => builder.ins().extractlane(chunks[0], 0u8),
            LaneRepr::PtrChunksF64 { .. } => {
                unreachable!("PtrChunksF64 leaked into scalar-ABI align_packed_f64")
            }
        };
        broadcast_scalar_f64_to_packed(builder, s, target)
    } else {
        get_packed_f64(builder, value_map, lhs)?
    };
    let r = if rhs_l == 1 && target > 1 {
        let s = match value_map.get(rhs).unwrap() {
            LaneRepr::Scalar(v) => v[0],
            LaneRepr::PackedF64 { tail: Some(t), .. } => *t,
            LaneRepr::PackedF64 { chunks, .. } => builder.ins().extractlane(chunks[0], 0u8),
            LaneRepr::PtrChunksF64 { .. } => {
                unreachable!("PtrChunksF64 leaked into scalar-ABI align_packed_f64")
            }
        };
        broadcast_scalar_f64_to_packed(builder, s, target)
    } else {
        get_packed_f64(builder, value_map, rhs)?
    };
    Ok((l, r))
}

/// Apply a polymorphic unary Cranelift op (fneg, sqrt, fabs, floor, ceil,
/// nearest, ...) to each F64X2 chunk plus the scalar tail. Produces a
/// `LaneRepr::PackedF64`.
fn elementwise_unop_packed_f64(
    builder: &mut FunctionBuilder,
    x: &PackedF64Chunks,
    f: impl Fn(&mut FunctionBuilder, Value) -> Value,
) -> LaneRepr {
    let mut chunks = Vec::with_capacity(x.chunks.len());
    for c in &x.chunks {
        chunks.push(f(builder, *c));
    }
    let tail = x.tail.map(|t| f(builder, t));
    LaneRepr::PackedF64 {
        chunks,
        tail,
        n: x.n,
    }
}

/// Emit one polymorphic Cranelift op per chunk (plus one scalar op for the
/// tail). The supplied closure is invoked with both F64X2 and f64 values, so
/// only polymorphic ops (fadd, fsub, fmul, fdiv, fmin, fmax, etc.) are safe.
fn elementwise_binop_packed_f64(
    builder: &mut FunctionBuilder,
    l: &PackedF64Chunks,
    r: &PackedF64Chunks,
    f: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> LaneRepr {
    assert_eq!(l.n, r.n, "elementwise_binop_packed_f64: shape mismatch");
    let mut chunks = Vec::with_capacity(l.chunks.len());
    for (lc, rc) in l.chunks.iter().zip(r.chunks.iter()) {
        chunks.push(f(builder, *lc, *rc));
    }
    let tail = match (l.tail, r.tail) {
        (Some(lt), Some(rt)) => Some(f(builder, lt, rt)),
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            unreachable!("align_packed_f64 ensures matched tail state")
        }
    };
    LaneRepr::PackedF64 {
        chunks,
        tail,
        n: l.n,
    }
}

/// Pack a flat slice of scalar f64 Values into a `PackedF64Chunks` view.
/// Used by dot product kernels that receive scalar inputs but want to emit
/// vector fmul+fadd.
fn pack_f64_slice(builder: &mut FunctionBuilder, scalars: &[Value]) -> PackedF64Chunks {
    let n = scalars.len();
    let mut chunks = Vec::with_capacity(n / 2);
    let mut i = 0;
    while i + 2 <= n {
        chunks.push(pack_f64x2(builder, scalars[i], scalars[i + 1]));
        i += 2;
    }
    let tail = if i < n { Some(scalars[i]) } else { None };
    PackedF64Chunks { chunks, tail, n }
}

/// Compute `sum(l[i] * r[i]) for i in 0..n` using `fmul.F64X2` per chunk and
/// a horizontal reduce at the end. Returns a single scalar f64 Value.
fn packed_dot_f64(
    builder: &mut FunctionBuilder,
    l: &PackedF64Chunks,
    r: &PackedF64Chunks,
) -> Value {
    assert_eq!(l.n, r.n, "packed_dot_f64: shape mismatch");
    // Vector phase: chunkwise fmul, accumulate as F64X2.
    let mut vec_acc: Option<Value> = None;
    for (lc, rc) in l.chunks.iter().zip(r.chunks.iter()) {
        let prod = builder.ins().fmul(*lc, *rc);
        vec_acc = Some(match vec_acc {
            Some(a) => builder.ins().fadd(a, prod),
            None => prod,
        });
    }
    // Horizontal reduce: (lane0 + lane1) → scalar f64.
    let mut scalar_acc = match vec_acc {
        Some(v) => {
            let a = builder.ins().extractlane(v, 0u8);
            let b = builder.ins().extractlane(v, 1u8);
            builder.ins().fadd(a, b)
        }
        None => builder.ins().f64const(0.0),
    };
    // Scalar tail.
    match (l.tail, r.tail) {
        (Some(lt), Some(rt)) => {
            let prod = builder.ins().fmul(lt, rt);
            scalar_acc = builder.ins().fadd(scalar_acc, prod);
        }
        (None, None) => {}
        _ => unreachable!("dot tail mismatch"),
    }
    scalar_acc
}

/// Emit fcmp+bitselect for each vector chunk (NaN-propagation matches the
/// scalar `fcmp cc a b ? a : b` semantics that StableHLO Maximum/Minimum
/// rely on) and fcmp+select for the scalar tail. Used by Maximum/Minimum.
///
/// Cranelift's `bitselect` requires the mask type to match the data type,
/// but vector `fcmp` returns an `I64X2` mask. We `bitcast` the mask to
/// `F64X2` (same bit pattern; `bitselect` only examines bits, so the
/// all-ones / all-zeros lane bit pattern produces the correct per-bit
/// select).
fn elementwise_cmp_select_packed_f64(
    builder: &mut FunctionBuilder,
    l: &PackedF64Chunks,
    r: &PackedF64Chunks,
    cc: FloatCC,
) -> LaneRepr {
    assert_eq!(
        l.n, r.n,
        "elementwise_cmp_select_packed_f64: shape mismatch"
    );
    let mut chunks = Vec::with_capacity(l.chunks.len());
    for (lc, rc) in l.chunks.iter().zip(r.chunks.iter()) {
        let cmp_i = builder.ins().fcmp(cc, *lc, *rc);
        let cmp_f = builder.ins().bitcast(types::F64X2, MemFlags::new(), cmp_i);
        chunks.push(builder.ins().bitselect(cmp_f, *lc, *rc));
    }
    let tail = match (l.tail, r.tail) {
        (Some(lt), Some(rt)) => {
            let cmp = builder.ins().fcmp(cc, lt, rt);
            Some(builder.ins().select(cmp, lt, rt))
        }
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            unreachable!("align_packed_f64 ensures matched tail state")
        }
    };
    LaneRepr::PackedF64 {
        chunks,
        tail,
        n: l.n,
    }
}

fn lower_int_binop(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
    f: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> Result<Vec<LaneRepr>, String> {
    let l = get_vals(builder, value_map, lhs)?.to_vec();
    let r = get_vals(builder, value_map, rhs)?.to_vec();
    Ok(vec![LaneRepr::scalar(elementwise_binop(
        builder, &l, &r, f,
    ))])
}

fn lower_libm_unary(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    operand: &ValueId,
    func_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let vals = get_vals(builder, value_map, operand)?.to_vec();
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    // Transcendental path bookkeeping: mode 0 = libm scalar fallback.
    // Bracket the whole per-element loop so time attribution captures
    // the full scalar cost; wrap each libm call in a call_begin/end
    // pair for per-function accounting.
    emit_xcend_begin(builder, jit_module, 0);
    let out: Vec<Value> = vals
        .iter()
        .map(|&v| {
            emit_call_begin(builder, jit_module);
            let call = builder.ins().call(func_ref, &[v]);
            let r = builder.inst_results(call)[0];
            emit_call_end(builder, jit_module);
            r
        })
        .collect();
    emit_xcend_end(builder, jit_module);
    Ok(vec![LaneRepr::scalar(out)])
}

/// Lower a unary f64 transcendental by marshaling scalars through a
/// stack buffer and calling the SIMD-backed tensor_rt variant. Used
/// for sin / cos / exp / log / ... on scalar-ABI f64 tensors with
/// 2+ lanes. Falls back to `lower_libm_unary` for n < 2.
fn lower_trt_simd_unary_f64(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    operand: &ValueId,
    trt_fn_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let vals = get_vals(builder, value_map, operand)?.to_vec();
    let n = vals.len();
    let total_bytes = 2 * n * 8;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let src_base = builder.ins().stack_addr(ptr_type(), ss, 0);
    let dst_base = builder.ins().stack_addr(ptr_type(), ss, (n * 8) as i32);
    for (i, &v) in vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, src_base, (i * 8) as i32);
    }
    let func_ref = jit_module.declare_func_in_func(trt_fn_id, builder.func);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    // Transcendental path bookkeeping: mode 1 = wide-SIMD batch.
    // Bracket to measure the batch duration.
    emit_xcend_begin(builder, jit_module, 1);
    emit_call_begin(builder, jit_module);
    builder.ins().call(func_ref, &[dst_base, src_base, n_val]);
    emit_call_end(builder, jit_module);
    emit_xcend_end(builder, jit_module);
    let out: Vec<Value> = (0..n)
        .map(|i| {
            builder
                .ins()
                .load(types::F64, MemFlags::trusted(), dst_base, (i * 8) as i32)
        })
        .collect();
    Ok(vec![LaneRepr::scalar(out)])
}

/// Lower a binary f64 transcendental via the tensor_rt SIMD variant (2-input).
/// Same marshaling pattern as `lower_trt_simd_unary_f64`.
fn lower_trt_simd_binary_f64(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
    trt_fn_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let l = get_vals(builder, value_map, lhs)?.to_vec();
    let r = get_vals(builder, value_map, rhs)?.to_vec();
    let n = l.len().max(r.len());
    let total_bytes = 3 * n * 8;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let a_base = builder.ins().stack_addr(ptr_type(), ss, 0);
    let b_base = builder.ins().stack_addr(ptr_type(), ss, (n * 8) as i32);
    let dst_base = builder.ins().stack_addr(ptr_type(), ss, (2 * n * 8) as i32);
    for i in 0..n {
        let lv = if i < l.len() { l[i] } else { l[0] };
        let rv = if i < r.len() { r[i] } else { r[0] };
        builder
            .ins()
            .store(MemFlags::trusted(), lv, a_base, (i * 8) as i32);
        builder
            .ins()
            .store(MemFlags::trusted(), rv, b_base, (i * 8) as i32);
    }
    let func_ref = jit_module.declare_func_in_func(trt_fn_id, builder.func);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    // Transcendental path bookkeeping: mode 1 = wide-SIMD batch (binary).
    // Bracket to measure the batch duration.
    emit_xcend_begin(builder, jit_module, 1);
    emit_call_begin(builder, jit_module);
    builder
        .ins()
        .call(func_ref, &[dst_base, a_base, b_base, n_val]);
    emit_call_end(builder, jit_module);
    emit_xcend_end(builder, jit_module);
    let out: Vec<Value> = (0..n)
        .map(|i| {
            builder
                .ins()
                .load(types::F64, MemFlags::trusted(), dst_base, (i * 8) as i32)
        })
        .collect();
    Ok(vec![LaneRepr::scalar(out)])
}

fn lower_libm_binary(
    builder: &mut FunctionBuilder,
    value_map: &mut HashMap<ValueId, LaneRepr>,
    lhs: &ValueId,
    rhs: &ValueId,
    func_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let l = get_vals(builder, value_map, lhs)?.to_vec();
    let r = get_vals(builder, value_map, rhs)?.to_vec();
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    let n = l.len().max(r.len());
    // Count as libm-scalar transcendentals; per-call begin/end so
    // time is attributed to the enclosing function body.
    emit_xcend_begin(builder, jit_module, 0);
    let out: Vec<Value> = (0..n)
        .map(|i| {
            let lv = if i < l.len() { l[i] } else { l[0] };
            let rv = if i < r.len() { r[i] } else { r[0] };
            emit_call_begin(builder, jit_module);
            let call = builder.ins().call(func_ref, &[lv, rv]);
            let r = builder.inst_results(call)[0];
            emit_call_end(builder, jit_module);
            r
        })
        .collect();
    emit_xcend_end(builder, jit_module);
    Ok(vec![LaneRepr::scalar(out)])
}

fn convert_value(
    builder: &mut FunctionBuilder,
    v: Value,
    src: &ElementType,
    dst: &ElementType,
) -> Value {
    match (src, dst) {
        _ if src == dst => v,
        (ElementType::F64, ElementType::I32) => builder.ins().fcvt_to_sint(types::I32, v),
        (ElementType::F64, ElementType::I64) => builder.ins().fcvt_to_sint(types::I64, v),
        (ElementType::F64, ElementType::UI32) => builder.ins().fcvt_to_uint(types::I32, v),
        (ElementType::F64, ElementType::F32) => builder.ins().fdemote(types::F32, v),
        (ElementType::F32, ElementType::F64) => builder.ins().fpromote(types::F64, v),
        (ElementType::I64, ElementType::F64) => builder.ins().fcvt_from_sint(types::F64, v),
        (ElementType::I64, ElementType::F32) => {
            let f = builder.ins().fcvt_from_sint(types::F64, v);
            builder.ins().fdemote(types::F32, f)
        }
        (ElementType::I32, ElementType::F64) => {
            let ext = builder.ins().sextend(types::I64, v);
            builder.ins().fcvt_from_sint(types::F64, ext)
        }
        (ElementType::UI32, ElementType::F64) => {
            let ext = builder.ins().uextend(types::I64, v);
            builder.ins().fcvt_from_uint(types::F64, ext)
        }
        (ElementType::I32, ElementType::I64) => builder.ins().sextend(types::I64, v),
        (ElementType::I32, ElementType::UI32) | (ElementType::UI32, ElementType::I32) => v,
        (ElementType::I64, ElementType::UI64) | (ElementType::UI64, ElementType::I64) => v,
        (ElementType::I64, ElementType::I32) => builder.ins().ireduce(types::I32, v),
        (ElementType::I64, ElementType::UI32)
        | (ElementType::UI64, ElementType::UI32)
        | (ElementType::UI64, ElementType::I32) => builder.ins().ireduce(types::I32, v),
        (ElementType::UI32, ElementType::I64) | (ElementType::UI32, ElementType::UI64) => {
            builder.ins().uextend(types::I64, v)
        }
        (ElementType::I1, ElementType::I32) | (ElementType::I1, ElementType::UI32) => {
            builder.ins().uextend(types::I32, v)
        }
        (ElementType::I1, ElementType::I64) | (ElementType::I1, ElementType::UI64) => {
            builder.ins().uextend(types::I64, v)
        }
        (ElementType::I1, ElementType::F64) => {
            let ext = builder.ins().uextend(types::I64, v);
            builder.ins().fcvt_from_uint(types::F64, ext)
        }
        (ElementType::I1, ElementType::F32) => {
            let ext = builder.ins().uextend(types::I32, v);
            builder.ins().fcvt_from_uint(types::F32, ext)
        }
        (ElementType::I32, ElementType::I1) => builder.ins().ireduce(types::I8, v),
        (ElementType::I64, ElementType::I1) => builder.ins().ireduce(types::I8, v),
        (ElementType::F64, ElementType::I1) => {
            let i = builder.ins().fcvt_to_sint(types::I32, v);
            builder.ins().ireduce(types::I8, i)
        }
        (ElementType::UI64, ElementType::F64) => builder.ins().fcvt_from_uint(types::F64, v),
        (ElementType::F64, ElementType::UI64) => builder.ins().fcvt_to_uint(types::I64, v),
        (ElementType::I32, ElementType::F32) => builder.ins().fcvt_from_sint(types::F32, v),
        _ => v,
    }
}

// ---------------------------------------------------------------------------
// Shape operation helpers
// ---------------------------------------------------------------------------

fn broadcast_values(
    vals: &[Value],
    target_shape: &[i64],
    broadcast_dims: &[i64],
    src_shape: &[i64],
) -> Vec<Value> {
    let n: usize = target_shape.iter().product::<i64>() as usize;
    if vals.len() == 1 {
        return vec![vals[0]; n];
    }
    if vals.len() == n && broadcast_dims.is_empty() {
        return vals.to_vec();
    }

    let out_rank = target_shape.len();
    let mut out_strides = vec![1usize; out_rank];
    for i in (0..out_rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1] as usize;
    }

    let src_rank = src_shape.len();
    let mut src_strides = vec![1usize; src_rank.max(1)];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let mut out = Vec::with_capacity(n);
    for flat_out in 0..n {
        let mut remaining = flat_out;
        let mut src_flat = 0;
        for (d, &stride) in out_strides.iter().enumerate() {
            let idx = remaining / stride;
            remaining %= stride;
            if let Some(pos) = broadcast_dims.iter().position(|&bd| bd as usize == d)
                && pos < src_rank
                && src_shape[pos] > 1
            {
                src_flat += idx * src_strides[pos];
            }
        }
        out.push(vals[src_flat.min(vals.len() - 1)]);
    }
    out
}

fn lower_concatenate_nd(
    parts: &[(Vec<Value>, TensorType)],
    dim: usize,
    out_ty: &TensorType,
) -> TensorVals {
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();
    let rank = out_shape.len();
    let n = out_ty.num_elements();

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let part_infos: Vec<(Vec<usize>, Vec<usize>)> = parts
        .iter()
        .map(|(_, ty)| {
            let shape: Vec<usize> = ty.shape.iter().map(|&d| d as usize).collect();
            let mut strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            (shape, strides)
        })
        .collect();

    let mut result = Vec::with_capacity(n);
    for flat_out in 0..n {
        let mut remaining = flat_out;
        let mut out_indices = vec![0usize; rank];
        for d in 0..rank {
            out_indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        let concat_idx = out_indices[dim];
        let mut part_idx = 0;
        let mut offset_in_dim = 0;
        for (i, (shape, _)) in part_infos.iter().enumerate() {
            if concat_idx < offset_in_dim + shape[dim] {
                part_idx = i;
                break;
            }
            offset_in_dim += shape[dim];
        }

        let local_dim_idx = concat_idx - offset_in_dim;
        let (ref _shape, ref strides) = part_infos[part_idx];
        let mut src_flat = 0;
        for d in 0..rank {
            let idx = if d == dim {
                local_dim_idx
            } else {
                out_indices[d]
            };
            src_flat += idx * strides[d];
        }

        result.push(parts[part_idx].0[src_flat]);
    }
    result
}

fn slice_tensor(vals: &[Value], src_shape: &[i64], starts: &[i64], limits: &[i64]) -> Vec<Value> {
    if src_shape.is_empty() {
        return vals.to_vec();
    }
    let rank = src_shape.len();
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let slice_shape: Vec<usize> = (0..rank)
        .map(|d| {
            let s = starts.get(d).copied().unwrap_or(0) as usize;
            let l = limits.get(d).copied().unwrap_or(src_shape[d]) as usize;
            l - s
        })
        .collect();
    let n_out: usize = slice_shape.iter().product();
    let mut out = Vec::with_capacity(n_out);

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * slice_shape[i + 1];
    }

    for flat_out in 0..n_out {
        let mut src_flat = 0;
        let mut remaining = flat_out;
        for d in 0..rank {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            let src_coord = coord + starts.get(d).copied().unwrap_or(0) as usize;
            src_flat += src_coord * src_strides[d];
        }
        if src_flat < vals.len() {
            out.push(vals[src_flat]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dot product / matrix operations
// ---------------------------------------------------------------------------

fn lower_dot_general(
    builder: &mut FunctionBuilder,
    l: &[Value],
    r: &[Value],
    l_ty: &TensorType,
    r_ty: &TensorType,
    out_ty: &TensorType,
    dims: &DotDims,
) -> TensorVals {
    let n = out_ty.num_elements();

    // Batched dot product: batching_dims=[0]x[0], contracting_dims=[1]x[1]
    // tensor<BxK> . tensor<BxK> -> tensor<B>
    if !dims.lhs_batch.is_empty()
        && l_ty.rank() == 2
        && r_ty.rank() == 2
        && dims.lhs_batch == [0]
        && dims.rhs_batch == [0]
    {
        let batch = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        let mut out = Vec::new();
        for b in 0..batch {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for i in 0..k {
                let lv = l[b * k + i];
                let rv = r[b * k + i];
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.is_scalar() && r_ty.is_scalar() {
        return vec![builder.ins().fmul(l[0], r[0])];
    }

    if l_ty.rank() == 1 && r_ty.rank() == 1 {
        let zero = builder.ins().f64const(0.0);
        let mut acc = zero;
        for i in 0..l.len().min(r.len()) {
            let prod = builder.ins().fmul(l[i], r[i]);
            acc = builder.ins().fadd(acc, prod);
        }
        return vec![acc];
    }

    if l_ty.rank() == 1 && r_ty.rank() == 2 {
        let rhs_contract_dim = dims.rhs_contracting[0] as usize;
        let k = l_ty.shape[0] as usize;
        let rhs_rows = r_ty.shape[0] as usize;
        let rhs_cols = r_ty.shape[1] as usize;
        let out_size = if rhs_contract_dim == 1 {
            rhs_rows
        } else {
            rhs_cols
        };
        let mut out = Vec::new();
        for i in 0..out_size {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for j in 0..k {
                let lv = l[j];
                let rv = if rhs_contract_dim == 1 {
                    r[i * rhs_cols + j]
                } else {
                    r[j * rhs_cols + i]
                };
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.rank() == 2 && r_ty.rank() == 1 {
        let rows = l_ty.shape[0] as usize;
        let cols = l_ty.shape[1] as usize;
        let mut out = Vec::new();
        for row in 0..rows {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for col in 0..cols {
                let lv = l[row * cols + col];
                let rv = r[col];
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.rank() == 2 && r_ty.rank() == 2 {
        let m = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        let n_cols = r_ty.shape[1] as usize;
        let mut out = Vec::new();
        for row in 0..m {
            for col in 0..n_cols {
                let zero = builder.ins().f64const(0.0);
                let mut acc = zero;
                for i in 0..k {
                    let lv = l[row * k + i];
                    let rv = r[i * n_cols + col];
                    let prod = builder.ins().fmul(lv, rv);
                    acc = builder.ins().fadd(acc, prod);
                }
                out.push(acc);
            }
        }
        return out;
    }

    let zero = builder.ins().f64const(0.0);
    vec![zero; n]
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

fn lower_reduce(
    builder: &mut FunctionBuilder,
    vals: &[Value],
    init_vals: &[Value],
    src_ty: &TensorType,
    out_ty: &TensorType,
    op: &ReduceOp,
    dimensions: &[i64],
) -> TensorVals {
    let n_out = out_ty.num_elements();

    if src_ty.rank() == 1 && dimensions == [0] {
        let mut acc = init_vals[0];
        for &v in vals {
            acc = apply_reduce_op(builder, acc, v, op);
        }
        return vec![acc];
    }

    if src_ty.rank() == 2 && dimensions == [0] {
        let rows = src_ty.shape[0] as usize;
        let cols = src_ty.shape[1] as usize;
        let mut out = Vec::new();
        for c in 0..cols {
            let mut acc = if c < init_vals.len() {
                init_vals[c]
            } else {
                init_vals[0]
            };
            for r in 0..rows {
                let idx = r * cols + c;
                if idx < vals.len() {
                    acc = apply_reduce_op(builder, acc, vals[idx], op);
                }
            }
            out.push(acc);
        }
        return out;
    }

    if src_ty.rank() == 2 && dimensions == [1] {
        let rows = src_ty.shape[0] as usize;
        let cols = src_ty.shape[1] as usize;
        let mut out = Vec::new();
        for r in 0..rows {
            let mut acc = init_vals[0];
            for c in 0..cols {
                let idx = r * cols + c;
                if idx < vals.len() {
                    acc = apply_reduce_op(builder, acc, vals[idx], op);
                }
            }
            out.push(acc);
        }
        return out;
    }

    init_vals[..n_out.min(init_vals.len())].to_vec()
}

fn apply_reduce_op(builder: &mut FunctionBuilder, acc: Value, v: Value, op: &ReduceOp) -> Value {
    let val_type = builder.func.dfg.value_type(acc);
    match op {
        ReduceOp::Add => {
            if val_type.is_float() {
                builder.ins().fadd(acc, v)
            } else {
                builder.ins().iadd(acc, v)
            }
        }
        ReduceOp::Minimum => {
            if val_type.is_float() {
                let cmp = builder.ins().fcmp(FloatCC::LessThan, acc, v);
                builder.ins().select(cmp, acc, v)
            } else {
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, acc, v);
                builder.ins().select(cmp, acc, v)
            }
        }
        ReduceOp::Maximum => {
            if val_type.is_float() {
                let cmp = builder.ins().fcmp(FloatCC::GreaterThan, acc, v);
                builder.ins().select(cmp, acc, v)
            } else {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, acc, v);
                builder.ins().select(cmp, acc, v)
            }
        }
        ReduceOp::And => builder.ins().band(acc, v),
        ReduceOp::Or => builder.ins().bor(acc, v),
    }
}

// ---------------------------------------------------------------------------
// Gather — stack-slot based element lookup for small tensors
// ---------------------------------------------------------------------------

fn lower_gather(
    builder: &mut FunctionBuilder,
    operand: &[Value],
    indices: &[Value],
    src_ty: &TensorType,
    idx_ty: &TensorType,
    out_ty: &TensorType,
    dims: &GatherDims,
    slice_sizes: &[i64],
) -> TensorVals {
    let n = out_ty.num_elements();
    let et = out_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    if operand.is_empty() || indices.is_empty() {
        return vec![make_zero(builder, et); n];
    }

    let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
    let src_rank = src_shape.len();
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();
    let out_rank = out_shape.len();
    let idx_shape: Vec<usize> = idx_ty.shape.iter().map(|&d| d as usize).collect();
    let index_vector_dim = dims.index_vector_dim as usize;

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut idx_strides = vec![1usize; idx_shape.len()];
    for i in (0..idx_shape.len().saturating_sub(1)).rev() {
        idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
    }

    let total_bytes = operand.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in operand.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
    }

    // Batch dims of the indices tensor = all dims except index_vector_dim
    let idx_rank = idx_shape.len();
    let batch_dims: Vec<usize> = (0..idx_rank).filter(|&d| d != index_vector_dim).collect();
    let index_depth = if index_vector_dim < idx_rank {
        idx_shape[index_vector_dim]
    } else {
        1
    };

    // Batch shape = indices shape with index_vector_dim removed
    let batch_shape: Vec<usize> = batch_dims.iter().map(|&d| idx_shape[d]).collect();
    let batch_rank = batch_shape.len();

    // Offset shape = slice_sizes with collapsed dims removed
    let offset_shape: Vec<usize> = (0..src_rank)
        .filter(|d| !dims.collapsed_slice_dims.contains(&(*d as i64)))
        .map(|d| slice_sizes[d] as usize)
        .collect();
    let offset_rank = offset_shape.len();

    let mut results = Vec::with_capacity(n);

    for flat_out in 0..n {
        // Decompose flat output index into multi-index
        let mut out_indices = vec![0usize; out_rank];
        {
            let mut rem = flat_out;
            for d in (0..out_rank).rev() {
                if out_shape[d] > 0 {
                    out_indices[d] = rem % out_shape[d];
                    rem /= out_shape[d];
                }
            }
        }

        // Split output indices into batch part and offset part
        let mut batch_idx = vec![0usize; batch_rank];
        let mut offset_idx = vec![0usize; offset_rank];
        let offset_dims = &dims.offset_dims;
        let mut bi = 0;
        let mut oi = 0;
        for (d, &out_idx) in out_indices.iter().enumerate() {
            if offset_dims.contains(&(d as i64)) {
                if oi < offset_rank {
                    offset_idx[oi] = out_idx;
                    oi += 1;
                }
            } else if bi < batch_rank {
                batch_idx[bi] = out_idx;
                bi += 1;
            }
        }

        // Look up start indices from the indices tensor
        let mut start_index = vec![0usize; index_depth];
        for (k, si) in start_index.iter_mut().enumerate() {
            let mut idx_multi = vec![0usize; idx_rank];
            let mut b = 0;
            for (d, im) in idx_multi.iter_mut().enumerate() {
                if d == index_vector_dim {
                    *im = k;
                } else {
                    if b < batch_rank {
                        *im = batch_idx[b];
                    }
                    b += 1;
                }
            }
            let mut flat_idx = 0;
            for (d, &im) in idx_multi.iter().enumerate() {
                flat_idx += im * idx_strides[d];
            }
            *si = flat_idx;
        }

        // Build source multi-index
        let mut src_idx = vec![0usize; src_rank];

        // Place start indices via start_index_map (runtime values)
        // Place offset indices into non-collapsed dims
        let mut oi2 = 0;
        for (d, si) in src_idx.iter_mut().enumerate() {
            if !dims.collapsed_slice_dims.contains(&(d as i64)) && oi2 < offset_rank {
                *si = offset_idx[oi2];
                oi2 += 1;
            }
        }

        // The start_index values are RUNTIME (SSA) values from the indices tensor.
        // We need to compute the source address dynamically.
        // Static part: offset contribution to flat index
        let mut static_offset = 0usize;
        for d in 0..src_rank {
            if !dims.collapsed_slice_dims.contains(&(d as i64)) {
                static_offset += src_idx[d] * src_strides[d];
            }
        }

        // Dynamic part: start_index contributions
        // For each k in start_index_map, add indices[start_index[k]] * src_strides[start_index_map[k]]
        let static_byte_off = (static_offset * elem_sz) as i32;
        let mut addr = builder.ins().iadd_imm(base, static_byte_off as i64);

        for (k, &mapped_dim) in dims.start_index_map.iter().enumerate() {
            if k >= index_depth {
                break;
            }
            let flat_idx = start_index[k];
            if flat_idx >= indices.len() {
                continue;
            }
            let raw_idx = indices[flat_idx];
            let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
                raw_idx
            } else {
                builder.ins().sextend(types::I64, raw_idx)
            };
            let stride_bytes = (src_strides[mapped_dim as usize] * elem_sz) as i64;
            let byte_off = builder.ins().imul_imm(idx_i64, stride_bytes);
            addr = builder.ins().iadd(addr, byte_off);
        }

        let v = builder.ins().load(ct, MemFlags::trusted(), addr, 0);
        results.push(v);
    }

    results
}

fn lower_transpose(
    vals: &[Value],
    src_ty: &TensorType,
    out_ty: &TensorType,
    permutation: &[i64],
) -> TensorVals {
    let n = out_ty.num_elements();
    if vals.len() != n || src_ty.shape.len() != permutation.len() {
        return vals.to_vec();
    }

    let rank = src_ty.shape.len();
    let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let mut result = vec![vals[0]; n];
    for (flat_out, slot) in result.iter_mut().enumerate() {
        let mut remaining = flat_out;
        let mut out_indices = vec![0usize; rank];
        for d in 0..rank {
            out_indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }
        let mut src_flat = 0;
        for d in 0..rank {
            let src_dim = permutation[d] as usize;
            src_flat += out_indices[d] * src_strides[src_dim];
        }
        *slot = vals[src_flat];
    }
    result
}

fn lower_dynamic_slice(
    builder: &mut FunctionBuilder,
    vals: &[Value],
    start_indices: &[Value],
    src_ty: &TensorType,
    slice_sizes: &[i64],
) -> TensorVals {
    let et = src_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    let total_bytes = vals.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
    }

    let rank = src_ty.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * src_ty.shape[i + 1] as usize;
    }

    // Compute the flat byte offset from dynamic start indices.
    // Per StableHLO spec, indices are clamped to [0, dim_size - slice_size].
    let mut flat_offset = builder.ins().iconst(types::I64, 0);
    for d in 0..rank {
        if d < start_indices.len() {
            let idx = start_indices[d];
            let idx_i64 = if builder.func.dfg.value_type(idx) == types::I64 {
                idx
            } else {
                builder.ins().sextend(types::I64, idx)
            };
            let max_idx = src_ty.shape[d] - slice_sizes.get(d).copied().unwrap_or(1);
            let max_val = builder.ins().iconst(types::I64, max_idx);
            let zero = builder.ins().iconst(types::I64, 0);
            let clamped_lo = {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, idx_i64, zero);
                builder.ins().select(cmp, idx_i64, zero)
            };
            let clamped = {
                let cmp = builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, clamped_lo, max_val);
                builder.ins().select(cmp, clamped_lo, max_val)
            };
            let stride_bytes = (strides[d] * elem_sz) as i64;
            let contrib = builder.ins().imul_imm(clamped, stride_bytes);
            flat_offset = builder.ins().iadd(flat_offset, contrib);
        }
    }
    let slice_base = builder.ins().iadd(base, flat_offset);

    let out_n: usize = slice_sizes.iter().product::<i64>() as usize;
    let mut results = Vec::with_capacity(out_n);
    for i in 0..out_n {
        let v = builder
            .ins()
            .load(ct, MemFlags::trusted(), slice_base, (i * elem_sz) as i32);
        results.push(v);
    }
    results
}

fn lower_dynamic_update_slice(
    builder: &mut FunctionBuilder,
    base_vals: &[Value],
    update_vals: &[Value],
    start_indices: &[Value],
    src_ty: &TensorType,
    _upd_ty: &TensorType,
) -> TensorVals {
    let et = src_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    let total_bytes = base_vals.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let base_addr = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in base_vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base_addr, (i * elem_sz) as i32);
    }

    let rank = src_ty.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * src_ty.shape[i + 1] as usize;
    }

    let upd_shape = &_upd_ty.shape;
    let mut flat_offset = builder.ins().iconst(types::I64, 0);
    for d in 0..rank.min(start_indices.len()) {
        let idx = start_indices[d];
        let idx_i64 = if builder.func.dfg.value_type(idx) == types::I64 {
            idx
        } else {
            builder.ins().sextend(types::I64, idx)
        };
        let upd_dim = upd_shape.get(d).copied().unwrap_or(1);
        let max_idx = src_ty.shape[d] - upd_dim;
        let max_val = builder.ins().iconst(types::I64, max_idx);
        let zero = builder.ins().iconst(types::I64, 0);
        let clamped_lo = {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, idx_i64, zero);
            builder.ins().select(cmp, idx_i64, zero)
        };
        let clamped = {
            let cmp = builder
                .ins()
                .icmp(IntCC::SignedLessThan, clamped_lo, max_val);
            builder.ins().select(cmp, clamped_lo, max_val)
        };
        let stride_bytes = (strides[d] * elem_sz) as i64;
        let contrib = builder.ins().imul_imm(clamped, stride_bytes);
        flat_offset = builder.ins().iadd(flat_offset, contrib);
    }
    let update_addr = builder.ins().iadd(base_addr, flat_offset);

    for (i, &v) in update_vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, update_addr, (i * elem_sz) as i32);
    }

    let mut results = Vec::with_capacity(base_vals.len());
    for i in 0..base_vals.len() {
        let v = builder
            .ins()
            .load(ct, MemFlags::trusted(), base_addr, (i * elem_sz) as i32);
        results.push(v);
    }
    results
}

fn lower_custom_call(
    builder: &mut FunctionBuilder,
    call_target: &str,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<LaneRepr>, String> {
    if call_target.starts_with("lapack_dgesdd") {
        return lower_svd_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgetrf") {
        return lower_lu_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dtrsm") {
        return lower_trsm_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            backend_config,
        );
    }
    if call_target.starts_with("lapack_dpotrf") {
        return lower_cholesky_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            backend_config,
        );
    }
    if call_target.starts_with("lapack_dgeqrf") {
        return lower_qr_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dorgqr") {
        return lower_orgqr_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dsyevd") {
        return lower_syevd_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgesv") {
        return lower_gesv_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dpotrs") {
        return lower_potrs_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            backend_config,
        );
    }
    if call_target.starts_with("lapack_dgelsd") {
        return lower_gelsd_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgeev") {
        return lower_geev_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgesvd") {
        return lower_gesvd_custom_call(builder, operands, value_map, type_map, jit_module);
    }
    Err(format!("unsupported custom_call target: {call_target}"))
}

// ---------------------------------------------------------------------------
// Host LAPACK functions backed by faer
// ---------------------------------------------------------------------------

fn row_major_to_col_major(row: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut col = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            col[j * m + i] = row[i * n + j];
        }
    }
    col
}

fn col_major_to_row_major(col: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut row = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            row[i * n + j] = col[j * m + i];
        }
    }
    row
}

extern "C" fn cranelift_svd(
    a_ptr: *const f64,
    n: usize,
    u_ptr: *mut f64,
    s_ptr: *mut f64,
    vt_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let u_out = unsafe { std::slice::from_raw_parts_mut(u_ptr, n * n) };
    let s_out = unsafe { std::slice::from_raw_parts_mut(s_ptr, n) };
    let vt_out = unsafe { std::slice::from_raw_parts_mut(vt_ptr, n * n) };

    // Input is row-major from the IR; faer needs column-major
    let col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice(&col_data, n, n);
    let svd = mat.thin_svd();

    // U output: convert from faer column-major back to row-major for the IR
    let u_col = svd.u();
    for i in 0..n {
        for j in 0..n {
            u_out[i * n + j] = u_col.read(i, j);
        }
    }
    let s_diag = svd.s_diagonal();
    for (i, val) in s_out.iter_mut().enumerate() {
        *val = s_diag.read(i);
    }
    // VT output: row i of VT = column i of V transposed
    let v = svd.v();
    for i in 0..n {
        for j in 0..n {
            vt_out[i * n + j] = v.read(j, i);
        }
    }

    unsafe { *info_ptr = 0 };
}

extern "C" fn cranelift_lu(
    a_ptr: *const f64,
    m: usize,
    n: usize,
    lu_ptr: *mut f64,
    ipiv_ptr: *mut i32,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let lu_out = unsafe { std::slice::from_raw_parts_mut(lu_ptr, m * n) };
    let ipiv_out = unsafe { std::slice::from_raw_parts_mut(ipiv_ptr, m.min(n)) };

    let min_mn = m.min(n);

    // Manual LU with partial pivoting producing LAPACK-compatible ipiv (1-indexed)
    let mut mat = vec![0.0f64; m * n];
    mat.copy_from_slice(a);
    let idx = |r: usize, c: usize| r * n + c;

    for k in 0..min_mn {
        // Find pivot
        let mut max_val = mat[idx(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..m {
            let v = mat[idx(i, k)].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        // LAPACK 1-indexed pivot
        ipiv_out[k] = max_row as i32 + 1;

        // Swap rows k and max_row
        if max_row != k {
            for j in 0..n {
                mat.swap(idx(k, j), idx(max_row, j));
            }
        }

        let pivot = mat[idx(k, k)];
        if pivot.abs() < LU_PIVOT_EPSILON {
            continue;
        }

        for i in (k + 1)..m {
            mat[idx(i, k)] /= pivot;
            let factor = mat[idx(i, k)];
            for j in (k + 1)..n {
                mat[idx(i, j)] -= factor * mat[idx(k, j)];
            }
        }
    }

    lu_out.copy_from_slice(&mat);
    unsafe { *info_ptr = 0 };
}

extern "C" fn cranelift_trsm(
    a_ptr: *const f64,
    b_ptr: *const f64,
    m: usize,
    n: usize,
    nrhs: usize,
    uplo: u8,
    diag: u8,
    out_ptr: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, m * nrhs) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, m * nrhs) };

    let a_col = row_major_to_col_major(a, m, n);
    let a_mat = faer::mat::from_column_major_slice(&a_col, m, n);

    let mut x_col = row_major_to_col_major(b, m, nrhs);
    let x_mat = faer::mat::from_column_major_slice_mut(&mut x_col, m, nrhs);

    let is_lower = uplo == b'L';
    let is_unit = diag == b'U';

    if is_lower {
        if is_unit {
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                a_mat,
                x_mat,
                faer::Parallelism::None,
            );
        } else {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                a_mat,
                x_mat,
                faer::Parallelism::None,
            );
        }
    } else if is_unit {
        faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            a_mat,
            x_mat,
            faer::Parallelism::None,
        );
    } else {
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
            a_mat,
            x_mat,
            faer::Parallelism::None,
        );
    }

    let result = col_major_to_row_major(&x_col, m, nrhs);
    out.copy_from_slice(&result);
}

/// Runtime Cholesky factor. When `lower != 0` writes the lower factor L
/// such that A = L·L^T; otherwise writes the upper factor U = L^T such
/// that A = U^T·U. The opposite triangle is explicitly zeroed.
extern "C" fn cranelift_cholesky(
    a_ptr: *const f64,
    n: usize,
    lower: i32,
    out_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * n) };

    let mut col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice_mut(&mut col_data, n, n);

    let req = faer::linalg::cholesky::llt::compute::cholesky_in_place_req::<f64>(
        n,
        faer::Parallelism::None,
        Default::default(),
    )
    .unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    let result = faer::linalg::cholesky::llt::compute::cholesky_in_place(
        mat,
        Default::default(),
        faer::Parallelism::None,
        stack,
        Default::default(),
    );

    // col_data now holds L in its lower triangle with the strict upper left as
    // garbage by faer. Element (row, col) in column-major is at col*n + row,
    // so L's strict upper = { (row, col) : row < col } and strict lower =
    // { (row, col) : row > col }.
    if lower != 0 {
        for col in 0..n {
            for row in 0..col {
                col_data[col * n + row] = 0.0;
            }
        }
    } else {
        // U = L^T. Swap (row, col) with (col, row) for row > col, then zero
        // the strict lower of the resulting U.
        for col in 0..n {
            for row in (col + 1)..n {
                col_data.swap(col * n + row, row * n + col);
            }
        }
        for col in 0..n {
            for row in (col + 1)..n {
                col_data[col * n + row] = 0.0;
            }
        }
    }

    let row_data = col_major_to_row_major(&col_data, n, n);
    out.copy_from_slice(&row_data);

    unsafe { *info_ptr = if result.is_ok() { 0 } else { 1 } };
}

extern "C" fn cranelift_qr(
    a_ptr: *const f64,
    m: usize,
    n: usize,
    qr_ptr: *mut f64,
    tau_ptr: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let qr_out = unsafe { std::slice::from_raw_parts_mut(qr_ptr, m * n) };
    let tau_out = unsafe { std::slice::from_raw_parts_mut(tau_ptr, m.min(n)) };

    let mut col_data = row_major_to_col_major(a, m, n);
    let mut factors = faer::mat::from_column_major_slice_mut(&mut col_data, m, n);

    let min_mn = m.min(n);
    let blocksize = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<f64>(m, n);
    let mut h_data = vec![0.0f64; blocksize * min_mn];
    let mut householder = faer::mat::from_column_major_slice_mut(&mut h_data, blocksize, min_mn);

    let params = Default::default();
    let parallelism = faer::Parallelism::None;
    let req = faer::linalg::qr::no_pivoting::compute::qr_in_place_req::<f64>(
        m,
        n,
        blocksize,
        parallelism,
        params,
    )
    .unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    faer::linalg::qr::no_pivoting::compute::qr_in_place(
        factors.as_mut(),
        householder.as_mut(),
        parallelism,
        stack,
        params,
    );

    let row_data = col_major_to_row_major(&col_data, m, n);
    qr_out.copy_from_slice(&row_data);

    // Store first row of householder block as tau (the reflector coefficients)
    for i in 0..min_mn {
        tau_out[i] = h_data[i * blocksize];
    }
}

extern "C" fn cranelift_orgqr(
    qr_ptr: *const f64,
    tau_ptr: *const f64,
    m: usize,
    n: usize,
    q_ptr: *mut f64,
) {
    let qr_data = unsafe { std::slice::from_raw_parts(qr_ptr, m * n) };
    let tau = unsafe { std::slice::from_raw_parts(tau_ptr, m.min(n)) };
    let q_out = unsafe { std::slice::from_raw_parts_mut(q_ptr, m * n) };

    let min_mn = m.min(n);
    let blocksize = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<f64>(m, n);

    // Rebuild the full block householder factor matrix from the scalar tau values
    // The first row of each block column contains the tau value
    let mut h_data = vec![0.0f64; blocksize * min_mn];
    for i in 0..min_mn {
        h_data[i * blocksize] = tau[i];
    }
    let householder = faer::mat::from_column_major_slice(&h_data, blocksize, min_mn);

    let qr_col = row_major_to_col_major(qr_data, m, n);
    let factors = faer::mat::from_column_major_slice(&qr_col, m, n);

    let mut q = faer::Mat::<f64>::zeros(m, m);
    q.as_mut().diagonal_mut().column_vector_mut().fill(1.0);

    let req = faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_req::<f64>(
        m, blocksize, m,
    ).unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        factors,
        householder,
        faer::Conj::No,
        q.as_mut(),
        faer::Parallelism::None,
        stack,
    );

    for i in 0..m {
        for j in 0..n {
            q_out[i * n + j] = q.read(i, j);
        }
    }
}

extern "C" fn cranelift_syevd(
    a_ptr: *const f64,
    n: usize,
    eigvecs_ptr: *mut f64,
    eigvals_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let eigvecs_out = unsafe { std::slice::from_raw_parts_mut(eigvecs_ptr, n * n) };
    let eigvals_out = unsafe { std::slice::from_raw_parts_mut(eigvals_ptr, n) };

    let col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice(&col_data, n, n);
    let eigen = faer::linalg::solvers::SelfAdjointEigendecomposition::new(mat, faer::Side::Lower);

    let u = eigen.u();
    for i in 0..n {
        for j in 0..n {
            eigvecs_out[i * n + j] = u.read(i, j);
        }
    }
    let s = eigen.s();
    for (i, val) in eigvals_out.iter_mut().enumerate() {
        *val = s.column_vector().read(i);
    }

    unsafe { *info_ptr = 0 };
}

// ---------------------------------------------------------------------------
// Cranelift IR lowering for each LAPACK custom_call
// ---------------------------------------------------------------------------

fn store_f64_vals(
    builder: &mut FunctionBuilder,
    vals: &[cranelift_codegen::ir::Value],
    base: cranelift_codegen::ir::Value,
    offset: i32,
) {
    for (i, &v) in vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, offset + (i * 8) as i32);
    }
}

fn load_f64_scalar_lane(
    builder: &mut FunctionBuilder,
    count: usize,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> LaneRepr {
    LaneRepr::scalar(load_f64_vals(builder, count, base, offset))
}

fn load_f64_vals(
    builder: &mut FunctionBuilder,
    count: usize,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> Vec<cranelift_codegen::ir::Value> {
    (0..count)
        .map(|i| {
            builder.ins().load(
                types::F64,
                MemFlags::trusted(),
                base,
                offset + (i * 8) as i32,
            )
        })
        .collect()
}

fn load_i32_vals(
    builder: &mut FunctionBuilder,
    count: usize,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> Vec<cranelift_codegen::ir::Value> {
    (0..count)
        .map(|i| {
            builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                base,
                offset + (i * 4) as i32,
            )
        })
        .collect()
}

fn require_square(name: &str, vals: &[cranelift_codegen::ir::Value]) -> Result<usize, String> {
    let n = (vals.len() as f64).sqrt() as usize;
    if n * n != vals.len() {
        return Err(format!(
            "{name}: non-square matrix ({} elements)",
            vals.len()
        ));
    }
    Ok(n)
}

fn lapack_slot(
    builder: &mut FunctionBuilder,
    total_bytes: usize,
) -> (
    cranelift_codegen::ir::StackSlot,
    cranelift_codegen::ir::Value,
) {
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        SLOT_ALIGN,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);
    (ss, base)
}

fn lapack_call(
    builder: &mut FunctionBuilder,
    jit_module: &mut JITModule,
    symbol: &str,
    param_types: &[cranelift_codegen::ir::Type],
    args: &[cranelift_codegen::ir::Value],
) -> Result<(), String> {
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    for &ty in param_types {
        sig.params.push(AbiParam::new(ty));
    }
    let func_id = jit_module
        .declare_function(symbol, Linkage::Import, &sig)
        .map_err(|e| format!("declare {symbol}: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    builder.ins().call(func_ref, args);
    Ok(())
}

fn load_info_i32(
    builder: &mut FunctionBuilder,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> cranelift_codegen::ir::Value {
    builder
        .ins()
        .load(types::I32, MemFlags::trusted(), base, offset)
}

fn lower_svd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let n = require_square("SVD", a_vals)?;
    let f8 = 8;

    let (a_b, u_b, s_b, vt_b, info_b) = (n * n * f8, n * n * f8, n * f8, n * n * f8, 4);
    let (_, base) = lapack_slot(builder, a_b + u_b + s_b + vt_b + info_b);
    store_f64_vals(builder, a_vals, base, 0);

    let u_off = a_b as i32;
    let s_off = (a_b + u_b) as i32;
    let vt_off = (a_b + u_b + s_b) as i32;
    let info_off = (a_b + u_b + s_b + vt_b) as i32;
    let u_ptr = builder.ins().iadd_imm(base, u_off as i64);
    let s_ptr = builder.ins().iadd_imm(base, s_off as i64);
    let vt_ptr = builder.ins().iadd_imm(base, vt_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_svd",
        &[pt, types::I64, pt, pt, pt, pt],
        &[base, n_val, u_ptr, s_ptr, vt_ptr, info_ptr],
    )?;

    // XLA dgesdd_ffi convention: (A_overwritten, sigma, U, VT, info)
    let mut result_groups = vec![
        load_f64_scalar_lane(builder, n * n, base, u_off),
        load_f64_scalar_lane(builder, n, base, s_off),
        load_f64_scalar_lane(builder, n * n, base, u_off),
    ];
    if result_types.len() > 3 {
        result_groups.push(load_f64_scalar_lane(builder, n * n, base, vt_off));
    }
    if result_types.len() > 4 {
        result_groups.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, info_off,
        )]));
    }
    Ok(result_groups)
}

fn lower_lu_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    _result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let n = require_square("LU", a_vals)?;
    let (m, f8) = (n, 8);

    let (lu_b, ipiv_b, info_b) = (m * n * f8, m.min(n) * 4, 4);
    let (_, base) = lapack_slot(builder, lu_b + lu_b + ipiv_b + info_b);
    store_f64_vals(builder, a_vals, base, 0);

    let lu_off = lu_b as i32;
    let ipiv_off = (lu_b + lu_b) as i32;
    let info_off = (lu_b + lu_b + ipiv_b) as i32;
    let lu_ptr = builder.ins().iadd_imm(base, lu_off as i64);
    let ipiv_ptr = builder.ins().iadd_imm(base, ipiv_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_lu",
        &[pt, types::I64, types::I64, pt, pt, pt],
        &[base, m_val, n_val, lu_ptr, ipiv_ptr, info_ptr],
    )?;

    Ok(vec![
        load_f64_scalar_lane(builder, m * n, base, lu_off),
        LaneRepr::scalar(load_i32_vals(builder, m.min(n), base, ipiv_off)),
        LaneRepr::scalar(vec![load_info_i32(builder, base, info_off)]),
    ])
}

fn lower_trsm_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?.to_vec();
    let b_vals = get_vals(builder, value_map, &operands[1])?.to_vec();

    let rt = &result_types[0];
    let (m, nrhs) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("TRSM: expected 2D result".to_string());
    };
    let n = (a_vals.len() as f64).sqrt() as usize;
    let f8 = 8;

    let (a_b, b_b, out_b) = (a_vals.len() * f8, b_vals.len() * f8, m * nrhs * f8);
    let (_, base) = lapack_slot(builder, a_b + b_b + out_b);
    store_f64_vals(builder, &a_vals, base, 0);
    let b_off = a_b as i32;
    store_f64_vals(builder, &b_vals, base, b_off);
    let out_off = (a_b + b_b) as i32;
    let b_ptr = builder.ins().iadd_imm(base, b_off as i64);
    let out_ptr = builder.ins().iadd_imm(base, out_off as i64);

    let uplo = *backend_config.get("uplo").unwrap_or(&(b'L' as i64)) as u8;
    let diag = *backend_config.get("diag").unwrap_or(&(b'N' as i64)) as u8;
    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_imm = builder.ins().iconst(types::I64, n as i64);
    let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
    let uplo_val = builder.ins().iconst(types::I8, uplo as i64);
    let diag_val = builder.ins().iconst(types::I8, diag as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_trsm",
        &[
            pt,
            pt,
            types::I64,
            types::I64,
            types::I64,
            types::I8,
            types::I8,
            pt,
        ],
        &[
            base, b_ptr, m_val, n_imm, nrhs_val, uplo_val, diag_val, out_ptr,
        ],
    )?;

    Ok(vec![load_f64_scalar_lane(builder, m * nrhs, base, out_off)])
}

fn lower_cholesky_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    _result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<LaneRepr>, String> {
    // Decode `uplo` from the custom_call's backend_config: ASCII 'L' (76) =>
    // lower factor L, 'U' (85) => upper factor U. Absent => default to L, the
    // only form JAX emits today (upper is constructed outside the call via
    // stablehlo.transpose).
    let lower_flag: i32 = match backend_config.get("uplo").copied() {
        Some(76) | None => 1,
        Some(85) => 0,
        Some(other) => return Err(format!("cholesky: unsupported uplo {other}")),
    };

    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let a_ty = type_map
        .get(&operands[0])
        .ok_or_else(|| "cholesky: missing operand type".to_string())?;
    let shape = &a_ty.shape;
    if shape.len() < 2 {
        return Err("cholesky: input rank must be >= 2".to_string());
    }
    let n = shape[shape.len() - 1] as usize;
    if shape[shape.len() - 2] as usize != n {
        return Err(format!(
            "cholesky: last two dims must be square, got {}x{}",
            shape[shape.len() - 2],
            n,
        ));
    }
    let batch_size: usize = shape[..shape.len() - 2]
        .iter()
        .map(|&d| d as usize)
        .product();
    if a_vals.len() != batch_size * n * n {
        return Err(format!(
            "cholesky: operand has {} elements, expected {} * {}^2 = {}",
            a_vals.len(),
            batch_size,
            n,
            batch_size * n * n,
        ));
    }

    let f8 = 8;
    let mat_b = n * n * f8;
    let info_b = 4;
    let (_, base) = lapack_slot(builder, mat_b + mat_b + info_b);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    let lower_val = builder.ins().iconst(types::I32, i64::from(lower_flag));
    let pt = ptr_type();
    let out_off = mat_b as i32;
    let info_off = (mat_b + mat_b) as i32;
    let out_ptr = builder.ins().iadd_imm(base, out_off as i64);
    let info_ptr_val = builder.ins().iadd_imm(base, info_off as i64);

    let mut flat: Vec<cranelift_codegen::ir::Value> = Vec::with_capacity(batch_size * n * n);
    let mut infos: Vec<cranelift_codegen::ir::Value> = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let slice = &a_vals[b * n * n..(b + 1) * n * n];
        store_f64_vals(builder, slice, base, 0);
        lapack_call(
            builder,
            jit_module,
            "__cranelift_cholesky",
            &[pt, types::I64, types::I32, pt, pt],
            &[base, n_val, lower_val, out_ptr, info_ptr_val],
        )?;
        flat.extend(load_f64_vals(builder, n * n, base, out_off));
        infos.push(load_info_i32(builder, base, info_off));
    }

    Ok(vec![LaneRepr::scalar(flat), LaneRepr::scalar(infos)])
}

fn lower_qr_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let rt = &result_types[0];
    let (m, n) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("QR: expected 2D result".to_string());
    };
    let f8 = 8;

    let (a_b, qr_b, tau_b) = (m * n * f8, m * n * f8, m.min(n) * f8);
    let (_, base) = lapack_slot(builder, a_b + qr_b + tau_b);
    store_f64_vals(builder, a_vals, base, 0);

    let qr_off = a_b as i32;
    let tau_off = (a_b + qr_b) as i32;
    let qr_ptr = builder.ins().iadd_imm(base, qr_off as i64);
    let tau_ptr = builder.ins().iadd_imm(base, tau_off as i64);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_qr",
        &[pt, types::I64, types::I64, pt, pt],
        &[base, m_val, n_val, qr_ptr, tau_ptr],
    )?;

    Ok(vec![
        load_f64_scalar_lane(builder, m * n, base, qr_off),
        load_f64_scalar_lane(builder, m.min(n), base, tau_off),
    ])
}

fn lower_orgqr_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let qr_vals = get_vals(builder, value_map, &operands[0])?.to_vec();
    let tau_vals = get_vals(builder, value_map, &operands[1])?.to_vec();

    let rt = &result_types[0];
    let (m, n) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("ORGQR: expected 2D result".to_string());
    };
    let f8 = 8;

    let (qr_b, tau_b, q_b) = (qr_vals.len() * f8, tau_vals.len() * f8, m * n * f8);
    let (_, base) = lapack_slot(builder, qr_b + tau_b + q_b);
    store_f64_vals(builder, &qr_vals, base, 0);
    let tau_off = qr_b as i32;
    store_f64_vals(builder, &tau_vals, base, tau_off);
    let q_off = (qr_b + tau_b) as i32;
    let tau_ptr = builder.ins().iadd_imm(base, tau_off as i64);
    let q_ptr = builder.ins().iadd_imm(base, q_off as i64);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_orgqr",
        &[pt, pt, types::I64, types::I64, pt],
        &[base, tau_ptr, m_val, n_val, q_ptr],
    )?;

    Ok(vec![load_f64_scalar_lane(builder, m * n, base, q_off)])
}

fn lower_syevd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    _result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let n = require_square("SYEVD", a_vals)?;
    let f8 = 8;

    let (a_b, ev_b, ew_b, info_b) = (n * n * f8, n * n * f8, n * f8, 4);
    let (_, base) = lapack_slot(builder, a_b + ev_b + ew_b + info_b);
    store_f64_vals(builder, a_vals, base, 0);

    let ev_off = a_b as i32;
    let ew_off = (a_b + ev_b) as i32;
    let info_off = (a_b + ev_b + ew_b) as i32;
    let ev_ptr = builder.ins().iadd_imm(base, ev_off as i64);
    let ew_ptr = builder.ins().iadd_imm(base, ew_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_syevd",
        &[pt, types::I64, pt, pt, pt],
        &[base, n_val, ev_ptr, ew_ptr, info_ptr],
    )?;

    Ok(vec![
        load_f64_scalar_lane(builder, n * n, base, ev_off),
        load_f64_scalar_lane(builder, n, base, ew_off),
        LaneRepr::scalar(vec![load_info_i32(builder, base, info_off)]),
    ])
}

// ---------------------------------------------------------------------------
// GESV: General linear solve Ax = B
// ---------------------------------------------------------------------------

extern "C" fn cranelift_gesv(
    a_ptr: *const f64,
    b_ptr: *const f64,
    n: usize,
    nrhs: usize,
    x_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    use faer::prelude::*;
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, n * nrhs) };
    let a_col = row_major_to_col_major(a, n, n);
    let b_col = row_major_to_col_major(b, n, nrhs);
    let a_mat = faer::mat::from_column_major_slice(&a_col, n, n);
    let b_mat = faer::mat::from_column_major_slice(&b_col, n, nrhs);
    let lu = a_mat.partial_piv_lu();
    let x_mat = lu.solve(&b_mat);
    let x_out = unsafe { std::slice::from_raw_parts_mut(x_ptr, n * nrhs) };
    for j in 0..nrhs {
        for i in 0..n {
            x_out[i * nrhs + j] = x_mat.read(i, j);
        }
    }
    unsafe { *info_ptr = 0 };
}

fn lower_gesv_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?.to_vec();
    let b_vals = get_vals(builder, value_map, &operands[1])?.to_vec();
    let a_ty = type_map.get(&operands[0]).ok_or("gesv: missing A type")?;
    let n = a_ty.shape[0] as usize;
    let nrhs = if b_vals.len() / n > 0 {
        b_vals.len() / n
    } else {
        1
    };
    let f8 = 8;

    let a_b = n * n * f8;
    let b_b = n * nrhs * f8;
    let x_b = n * nrhs * f8;
    let info_b = 4;
    let (_, base) = lapack_slot(builder, a_b + b_b + x_b + info_b);
    store_f64_vals(builder, &a_vals, base, 0);
    store_f64_vals(builder, &b_vals, base, a_b as i32);

    let x_off = (a_b + b_b) as i32;
    let info_off = (a_b + b_b + x_b) as i32;
    let x_ptr = builder.ins().iadd_imm(base, x_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
    let b_ptr_val = builder.ins().iadd_imm(base, a_b as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_gesv",
        &[pt, pt, types::I64, types::I64, pt, pt],
        &[base, b_ptr_val, n_val, nrhs_val, x_ptr, info_ptr],
    )?;

    let mut results = Vec::new();
    if !result_types.is_empty() {
        results.push(load_f64_scalar_lane(builder, n * nrhs, base, x_off));
    }
    if result_types.len() > 1 {
        results.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, info_off,
        )]));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// POTRS: Cholesky-based solve (L from prior dpotrf)
// ---------------------------------------------------------------------------

extern "C" fn cranelift_potrs(
    l_ptr: *const f64,
    b_ptr: *const f64,
    n: usize,
    nrhs: usize,
    uplo: u8,
    x_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let l = unsafe { std::slice::from_raw_parts(l_ptr, n * n) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, n * nrhs) };
    let l_col = row_major_to_col_major(l, n, n);
    let b_col = row_major_to_col_major(b, n, nrhs);
    let x_out = unsafe { std::slice::from_raw_parts_mut(x_ptr, n * nrhs) };
    // Solve L * L^T * x = b via forward then back substitution
    let _is_lower = uplo == b'L';
    // Forward: L * y = b
    let mut y = b_col.clone();
    for k in 0..nrhs {
        for i in 0..n {
            let mut s = y[k * n + i];
            for j in 0..i {
                s -= l_col[i + j * n] * y[k * n + j];
            }
            y[k * n + i] = s / l_col[i + i * n];
        }
    }
    // Backward: L^T * x = y
    let mut x_col = y;
    for k in 0..nrhs {
        for i in (0..n).rev() {
            let mut s = x_col[k * n + i];
            for j in (i + 1)..n {
                s -= l_col[j + i * n] * x_col[k * n + j];
            }
            x_col[k * n + i] = s / l_col[i + i * n];
        }
    }
    let x_row = col_major_to_row_major(&x_col, n, nrhs);
    x_out.copy_from_slice(&x_row);
    unsafe { *info_ptr = 0 };
}

fn lower_potrs_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<LaneRepr>, String> {
    let l_vals = get_vals(builder, value_map, &operands[0])?.to_vec();
    let b_vals = get_vals(builder, value_map, &operands[1])?.to_vec();
    let l_ty = type_map.get(&operands[0]).ok_or("potrs: missing L type")?;
    let n = l_ty.shape[0] as usize;
    let nrhs = if b_vals.len() / n > 0 {
        b_vals.len() / n
    } else {
        1
    };
    let uplo = backend_config.get("uplo").copied().unwrap_or(76) as u8;
    let f8 = 8;

    let l_b = n * n * f8;
    let b_b = n * nrhs * f8;
    let x_b = n * nrhs * f8;
    let info_b = 4;
    let (_, base) = lapack_slot(builder, l_b + b_b + x_b + info_b);
    store_f64_vals(builder, &l_vals, base, 0);
    store_f64_vals(builder, &b_vals, base, l_b as i32);

    let x_off = (l_b + b_b) as i32;
    let info_off = (l_b + b_b + x_b) as i32;
    let x_ptr = builder.ins().iadd_imm(base, x_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);
    let b_ptr_val = builder.ins().iadd_imm(base, l_b as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
    let uplo_val = builder.ins().iconst(types::I8, uplo as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_potrs",
        &[pt, pt, types::I64, types::I64, types::I8, pt, pt],
        &[base, b_ptr_val, n_val, nrhs_val, uplo_val, x_ptr, info_ptr],
    )?;

    let mut results = Vec::new();
    if !result_types.is_empty() {
        results.push(load_f64_scalar_lane(builder, n * nrhs, base, x_off));
    }
    if result_types.len() > 1 {
        results.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, info_off,
        )]));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// GELSD: Least-squares solve via SVD
// ---------------------------------------------------------------------------

extern "C" fn cranelift_gelsd(
    a_ptr: *const f64,
    b_ptr: *const f64,
    m: usize,
    n: usize,
    nrhs: usize,
    x_ptr: *mut f64,
    s_ptr: *mut f64,
    rank_ptr: *mut i32,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, m * nrhs) };
    let a_col = row_major_to_col_major(a, m, n);
    let b_col = row_major_to_col_major(b, m, nrhs);
    let a_mat = faer::mat::from_column_major_slice(&a_col, m, n);
    let b_mat = faer::mat::from_column_major_slice(&b_col, m, nrhs);

    let svd = a_mat.thin_svd();
    let min_mn = m.min(n);
    let s_diag = svd.s_diagonal();
    let s_out = unsafe { std::slice::from_raw_parts_mut(s_ptr, min_mn) };
    for (i, s) in s_out.iter_mut().enumerate() {
        *s = s_diag.read(i);
    }

    let threshold =
        f64::EPSILON * (m.max(n) as f64) * s_out.iter().cloned().fold(0.0_f64, f64::max);
    let mut effective_rank = 0i32;
    let x_out = unsafe { std::slice::from_raw_parts_mut(x_ptr, n * nrhs) };
    for j in 0..nrhs {
        for i in 0..n {
            let mut val = 0.0;
            for k in 0..min_mn {
                let sv = s_diag.read(k);
                if sv > threshold {
                    let mut ut_b = 0.0;
                    for r in 0..m {
                        ut_b += svd.u().read(r, k) * b_mat.read(r, j);
                    }
                    val += svd.v().read(i, k) * ut_b / sv;
                }
            }
            x_out[i * nrhs + j] = val;
        }
    }
    for s in s_out.iter() {
        if *s > threshold {
            effective_rank += 1;
        }
    }
    unsafe { *rank_ptr = effective_rank };
    unsafe { *info_ptr = 0 };
}

fn lower_gelsd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?.to_vec();
    let b_vals = get_vals(builder, value_map, &operands[1])?.to_vec();
    let a_ty = type_map.get(&operands[0]).ok_or("gelsd: missing A type")?;
    let m = a_ty.shape[0] as usize;
    let n = a_ty.shape[1] as usize;
    let nrhs = if b_vals.len() / m > 0 {
        b_vals.len() / m
    } else {
        1
    };
    let min_mn = m.min(n);
    let f8 = 8;

    let a_b = m * n * f8;
    let b_b = m * nrhs * f8;
    let x_b = n * nrhs * f8;
    let s_b = min_mn * f8;
    let rank_b = 4;
    let info_b = 4;
    let total = a_b + b_b + x_b + s_b + rank_b + info_b;
    let (_, base) = lapack_slot(builder, total);
    store_f64_vals(builder, &a_vals, base, 0);
    store_f64_vals(builder, &b_vals, base, a_b as i32);

    let x_off = (a_b + b_b) as i32;
    let s_off = (a_b + b_b + x_b) as i32;
    let rank_off = (a_b + b_b + x_b + s_b) as i32;
    let info_off = (a_b + b_b + x_b + s_b + rank_b) as i32;

    let b_ptr_val = builder.ins().iadd_imm(base, a_b as i64);
    let x_ptr = builder.ins().iadd_imm(base, x_off as i64);
    let s_ptr = builder.ins().iadd_imm(base, s_off as i64);
    let rank_ptr = builder.ins().iadd_imm(base, rank_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);
    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_gelsd",
        &[pt, pt, types::I64, types::I64, types::I64, pt, pt, pt, pt],
        &[
            base, b_ptr_val, m_val, n_val, nrhs_val, x_ptr, s_ptr, rank_ptr, info_ptr,
        ],
    )?;

    let mut results = Vec::new();
    results.push(load_f64_scalar_lane(builder, n * nrhs, base, x_off));
    if result_types.len() > 1 {
        results.push(load_f64_scalar_lane(builder, min_mn, base, s_off));
    }
    if result_types.len() > 2 {
        results.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, rank_off,
        )]));
    }
    if result_types.len() > 3 {
        results.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, info_off,
        )]));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// GEEV: Non-symmetric eigendecomposition
// ---------------------------------------------------------------------------

extern "C" fn cranelift_geev(
    a_ptr: *const f64,
    n: usize,
    wr_ptr: *mut f64,
    wi_ptr: *mut f64,
    vr_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    use faer::complex_native::c64;
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let a_col = row_major_to_col_major(a, n, n);
    let a_mat = faer::mat::from_column_major_slice(&a_col, n, n);

    let evd = a_mat.eigendecomposition::<c64>();
    let eigenvalues = evd.s();
    let eigenvectors = evd.u();

    let wr = unsafe { std::slice::from_raw_parts_mut(wr_ptr, n) };
    let wi = unsafe { std::slice::from_raw_parts_mut(wi_ptr, n) };
    let vr = unsafe { std::slice::from_raw_parts_mut(vr_ptr, n * n) };

    for i in 0..n {
        let ev = eigenvalues.column_vector().read(i);
        wr[i] = ev.re;
        wi[i] = ev.im;
    }

    let mut i = 0;
    while i < n {
        if wi[i].abs() < 1e-15 {
            for j in 0..n {
                vr[j * n + i] = eigenvectors.read(j, i).re;
            }
            i += 1;
        } else {
            for j in 0..n {
                vr[j * n + i] = eigenvectors.read(j, i).re;
                vr[j * n + i + 1] = eigenvectors.read(j, i).im;
            }
            i += 2;
        }
    }
    unsafe { *info_ptr = 0 };
}

fn lower_geev_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let n = require_square("GEEV", a_vals)?;
    let f8 = 8;

    let a_b = n * n * f8;
    let wr_b = n * f8;
    let wi_b = n * f8;
    let vr_b = n * n * f8;
    let info_b = 4;
    let (_, base) = lapack_slot(builder, a_b + wr_b + wi_b + vr_b + info_b);
    store_f64_vals(builder, a_vals, base, 0);

    let wr_off = a_b as i32;
    let wi_off = (a_b + wr_b) as i32;
    let vr_off = (a_b + wr_b + wi_b) as i32;
    let info_off = (a_b + wr_b + wi_b + vr_b) as i32;

    let wr_ptr = builder.ins().iadd_imm(base, wr_off as i64);
    let wi_ptr = builder.ins().iadd_imm(base, wi_off as i64);
    let vr_ptr = builder.ins().iadd_imm(base, vr_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_geev",
        &[pt, types::I64, pt, pt, pt, pt],
        &[base, n_val, wr_ptr, wi_ptr, vr_ptr, info_ptr],
    )?;

    let mut results = Vec::new();
    results.push(load_f64_scalar_lane(builder, n, base, wr_off));
    if result_types.len() > 1 {
        results.push(load_f64_scalar_lane(builder, n, base, wi_off));
    }
    if result_types.len() > 2 {
        results.push(load_f64_scalar_lane(builder, n * n, base, vr_off));
    }
    if result_types.len() > 3 {
        results.push(LaneRepr::scalar(vec![load_info_i32(
            builder, base, info_off,
        )]));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// GESVD: Full SVD (m x m U, n x n V)
// ---------------------------------------------------------------------------

extern "C" fn cranelift_gesvd(
    a_ptr: *const f64,
    m: usize,
    n: usize,
    u_ptr: *mut f64,
    s_ptr: *mut f64,
    vt_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let a_col = row_major_to_col_major(a, m, n);
    let a_mat = faer::mat::from_column_major_slice(&a_col, m, n);
    let svd = a_mat.svd();

    let min_mn = m.min(n);
    let s_diag = svd.s_diagonal();
    let s_out = unsafe { std::slice::from_raw_parts_mut(s_ptr, min_mn) };
    for (i, s) in s_out.iter_mut().enumerate() {
        *s = s_diag.read(i);
    }

    let u_out = unsafe { std::slice::from_raw_parts_mut(u_ptr, m * m) };
    for i in 0..m {
        for j in 0..m {
            u_out[i * m + j] = svd.u().read(i, j);
        }
    }

    let vt_out = unsafe { std::slice::from_raw_parts_mut(vt_ptr, n * n) };
    for i in 0..n {
        for j in 0..n {
            vt_out[i * n + j] = svd.v().read(j, i);
        }
    }
    unsafe { *info_ptr = 0 };
}

fn lower_gesvd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    value_map: &mut HashMap<ValueId, LaneRepr>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<LaneRepr>, String> {
    let a_vals = get_vals(builder, value_map, &operands[0])?;
    let a_ty = type_map.get(&operands[0]).ok_or("gesvd: missing A type")?;
    let m = a_ty.shape[0] as usize;
    let n = a_ty.shape[1] as usize;
    let min_mn = m.min(n);
    let f8 = 8;

    let a_b = m * n * f8;
    let u_b = m * m * f8;
    let s_b = min_mn * f8;
    let vt_b = n * n * f8;
    let info_b = 4;
    let total = a_b + u_b + s_b + vt_b + info_b;
    let (_, base) = lapack_slot(builder, total);
    store_f64_vals(builder, a_vals, base, 0);

    let u_off = a_b as i32;
    let s_off = (a_b + u_b) as i32;
    let vt_off = (a_b + u_b + s_b) as i32;
    let info_off = (a_b + u_b + s_b + vt_b) as i32;

    let u_ptr = builder.ins().iadd_imm(base, u_off as i64);
    let s_ptr = builder.ins().iadd_imm(base, s_off as i64);
    let vt_ptr = builder.ins().iadd_imm(base, vt_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);
    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    let pt = ptr_type();
    lapack_call(
        builder,
        jit_module,
        "__cranelift_gesvd",
        &[pt, types::I64, types::I64, pt, pt, pt, pt],
        &[base, m_val, n_val, u_ptr, s_ptr, vt_ptr, info_ptr],
    )?;

    // XLA result ordering: (A_overwritten, sigma, U, VT, info)
    let results = vec![
        load_f64_scalar_lane(builder, m * n, base, 0),
        load_f64_scalar_lane(builder, min_mn, base, s_off),
        load_f64_scalar_lane(builder, m * m, base, u_off),
        load_f64_scalar_lane(builder, n * n, base, vt_off),
        LaneRepr::scalar(vec![load_info_i32(builder, base, info_off)]),
    ];
    Ok(results)
}
