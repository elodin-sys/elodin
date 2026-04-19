//! Single-caller inliner for cranelift-mlir.
//!
//! This is a narrow, IR-level pass that runs before compilation. It
//! walks the whole module to identify callees that are invoked from
//! exactly one site across all function bodies, and, if the callee is
//! small enough, splices the callee's body into the caller in place
//! of the `Call` instruction.
//!
//! ## Why single-caller specifically
//!
//! Callees issued from exactly one site can be inlined with zero
//! code-size growth (no duplication). For workloads dominated by a
//! few hot parent→callee edges this eliminates per-call
//! prologue/epilogue without expanding compile time. A general
//! inliner would trade that bound for marginal extra benefit.
//!
//! ## Correctness contract
//!
//! - The callee must have exactly one call-site across the whole module.
//! - The callee must terminate with a `Return` instruction.
//! - The callee must not recurse (directly or indirectly).
//! - The callee's body must be < `INLINE_MAX_BODY_INSTRS` IR instructions.
//! - `main` is never inlined.
//!
//! Substitution rule: every parameter ValueId in the callee body is
//! replaced by the caller's argument ValueId; every other body
//! ValueId gets a fresh id; the Return's operands get mapped to the
//! caller's `Call` result ValueIds so downstream uses resolve
//! correctly.
//!
use std::collections::{HashMap, HashSet};

use crate::const_fold::{count_body_instructions, rewrite_aliases_body};
use crate::ir::{FuncDef, InstrResult, Instruction, Module, ValueId};

/// Maximum callee body size (top-level instruction count) eligible
/// for inlining. 200 handles typical helper-sized callees without
/// risking compile-time explosion.
const INLINE_MAX_BODY_INSTRS: usize = 200;

/// Caller-size cap: skip inlining when the caller already exceeds
/// this many top-level IR instructions. Inlining into an already-
/// large caller hurts L1I and register-allocation locality enough
/// to erase the call-overhead win. 1200 covers small-helper patterns
/// while excluding multi-thousand-IR hot functions.
const INLINE_MAX_CALLER_BODY_INSTRS: usize = 1200;

/// Return value: number of Call sites that were inlined. Useful for
/// logging and tests.
pub fn inline_single_caller_callees(module: &mut Module) -> usize {
    // Pass 1: build callee → caller-function-names map across every body.
    // Callers that invoke a callee more than once count as multi-caller
    // — conservative and matches "single call site" contract.
    let plan = plan_single_caller_inlines(module);
    if plan.is_empty() {
        return 0;
    }

    // Pass 2: fresh-id generator seeded past the module's current max.
    let mut next_value_id = max_value_id_in_module(module) + 1;

    // Snapshot the callee bodies once; the subsequent mutation pass
    // modifies caller bodies but never changes callee bodies, so the
    // snapshot stays valid.
    let callee_defs = lookup_callees(module);

    let mut inlined_count = 0usize;
    for i in 0..module.functions.len() {
        let caller_name = module.functions[i].name.clone();
        loop {
            let inlined = try_inline_one_top_level_call(
                &mut module.functions[i],
                &caller_name,
                &plan,
                &callee_defs,
                &mut next_value_id,
            );
            if !inlined {
                break;
            }
            inlined_count += 1;
        }
    }
    inlined_count
}

/// Build callee-name → sole-caller-name plan for all callees invoked
/// from exactly one call site and whose body is within the size budget.
fn plan_single_caller_inlines(module: &Module) -> HashMap<String, String> {
    // callee -> Vec<caller, per-caller-count>
    let mut callers: HashMap<String, Vec<String>> = HashMap::new();
    for f in &module.functions {
        collect_callees(&f.body, &f.name, &mut callers);
    }

    let func_by_name: HashMap<&str, &FuncDef> = module
        .functions
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();

    let mut plan: HashMap<String, String> = HashMap::new();
    for (callee, sites) in callers.iter() {
        if sites.len() != 1 {
            continue;
        }
        if callee == "main" {
            continue;
        }
        let caller = &sites[0];
        // Defensively skip self-recursive inlines (which single-call
        // counting already excludes unless the callee calls itself
        // exactly once).
        if caller == callee {
            continue;
        }
        let Some(def) = func_by_name.get(callee.as_str()) else {
            continue;
        };
        // Body size budget.
        if count_body_instructions(&def.body) >= INLINE_MAX_BODY_INSTRS {
            continue;
        }
        // Caller-side size cap: inlining into an already-large caller
        // hurts cache locality more than the call-overhead saving it
        // nets. `func_by_name` snapshots the pre-pass module, so the
        // check reads the caller's pre-inline body — protecting
        // already-big callers without a cumulative tally.
        if let Some(caller_def) = func_by_name.get(caller.as_str())
            && count_body_instructions(&caller_def.body) >= INLINE_MAX_CALLER_BODY_INSTRS
        {
            continue;
        }
        // Must end with a Return (no early exits / missing terminator).
        if !body_ends_with_return(&def.body) {
            continue;
        }
        // Skip callees with nested body regions
        // (while/case/map/reduce_window/sort/scatter). Their inner
        // block parameters require careful ValueId renaming; the
        // common single-caller helpers don't contain them. Relaxable
        // in a follow-up.
        if body_has_nested_region(&def.body) {
            continue;
        }
        // Skip callees that contain instructions only supported on the
        // scalar-ABI lowering path. If such a callee were inlined into
        // a pointer-ABI caller (classifier picks ptr-ABI for large
        // tensors), the `lower_instruction_mem` path would bail with
        // "custom_call not yet supported" etc. The callee is already
        // compiled separately as scalar-ABI via the normal call path,
        // so not inlining here costs us the call overhead but stays
        // correct.
        if body_has_scalar_abi_only_ops(&def.body) {
            continue;
        }
        plan.insert(callee.clone(), caller.clone());
    }
    plan
}

fn lookup_callees(module: &Module) -> HashMap<String, FuncDef> {
    module
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.clone()))
        .collect()
}

/// Walk a body and record every caller of every callee. The passed-in
/// `caller` is the enclosing function name for top-level calls; nested
/// bodies (while/case/etc.) still count against the outer function.
fn collect_callees(body: &[InstrResult], caller: &str, out: &mut HashMap<String, Vec<String>>) {
    for ir in body {
        collect_callees_instr(&ir.instr, caller, out);
    }
}

fn collect_callees_instr(
    instr: &Instruction,
    caller: &str,
    out: &mut HashMap<String, Vec<String>>,
) {
    match instr {
        Instruction::Call { callee, .. } => {
            out.entry(callee.clone())
                .or_default()
                .push(caller.to_string());
        }
        Instruction::While {
            cond_body,
            loop_body,
            ..
        } => {
            collect_callees(cond_body, caller, out);
            collect_callees(loop_body, caller, out);
        }
        Instruction::Case { branches, .. } => {
            for b in branches {
                collect_callees(b, caller, out);
            }
        }
        Instruction::Map { body, .. } => collect_callees(body, caller, out),
        Instruction::ReduceWindow { body, .. } => collect_callees(body, caller, out),
        Instruction::SelectAndScatter {
            select_body,
            scatter_body,
            ..
        } => {
            collect_callees(select_body, caller, out);
            collect_callees(scatter_body, caller, out);
        }
        Instruction::Sort { comparator, .. } => collect_callees(comparator, caller, out),
        _ => {}
    }
}

fn body_ends_with_return(body: &[InstrResult]) -> bool {
    matches!(
        body.last().map(|ir| &ir.instr),
        Some(Instruction::Return { .. })
    )
}

fn body_has_nested_region(body: &[InstrResult]) -> bool {
    body.iter().any(|ir| {
        matches!(
            ir.instr,
            Instruction::While { .. }
                | Instruction::Case { .. }
                | Instruction::Map { .. }
                | Instruction::ReduceWindow { .. }
                | Instruction::SelectAndScatter { .. }
                | Instruction::Sort { .. }
        )
    })
}

/// Return true if the body contains any instruction that is only
/// supported on the scalar-ABI lowering path (not in
/// `lower_instruction_mem`). Inlining such a callee into a pointer-ABI
/// caller would produce "not supported" errors at compile time.
fn body_has_scalar_abi_only_ops(body: &[InstrResult]) -> bool {
    body.iter().any(|ir| {
        matches!(
            ir.instr,
            // LAPACK SVD / QR / EVD / Cholesky etc.
            Instruction::CustomCall { .. }
                | Instruction::CholeskyOp { .. }
                | Instruction::TriangularSolve { .. }
                // FFT, RNG, convolution — pointer-ABI doesn't always
                // handle these; exclude defensively.
                | Instruction::Fft { .. }
                | Instruction::Rng { .. }
                | Instruction::Convolution { .. }
                | Instruction::BatchNormInference { .. }
        )
    })
}

/// Scan the module for the largest-ValueId we've seen so we can allocate
/// fresh ones without collision.
fn max_value_id_in_module(module: &Module) -> u32 {
    let mut max_id = 0u32;
    for f in &module.functions {
        for (vid, _) in &f.params {
            max_id = max_id.max(vid.0);
        }
        scan_body_for_max_value_id(&f.body, &mut max_id);
    }
    max_id
}

fn scan_body_for_max_value_id(body: &[InstrResult], max_id: &mut u32) {
    for ir in body {
        for (vid, _) in &ir.values {
            *max_id = (*max_id).max(vid.0);
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                iter_arg_ids,
                ..
            } => {
                for v in iter_arg_ids {
                    *max_id = (*max_id).max(v.0);
                }
                scan_body_for_max_value_id(cond_body, max_id);
                scan_body_for_max_value_id(loop_body, max_id);
            }
            Instruction::Case { branches, .. } => {
                for b in branches {
                    scan_body_for_max_value_id(b, max_id);
                }
            }
            Instruction::Map { body, .. } => scan_body_for_max_value_id(body, max_id),
            Instruction::ReduceWindow { body, .. } => scan_body_for_max_value_id(body, max_id),
            Instruction::SelectAndScatter {
                select_body,
                scatter_body,
                ..
            } => {
                scan_body_for_max_value_id(select_body, max_id);
                scan_body_for_max_value_id(scatter_body, max_id);
            }
            Instruction::Sort { comparator, .. } => scan_body_for_max_value_id(comparator, max_id),
            _ => {}
        }
    }
}

/// Find the first top-level `Call` in this function's body that matches
/// the plan, and inline it. Only handles top-level calls (not nested in
/// `while` bodies etc.) to keep the pass narrow; in practice the
/// `closed_call_786` pattern sits at top-level. Returns true iff an
/// inlining happened.
fn try_inline_one_top_level_call(
    caller: &mut FuncDef,
    caller_name: &str,
    plan: &HashMap<String, String>,
    callee_defs: &HashMap<String, FuncDef>,
    next_value_id: &mut u32,
) -> bool {
    // Find the Call site.
    let idx = caller.body.iter().position(|ir| {
        matches!(&ir.instr, Instruction::Call { callee, .. }
            if plan.get(callee).is_some_and(|c| c == caller_name))
    });
    let Some(call_pos) = idx else {
        return false;
    };

    let call_ir = &caller.body[call_pos];
    let (callee_name, args) = match &call_ir.instr {
        Instruction::Call { callee, args } => (callee.clone(), args.clone()),
        _ => return false,
    };
    let call_results: Vec<ValueId> = call_ir.values.iter().map(|(v, _)| *v).collect();
    let Some(callee_def) = callee_defs.get(&callee_name) else {
        return false;
    };

    // Sanity: arity must match (parser should guarantee this, but
    // profiling is defensive).
    if args.len() != callee_def.params.len() {
        return false;
    }

    // Extract the Return operands (last instruction).
    let return_operands: Vec<ValueId> = match callee_def.body.last() {
        Some(ir) => {
            if let Instruction::Return { operands } = &ir.instr {
                operands.clone()
            } else {
                return false;
            }
        }
        None => return false,
    };

    // Skip edge case: a return operand that is also a parameter would
    // require us to introduce a move/copy (arg → call_result), which
    // this narrow pass doesn't do. Cases like `f(p0) { return p0 }`
    // are rare and trivially handled by the caller optimizing around.
    let param_ids: HashSet<ValueId> = callee_def.params.iter().map(|(v, _)| *v).collect();
    if return_operands.iter().any(|v| param_ids.contains(v)) {
        return false;
    }

    // TWO-PHASE SUBSTITUTION.
    //
    // The callee and caller ValueId namespaces overlap — both start
    // from 0 — so if we merge their mappings into one subst the alias
    // chain walker in `rewrite_aliases_body` will incorrectly follow
    // `callee_param(0) → caller_arg(2) → callee_body(2)_fresh` when
    // the intent is "callee_param(0) → caller_arg(2); stop there".
    //
    // Phase A: rename EVERY callee value (params + body locals +
    //   inner block params) to fresh ids. After this, the body uses
    //   only ids guaranteed not to overlap with the caller.
    // Phase B: map fresh_param_id → caller's actual arg value and
    //   fresh_return_operand_id → caller's call-result value. No
    //   overlap possible, so chain walking is safe.

    // --- Phase A subst: everything → fresh id. ---
    let mut phase_a: HashMap<ValueId, ValueId> = HashMap::new();
    for (pid, _) in &callee_def.params {
        phase_a.entry(*pid).or_insert_with(|| {
            let fresh = ValueId(*next_value_id);
            *next_value_id += 1;
            fresh
        });
    }
    collect_body_value_defs_into_subst(&callee_def.body, &mut phase_a, next_value_id);

    // Rewrite the callee body with Phase A subst (fresh everywhere).
    let mut inlined: Vec<InstrResult> = callee_def
        .body
        .iter()
        .filter(|ir| !matches!(ir.instr, Instruction::Return { .. }))
        .cloned()
        .collect();

    // Phase A apply: LHS + block params + operands all → fresh.
    for ir in inlined.iter_mut() {
        for (vid, _) in ir.values.iter_mut() {
            if let Some(&new) = phase_a.get(vid) {
                *vid = new;
            }
        }
    }
    rewrite_block_param_ids_body(&mut inlined, &phase_a);
    rewrite_aliases_body(&mut inlined, &phase_a);

    // --- Phase B subst: fresh_param → caller_arg, fresh_return → call_result. ---
    let mut phase_b: HashMap<ValueId, ValueId> = HashMap::new();
    for ((pid, _), &aid) in callee_def.params.iter().zip(args.iter()) {
        if let Some(&fresh) = phase_a.get(pid) {
            phase_b.insert(fresh, aid);
        }
    }
    for (ret_op, call_result) in return_operands.iter().zip(call_results.iter()) {
        if let Some(&fresh) = phase_a.get(ret_op) {
            phase_b.insert(fresh, *call_result);
        }
    }

    // Phase B apply: operands + block params → caller's values. LHS
    // rewrites for return-operand fresh ids so the defining
    // instruction in the inlined body assigns the caller's
    // call-result id directly.
    for ir in inlined.iter_mut() {
        for (vid, _) in ir.values.iter_mut() {
            if let Some(&new) = phase_b.get(vid) {
                *vid = new;
            }
        }
    }
    rewrite_block_param_ids_body(&mut inlined, &phase_b);
    rewrite_aliases_body(&mut inlined, &phase_b);

    // Splice: remove the Call at `call_pos`, insert the inlined body.
    let tail = caller.body.split_off(call_pos + 1);
    caller.body.pop(); // drop the Call
    caller.body.extend(inlined);
    caller.body.extend(tail);
    if crate::debug::enabled() {
        eprintln!(
            "[inliner] inlined {} into {} at pos {} (phase_a={}, phase_b={})",
            callee_name,
            caller_name,
            call_pos,
            phase_a.len(),
            phase_b.len()
        );
    }
    true
}

/// For every `(vid, _)` defined as a result in the callee body, insert
/// `subst[vid] = fresh_id` unless already present (e.g. parameter).
/// Also descends into nested bodies so inner ValueIds are renamed too.
fn collect_body_value_defs_into_subst(
    body: &[InstrResult],
    subst: &mut HashMap<ValueId, ValueId>,
    next_value_id: &mut u32,
) {
    for ir in body {
        for (vid, _) in &ir.values {
            subst.entry(*vid).or_insert_with(|| {
                let fresh = ValueId(*next_value_id);
                *next_value_id += 1;
                fresh
            });
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                iter_arg_ids,
                ..
            } => {
                for v in iter_arg_ids {
                    subst.entry(*v).or_insert_with(|| {
                        let fresh = ValueId(*next_value_id);
                        *next_value_id += 1;
                        fresh
                    });
                }
                collect_body_value_defs_into_subst(cond_body, subst, next_value_id);
                collect_body_value_defs_into_subst(loop_body, subst, next_value_id);
            }
            Instruction::Case { branches, .. } => {
                for b in branches {
                    collect_body_value_defs_into_subst(b, subst, next_value_id);
                }
            }
            Instruction::Map { body, .. } => {
                collect_body_value_defs_into_subst(body, subst, next_value_id)
            }
            Instruction::ReduceWindow { body, .. } => {
                collect_body_value_defs_into_subst(body, subst, next_value_id)
            }
            Instruction::SelectAndScatter {
                select_body,
                scatter_body,
                ..
            } => {
                collect_body_value_defs_into_subst(select_body, subst, next_value_id);
                collect_body_value_defs_into_subst(scatter_body, subst, next_value_id);
            }
            Instruction::Sort { comparator, .. } => {
                collect_body_value_defs_into_subst(comparator, subst, next_value_id)
            }
            _ => {}
        }
    }
    // Used operands that aren't defined (e.g. parameters referenced
    // inside nested bodies) are handled by the parameter subst added
    // before this helper runs.
    let _ = HashSet::<ValueId>::new(); // keep import silent in case
}

/// Rewrite inner-block-parameter ValueIds that `rewrite_aliases_body`
/// leaves untouched: `While::iter_arg_ids`, `Map::body_params`,
/// `ReduceWindow::body_params`, `SelectAndScatter::{select,scatter}_params`,
/// and `Sort::comparator_params`. These are define-sites for the
/// embedded body's block parameters; without remapping them, the body
/// references (which we DO rewrite via the alias helper) end up
/// pointing at fresh ids while the parameter declarations still hold
/// the callee's original ids.
fn rewrite_block_param_ids_body(body: &mut [InstrResult], subst: &HashMap<ValueId, ValueId>) {
    for ir in body.iter_mut() {
        rewrite_block_param_ids_instr(&mut ir.instr, subst);
    }
}

fn rewrite_block_param_ids_instr(instr: &mut Instruction, subst: &HashMap<ValueId, ValueId>) {
    let remap = |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    };
    match instr {
        Instruction::While {
            iter_arg_ids,
            cond_body,
            loop_body,
            ..
        } => {
            for v in iter_arg_ids.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(cond_body, subst);
            rewrite_block_param_ids_body(loop_body, subst);
        }
        Instruction::Case { branches, .. } => {
            for b in branches.iter_mut() {
                rewrite_block_param_ids_body(b, subst);
            }
        }
        Instruction::Map {
            body, body_params, ..
        } => {
            for v in body_params.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(body, subst);
        }
        Instruction::ReduceWindow {
            body, body_params, ..
        } => {
            for v in body_params.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(body, subst);
        }
        Instruction::SelectAndScatter {
            select_body,
            select_params,
            scatter_body,
            scatter_params,
            ..
        } => {
            for v in select_params.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(select_body, subst);
            for v in scatter_params.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(scatter_body, subst);
        }
        Instruction::Sort {
            comparator,
            comparator_params,
            ..
        } => {
            for v in comparator_params.iter_mut() {
                remap(v);
            }
            rewrite_block_param_ids_body(comparator, subst);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConstantValue, ScalarValue, TensorType};

    fn ty_f64() -> TensorType {
        TensorType::scalar(crate::ir::ElementType::F64)
    }

    fn mk_constant(id: u32, v: f64) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(id), ty_f64())],
            instr: Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(v)),
            },
        }
    }

    /// Caller has `a = f(x); return a * 2`, callee `f(p0) { return p0 + 1 }`.
    /// After inlining, caller should contain Add then Multiply, no Call.
    #[test]
    fn inlines_single_caller_callee() {
        let f = FuncDef {
            name: "f".into(),
            is_public: false,
            params: vec![(ValueId(0), ty_f64())],
            result_types: vec![ty_f64()],
            body: vec![
                mk_constant(1, 1.0),
                InstrResult {
                    values: vec![(ValueId(2), ty_f64())],
                    instr: Instruction::Add {
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(2)],
                    },
                },
            ],
            source_line: None,
        };
        let main = FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), ty_f64())],
            result_types: vec![ty_f64()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(200), ty_f64())],
                    instr: Instruction::Call {
                        callee: "f".into(),
                        args: vec![ValueId(100)],
                    },
                },
                mk_constant(201, 2.0),
                InstrResult {
                    values: vec![(ValueId(202), ty_f64())],
                    instr: Instruction::Multiply {
                        lhs: ValueId(200),
                        rhs: ValueId(201),
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(202)],
                    },
                },
            ],
            source_line: None,
        };
        let mut module = Module::new(vec![f, main]);
        let n = inline_single_caller_callees(&mut module);
        assert_eq!(n, 1);
        let main = module.functions.iter().find(|f| f.name == "main").unwrap();
        // After inlining: Constant (for the `1.0`), Add, Constant (2.0), Multiply, Return.
        // Most importantly, no Call.
        assert!(
            !main
                .body
                .iter()
                .any(|ir| matches!(ir.instr, Instruction::Call { .. })),
            "Call should be gone after inlining, body = {:#?}",
            main.body
        );
        // And the Multiply's lhs should resolve to the Add's result.
        let add_result_id = main
            .body
            .iter()
            .find_map(|ir| match &ir.instr {
                Instruction::Add { .. } => ir.values.first().map(|(v, _)| *v),
                _ => None,
            })
            .expect("Add should be present");
        let mul_lhs = main
            .body
            .iter()
            .find_map(|ir| match &ir.instr {
                Instruction::Multiply { lhs, .. } => Some(*lhs),
                _ => None,
            })
            .expect("Multiply should be present");
        assert_eq!(
            mul_lhs, add_result_id,
            "Multiply should consume Add's result (was the Call's output)"
        );
    }

    #[test]
    fn does_not_inline_multi_caller_callee() {
        let f = FuncDef {
            name: "f".into(),
            is_public: false,
            params: vec![(ValueId(0), ty_f64())],
            result_types: vec![ty_f64()],
            body: vec![InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(0)],
                },
            }],
            source_line: None,
        };
        // Two calls to `f` — f should not be inlined.
        let main = FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), ty_f64())],
            result_types: vec![ty_f64()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(200), ty_f64())],
                    instr: Instruction::Call {
                        callee: "f".into(),
                        args: vec![ValueId(100)],
                    },
                },
                InstrResult {
                    values: vec![(ValueId(201), ty_f64())],
                    instr: Instruction::Call {
                        callee: "f".into(),
                        args: vec![ValueId(100)],
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(201)],
                    },
                },
            ],
            source_line: None,
        };
        let mut module = Module::new(vec![f, main]);
        let n = inline_single_caller_callees(&mut module);
        assert_eq!(n, 0);
    }

    #[test]
    fn does_not_inline_oversize_callee() {
        // Build a callee with > INLINE_MAX_BODY_INSTRS instructions.
        let mut body: Vec<InstrResult> = (0..INLINE_MAX_BODY_INSTRS + 10)
            .map(|i| mk_constant(i as u32 + 10, i as f64))
            .collect();
        body.push(InstrResult {
            values: vec![],
            instr: Instruction::Return {
                operands: vec![ValueId(10)],
            },
        });
        let f = FuncDef {
            name: "big".into(),
            is_public: false,
            params: vec![(ValueId(0), ty_f64())],
            result_types: vec![ty_f64()],
            body,
            source_line: None,
        };
        let main = FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), ty_f64())],
            result_types: vec![ty_f64()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(200), ty_f64())],
                    instr: Instruction::Call {
                        callee: "big".into(),
                        args: vec![ValueId(100)],
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(200)],
                    },
                },
            ],
            source_line: None,
        };
        let mut module = Module::new(vec![f, main]);
        let n = inline_single_caller_callees(&mut module);
        assert_eq!(n, 0);
    }
}
