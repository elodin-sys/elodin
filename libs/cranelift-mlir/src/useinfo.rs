//! Dataflow "use-info" pre-pass for per-function bodies.
//!
//! Consumed by the result-write elision path in `lower.rs`. Given a
//! function body (a flat `[InstrResult]` slice,
//! top-level only), this pass answers three questions for each
//! `ValueId` the body produces:
//!
//! - How many times is it read? (`use_counts`)
//! - At what body index does its first consumer live? (`first_use_pos`,
//!   and equivalently `next_user` for a linear body).
//! - What kind of instruction is the first consumer? (`user_kind`,
//!   a short string tag).
//!
//! The elision decision in `lower_ptr_binop_elision_or_spill` keys on
//! `use_counts[v] == 1 && user_kind[v] in ELISION_FRIENDLY_SET` to
//! decide whether to keep a producer's result in SSA registers
//! (`LaneRepr::PtrChunksF64`) or spill it to a fresh stack slot.
//!
//! ## Nested regions
//!
//! Instructions that embed sub-bodies (`While`, `Case`, `Map`,
//! `Sort`, `ReduceWindow`, `SelectAndScatter`) are treated as a
//! single top-level use of every operand that flows into them —
//! we do NOT recurse into sub-bodies. This is the conservative
//! choice: if a top-level value is consumed inside a nested loop,
//! it must already be in memory form (pointer-ABI) when the loop
//! starts, so it's never eligible for elision. Tagging the use
//! with the outer instruction kind (e.g. `"while"`) gives
//! `user_kind[v]` a tag outside `ELISION_FRIENDLY_SET`, which is
//! exactly the correct decision.

use std::collections::HashMap;

use crate::const_fold::operand_ids;
use crate::ir::{InstrResult, Instruction, ValueId};

#[derive(Debug, Default, Clone)]
pub struct UseInfo {
    /// Total number of reads (across the body) of each ValueId that
    /// appears as an operand anywhere.
    pub use_counts: HashMap<ValueId, u32>,
    /// Body index of the first instruction that reads a given
    /// ValueId. For values with zero uses this entry is absent.
    pub first_use_pos: HashMap<ValueId, usize>,
    /// Alias for `first_use_pos` maintained for API clarity: the
    /// "next user" of a def is the first instruction after the def
    /// that reads it, which for a linear (SSA) body is the same
    /// position as `first_use_pos`. Kept as a separate map so callers
    /// that want "first user of any value, regardless of where it's
    /// defined" can use `first_use_pos` and callers explicitly
    /// consuming a def's next user can use `next_user` — the intent
    /// is the same but the naming makes the intent clear at callsites.
    pub next_user: HashMap<ValueId, usize>,
    /// Instruction-kind tag at `next_user[v]`, used by the elision
    /// check to classify whether the consumer can accept a PtrChunksF64
    /// input. See `ELISION_FRIENDLY_SET` in `lower.rs`.
    pub user_kind: HashMap<ValueId, &'static str>,
    /// Body index of the LAST instruction reading this ValueId. Used
    /// by `slot_pool::SlotPool` to decide when a result slot can be
    /// returned to the per-function free-list — after
    /// `last_use_pos[v]` completes, no later top-level instruction
    /// reads `v`, so the slot is safe to reuse for an allocation of
    /// the same (bytes, align).
    pub last_use_pos: HashMap<ValueId, usize>,
}

/// Build a `UseInfo` for a single function body.
///
/// Only top-level instructions are walked; when an instruction has
/// sub-bodies (`While`, `Case`, etc.) the operands that flow into
/// those sub-bodies are counted as a single use at the outer
/// instruction's position, tagged with the outer kind.
pub fn build_use_info(body: &[InstrResult]) -> UseInfo {
    let mut info = UseInfo::default();
    for (pos, ir) in body.iter().enumerate() {
        let kind = instr_kind(&ir.instr);
        for operand in operand_ids(&ir.instr) {
            *info.use_counts.entry(operand).or_insert(0) += 1;
            info.first_use_pos.entry(operand).or_insert(pos);
            info.next_user.entry(operand).or_insert(pos);
            info.user_kind.entry(operand).or_insert(kind);
            // Last-use position: unconditional insert so the latest
            // (max) body position wins. Over a linear walk this
            // naturally tracks the final reader of each vid.
            info.last_use_pos.insert(operand, pos);
        }
    }
    info
}

/// Short static tag describing an instruction's top-level kind.
/// Used by the elision decision to classify consumers.
pub(crate) fn instr_kind(instr: &Instruction) -> &'static str {
    match instr {
        Instruction::Constant { .. } => "constant",
        Instruction::Iota { .. } => "iota",
        Instruction::Add { .. } => "add",
        Instruction::Subtract { .. } => "subtract",
        Instruction::Multiply { .. } => "multiply",
        Instruction::Divide { .. } => "divide",
        Instruction::Maximum { .. } => "maximum",
        Instruction::Minimum { .. } => "minimum",
        Instruction::Atan2 { .. } => "atan2",
        Instruction::Remainder { .. } => "remainder",
        Instruction::Power { .. } => "power",
        Instruction::Xor { .. } => "xor",
        Instruction::Or { .. } => "or",
        Instruction::And { .. } => "and",
        Instruction::ShiftLeft { .. } => "shift_left",
        Instruction::ShiftRightLogical { .. } => "shift_right_logical",
        Instruction::ShiftRightArithmetic { .. } => "shift_right_arithmetic",
        Instruction::Compare { .. } => "compare",
        Instruction::Negate { .. } => "negate",
        Instruction::Sqrt { .. } => "sqrt",
        Instruction::Rsqrt { .. } => "rsqrt",
        Instruction::Abs { .. } => "abs",
        Instruction::Sign { .. } => "sign",
        Instruction::Sine { .. } => "sine",
        Instruction::Cosine { .. } => "cosine",
        Instruction::Tan { .. } => "tan",
        Instruction::Tanh { .. } => "tanh",
        Instruction::Sinh { .. } => "sinh",
        Instruction::Cosh { .. } => "cosh",
        Instruction::Asin { .. } => "asin",
        Instruction::Acos { .. } => "acos",
        Instruction::Atan { .. } => "atan",
        Instruction::Exponential { .. } => "exp",
        Instruction::Log { .. } => "log",
        Instruction::Log1p { .. } => "log1p",
        Instruction::Expm1 { .. } => "expm1",
        Instruction::Cbrt { .. } => "cbrt",
        Instruction::Erfc { .. } => "erfc",
        Instruction::ErfInv { .. } => "erf_inv",
        Instruction::IsFinite { .. } => "is_finite",
        Instruction::Not { .. } => "not",
        Instruction::Floor { .. } => "floor",
        Instruction::Ceil { .. } => "ceil",
        Instruction::RoundNearestEven { .. } => "nearest",
        Instruction::Reshape { .. } => "reshape",
        Instruction::Convert { .. } => "convert",
        Instruction::BitcastConvert { .. } => "bitcast_convert",
        Instruction::BroadcastInDim { .. } => "broadcast_in_dim",
        Instruction::Slice { .. } => "slice",
        Instruction::Transpose { .. } => "transpose",
        Instruction::Reverse { .. } => "reverse",
        Instruction::Select { .. } => "select",
        Instruction::Clamp { .. } => "clamp",
        Instruction::DotGeneral { .. } => "dot_general",
        Instruction::Reduce { .. } => "reduce",
        Instruction::ReduceArgminmax { .. } => "reduce_argminmax",
        Instruction::Concatenate { .. } => "concatenate",
        Instruction::Call { .. } => "call",
        Instruction::While { .. } => "while",
        Instruction::Case { .. } => "case",
        Instruction::Return { .. } => "return",
        Instruction::Gather { .. } => "gather",
        Instruction::DynamicSlice { .. } => "dynamic_slice",
        Instruction::DynamicUpdateSlice { .. } => "dynamic_update_slice",
        Instruction::Pad { .. } => "pad",
        Instruction::Scatter { .. } => "scatter",
        Instruction::CustomCall { .. } => "custom_call",
        Instruction::Sort { .. } => "sort",
        Instruction::Map { .. } => "map",
        Instruction::ReduceWindow { .. } => "reduce_window",
        Instruction::SelectAndScatter { .. } => "select_and_scatter",
        Instruction::Convolution { .. } => "convolution",
        Instruction::CholeskyOp { .. } => "cholesky",
        Instruction::TriangularSolve { .. } => "triangular_solve",
        Instruction::Fft { .. } => "fft",
        Instruction::BatchNormInference { .. } => "batch_norm_inference",
        Instruction::RealDynamicSlice { .. } => "real_dynamic_slice",
        Instruction::Rng { .. } => "rng",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ConstantValue, ElementType, InstrResult, Instruction, ScalarValue, TensorType, ValueId,
    };

    fn scalar_ty() -> TensorType {
        TensorType {
            shape: vec![1],
            element_type: ElementType::F64,
        }
    }

    fn mk_const(id: u32) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(id), scalar_ty())],
            instr: Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(0.0)),
            },
        }
    }

    fn mk_add(id: u32, lhs: u32, rhs: u32) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(id), scalar_ty())],
            instr: Instruction::Add {
                lhs: ValueId(lhs),
                rhs: ValueId(rhs),
            },
        }
    }

    fn mk_mul(id: u32, lhs: u32, rhs: u32) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(id), scalar_ty())],
            instr: Instruction::Multiply {
                lhs: ValueId(lhs),
                rhs: ValueId(rhs),
            },
        }
    }

    fn mk_return(ids: &[u32]) -> InstrResult {
        InstrResult {
            values: vec![],
            instr: Instruction::Return {
                operands: ids.iter().copied().map(ValueId).collect(),
            },
        }
    }

    #[test]
    fn single_use_def_followed_by_consumer() {
        // %0 = constant; %1 = constant; %2 = add(%0, %1); return(%2)
        let body = vec![mk_const(0), mk_const(1), mk_add(2, 0, 1), mk_return(&[2])];
        let info = build_use_info(&body);
        assert_eq!(info.use_counts[&ValueId(0)], 1);
        assert_eq!(info.use_counts[&ValueId(1)], 1);
        assert_eq!(info.use_counts[&ValueId(2)], 1);
        assert_eq!(info.first_use_pos[&ValueId(0)], 2);
        assert_eq!(info.first_use_pos[&ValueId(2)], 3);
        assert_eq!(info.next_user[&ValueId(0)], 2);
        assert_eq!(info.user_kind[&ValueId(0)], "add");
        assert_eq!(info.user_kind[&ValueId(2)], "return");
    }

    #[test]
    fn multi_use_counts() {
        // %0 = constant; %1 = add(%0, %0); %2 = mul(%0, %1); return(%2)
        let body = vec![
            mk_const(0),
            mk_add(1, 0, 0),
            mk_mul(2, 0, 1),
            mk_return(&[2]),
        ];
        let info = build_use_info(&body);
        // %0 appears in add(lhs,rhs) twice + mul(lhs) once = 3 uses.
        assert_eq!(info.use_counts[&ValueId(0)], 3);
        assert_eq!(info.use_counts[&ValueId(1)], 1);
        assert_eq!(info.first_use_pos[&ValueId(0)], 1);
        assert_eq!(info.user_kind[&ValueId(0)], "add");
    }

    #[test]
    fn last_use_pos_tracks_final_reader() {
        // %0 const; %1 const; %2 = add(%0, %1); %3 = mul(%2, %0);
        // return(%3). %0 is read at positions 2 and 3; last_use == 3.
        // %2 read only at position 3; last == first == 3.
        // %1 read only at position 2; last == first == 2.
        let body = vec![
            mk_const(0),
            mk_const(1),
            mk_add(2, 0, 1),
            mk_mul(3, 2, 0),
            mk_return(&[3]),
        ];
        let info = build_use_info(&body);
        assert_eq!(info.last_use_pos[&ValueId(0)], 3);
        assert_eq!(info.last_use_pos[&ValueId(1)], 2);
        assert_eq!(info.last_use_pos[&ValueId(2)], 3);
        assert_eq!(info.last_use_pos[&ValueId(3)], 4);
        // first_use_pos stays at the FIRST read.
        assert_eq!(info.first_use_pos[&ValueId(0)], 2);
        assert_eq!(info.first_use_pos[&ValueId(1)], 2);
    }

    #[test]
    fn no_use_def_is_absent() {
        // %0 = constant; %1 = constant; return(%0). %1 never read.
        let body = vec![mk_const(0), mk_const(1), mk_return(&[0])];
        let info = build_use_info(&body);
        assert_eq!(info.use_counts.get(&ValueId(1)).copied().unwrap_or(0), 0);
        assert!(!info.first_use_pos.contains_key(&ValueId(1)));
    }

    #[test]
    fn adjacent_single_use_add_to_mul_chain() {
        // %0 = constant; %1 = constant; %2 = add(%0,%1);
        // %3 = mul(%2, %0); return(%3).
        // %2 has use_counts=1 and user_kind="multiply" — elision-friendly.
        let body = vec![
            mk_const(0),
            mk_const(1),
            mk_add(2, 0, 1),
            mk_mul(3, 2, 0),
            mk_return(&[3]),
        ];
        let info = build_use_info(&body);
        assert_eq!(info.use_counts[&ValueId(2)], 1);
        assert_eq!(info.user_kind[&ValueId(2)], "multiply");
        assert_eq!(info.next_user[&ValueId(2)], 3);
        assert_eq!(info.first_use_pos[&ValueId(2)], 3);
    }

    #[test]
    fn call_user_kind_blocks_elision() {
        // %0 = constant; %1 = call(%0); return(%1). %0's user_kind is "call",
        // which is NOT in the elision-friendly set (see lower.rs).
        let body = vec![
            mk_const(0),
            InstrResult {
                values: vec![(ValueId(1), scalar_ty())],
                instr: Instruction::Call {
                    callee: "some_fn".to_string(),
                    args: vec![ValueId(0)],
                },
            },
            mk_return(&[1]),
        ];
        let info = build_use_info(&body);
        assert_eq!(info.user_kind[&ValueId(0)], "call");
    }

    #[test]
    fn while_user_kind_blocks_elision() {
        // %0 = constant; while(init=%0) { ... }; return(...).
        // %0's user_kind is "while" — must be a pointer on entry,
        // so never eligible for elision.
        let body = vec![
            mk_const(0),
            InstrResult {
                values: vec![(ValueId(1), scalar_ty())],
                instr: Instruction::While {
                    init_values: vec![ValueId(0)],
                    iter_arg_ids: vec![ValueId(10)],
                    cond_body: vec![],
                    loop_body: vec![],
                },
            },
            mk_return(&[1]),
        ];
        let info = build_use_info(&body);
        assert_eq!(info.user_kind[&ValueId(0)], "while");
    }
}
