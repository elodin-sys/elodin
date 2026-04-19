//! Compile-time constant folding for the cranelift-mlir IR.
//!
//! Runs between parsing and lowering. Rewrites `Instruction` variants whose
//! operands are known constants into `Instruction::Constant`, then drops
//! instructions that are no longer reachable from any live result.
//!
//! The lowering pipeline already handles every `ConstantValue` variant
//! (`DenseScalar`, `DenseArray`, `DenseSplat`) efficiently, so folding is a
//! pure IR rewrite with no change to `lower.rs`. Correctness is gated by
//! `tests/checkpoint_test.rs::verify_checkpoint` and the full regression
//! suite; see `ARCHITECTURE.md` for the rule taxonomy and measured impact.

use std::collections::{HashMap, HashSet};

use crate::ir::{
    CompareDirection, CompareType, ConstantValue, ElementType, FuncDef, InstrResult, Instruction,
    Module, ScalarValue, TensorType, ValueId,
};

/// Byte-size cap for synthesizing `DenseArray` constants in the fold pass.
/// Matches the lower.rs large-constant threshold so folded arrays that are
/// still "small" stay stack-backed and keep their placement decisions
/// consistent with the existing pipeline.
const FOLD_ARRAY_BYTE_CAP: usize = 1024;

/// Measurement + logging output of a fold pass.
#[derive(Debug, Clone, Default)]
pub struct FoldReport {
    /// Per-rule tally of how many instructions were folded or removed.
    pub counts_by_rule: HashMap<&'static str, usize>,
    /// Total IR instruction count (including nested bodies) before folding.
    pub total_instructions_before: usize,
    /// Total IR instruction count after folding.
    pub total_instructions_after: usize,
}

impl FoldReport {
    /// Increment the tally for a named rule.
    pub fn bump(&mut self, rule: &'static str) {
        *self.counts_by_rule.entry(rule).or_insert(0) += 1;
    }

    pub fn instructions_removed(&self) -> usize {
        self.total_instructions_before
            .saturating_sub(self.total_instructions_after)
    }

    pub fn percent_removed(&self) -> f64 {
        if self.total_instructions_before == 0 {
            return 0.0;
        }
        (self.instructions_removed() as f64) * 100.0 / (self.total_instructions_before as f64)
    }

    /// One-line summary, always emitted during `fold_module`.
    pub fn fmt_summary(&self) -> String {
        format!(
            "fold: {} -> {} instr (-{}, -{:.1}%)",
            self.total_instructions_before,
            self.total_instructions_after,
            self.instructions_removed(),
            self.percent_removed(),
        )
    }

    /// Sorted "rule=count, ..." histogram. Emitted only when debug
    /// mode is active.
    pub fn fmt_histogram(&self) -> String {
        let mut entries: Vec<(&&str, &usize)> = self.counts_by_rule.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut s = String::new();
        for (rule, count) in entries {
            if !s.is_empty() {
                s.push_str(", ");
            }
            s.push_str(&format!("{rule}={count}"));
        }
        if s.is_empty() {
            "(empty)".to_string()
        } else {
            s
        }
    }
}

/// Recursively count every `Instruction` in a module, descending into the
/// embedded bodies so the metric reflects the full workload the lowering
/// pipeline must process.
pub fn count_instructions(module: &Module) -> usize {
    module.functions.iter().map(count_func_instructions).sum()
}

fn count_func_instructions(func: &FuncDef) -> usize {
    count_body_instructions(&func.body)
}

pub(crate) fn count_body_instructions(body: &[InstrResult]) -> usize {
    let mut total = body.len();
    for ir in body {
        total += count_nested(&ir.instr);
    }
    total
}

fn count_nested(instr: &Instruction) -> usize {
    match instr {
        Instruction::While {
            cond_body,
            loop_body,
            ..
        } => count_body_instructions(cond_body) + count_body_instructions(loop_body),
        Instruction::Case { branches, .. } => {
            branches.iter().map(|b| count_body_instructions(b)).sum()
        }
        Instruction::Map { body, .. } => count_body_instructions(body),
        Instruction::ReduceWindow { body, .. } => count_body_instructions(body),
        Instruction::SelectAndScatter {
            select_body,
            scatter_body,
            ..
        } => count_body_instructions(select_body) + count_body_instructions(scatter_body),
        Instruction::Sort { comparator, .. } => count_body_instructions(comparator),
        _ => 0,
    }
}

/// Dry-run measurement. Leaves the module unchanged.
pub fn measure(module: &Module) -> FoldReport {
    let total = count_instructions(module);
    FoldReport {
        counts_by_rule: HashMap::new(),
        total_instructions_before: total,
        total_instructions_after: total,
    }
}

/// Fold the module in-place. Returns a `FoldReport` with measurement and
/// rule-level tallies. Always emits a one-line summary; emits a per-rule
/// histogram when debug mode is active (`ELODIN_CRANELIFT_DEBUG_DIR`).
pub fn fold_module(module: &mut Module) -> FoldReport {
    let mut report = FoldReport {
        total_instructions_before: count_instructions(module),
        ..FoldReport::default()
    };

    for func in &mut module.functions {
        fold_func(func, &mut report);
    }

    report.total_instructions_after = count_instructions(module);
    log_report(&report);
    report
}

/// Maximum number of fold+DCE iterations per function. Caps pathological
/// cascades; in practice, 2-3 iterations reach a fixed point on every MLIR
/// we have measured.
const MAX_FOLD_ITERATIONS: usize = 8;

fn fold_func(func: &mut FuncDef, report: &mut FoldReport) {
    // Seed the type map with function parameters; every InstrResult in the
    // body extends it as we walk.
    let mut type_of: HashMap<ValueId, TensorType> = HashMap::new();
    for (vid, ty) in &func.params {
        type_of.insert(*vid, ty.clone());
    }

    for _ in 0..MAX_FOLD_ITERATIONS {
        let before_len = count_body_instructions(&func.body);
        let mut local = FoldReport::default();
        let mut env: HashMap<ValueId, ConstantValue> = HashMap::new();
        let mut aliases: HashMap<ValueId, ValueId> = HashMap::new();

        fold_body(
            &mut func.body,
            &mut env,
            &mut aliases,
            &mut type_of,
            &mut local,
        );

        if !aliases.is_empty() {
            rewrite_aliases_body(&mut func.body, &aliases);
        }

        dce_body(&mut func.body, &mut local);

        for (k, v) in &local.counts_by_rule {
            *report.counts_by_rule.entry(k).or_insert(0) += v;
        }

        let changed =
            !local.counts_by_rule.is_empty() || count_body_instructions(&func.body) != before_len;
        if !changed {
            break;
        }
    }
}

/// Walk a body in program order, folding each instruction where possible and
/// recording result `ValueId`s that are known constants so downstream ops can
/// fold against them.
fn fold_body(
    body: &mut [InstrResult],
    env: &mut HashMap<ValueId, ConstantValue>,
    aliases: &mut HashMap<ValueId, ValueId>,
    type_of: &mut HashMap<ValueId, TensorType>,
    report: &mut FoldReport,
) {
    for ir in body.iter_mut() {
        // Record result types so later array folds / identities can reference
        // the source shape without having to re-derive it.
        for (vid, ty) in &ir.values {
            type_of.insert(*vid, ty.clone());
        }

        // Fold nested bodies first; they run under a snapshot of the current env.
        fold_nested_bodies(&mut ir.instr, env, aliases, type_of, report);

        // Try to fold this instruction using the current env.
        if let Some((folded_value, rule)) = try_fold(&ir.instr, env, type_of, &ir.values) {
            report.bump(rule);
            if let Some((result_vid, _)) = ir.values.first() {
                env.insert(*result_vid, folded_value.clone());
            }
            ir.instr = Instruction::Constant {
                value: folded_value,
            };
            continue;
        }

        // Identity detection: record an alias so the post-pass
        // substitutes downstream uses with the upstream value
        // (e.g. `x * 1` → `x`).
        if let Some((alias_target, rule)) = try_identity(&ir.instr, env, type_of, &ir.values) {
            report.bump(rule);
            if let Some((result_vid, _)) = ir.values.first() {
                aliases.insert(*result_vid, alias_target);
                if let Some(cv) = env.get(&alias_target).cloned() {
                    env.insert(*result_vid, cv);
                }
            }
            continue;
        }

        if let Instruction::Constant { value } = &ir.instr
            && let Some((result_vid, _)) = ir.values.first()
        {
            env.insert(*result_vid, value.clone());
        }
    }
}

/// Recurse folding into every embedded body. Outer-scope values stay visible
/// inside nested bodies (SSA), but new values created inside a body should
/// not leak to siblings — each nested body gets its own cloned env.
fn fold_nested_bodies(
    instr: &mut Instruction,
    env: &HashMap<ValueId, ConstantValue>,
    aliases: &HashMap<ValueId, ValueId>,
    type_of: &HashMap<ValueId, TensorType>,
    report: &mut FoldReport,
) {
    match instr {
        Instruction::While {
            cond_body,
            loop_body,
            ..
        } => {
            fold_body(
                cond_body,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
            fold_body(
                loop_body,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
        }
        Instruction::Case { branches, .. } => {
            for branch in branches.iter_mut() {
                fold_body(
                    branch,
                    &mut env.clone(),
                    &mut aliases.clone(),
                    &mut type_of.clone(),
                    report,
                );
            }
        }
        Instruction::Map { body, .. } | Instruction::ReduceWindow { body, .. } => {
            fold_body(
                body,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
        }
        Instruction::SelectAndScatter {
            select_body,
            scatter_body,
            ..
        } => {
            fold_body(
                select_body,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
            fold_body(
                scatter_body,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
        }
        Instruction::Sort { comparator, .. } => {
            fold_body(
                comparator,
                &mut env.clone(),
                &mut aliases.clone(),
                &mut type_of.clone(),
                report,
            );
        }
        _ => {}
    }
}

/// Fold rules. Returns the new `ConstantValue` and the rule name if
/// the instruction is foldable given the current constant env.
fn try_fold(
    instr: &Instruction,
    env: &HashMap<ValueId, ConstantValue>,
    type_of: &HashMap<ValueId, TensorType>,
    results: &[(ValueId, TensorType)],
) -> Option<(ConstantValue, &'static str)> {
    let result_ty = &results.first()?.1;

    match instr {
        Instruction::BroadcastInDim { operand, .. } => {
            let src = env.get(operand)?;
            match src {
                ConstantValue::DenseScalar(v) | ConstantValue::DenseSplat(v, _) => Some((
                    ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                    "broadcast_scalar",
                )),
                ConstantValue::DenseArray(_) => None,
            }
        }

        Instruction::Convert { operand } => {
            let src = env.get(operand)?;
            match src {
                ConstantValue::DenseScalar(v) => {
                    let coerced = coerce_scalar(v, result_ty.element_type)?;
                    Some((ConstantValue::DenseScalar(coerced), "convert_scalar"))
                }
                ConstantValue::DenseSplat(v, _) => {
                    let coerced = coerce_scalar(v, result_ty.element_type)?;
                    Some((
                        ConstantValue::DenseSplat(coerced, result_ty.clone()),
                        "convert_splat",
                    ))
                }
                ConstantValue::DenseArray(_) => None,
            }
        }

        Instruction::Reshape { operand } => {
            let src = env.get(operand)?;
            match src {
                ConstantValue::DenseScalar(v) => {
                    if result_ty.is_scalar() {
                        Some((ConstantValue::DenseScalar(v.clone()), "reshape_scalar"))
                    } else {
                        Some((
                            ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                            "reshape_scalar_to_splat",
                        ))
                    }
                }
                ConstantValue::DenseSplat(v, _) => Some((
                    ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                    "reshape_splat",
                )),
                ConstantValue::DenseArray(arr) => {
                    if arr.len() == result_ty.num_elements() {
                        Some((
                            ConstantValue::DenseArray(arr.clone()),
                            "reshape_dense_array",
                        ))
                    } else {
                        None
                    }
                }
            }
        }

        Instruction::Iota { dimension } => {
            if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                return None;
            }
            let values = iota_values(result_ty, *dimension);
            Some((ConstantValue::DenseArray(values), "iota_small"))
        }

        // ----------- Scalar arithmetic and compare -----------

        // Unary float ops.
        Instruction::Negate { operand } => unary_fold(env, operand, result_ty, "negate", |s| {
            finite_unary_f64(s, |x| -x).or_else(|| integer_unary(s, |i| i.wrapping_neg()))
        }),
        Instruction::Abs { operand } => unary_fold(env, operand, result_ty, "abs", |s| {
            finite_unary_f64(s, |x| x.abs()).or_else(|| integer_unary(s, |i| i.wrapping_abs()))
        }),
        Instruction::Sqrt { operand } => unary_fold(env, operand, result_ty, "sqrt", |s| {
            finite_unary_f64(s, f64::sqrt)
        }),
        Instruction::Rsqrt { operand } => unary_fold(env, operand, result_ty, "rsqrt", |s| {
            finite_unary_f64(s, |x| 1.0 / x.sqrt())
        }),
        Instruction::Floor { operand } => unary_fold(env, operand, result_ty, "floor", |s| {
            finite_unary_f64(s, f64::floor)
        }),
        Instruction::Ceil { operand } => unary_fold(env, operand, result_ty, "ceil", |s| {
            finite_unary_f64(s, f64::ceil)
        }),
        Instruction::RoundNearestEven { operand } => {
            unary_fold(env, operand, result_ty, "round_nearest_even", |s| {
                finite_unary_f64(s, round_nearest_even_f64)
            })
        }
        Instruction::Sign { operand } => unary_fold(env, operand, result_ty, "sign", |s| {
            finite_unary_f64(s, f64_sign)
        }),
        Instruction::Sine { operand } => unary_fold(env, operand, result_ty, "sine", |s| {
            finite_unary_f64(s, f64::sin)
        }),
        Instruction::Cosine { operand } => unary_fold(env, operand, result_ty, "cosine", |s| {
            finite_unary_f64(s, f64::cos)
        }),
        Instruction::Tan { operand } => unary_fold(env, operand, result_ty, "tan", |s| {
            finite_unary_f64(s, f64::tan)
        }),
        Instruction::Tanh { operand } => unary_fold(env, operand, result_ty, "tanh", |s| {
            finite_unary_f64(s, f64::tanh)
        }),
        Instruction::Sinh { operand } => unary_fold(env, operand, result_ty, "sinh", |s| {
            finite_unary_f64(s, f64::sinh)
        }),
        Instruction::Cosh { operand } => unary_fold(env, operand, result_ty, "cosh", |s| {
            finite_unary_f64(s, f64::cosh)
        }),
        Instruction::Asin { operand } => unary_fold(env, operand, result_ty, "asin", |s| {
            finite_unary_f64(s, f64::asin)
        }),
        Instruction::Acos { operand } => unary_fold(env, operand, result_ty, "acos", |s| {
            finite_unary_f64(s, f64::acos)
        }),
        Instruction::Atan { operand } => unary_fold(env, operand, result_ty, "atan", |s| {
            finite_unary_f64(s, f64::atan)
        }),
        Instruction::Exponential { operand } => unary_fold(env, operand, result_ty, "exp", |s| {
            finite_unary_f64(s, f64::exp)
        }),
        Instruction::Log { operand } => unary_fold(env, operand, result_ty, "log", |s| {
            finite_unary_f64(s, f64::ln)
        }),
        Instruction::Log1p { operand } => unary_fold(env, operand, result_ty, "log1p", |s| {
            finite_unary_f64(s, f64::ln_1p)
        }),
        Instruction::Expm1 { operand } => unary_fold(env, operand, result_ty, "expm1", |s| {
            finite_unary_f64(s, f64::exp_m1)
        }),
        Instruction::Cbrt { operand } => unary_fold(env, operand, result_ty, "cbrt", |s| {
            finite_unary_f64(s, f64::cbrt)
        }),
        Instruction::IsFinite { operand } => {
            // Returns I1; always safe to evaluate from a known scalar.
            unary_fold(env, operand, result_ty, "is_finite", |s| {
                Some(ScalarValue::I1(is_finite_scalar(s)))
            })
        }
        Instruction::Not { operand } => unary_fold(env, operand, result_ty, "not", |s| match s {
            ScalarValue::I1(b) => Some(ScalarValue::I1(!*b)),
            ScalarValue::I64(x) => Some(ScalarValue::I64(!*x)),
            ScalarValue::I32(x) => Some(ScalarValue::I32(!*x)),
            ScalarValue::UI64(x) => Some(ScalarValue::UI64(!*x)),
            ScalarValue::UI32(x) => Some(ScalarValue::UI32(!*x)),
            _ => None,
        }),

        // Binary arithmetic.
        Instruction::Add { lhs, rhs } => binary_fold(env, lhs, rhs, result_ty, "add", |a, b| {
            finite_binary_f64(a, b, |x, y| x + y)
                .or_else(|| integer_binary(a, b, |x, y| x.wrapping_add(y)))
        }),
        Instruction::Subtract { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "subtract", |a, b| {
                finite_binary_f64(a, b, |x, y| x - y)
                    .or_else(|| integer_binary(a, b, |x, y| x.wrapping_sub(y)))
            })
        }
        Instruction::Multiply { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "multiply", |a, b| {
                finite_binary_f64(a, b, |x, y| x * y)
                    .or_else(|| integer_binary(a, b, |x, y| x.wrapping_mul(y)))
            })
        }
        Instruction::Divide { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "divide", |a, b| {
                if is_integer_scalar(b) && scalar_as_i64(b).unwrap_or(0) == 0 {
                    return None;
                }
                finite_binary_f64(a, b, |x, y| x / y)
                    .or_else(|| integer_binary(a, b, |x, y| x.wrapping_div(y)))
            })
        }
        Instruction::Remainder { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "remainder", |a, b| {
                if is_integer_scalar(b) && scalar_as_i64(b).unwrap_or(0) == 0 {
                    return None;
                }
                finite_binary_f64(a, b, |x, y| x % y)
                    .or_else(|| integer_binary(a, b, |x, y| x.wrapping_rem(y)))
            })
        }
        Instruction::Maximum { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "maximum", |a, b| {
                finite_binary_f64(a, b, f64::max).or_else(|| integer_binary(a, b, i64::max))
            })
        }
        Instruction::Minimum { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "minimum", |a, b| {
                finite_binary_f64(a, b, f64::min).or_else(|| integer_binary(a, b, i64::min))
            })
        }
        Instruction::Atan2 { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "atan2", |a, b| {
                finite_binary_f64(a, b, f64::atan2)
            })
        }
        Instruction::Power { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "power", |a, b| {
                finite_binary_f64(a, b, f64::powf)
            })
        }

        // Bitwise.
        Instruction::Xor { lhs, rhs } => binary_fold(env, lhs, rhs, result_ty, "xor", |a, b| {
            bitwise_binary(a, b, |x, y| x ^ y)
        }),
        Instruction::Or { lhs, rhs } => binary_fold(env, lhs, rhs, result_ty, "or", |a, b| {
            bitwise_binary(a, b, |x, y| x | y)
        }),
        Instruction::And { lhs, rhs } => binary_fold(env, lhs, rhs, result_ty, "and", |a, b| {
            bitwise_binary(a, b, |x, y| x & y)
        }),
        Instruction::ShiftLeft { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "shift_left", |a, b| {
                shift_binary(a, b, |x, n| x.wrapping_shl(n))
            })
        }
        Instruction::ShiftRightLogical { lhs, rhs } => {
            binary_fold(env, lhs, rhs, result_ty, "shift_right_logical", |a, b| {
                shift_binary_unsigned(a, b, |x, n| x.wrapping_shr(n))
            })
        }
        Instruction::ShiftRightArithmetic { lhs, rhs } => binary_fold(
            env,
            lhs,
            rhs,
            result_ty,
            "shift_right_arithmetic",
            |a, b| shift_binary_signed(a, b, |x, n| x.wrapping_shr(n)),
        ),

        // Compare always produces I1.
        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => {
            let l = env.get(lhs)?;
            let r = env.get(rhs)?;
            let (a, b) = match (l, r) {
                (ConstantValue::DenseScalar(a), ConstantValue::DenseScalar(b)) => {
                    (a.clone(), b.clone())
                }
                (ConstantValue::DenseSplat(a, ta), ConstantValue::DenseSplat(b, tb))
                    if ta.shape == tb.shape =>
                {
                    (a.clone(), b.clone())
                }
                _ => return None,
            };
            let result_bool = eval_compare(&a, &b, *direction, *compare_type)?;
            let value = if result_ty.is_scalar() {
                ConstantValue::DenseScalar(ScalarValue::I1(result_bool))
            } else {
                ConstantValue::DenseSplat(ScalarValue::I1(result_bool), result_ty.clone())
            };
            Some((value, "compare"))
        }

        // Select picks an arm when the condition is a known boolean constant
        // and both arms are themselves constants.
        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let cond_v = env.get(cond)?;
            let cond_bool = match cond_v {
                ConstantValue::DenseScalar(ScalarValue::I1(b)) => *b,
                ConstantValue::DenseSplat(ScalarValue::I1(b), _) => *b,
                _ => return None,
            };
            let chosen = if cond_bool { on_true } else { on_false };
            let chosen_const = env.get(chosen)?;
            Some((chosen_const.clone(), "select_constant"))
        }

        // ----------- Array folds (size-capped) -----------
        Instruction::Transpose {
            operand,
            permutation,
        } => {
            let src_cv = env.get(operand)?;
            match src_cv {
                ConstantValue::DenseSplat(v, _) => Some((
                    ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                    "transpose_splat",
                )),
                ConstantValue::DenseArray(arr) => {
                    let src_shape = &type_of.get(operand)?.shape;
                    if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                        return None;
                    }
                    let out = transpose_dense_array(arr, src_shape, permutation)?;
                    Some((ConstantValue::DenseArray(out), "transpose_dense_array"))
                }
                _ => None,
            }
        }

        Instruction::Reverse {
            operand,
            dimensions,
        } => {
            let src_cv = env.get(operand)?;
            match src_cv {
                ConstantValue::DenseSplat(v, _) => Some((
                    ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                    "reverse_splat",
                )),
                ConstantValue::DenseArray(arr) => {
                    let src_shape = &type_of.get(operand)?.shape;
                    if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                        return None;
                    }
                    let out = reverse_dense_array(arr, src_shape, dimensions)?;
                    Some((ConstantValue::DenseArray(out), "reverse_dense_array"))
                }
                _ => None,
            }
        }

        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => {
            let src_cv = env.get(operand)?;
            match src_cv {
                ConstantValue::DenseSplat(v, _) => Some((
                    ConstantValue::DenseSplat(v.clone(), result_ty.clone()),
                    "slice_splat",
                )),
                ConstantValue::DenseArray(arr) => {
                    let src_shape = &type_of.get(operand)?.shape;
                    if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                        return None;
                    }
                    let out = slice_dense_array(arr, src_shape, start_indices, limit_indices)?;
                    Some((ConstantValue::DenseArray(out), "slice_dense_array"))
                }
                _ => None,
            }
        }

        Instruction::Concatenate {
            operands,
            dimension,
        } => {
            // All operands must be DenseArray (or convertible to arrays of
            // the same scalar type) and their source shapes must be known.
            if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                return None;
            }
            let mut sources: Vec<(&[ScalarValue], &[i64])> = Vec::with_capacity(operands.len());
            for vid in operands {
                let cv = env.get(vid)?;
                let arr = match cv {
                    ConstantValue::DenseArray(a) => a.as_slice(),
                    _ => return None,
                };
                let shape = type_of.get(vid)?.shape.as_slice();
                sources.push((arr, shape));
            }
            let out = concat_dense_arrays(&sources, *dimension)?;
            Some((ConstantValue::DenseArray(out), "concat_dense_arrays"))
        }

        Instruction::Pad {
            operand,
            padding_value,
            low,
            high,
            interior,
        } => {
            let src_cv = env.get(operand)?;
            let src_arr = match src_cv {
                ConstantValue::DenseArray(arr) => arr,
                _ => return None,
            };
            let pad_cv = env.get(padding_value)?;
            let pad_scalar = match pad_cv {
                ConstantValue::DenseScalar(s) | ConstantValue::DenseSplat(s, _) => s,
                _ => return None,
            };
            let src_shape = &type_of.get(operand)?.shape;
            if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                return None;
            }
            let out = pad_dense_array(src_arr, src_shape, pad_scalar, low, high, interior)?;
            Some((ConstantValue::DenseArray(out), "pad_dense_array"))
        }

        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => {
            let src_cv = env.get(operand)?;
            let src_arr = match src_cv {
                ConstantValue::DenseArray(arr) => arr,
                _ => return None,
            };
            let src_shape = &type_of.get(operand)?.shape;
            if result_ty.byte_size() > FOLD_ARRAY_BYTE_CAP {
                return None;
            }
            // Every start index must itself be a known constant scalar.
            let mut starts = Vec::with_capacity(start_indices.len());
            for sid in start_indices {
                let s = env.get(sid)?;
                let scalar = match s {
                    ConstantValue::DenseScalar(v) | ConstantValue::DenseSplat(v, _) => v,
                    _ => return None,
                };
                starts.push(scalar_as_i64(scalar)?);
            }
            // Clamp starts to stay within bounds, matching StableHLO semantics.
            let mut limits = Vec::with_capacity(slice_sizes.len());
            let mut clamped_starts = Vec::with_capacity(starts.len());
            for (i, &sz) in slice_sizes.iter().enumerate() {
                let dim = src_shape.get(i).copied().unwrap_or(0);
                let lo = starts[i].clamp(0, (dim - sz).max(0));
                clamped_starts.push(lo);
                limits.push(lo + sz);
            }
            let out = slice_dense_array(src_arr, src_shape, &clamped_starts, &limits)?;
            Some((ConstantValue::DenseArray(out), "dynamic_slice_dense_array"))
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scalar-op fold helpers
// ---------------------------------------------------------------------------

/// Run a unary scalar fold on a value in `env`, producing a new
/// `ConstantValue` of the result type. Handles `DenseScalar` and `DenseSplat`
/// operands; the eval closure takes the inner scalar and returns the new
/// inner scalar (or `None` to skip the fold).
fn unary_fold(
    env: &HashMap<ValueId, ConstantValue>,
    operand: &ValueId,
    result_ty: &TensorType,
    rule: &'static str,
    eval: impl Fn(&ScalarValue) -> Option<ScalarValue>,
) -> Option<(ConstantValue, &'static str)> {
    let src = env.get(operand)?;
    let (src_scalar, splat_ty) = match src {
        ConstantValue::DenseScalar(s) => (s.clone(), None),
        ConstantValue::DenseSplat(s, t) => (s.clone(), Some(t.clone())),
        ConstantValue::DenseArray(_) => return None,
    };
    let out_scalar = eval(&src_scalar)?;
    // Coerce the eval output into the requested result type when sensible.
    let out_coerced = coerce_to_element_type(out_scalar, result_ty.element_type)?;
    let value = if splat_ty.is_some() && !result_ty.is_scalar() {
        ConstantValue::DenseSplat(out_coerced, result_ty.clone())
    } else if result_ty.is_scalar() {
        ConstantValue::DenseScalar(out_coerced)
    } else {
        ConstantValue::DenseSplat(out_coerced, result_ty.clone())
    };
    Some((value, rule))
}

/// Two-operand variant of [`unary_fold`]. Both operands must be constant,
/// and both must be either matching `DenseScalar`s or `DenseSplat`s with the
/// same shape. Broadcast across mismatched shapes is not attempted — in
/// StableHLO the lhs/rhs shapes match by construction.
fn binary_fold(
    env: &HashMap<ValueId, ConstantValue>,
    lhs: &ValueId,
    rhs: &ValueId,
    result_ty: &TensorType,
    rule: &'static str,
    eval: impl Fn(&ScalarValue, &ScalarValue) -> Option<ScalarValue>,
) -> Option<(ConstantValue, &'static str)> {
    let l = env.get(lhs)?;
    let r = env.get(rhs)?;
    let (ls, rs) = match (l, r) {
        (ConstantValue::DenseScalar(a), ConstantValue::DenseScalar(b)) => (a.clone(), b.clone()),
        (ConstantValue::DenseSplat(a, ta), ConstantValue::DenseSplat(b, tb))
            if ta.shape == tb.shape =>
        {
            (a.clone(), b.clone())
        }
        _ => return None,
    };
    let out_scalar = eval(&ls, &rs)?;
    let out_coerced = coerce_to_element_type(out_scalar, result_ty.element_type)?;
    let value = if result_ty.is_scalar() {
        ConstantValue::DenseScalar(out_coerced)
    } else {
        ConstantValue::DenseSplat(out_coerced, result_ty.clone())
    };
    Some((value, rule))
}

/// Map a unary f64 op; returns None if either operand or result is non-finite.
fn finite_unary_f64(s: &ScalarValue, op: impl Fn(f64) -> f64) -> Option<ScalarValue> {
    if !is_float_scalar(s) {
        return None;
    }
    let x = scalar_as_f64(s);
    if !x.is_finite() {
        return None;
    }
    let y = op(x);
    if !y.is_finite() {
        return None;
    }
    Some(match s {
        ScalarValue::F64(_) => ScalarValue::F64(y),
        ScalarValue::F32(_) => ScalarValue::F32(y as f32),
        _ => unreachable!(),
    })
}

/// Map a binary f64 op; returns None if any operand or the result is non-finite.
fn finite_binary_f64(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(f64, f64) -> f64,
) -> Option<ScalarValue> {
    if !is_float_scalar(a) || !is_float_scalar(b) {
        return None;
    }
    let (x, y) = (scalar_as_f64(a), scalar_as_f64(b));
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let r = op(x, y);
    if !r.is_finite() {
        return None;
    }
    // Preserve the input element type. lhs and rhs share type in StableHLO.
    Some(match a {
        ScalarValue::F64(_) => ScalarValue::F64(r),
        ScalarValue::F32(_) => ScalarValue::F32(r as f32),
        _ => unreachable!(),
    })
}

fn integer_unary(s: &ScalarValue, op: impl Fn(i64) -> i64) -> Option<ScalarValue> {
    match s {
        ScalarValue::I64(x) => Some(ScalarValue::I64(op(*x))),
        ScalarValue::I32(x) => Some(ScalarValue::I32(op(*x as i64) as i32)),
        _ => None,
    }
}

fn integer_binary(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(i64, i64) -> i64,
) -> Option<ScalarValue> {
    match (a, b) {
        (ScalarValue::I64(x), ScalarValue::I64(y)) => Some(ScalarValue::I64(op(*x, *y))),
        (ScalarValue::I32(x), ScalarValue::I32(y)) => {
            Some(ScalarValue::I32(op(*x as i64, *y as i64) as i32))
        }
        _ => None,
    }
}

fn bitwise_binary(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(u64, u64) -> u64,
) -> Option<ScalarValue> {
    match (a, b) {
        (ScalarValue::I1(x), ScalarValue::I1(y)) => {
            Some(ScalarValue::I1(op(*x as u64, *y as u64) != 0))
        }
        (ScalarValue::I64(x), ScalarValue::I64(y)) => {
            Some(ScalarValue::I64(op(*x as u64, *y as u64) as i64))
        }
        (ScalarValue::I32(x), ScalarValue::I32(y)) => {
            Some(ScalarValue::I32(
                op(*x as u32 as u64, *y as u32 as u64) as u32 as i32,
            ))
        }
        (ScalarValue::UI64(x), ScalarValue::UI64(y)) => Some(ScalarValue::UI64(op(*x, *y))),
        (ScalarValue::UI32(x), ScalarValue::UI32(y)) => {
            Some(ScalarValue::UI32(op(*x as u64, *y as u64) as u32))
        }
        _ => None,
    }
}

fn shift_binary(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(u64, u32) -> u64,
) -> Option<ScalarValue> {
    let amount = scalar_as_u64(b)? as u32;
    match a {
        ScalarValue::I64(x) => Some(ScalarValue::I64(op(*x as u64, amount) as i64)),
        ScalarValue::I32(x) => Some(ScalarValue::I32(op(*x as u32 as u64, amount) as u32 as i32)),
        ScalarValue::UI64(x) => Some(ScalarValue::UI64(op(*x, amount))),
        ScalarValue::UI32(x) => Some(ScalarValue::UI32(op(*x as u64, amount) as u32)),
        _ => None,
    }
}

fn shift_binary_unsigned(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(u64, u32) -> u64,
) -> Option<ScalarValue> {
    shift_binary(a, b, op)
}

fn shift_binary_signed(
    a: &ScalarValue,
    b: &ScalarValue,
    op: impl Fn(i64, u32) -> i64,
) -> Option<ScalarValue> {
    let amount = scalar_as_u64(b)? as u32;
    match a {
        ScalarValue::I64(x) => Some(ScalarValue::I64(op(*x, amount))),
        ScalarValue::I32(x) => Some(ScalarValue::I32(op(*x as i64, amount) as i32)),
        _ => None,
    }
}

fn is_float_scalar(s: &ScalarValue) -> bool {
    matches!(s, ScalarValue::F64(_) | ScalarValue::F32(_))
}

fn is_integer_scalar(s: &ScalarValue) -> bool {
    matches!(
        s,
        ScalarValue::I64(_) | ScalarValue::I32(_) | ScalarValue::UI64(_) | ScalarValue::UI32(_)
    )
}

fn is_finite_scalar(s: &ScalarValue) -> bool {
    match s {
        ScalarValue::F64(x) => x.is_finite(),
        ScalarValue::F32(x) => x.is_finite(),
        _ => true,
    }
}

fn f64_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

fn round_nearest_even_f64(x: f64) -> f64 {
    // StableHLO `round_nearest_even` = banker's rounding.
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

fn eval_compare(
    a: &ScalarValue,
    b: &ScalarValue,
    direction: CompareDirection,
    compare_type: CompareType,
) -> Option<bool> {
    match compare_type {
        CompareType::Float | CompareType::TotalOrder => {
            if !is_float_scalar(a) || !is_float_scalar(b) {
                return None;
            }
            let (x, y) = (scalar_as_f64(a), scalar_as_f64(b));
            Some(match direction {
                CompareDirection::Eq => x == y,
                CompareDirection::Ne => x != y,
                CompareDirection::Lt => x < y,
                CompareDirection::Le => x <= y,
                CompareDirection::Gt => x > y,
                CompareDirection::Ge => x >= y,
            })
        }
        CompareType::Signed => {
            let (x, y) = (scalar_as_i64(a)?, scalar_as_i64(b)?);
            Some(match direction {
                CompareDirection::Eq => x == y,
                CompareDirection::Ne => x != y,
                CompareDirection::Lt => x < y,
                CompareDirection::Le => x <= y,
                CompareDirection::Gt => x > y,
                CompareDirection::Ge => x >= y,
            })
        }
        CompareType::Unsigned => {
            let (x, y) = (scalar_as_u64(a)?, scalar_as_u64(b)?);
            Some(match direction {
                CompareDirection::Eq => x == y,
                CompareDirection::Ne => x != y,
                CompareDirection::Lt => x < y,
                CompareDirection::Le => x <= y,
                CompareDirection::Gt => x > y,
                CompareDirection::Ge => x >= y,
            })
        }
    }
}

/// Coerce a freshly computed ScalarValue into the declared result element
/// type. Reuses `coerce_scalar` so we reject out-of-range conversions.
fn coerce_to_element_type(s: ScalarValue, to: ElementType) -> Option<ScalarValue> {
    // Fast path: scalar is already the right element type.
    let current = match s {
        ScalarValue::F64(_) => ElementType::F64,
        ScalarValue::F32(_) => ElementType::F32,
        ScalarValue::I64(_) => ElementType::I64,
        ScalarValue::I32(_) => ElementType::I32,
        ScalarValue::UI64(_) => ElementType::UI64,
        ScalarValue::UI32(_) => ElementType::UI32,
        ScalarValue::I1(_) => ElementType::I1,
    };
    if current == to {
        return Some(s);
    }
    coerce_scalar(&s, to)
}

/// Evaluate an element-wise coercion between two `ElementType`s. Mirrors the
/// behaviour of the runtime `convert_*` helpers for representable values.
fn coerce_scalar(v: &ScalarValue, to: ElementType) -> Option<ScalarValue> {
    let out = match to {
        ElementType::F64 => ScalarValue::F64(scalar_as_f64(v)),
        ElementType::F32 => ScalarValue::F32(scalar_as_f64(v) as f32),
        ElementType::I64 => ScalarValue::I64(scalar_as_i64(v)?),
        ElementType::I32 => ScalarValue::I32(scalar_as_i64(v)? as i32),
        ElementType::UI64 => ScalarValue::UI64(scalar_as_u64(v)?),
        ElementType::UI32 => ScalarValue::UI32(scalar_as_u64(v)? as u32),
        ElementType::I1 => ScalarValue::I1(scalar_as_bool(v)),
    };
    Some(out)
}

fn scalar_as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::F64(x) => *x,
        ScalarValue::F32(x) => *x as f64,
        ScalarValue::I64(x) => *x as f64,
        ScalarValue::I32(x) => *x as f64,
        ScalarValue::UI64(x) => *x as f64,
        ScalarValue::UI32(x) => *x as f64,
        ScalarValue::I1(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Convert a scalar value to i64 if the conversion is lossless or the
/// result is representable. Returns None when the value is non-finite or
/// out of range — for those edge cases we keep the instruction in place
/// and let the runtime handle it.
fn scalar_as_i64(v: &ScalarValue) -> Option<i64> {
    match v {
        ScalarValue::I64(x) => Some(*x),
        ScalarValue::I32(x) => Some(*x as i64),
        ScalarValue::UI64(x) => i64::try_from(*x).ok(),
        ScalarValue::UI32(x) => Some(*x as i64),
        ScalarValue::F64(x) => {
            if x.is_finite() && *x >= i64::MIN as f64 && *x <= i64::MAX as f64 {
                Some(x.trunc() as i64)
            } else {
                None
            }
        }
        ScalarValue::F32(x) => {
            if x.is_finite() && *x >= i64::MIN as f32 && *x <= i64::MAX as f32 {
                Some(x.trunc() as i64)
            } else {
                None
            }
        }
        ScalarValue::I1(b) => Some(if *b { 1 } else { 0 }),
    }
}

fn scalar_as_u64(v: &ScalarValue) -> Option<u64> {
    match v {
        ScalarValue::UI64(x) => Some(*x),
        ScalarValue::UI32(x) => Some(*x as u64),
        ScalarValue::I64(x) => u64::try_from(*x).ok(),
        ScalarValue::I32(x) => u64::try_from(*x).ok(),
        ScalarValue::F64(x) => {
            if x.is_finite() && *x >= 0.0 && *x <= u64::MAX as f64 {
                Some(x.trunc() as u64)
            } else {
                None
            }
        }
        ScalarValue::F32(x) => {
            if x.is_finite() && *x >= 0.0 && *x <= u64::MAX as f32 {
                Some(x.trunc() as u64)
            } else {
                None
            }
        }
        ScalarValue::I1(b) => Some(if *b { 1 } else { 0 }),
    }
}

fn scalar_as_bool(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::I1(b) => *b,
        ScalarValue::I64(x) => *x != 0,
        ScalarValue::I32(x) => *x != 0,
        ScalarValue::UI64(x) => *x != 0,
        ScalarValue::UI32(x) => *x != 0,
        ScalarValue::F64(x) => *x != 0.0,
        ScalarValue::F32(x) => *x != 0.0,
    }
}

/// Materialize an `Iota { dimension }` result as a flat `Vec<ScalarValue>`
/// in row-major order (matching the IR's storage convention).
fn iota_values(ty: &TensorType, dimension: i64) -> Vec<ScalarValue> {
    let n = ty.num_elements();
    let rank = ty.rank();
    let d = dimension as usize;
    let mut strides = vec![1i64; rank.max(1)];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * ty.shape[i + 1];
    }

    let mut values = Vec::with_capacity(n);
    for flat in 0..n {
        let coord = if d < rank {
            (flat as i64 / strides[d]) % ty.shape[d]
        } else {
            0
        };
        values.push(scalar_of_element_type(ty.element_type, coord));
    }
    values
}

fn scalar_of_element_type(et: ElementType, value: i64) -> ScalarValue {
    match et {
        ElementType::F64 => ScalarValue::F64(value as f64),
        ElementType::F32 => ScalarValue::F32(value as f32),
        ElementType::I64 => ScalarValue::I64(value),
        ElementType::I32 => ScalarValue::I32(value as i32),
        ElementType::UI64 => ScalarValue::UI64(value as u64),
        ElementType::UI32 => ScalarValue::UI32(value as u32),
        ElementType::I1 => ScalarValue::I1(value != 0),
    }
}

// ---------------------------------------------------------------------------
// Array folds
// ---------------------------------------------------------------------------

/// Flat row-major index given an N-D coordinate and shape.
fn flat_index(coord: &[i64], shape: &[i64]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        idx += (coord[i] as usize) * stride;
        stride *= shape[i] as usize;
    }
    idx
}

/// Unravel a flat index back into an N-D coordinate for a given shape.
fn unravel_index(mut flat: usize, shape: &[i64]) -> Vec<i64> {
    let mut coord = vec![0i64; shape.len()];
    for i in (0..shape.len()).rev() {
        let dim = shape[i] as usize;
        coord[i] = (flat % dim) as i64;
        flat /= dim;
    }
    coord
}

fn transpose_dense_array(
    arr: &[ScalarValue],
    src_shape: &[i64],
    permutation: &[i64],
) -> Option<Vec<ScalarValue>> {
    if permutation.len() != src_shape.len() {
        return None;
    }
    let out_shape: Vec<i64> = permutation.iter().map(|&p| src_shape[p as usize]).collect();
    let n: usize = out_shape.iter().product::<i64>().max(1) as usize;
    if n != arr.len() {
        // Guard against inconsistent inputs.
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let out_coord = unravel_index(flat, &out_shape);
        let mut src_coord = vec![0i64; src_shape.len()];
        for (out_dim, &src_dim) in permutation.iter().enumerate() {
            src_coord[src_dim as usize] = out_coord[out_dim];
        }
        out.push(arr[flat_index(&src_coord, src_shape)].clone());
    }
    Some(out)
}

fn reverse_dense_array(
    arr: &[ScalarValue],
    src_shape: &[i64],
    dimensions: &[i64],
) -> Option<Vec<ScalarValue>> {
    let n: usize = src_shape.iter().product::<i64>().max(1) as usize;
    if n != arr.len() {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut coord = unravel_index(flat, src_shape);
        for &d in dimensions {
            let du = d as usize;
            coord[du] = src_shape[du] - 1 - coord[du];
        }
        out.push(arr[flat_index(&coord, src_shape)].clone());
    }
    Some(out)
}

fn slice_dense_array(
    arr: &[ScalarValue],
    src_shape: &[i64],
    start: &[i64],
    limit: &[i64],
) -> Option<Vec<ScalarValue>> {
    if start.len() != src_shape.len() || limit.len() != src_shape.len() {
        return None;
    }
    let out_shape: Vec<i64> = start
        .iter()
        .zip(limit)
        .map(|(s, l)| (l - s).max(0))
        .collect();
    let n: usize = out_shape.iter().product::<i64>().max(1) as usize;
    if n == 0 {
        return Some(Vec::new());
    }
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let out_coord = unravel_index(flat, &out_shape);
        let src_coord: Vec<i64> = out_coord.iter().zip(start).map(|(c, s)| c + s).collect();
        out.push(arr[flat_index(&src_coord, src_shape)].clone());
    }
    Some(out)
}

fn concat_dense_arrays(
    sources: &[(&[ScalarValue], &[i64])],
    dimension: i64,
) -> Option<Vec<ScalarValue>> {
    if sources.is_empty() {
        return Some(Vec::new());
    }
    let rank = sources[0].1.len();
    let dim = dimension as usize;
    if dim >= rank {
        return None;
    }
    // Every source must share every non-concat dim with the first source.
    let first_shape = sources[0].1;
    for (_, shape) in &sources[1..] {
        if shape.len() != rank {
            return None;
        }
        for i in 0..rank {
            if i != dim && shape[i] != first_shape[i] {
                return None;
            }
        }
    }
    let mut out_shape: Vec<i64> = first_shape.to_vec();
    out_shape[dim] = sources.iter().map(|(_, s)| s[dim]).sum();
    let n: usize = out_shape.iter().product::<i64>().max(1) as usize;
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut coord = unravel_index(flat, &out_shape);
        // Find the source whose dimension range contains coord[dim].
        let mut offset = coord[dim];
        let mut chosen: Option<(usize, i64)> = None;
        for (i, (_, shape)) in sources.iter().enumerate() {
            let sz = shape[dim];
            if offset < sz {
                chosen = Some((i, offset));
                break;
            }
            offset -= sz;
        }
        let (i, local) = chosen?;
        coord[dim] = local;
        out.push(sources[i].0[flat_index(&coord, sources[i].1)].clone());
    }
    Some(out)
}

fn pad_dense_array(
    arr: &[ScalarValue],
    src_shape: &[i64],
    padding_value: &ScalarValue,
    low: &[i64],
    high: &[i64],
    interior: &[i64],
) -> Option<Vec<ScalarValue>> {
    let rank = src_shape.len();
    if low.len() != rank || high.len() != rank || interior.len() != rank {
        return None;
    }
    // Output dim = low + src*(interior+1) - interior + high (per StableHLO).
    let mut out_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let stride = interior[i] + 1;
        let src_contribution = src_shape[i] * stride - interior[i];
        out_shape.push(low[i] + src_contribution + high[i]);
    }
    let n: usize = out_shape.iter().product::<i64>().max(1) as usize;
    let mut out = vec![padding_value.clone(); n];
    for (src_flat, value) in arr.iter().enumerate() {
        let src_coord = unravel_index(src_flat, src_shape);
        let mut out_coord = Vec::with_capacity(rank);
        for i in 0..rank {
            out_coord.push(low[i] + src_coord[i] * (interior[i] + 1));
        }
        let out_flat = flat_index(&out_coord, &out_shape);
        out[out_flat] = value.clone();
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Algebraic identities (produce a ValueId alias, not a Constant)
// ---------------------------------------------------------------------------

fn try_identity(
    instr: &Instruction,
    env: &HashMap<ValueId, ConstantValue>,
    type_of: &HashMap<ValueId, TensorType>,
    results: &[(ValueId, TensorType)],
) -> Option<(ValueId, &'static str)> {
    let result_ty = &results.first()?.1;

    match instr {
        Instruction::Multiply { lhs, rhs } => {
            if env.get(rhs).is_some_and(is_all_one_constant) {
                Some((*lhs, "identity_mul_one"))
            } else if env.get(lhs).is_some_and(is_all_one_constant) {
                Some((*rhs, "identity_mul_one"))
            } else {
                None
            }
        }
        Instruction::Add { lhs, rhs } => {
            if env.get(rhs).is_some_and(is_all_zero_constant) {
                Some((*lhs, "identity_add_zero"))
            } else if env.get(lhs).is_some_and(is_all_zero_constant) {
                Some((*rhs, "identity_add_zero"))
            } else {
                None
            }
        }
        Instruction::Subtract { lhs, rhs } => {
            if env.get(rhs).is_some_and(is_all_zero_constant) {
                Some((*lhs, "identity_sub_zero"))
            } else {
                None
            }
        }
        Instruction::Divide { lhs, rhs } => {
            if env.get(rhs).is_some_and(is_all_one_constant) {
                Some((*lhs, "identity_div_one"))
            } else {
                None
            }
        }
        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let cond_v = env.get(cond)?;
            let cond_bool = match cond_v {
                ConstantValue::DenseScalar(ScalarValue::I1(b))
                | ConstantValue::DenseSplat(ScalarValue::I1(b), _) => *b,
                _ => return None,
            };
            // The full-constant case is handled earlier in `try_fold`; here we
            // alias the chosen branch even when it isn't a known constant.
            if cond_bool {
                Some((*on_true, "identity_select_true"))
            } else {
                Some((*on_false, "identity_select_false"))
            }
        }
        Instruction::Convert { operand } => {
            let src_ty = type_of.get(operand)?;
            if src_ty.element_type == result_ty.element_type && src_ty.shape == result_ty.shape {
                Some((*operand, "identity_convert_same_type"))
            } else {
                None
            }
        }
        Instruction::Reshape { operand } => {
            let src_ty = type_of.get(operand)?;
            if src_ty.shape == result_ty.shape {
                Some((*operand, "identity_reshape_same_shape"))
            } else {
                None
            }
        }
        Instruction::BroadcastInDim { operand, .. } => {
            let src_ty = type_of.get(operand)?;
            if src_ty.shape == result_ty.shape {
                Some((*operand, "identity_broadcast_same_shape"))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_all_zero_constant(cv: &ConstantValue) -> bool {
    match cv {
        ConstantValue::DenseScalar(s) | ConstantValue::DenseSplat(s, _) => is_scalar_zero(s),
        ConstantValue::DenseArray(_) => false,
    }
}

fn is_all_one_constant(cv: &ConstantValue) -> bool {
    match cv {
        ConstantValue::DenseScalar(s) | ConstantValue::DenseSplat(s, _) => is_scalar_one(s),
        ConstantValue::DenseArray(_) => false,
    }
}

fn is_scalar_zero(s: &ScalarValue) -> bool {
    match s {
        ScalarValue::F64(v) => *v == 0.0,
        ScalarValue::F32(v) => *v == 0.0,
        ScalarValue::I64(v) => *v == 0,
        ScalarValue::I32(v) => *v == 0,
        ScalarValue::UI64(v) => *v == 0,
        ScalarValue::UI32(v) => *v == 0,
        ScalarValue::I1(v) => !*v,
    }
}

fn is_scalar_one(s: &ScalarValue) -> bool {
    match s {
        ScalarValue::F64(v) => *v == 1.0,
        ScalarValue::F32(v) => *v == 1.0,
        ScalarValue::I64(v) => *v == 1,
        ScalarValue::I32(v) => *v == 1,
        ScalarValue::UI64(v) => *v == 1,
        ScalarValue::UI32(v) => *v == 1,
        ScalarValue::I1(v) => *v,
    }
}

// ---------------------------------------------------------------------------
// Alias rewriting
// ---------------------------------------------------------------------------

/// Resolve an alias chain: walk through the table until a value ID no longer
/// has an entry. Paths in the table are always strictly upstream, so the
/// loop terminates in O(length of chain).
fn resolve_alias(mut v: ValueId, aliases: &HashMap<ValueId, ValueId>) -> ValueId {
    while let Some(&next) = aliases.get(&v) {
        if next == v {
            break;
        }
        v = next;
    }
    v
}

/// Rewrite every operand reference in every instruction (including embedded
/// bodies) so that values pointing at an aliased result now point at the
/// upstream value directly.
pub(crate) fn rewrite_aliases_body(body: &mut [InstrResult], aliases: &HashMap<ValueId, ValueId>) {
    for ir in body.iter_mut() {
        rewrite_aliases_instr(&mut ir.instr, aliases);
    }
}

fn rewrite_aliases_instr(instr: &mut Instruction, aliases: &HashMap<ValueId, ValueId>) {
    let remap = |v: &mut ValueId| *v = resolve_alias(*v, aliases);

    match instr {
        Instruction::Constant { .. } | Instruction::Iota { .. } => {}
        Instruction::Add { lhs, rhs }
        | Instruction::Subtract { lhs, rhs }
        | Instruction::Multiply { lhs, rhs }
        | Instruction::Divide { lhs, rhs }
        | Instruction::Maximum { lhs, rhs }
        | Instruction::Minimum { lhs, rhs }
        | Instruction::Atan2 { lhs, rhs }
        | Instruction::Remainder { lhs, rhs }
        | Instruction::Power { lhs, rhs }
        | Instruction::Xor { lhs, rhs }
        | Instruction::Or { lhs, rhs }
        | Instruction::And { lhs, rhs }
        | Instruction::ShiftLeft { lhs, rhs }
        | Instruction::ShiftRightLogical { lhs, rhs }
        | Instruction::ShiftRightArithmetic { lhs, rhs }
        | Instruction::Compare { lhs, rhs, .. } => {
            remap(lhs);
            remap(rhs);
        }
        Instruction::Negate { operand }
        | Instruction::Sqrt { operand }
        | Instruction::Rsqrt { operand }
        | Instruction::Abs { operand }
        | Instruction::Sign { operand }
        | Instruction::Sine { operand }
        | Instruction::Cosine { operand }
        | Instruction::Tan { operand }
        | Instruction::Tanh { operand }
        | Instruction::Sinh { operand }
        | Instruction::Cosh { operand }
        | Instruction::Asin { operand }
        | Instruction::Acos { operand }
        | Instruction::Atan { operand }
        | Instruction::Exponential { operand }
        | Instruction::Log { operand }
        | Instruction::Log1p { operand }
        | Instruction::Expm1 { operand }
        | Instruction::Cbrt { operand }
        | Instruction::Erfc { operand }
        | Instruction::ErfInv { operand }
        | Instruction::IsFinite { operand }
        | Instruction::Not { operand }
        | Instruction::Floor { operand }
        | Instruction::Ceil { operand }
        | Instruction::RoundNearestEven { operand }
        | Instruction::Reshape { operand }
        | Instruction::Convert { operand }
        | Instruction::BitcastConvert { operand }
        | Instruction::BroadcastInDim { operand, .. }
        | Instruction::Slice { operand, .. }
        | Instruction::Transpose { operand, .. }
        | Instruction::Reverse { operand, .. } => {
            remap(operand);
        }
        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            remap(cond);
            remap(on_true);
            remap(on_false);
        }
        Instruction::Clamp { operand, min, max } => {
            remap(operand);
            remap(min);
            remap(max);
        }
        Instruction::DotGeneral { lhs, rhs, .. } => {
            remap(lhs);
            remap(rhs);
        }
        Instruction::Reduce { operand, init, .. } => {
            remap(operand);
            remap(init);
        }
        Instruction::ReduceArgminmax {
            values, indices, ..
        } => {
            remap(values);
            remap(indices);
        }
        Instruction::Concatenate { operands, .. } => {
            for v in operands.iter_mut() {
                remap(v);
            }
        }
        Instruction::Call { args, .. } => {
            for v in args.iter_mut() {
                remap(v);
            }
        }
        Instruction::While {
            init_values,
            cond_body,
            loop_body,
            ..
        } => {
            for v in init_values.iter_mut() {
                remap(v);
            }
            rewrite_aliases_body(cond_body, aliases);
            rewrite_aliases_body(loop_body, aliases);
        }
        Instruction::Case { index, branches } => {
            remap(index);
            for b in branches.iter_mut() {
                rewrite_aliases_body(b, aliases);
            }
        }
        Instruction::Return { operands } => {
            for v in operands.iter_mut() {
                remap(v);
            }
        }
        Instruction::Gather {
            operand, indices, ..
        } => {
            remap(operand);
            remap(indices);
        }
        Instruction::DynamicSlice {
            operand,
            start_indices,
            ..
        } => {
            remap(operand);
            for v in start_indices.iter_mut() {
                remap(v);
            }
        }
        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            remap(operand);
            remap(update);
            for v in start_indices.iter_mut() {
                remap(v);
            }
        }
        Instruction::Pad {
            operand,
            padding_value,
            ..
        } => {
            remap(operand);
            remap(padding_value);
        }
        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => {
            remap(operand);
            remap(indices);
            remap(updates);
        }
        Instruction::CustomCall { operands, .. } => {
            for v in operands.iter_mut() {
                remap(v);
            }
        }
        Instruction::Sort {
            inputs, comparator, ..
        } => {
            for v in inputs.iter_mut() {
                remap(v);
            }
            rewrite_aliases_body(comparator, aliases);
        }
        Instruction::Map {
            inputs, body: b, ..
        } => {
            for v in inputs.iter_mut() {
                remap(v);
            }
            rewrite_aliases_body(b, aliases);
        }
        Instruction::ReduceWindow {
            operands,
            init_values,
            body: b,
            ..
        } => {
            for v in operands.iter_mut() {
                remap(v);
            }
            for v in init_values.iter_mut() {
                remap(v);
            }
            rewrite_aliases_body(b, aliases);
        }
        Instruction::SelectAndScatter {
            operand,
            source,
            init_value,
            select_body,
            scatter_body,
            ..
        } => {
            remap(operand);
            remap(source);
            remap(init_value);
            rewrite_aliases_body(select_body, aliases);
            rewrite_aliases_body(scatter_body, aliases);
        }
        Instruction::Convolution { lhs, rhs, .. } => {
            remap(lhs);
            remap(rhs);
        }
        Instruction::CholeskyOp { operand, .. } => {
            remap(operand);
        }
        Instruction::TriangularSolve { a, b, .. } => {
            remap(a);
            remap(b);
        }
        Instruction::Fft { operand, .. } => {
            remap(operand);
        }
        Instruction::BatchNormInference {
            operand,
            scale,
            offset,
            mean,
            variance,
            ..
        } => {
            remap(operand);
            remap(scale);
            remap(offset);
            remap(mean);
            remap(variance);
        }
        Instruction::RealDynamicSlice {
            operand,
            start_indices,
            limit_indices,
            strides,
        } => {
            remap(operand);
            remap(start_indices);
            remap(limit_indices);
            remap(strides);
        }
        Instruction::Rng { operands, .. } => {
            for v in operands.iter_mut() {
                remap(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dead-code elimination (conservative)
// ---------------------------------------------------------------------------

/// Drop instructions whose result `ValueId`s are not referenced anywhere in
/// the surrounding function (including nested bodies). Only pure instructions
/// are dropped; anything terminator-like, call-like, or containing an
/// embedded region stays.
fn dce_body(body: &mut Vec<InstrResult>, report: &mut FoldReport) {
    let mut used: HashSet<ValueId> = HashSet::new();
    collect_used_value_ids(body, &mut used);

    body.retain(|ir| {
        if !is_removable(&ir.instr) {
            return true;
        }
        let any_output_used = ir.values.iter().any(|(id, _)| used.contains(id));
        if any_output_used {
            true
        } else {
            report.bump("dce_unused_constant");
            false
        }
    });
}

/// Collect every `ValueId` that appears as an operand anywhere in `body`,
/// descending into embedded bodies. These are the "live" values.
fn collect_used_value_ids(body: &[InstrResult], out: &mut HashSet<ValueId>) {
    for ir in body {
        for id in operand_ids(&ir.instr) {
            out.insert(id);
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                ..
            } => {
                collect_used_value_ids(cond_body, out);
                collect_used_value_ids(loop_body, out);
            }
            Instruction::Case { branches, .. } => {
                for branch in branches {
                    collect_used_value_ids(branch, out);
                }
            }
            Instruction::Map { body: b, .. } | Instruction::ReduceWindow { body: b, .. } => {
                collect_used_value_ids(b, out);
            }
            Instruction::SelectAndScatter {
                select_body,
                scatter_body,
                ..
            } => {
                collect_used_value_ids(select_body, out);
                collect_used_value_ids(scatter_body, out);
            }
            Instruction::Sort { comparator, .. } => {
                collect_used_value_ids(comparator, out);
            }
            _ => {}
        }
    }
}

/// Return `true` for pure-and-self-contained instructions whose
/// removal is always safe when their outputs are unused. The DCE
/// pass never touches anything outside this whitelist — the set is
/// kept tight so customer workloads cannot silently lose a side
/// effect.
fn is_removable(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::Constant { .. }
            | Instruction::Add { .. }
            | Instruction::Subtract { .. }
            | Instruction::Multiply { .. }
            | Instruction::Divide { .. }
            | Instruction::Negate { .. }
            | Instruction::Sqrt { .. }
            | Instruction::Maximum { .. }
            | Instruction::Minimum { .. }
            | Instruction::Compare { .. }
            | Instruction::Select { .. }
            | Instruction::Reshape { .. }
            | Instruction::BroadcastInDim { .. }
            | Instruction::Slice { .. }
            | Instruction::Concatenate { .. }
            | Instruction::DotGeneral { .. }
            | Instruction::Reduce { .. }
            | Instruction::ReduceArgminmax { .. }
            | Instruction::Convert { .. }
            | Instruction::BitcastConvert { .. }
            | Instruction::Iota { .. }
            | Instruction::Xor { .. }
            | Instruction::Or { .. }
            | Instruction::And { .. }
            | Instruction::ShiftLeft { .. }
            | Instruction::ShiftRightLogical { .. }
            | Instruction::ShiftRightArithmetic { .. }
            | Instruction::ErfInv { .. }
            | Instruction::Gather { .. }
            | Instruction::Transpose { .. }
            | Instruction::DynamicSlice { .. }
            | Instruction::Sine { .. }
            | Instruction::Cosine { .. }
            | Instruction::Atan2 { .. }
            | Instruction::Abs { .. }
            | Instruction::Sign { .. }
            | Instruction::Remainder { .. }
            | Instruction::Acos { .. }
            | Instruction::Exponential { .. }
            | Instruction::Log { .. }
            | Instruction::Clamp { .. }
            | Instruction::Power { .. }
            | Instruction::Reverse { .. }
            | Instruction::Tanh { .. }
            | Instruction::Tan { .. }
            | Instruction::Floor { .. }
            | Instruction::RoundNearestEven { .. }
            | Instruction::Pad { .. }
            | Instruction::Rsqrt { .. }
            | Instruction::Log1p { .. }
            | Instruction::IsFinite { .. }
            | Instruction::Not { .. }
            | Instruction::Ceil { .. }
            | Instruction::Asin { .. }
            | Instruction::Atan { .. }
            | Instruction::Sinh { .. }
            | Instruction::Cosh { .. }
            | Instruction::Erfc { .. }
            | Instruction::Expm1 { .. }
            | Instruction::Cbrt { .. }
            | Instruction::BatchNormInference { .. }
            | Instruction::RealDynamicSlice { .. }
            | Instruction::CholeskyOp { .. }
            | Instruction::TriangularSolve { .. }
            | Instruction::Fft { .. }
            | Instruction::Rng { .. }
    )
    // Not included (always kept): Return, Call, CustomCall, Scatter,
    // DynamicUpdateSlice, While, Case, Map, ReduceWindow, SelectAndScatter,
    // Sort. These either terminate a block, call external code, write to
    // a buffer, or carry an embedded body whose effects we conservatively
    // preserve.
}

/// Exhaustive operand extraction for every `Instruction` variant. This is
/// the single source of truth for "what does this instruction depend on"
/// used by the DCE walk.
pub(crate) fn operand_ids(instr: &Instruction) -> Vec<ValueId> {
    match instr {
        Instruction::Constant { .. } | Instruction::Iota { .. } => vec![],
        Instruction::Add { lhs, rhs }
        | Instruction::Subtract { lhs, rhs }
        | Instruction::Multiply { lhs, rhs }
        | Instruction::Divide { lhs, rhs }
        | Instruction::Maximum { lhs, rhs }
        | Instruction::Minimum { lhs, rhs }
        | Instruction::Atan2 { lhs, rhs }
        | Instruction::Remainder { lhs, rhs }
        | Instruction::Power { lhs, rhs }
        | Instruction::Xor { lhs, rhs }
        | Instruction::Or { lhs, rhs }
        | Instruction::And { lhs, rhs }
        | Instruction::ShiftLeft { lhs, rhs }
        | Instruction::ShiftRightLogical { lhs, rhs }
        | Instruction::ShiftRightArithmetic { lhs, rhs }
        | Instruction::Compare { lhs, rhs, .. } => vec![*lhs, *rhs],
        Instruction::Negate { operand }
        | Instruction::Sqrt { operand }
        | Instruction::Rsqrt { operand }
        | Instruction::Abs { operand }
        | Instruction::Sign { operand }
        | Instruction::Sine { operand }
        | Instruction::Cosine { operand }
        | Instruction::Tan { operand }
        | Instruction::Tanh { operand }
        | Instruction::Sinh { operand }
        | Instruction::Cosh { operand }
        | Instruction::Asin { operand }
        | Instruction::Acos { operand }
        | Instruction::Atan { operand }
        | Instruction::Exponential { operand }
        | Instruction::Log { operand }
        | Instruction::Log1p { operand }
        | Instruction::Expm1 { operand }
        | Instruction::Cbrt { operand }
        | Instruction::Erfc { operand }
        | Instruction::ErfInv { operand }
        | Instruction::IsFinite { operand }
        | Instruction::Not { operand }
        | Instruction::Floor { operand }
        | Instruction::Ceil { operand }
        | Instruction::RoundNearestEven { operand }
        | Instruction::Reshape { operand }
        | Instruction::Convert { operand }
        | Instruction::BitcastConvert { operand }
        | Instruction::BroadcastInDim { operand, .. }
        | Instruction::Slice { operand, .. }
        | Instruction::Transpose { operand, .. }
        | Instruction::Reverse { operand, .. } => vec![*operand],
        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => vec![*cond, *on_true, *on_false],
        Instruction::Clamp { operand, min, max } => vec![*operand, *min, *max],
        Instruction::DotGeneral { lhs, rhs, .. } => vec![*lhs, *rhs],
        Instruction::Reduce { operand, init, .. } => vec![*operand, *init],
        Instruction::ReduceArgminmax {
            values, indices, ..
        } => vec![*values, *indices],
        Instruction::Concatenate { operands, .. } => operands.clone(),
        Instruction::Call { args, .. } => args.clone(),
        Instruction::While {
            init_values,
            cond_body,
            loop_body,
            ..
        } => {
            let mut ids = init_values.clone();
            collect_body_operands(cond_body, &mut ids);
            collect_body_operands(loop_body, &mut ids);
            ids
        }
        Instruction::Case { index, branches } => {
            let mut ids = vec![*index];
            for b in branches {
                collect_body_operands(b, &mut ids);
            }
            ids
        }
        Instruction::Return { operands } => operands.clone(),
        Instruction::Gather {
            operand, indices, ..
        } => vec![*operand, *indices],
        Instruction::DynamicSlice {
            operand,
            start_indices,
            ..
        } => {
            let mut v = vec![*operand];
            v.extend_from_slice(start_indices);
            v
        }
        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            let mut v = vec![*operand, *update];
            v.extend_from_slice(start_indices);
            v
        }
        Instruction::Pad {
            operand,
            padding_value,
            ..
        } => vec![*operand, *padding_value],
        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => vec![*operand, *indices, *updates],
        Instruction::CustomCall { operands, .. } => operands.clone(),
        Instruction::Sort {
            inputs, comparator, ..
        } => {
            let mut v = inputs.clone();
            collect_body_operands(comparator, &mut v);
            v
        }
        Instruction::Map {
            inputs, body: b, ..
        } => {
            let mut v = inputs.clone();
            collect_body_operands(b, &mut v);
            v
        }
        Instruction::ReduceWindow {
            operands,
            init_values,
            body: b,
            ..
        } => {
            let mut v = operands.clone();
            v.extend_from_slice(init_values);
            collect_body_operands(b, &mut v);
            v
        }
        Instruction::SelectAndScatter {
            operand,
            source,
            init_value,
            select_body,
            scatter_body,
            ..
        } => {
            let mut v = vec![*operand, *source, *init_value];
            collect_body_operands(select_body, &mut v);
            collect_body_operands(scatter_body, &mut v);
            v
        }
        Instruction::Convolution { lhs, rhs, .. } => vec![*lhs, *rhs],
        Instruction::CholeskyOp { operand, .. } => vec![*operand],
        Instruction::TriangularSolve { a, b, .. } => vec![*a, *b],
        Instruction::Fft { operand, .. } => vec![*operand],
        Instruction::BatchNormInference {
            operand,
            scale,
            offset,
            mean,
            variance,
            ..
        } => vec![*operand, *scale, *offset, *mean, *variance],
        Instruction::RealDynamicSlice {
            operand,
            start_indices,
            limit_indices,
            strides,
        } => vec![*operand, *start_indices, *limit_indices, *strides],
        Instruction::Rng { operands, .. } => operands.clone(),
    }
}

fn collect_body_operands(body: &[InstrResult], out: &mut Vec<ValueId>) {
    for ir in body {
        out.extend(operand_ids(&ir.instr));
    }
}

fn log_report(report: &FoldReport) {
    eprintln!("[elodin-cranelift] {}", report.fmt_summary());
    if crate::debug::enabled() {
        eprintln!(
            "[elodin-cranelift] fold histogram: {}",
            report.fmt_histogram()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConstantValue, ElementType, ScalarValue, TensorType, ValueId};

    fn f64_scalar_ty() -> TensorType {
        TensorType::scalar(ElementType::F64)
    }

    fn f64_tensor(shape: Vec<i64>) -> TensorType {
        TensorType {
            shape,
            element_type: ElementType::F64,
        }
    }

    fn mk_constant(id: u32, value: f64) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(id), f64_scalar_ty())],
            instr: Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(value)),
            },
        }
    }

    fn mk_func(body: Vec<InstrResult>) -> FuncDef {
        FuncDef {
            name: "test".into(),
            is_public: true,
            params: vec![],
            result_types: vec![],
            body,
            source_line: None,
        }
    }

    fn mk_module(body: Vec<InstrResult>) -> Module {
        Module::new(vec![mk_func(body)])
    }

    // ----- Measurement tests -----

    #[test]
    fn count_flat_module() {
        let m = mk_module(vec![
            mk_constant(0, 1.0),
            mk_constant(1, 2.0),
            mk_constant(2, 3.0),
        ]);
        assert_eq!(count_instructions(&m), 3);
    }

    #[test]
    fn count_recurses_into_while_bodies() {
        let m = Module::new(vec![FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![],
            result_types: vec![],
            body: vec![InstrResult {
                values: vec![(ValueId(0), f64_scalar_ty())],
                instr: Instruction::While {
                    cond_body: vec![mk_constant(1, 1.0), mk_constant(2, 2.0)],
                    loop_body: vec![mk_constant(3, 3.0)],
                    init_values: vec![],
                    iter_arg_ids: vec![],
                },
            }],
            source_line: None,
        }]);
        assert_eq!(count_instructions(&m), 4);
    }

    #[test]
    fn report_percent_removed_zero_when_empty() {
        let r = FoldReport::default();
        assert_eq!(r.percent_removed(), 0.0);
        assert_eq!(r.fmt_histogram(), "(empty)");
    }

    #[test]
    fn report_histogram_sorts_by_count_desc() {
        let mut r = FoldReport::default();
        r.bump("rule_b");
        r.bump("rule_a");
        r.bump("rule_a");
        r.bump("rule_c");
        r.bump("rule_c");
        r.bump("rule_c");
        assert_eq!(r.fmt_histogram(), "rule_c=3, rule_a=2, rule_b=1");
    }

    // ----- Scalar-constant fold-rule tests -----

    #[test]
    fn broadcast_scalar_folds_to_splat() {
        let mut m = mk_module(vec![
            mk_constant(0, 7.0),
            InstrResult {
                values: vec![(ValueId(1), f64_tensor(vec![6]))],
                instr: Instruction::BroadcastInDim {
                    operand: ValueId(0),
                    broadcast_dims: vec![],
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        let r = fold_module(&mut m);
        assert_eq!(r.counts_by_rule.get("broadcast_scalar").copied(), Some(1));

        let body = &m.functions[0].body;
        // Position 0: original constant is dead, should be DCE'd out. Position 0
        // is now the folded broadcast (a splat constant). Position 1 is Return.
        assert_eq!(body.len(), 2);
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseSplat(ScalarValue::F64(v), ty),
            } => {
                assert_eq!(*v, 7.0);
                assert_eq!(ty.shape, vec![6]);
            }
            other => panic!("expected Constant DenseSplat, got {other:?}"),
        }
    }

    #[test]
    fn convert_scalar_f64_to_f32() {
        let mut m = mk_module(vec![
            mk_constant(0, 1.5),
            InstrResult {
                values: vec![(ValueId(1), TensorType::scalar(ElementType::F32))],
                instr: Instruction::Convert {
                    operand: ValueId(0),
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        let r = fold_module(&mut m);
        assert_eq!(r.counts_by_rule.get("convert_scalar").copied(), Some(1));
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F32(v)),
            } => assert_eq!(*v, 1.5),
            other => panic!("expected Constant DenseScalar F32, got {other:?}"),
        }
    }

    #[test]
    fn reshape_scalar_to_tensor_becomes_splat() {
        let mut m = mk_module(vec![
            mk_constant(0, 3.0),
            InstrResult {
                values: vec![(ValueId(1), f64_tensor(vec![1]))],
                instr: Instruction::Reshape {
                    operand: ValueId(0),
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseSplat(ScalarValue::F64(v), ty),
            } => {
                assert_eq!(*v, 3.0);
                assert_eq!(ty.shape, vec![1]);
            }
            other => panic!("expected DenseSplat, got {other:?}"),
        }
    }

    #[test]
    fn iota_small_folds_to_dense_array() {
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(
                    ValueId(0),
                    TensorType {
                        shape: vec![5],
                        element_type: ElementType::I64,
                    },
                )],
                instr: Instruction::Iota { dimension: 0 },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(0)],
                },
            },
        ]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseArray(arr),
            } => {
                let got: Vec<i64> = arr
                    .iter()
                    .map(|s| match s {
                        ScalarValue::I64(v) => *v,
                        _ => panic!("expected I64"),
                    })
                    .collect();
                assert_eq!(got, vec![0, 1, 2, 3, 4]);
            }
            other => panic!("expected DenseArray, got {other:?}"),
        }
    }

    #[test]
    fn iota_above_cap_does_not_fold() {
        // f64 * 1000 elements = 8000 bytes > 1024 cap.
        let ty = TensorType {
            shape: vec![1000],
            element_type: ElementType::F64,
        };
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), ty)],
                instr: Instruction::Iota { dimension: 0 },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(0)],
                },
            },
        ]);
        fold_module(&mut m);
        assert!(matches!(
            m.functions[0].body[0].instr,
            Instruction::Iota { .. }
        ));
    }

    #[test]
    fn dce_drops_unused_constant_after_fold() {
        // Constant %0 feeds a broadcast. Once the broadcast is folded, the
        // original constant is unused and DCE removes it.
        let mut m = mk_module(vec![
            mk_constant(0, 2.5),
            InstrResult {
                values: vec![(ValueId(1), f64_tensor(vec![4]))],
                instr: Instruction::BroadcastInDim {
                    operand: ValueId(0),
                    broadcast_dims: vec![],
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        let r = fold_module(&mut m);
        assert!(
            r.counts_by_rule
                .get("dce_unused_constant")
                .copied()
                .unwrap_or(0)
                >= 1
        );
        // Only the folded splat + Return remain.
        assert_eq!(m.functions[0].body.len(), 2);
        assert!(matches!(
            m.functions[0].body[0].instr,
            Instruction::Constant { .. }
        ));
        assert!(matches!(
            m.functions[0].body[1].instr,
            Instruction::Return { .. }
        ));
    }

    #[test]
    fn dce_preserves_constants_referenced_by_nested_body() {
        // %0 is referenced inside the while cond body by a Compare against a
        // function parameter (not a constant), so the Compare cannot fold and
        // the constant must stay live across the nested-body boundary.
        let mut m = Module::new(vec![FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), f64_scalar_ty())],
            result_types: vec![f64_scalar_ty()],
            body: vec![
                mk_constant(0, 1.0),
                InstrResult {
                    values: vec![(ValueId(10), f64_scalar_ty())],
                    instr: Instruction::While {
                        cond_body: vec![InstrResult {
                            values: vec![(ValueId(11), TensorType::scalar(ElementType::I1))],
                            instr: Instruction::Compare {
                                lhs: ValueId(0),
                                rhs: ValueId(100),
                                direction: CompareDirection::Eq,
                                compare_type: CompareType::Float,
                            },
                        }],
                        loop_body: vec![],
                        init_values: vec![],
                        iter_arg_ids: vec![],
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(10)],
                    },
                },
            ],
            source_line: None,
        }]);
        fold_module(&mut m);
        // The constant stays: it's referenced by a Compare whose rhs is a
        // (non-constant) function parameter, so the Compare cannot fold.
        assert!(matches!(
            m.functions[0].body[0].instr,
            Instruction::Constant { .. }
        ));
    }

    // ----- Scalar arithmetic tests -----

    fn mk_instr(vid: u32, ty: TensorType, instr: Instruction) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(vid), ty)],
            instr,
        }
    }

    #[test]
    fn add_scalar_f64() {
        let mut m = mk_module(vec![
            mk_constant(0, 2.0),
            mk_constant(1, 3.0),
            mk_instr(
                2,
                f64_scalar_ty(),
                Instruction::Add {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(2)],
                },
            },
        ]);
        let r = fold_module(&mut m);
        assert_eq!(r.counts_by_rule.get("add").copied(), Some(1));
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(v)),
            } => assert_eq!(*v, 5.0),
            other => panic!("expected folded add, got {other:?}"),
        }
    }

    #[test]
    fn divide_zero_rhs_does_not_fold() {
        // Integer divide by zero: must stay in place so the runtime handles it.
        let i64_ty = TensorType::scalar(ElementType::I64);
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), i64_ty.clone())],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I64(42)),
                },
            },
            InstrResult {
                values: vec![(ValueId(1), i64_ty.clone())],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I64(0)),
                },
            },
            mk_instr(
                2,
                i64_ty,
                Instruction::Divide {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(2)],
                },
            },
        ]);
        fold_module(&mut m);
        assert!(matches!(
            m.functions[0].body[2].instr,
            Instruction::Divide { .. }
        ));
    }

    #[test]
    fn divide_by_zero_float_does_not_fold() {
        // 0.0 / 0.0 -> NaN; skip fold.
        let mut m = mk_module(vec![
            mk_constant(0, 0.0),
            mk_constant(1, 0.0),
            mk_instr(
                2,
                f64_scalar_ty(),
                Instruction::Divide {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(2)],
                },
            },
        ]);
        fold_module(&mut m);
        assert!(matches!(
            m.functions[0].body[2].instr,
            Instruction::Divide { .. }
        ));
    }

    #[test]
    fn sqrt_negative_does_not_fold() {
        let mut m = mk_module(vec![
            mk_constant(0, -1.0),
            mk_instr(
                1,
                f64_scalar_ty(),
                Instruction::Sqrt {
                    operand: ValueId(0),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        fold_module(&mut m);
        // The sqrt stays because -1.0 produces NaN.
        assert!(matches!(
            m.functions[0].body[1].instr,
            Instruction::Sqrt { .. }
        ));
    }

    #[test]
    fn cascade_chain_of_negates_collapses() {
        // negate(negate(negate(negate(negate(1.0))))) = -1.0
        let mut m = mk_module(vec![
            mk_constant(0, 1.0),
            mk_instr(
                1,
                f64_scalar_ty(),
                Instruction::Negate {
                    operand: ValueId(0),
                },
            ),
            mk_instr(
                2,
                f64_scalar_ty(),
                Instruction::Negate {
                    operand: ValueId(1),
                },
            ),
            mk_instr(
                3,
                f64_scalar_ty(),
                Instruction::Negate {
                    operand: ValueId(2),
                },
            ),
            mk_instr(
                4,
                f64_scalar_ty(),
                Instruction::Negate {
                    operand: ValueId(3),
                },
            ),
            mk_instr(
                5,
                f64_scalar_ty(),
                Instruction::Negate {
                    operand: ValueId(4),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(5)],
                },
            },
        ]);
        let r = fold_module(&mut m);
        assert_eq!(r.counts_by_rule.get("negate").copied(), Some(5));
        let body = &m.functions[0].body;
        // All five negates + four intermediate constants collapse to one Constant + Return.
        assert_eq!(body.len(), 2);
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(v)),
            } => assert_eq!(*v, -1.0),
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    #[test]
    fn compare_scalar_float() {
        let mut m = mk_module(vec![
            mk_constant(0, 3.0),
            mk_constant(1, 2.0),
            mk_instr(
                2,
                TensorType::scalar(ElementType::I1),
                Instruction::Compare {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                    direction: CompareDirection::Gt,
                    compare_type: CompareType::Float,
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(2)],
                },
            },
        ]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::I1(b)),
            } => assert!(*b),
            other => panic!("expected Constant I1 true, got {other:?}"),
        }
    }

    #[test]
    fn select_constant_cond_picks_branch() {
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), TensorType::scalar(ElementType::I1))],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I1(true)),
                },
            },
            mk_constant(1, 10.0),
            mk_constant(2, 20.0),
            mk_instr(
                3,
                f64_scalar_ty(),
                Instruction::Select {
                    cond: ValueId(0),
                    on_true: ValueId(1),
                    on_false: ValueId(2),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(3)],
                },
            },
        ]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        // Should fold to constant 10.0.
        match body
            .iter()
            .find(|ir| ir.values.iter().any(|(id, _)| *id == ValueId(3)))
            .expect("result value present")
            .instr
            .clone()
        {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(v)),
            } => assert_eq!(v, 10.0),
            other => panic!("expected Constant 10.0, got {other:?}"),
        }
    }

    #[test]
    fn xor_integer_scalar() {
        let i32_ty = TensorType::scalar(ElementType::I32);
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), i32_ty.clone())],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I32(0b1100)),
                },
            },
            InstrResult {
                values: vec![(ValueId(1), i32_ty.clone())],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I32(0b1010)),
                },
            },
            mk_instr(
                2,
                i32_ty,
                Instruction::Xor {
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            ),
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(2)],
                },
            },
        ]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::I32(v)),
            } => assert_eq!(*v, 0b0110),
            other => panic!("expected I32 Constant, got {other:?}"),
        }
    }

    // ----- Array-fold tests -----

    #[test]
    fn transpose_dense_array_2d() {
        // 2x3 array, transpose to 3x2.
        let arr: Vec<ScalarValue> = (0..6).map(|i| ScalarValue::F64(i as f64)).collect();
        let src_shape = vec![2, 3];
        let out = transpose_dense_array(&arr, &src_shape, &[1, 0]).expect("fold");
        let got: Vec<f64> = out
            .into_iter()
            .map(|s| match s {
                ScalarValue::F64(v) => v,
                _ => panic!(),
            })
            .collect();
        // [0,1,2,3,4,5] with shape 2x3 row-major is:
        //   [[0,1,2],
        //    [3,4,5]]
        // transposed to 3x2:
        //   [[0,3],
        //    [1,4],
        //    [2,5]]
        // flattened row-major: [0,3,1,4,2,5]
        assert_eq!(got, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn reverse_dense_array_1d() {
        let arr: Vec<ScalarValue> = (0..4).map(|i| ScalarValue::F64(i as f64)).collect();
        let out = reverse_dense_array(&arr, &[4], &[0]).expect("fold");
        let got: Vec<f64> = out
            .into_iter()
            .map(|s| match s {
                ScalarValue::F64(v) => v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec![3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn slice_dense_array_1d() {
        let arr: Vec<ScalarValue> = (0..5).map(|i| ScalarValue::F64(i as f64)).collect();
        let out = slice_dense_array(&arr, &[5], &[1], &[4]).expect("fold");
        let got: Vec<f64> = out
            .into_iter()
            .map(|s| match s {
                ScalarValue::F64(v) => v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_dense_arrays_along_dim0() {
        let a: Vec<ScalarValue> = (0..3).map(|i| ScalarValue::F64(i as f64)).collect();
        let b: Vec<ScalarValue> = (10..13).map(|i| ScalarValue::F64(i as f64)).collect();
        let out = concat_dense_arrays(&[(&a, &[3]), (&b, &[3])], 0).expect("fold");
        let got: Vec<f64> = out
            .into_iter()
            .map(|s| match s {
                ScalarValue::F64(v) => v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn pad_dense_array_1d() {
        let arr: Vec<ScalarValue> = (0..3).map(|i| ScalarValue::F64(i as f64)).collect();
        // src=[0,1,2], low=1, high=2, interior=0 -> [99, 0, 1, 2, 99, 99]
        let out =
            pad_dense_array(&arr, &[3], &ScalarValue::F64(99.0), &[1], &[2], &[0]).expect("fold");
        let got: Vec<f64> = out
            .into_iter()
            .map(|s| match s {
                ScalarValue::F64(v) => v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec![99.0, 0.0, 1.0, 2.0, 99.0, 99.0]);
    }

    #[test]
    fn transpose_over_size_cap_does_not_fold_via_try_fold() {
        // Size > 1 KB -> should stay as Transpose instruction.
        let shape = vec![200];
        let arr: Vec<ScalarValue> = (0..200).map(|i| ScalarValue::F64(i as f64)).collect();
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), f64_tensor(shape.clone()))],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseArray(arr),
                },
            },
            InstrResult {
                values: vec![(ValueId(1), f64_tensor(vec![200]))],
                instr: Instruction::Transpose {
                    operand: ValueId(0),
                    permutation: vec![0],
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        fold_module(&mut m);
        // Transpose over 1 KB: stays as Transpose (not folded).
        let body = &m.functions[0].body;
        assert!(
            body.iter()
                .any(|ir| matches!(ir.instr, Instruction::Transpose { .. })),
            "expected Transpose to survive over the size cap"
        );
    }

    // ----- Identity-rewriting tests -----

    #[test]
    fn identity_multiply_by_one_aliases_lhs() {
        // Multiply(arg0, const 1.0) -> arg0. The multiply should be
        // removed and the Return should reference arg0 directly.
        let f64_ty = f64_scalar_ty();
        let mut m = Module::new(vec![FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), f64_ty.clone())],
            result_types: vec![f64_ty.clone()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(0), f64_ty.clone())],
                    instr: Instruction::Constant {
                        value: ConstantValue::DenseScalar(ScalarValue::F64(1.0)),
                    },
                },
                InstrResult {
                    values: vec![(ValueId(1), f64_ty)],
                    instr: Instruction::Multiply {
                        lhs: ValueId(100),
                        rhs: ValueId(0),
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(1)],
                    },
                },
            ],
            source_line: None,
        }]);
        let r = fold_module(&mut m);
        assert_eq!(r.counts_by_rule.get("identity_mul_one").copied(), Some(1));
        // After folding, body should contain just the Return referencing arg0
        // (the constant and multiply both DCE'd).
        let body = &m.functions[0].body;
        assert!(
            body.iter()
                .all(|ir| !matches!(ir.instr, Instruction::Multiply { .. }))
        );
        // The Return now references arg0 (ValueId 100), not ValueId 1.
        match body
            .iter()
            .find_map(|ir| {
                if let Instruction::Return { operands } = &ir.instr {
                    operands.first().copied()
                } else {
                    None
                }
            })
            .expect("return")
        {
            ValueId(100) => {}
            other => panic!("expected Return to reference arg0, got {other:?}"),
        }
    }

    #[test]
    fn identity_select_true_aliases_true_branch() {
        let f64_ty = f64_scalar_ty();
        let mut m = Module::new(vec![FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![
                (ValueId(100), f64_ty.clone()),
                (ValueId(101), f64_ty.clone()),
            ],
            result_types: vec![f64_ty.clone()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(0), TensorType::scalar(ElementType::I1))],
                    instr: Instruction::Constant {
                        value: ConstantValue::DenseScalar(ScalarValue::I1(true)),
                    },
                },
                InstrResult {
                    values: vec![(ValueId(1), f64_ty)],
                    instr: Instruction::Select {
                        cond: ValueId(0),
                        on_true: ValueId(100),
                        on_false: ValueId(101),
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(1)],
                    },
                },
            ],
            source_line: None,
        }]);
        fold_module(&mut m);
        let body = &m.functions[0].body;
        assert!(
            body.iter()
                .all(|ir| !matches!(ir.instr, Instruction::Select { .. }))
        );
        // Return now references arg0 (on_true path).
        match body
            .iter()
            .find_map(|ir| {
                if let Instruction::Return { operands } = &ir.instr {
                    operands.first().copied()
                } else {
                    None
                }
            })
            .expect("return")
        {
            ValueId(100) => {}
            other => panic!("expected Return to reference on_true, got {other:?}"),
        }
    }

    #[test]
    fn identity_reshape_same_shape_aliases_operand() {
        let f64_ty = f64_scalar_ty();
        let mut m = Module::new(vec![FuncDef {
            name: "main".into(),
            is_public: true,
            params: vec![(ValueId(100), f64_ty.clone())],
            result_types: vec![f64_ty.clone()],
            body: vec![
                InstrResult {
                    values: vec![(ValueId(0), f64_ty)],
                    instr: Instruction::Reshape {
                        operand: ValueId(100),
                    },
                },
                InstrResult {
                    values: vec![],
                    instr: Instruction::Return {
                        operands: vec![ValueId(0)],
                    },
                },
            ],
            source_line: None,
        }]);
        let r = fold_module(&mut m);
        assert_eq!(
            r.counts_by_rule.get("identity_reshape_same_shape").copied(),
            Some(1)
        );
        let body = &m.functions[0].body;
        assert!(
            body.iter()
                .all(|ir| !matches!(ir.instr, Instruction::Reshape { .. }))
        );
    }

    // ----- End-to-end cascade test -----

    #[test]
    fn cascade_phase1_phase2_phase3() {
        // Transpose(BroadcastInDim(DenseScalar(3.0), 3x3)) then Select(true, x, _)
        // All three phases should cooperate to collapse this to a single splat constant.
        let tensor_3x3 = f64_tensor(vec![3, 3]);
        let mut m = mk_module(vec![
            mk_constant(0, 3.0),
            InstrResult {
                values: vec![(ValueId(1), tensor_3x3.clone())],
                instr: Instruction::BroadcastInDim {
                    operand: ValueId(0),
                    broadcast_dims: vec![],
                },
            },
            InstrResult {
                values: vec![(ValueId(2), tensor_3x3.clone())],
                instr: Instruction::Transpose {
                    operand: ValueId(1),
                    permutation: vec![1, 0],
                },
            },
            InstrResult {
                values: vec![(ValueId(3), TensorType::scalar(ElementType::I1))],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseScalar(ScalarValue::I1(true)),
                },
            },
            // A second input branch for the Select. Shape must match result_ty.
            InstrResult {
                values: vec![(ValueId(4), tensor_3x3.clone())],
                instr: Instruction::BroadcastInDim {
                    operand: ValueId(0),
                    broadcast_dims: vec![],
                },
            },
            InstrResult {
                values: vec![(ValueId(5), tensor_3x3)],
                instr: Instruction::Select {
                    cond: ValueId(3),
                    on_true: ValueId(2),
                    on_false: ValueId(4),
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(5)],
                },
            },
        ]);
        fold_module(&mut m);
        // After folding, the body should contain a single Constant DenseSplat
        // (the 3x3 splat of 3.0) plus a Return. Everything else collapsed.
        let body = &m.functions[0].body;
        let splat_count = body
            .iter()
            .filter(|ir| {
                matches!(
                    &ir.instr,
                    Instruction::Constant {
                        value: ConstantValue::DenseSplat(_, _),
                    }
                )
            })
            .count();
        assert!(
            splat_count >= 1,
            "expected at least one DenseSplat constant"
        );
        assert!(
            body.iter()
                .all(|ir| !matches!(ir.instr, Instruction::Transpose { .. }))
        );
        assert!(
            body.iter()
                .all(|ir| !matches!(ir.instr, Instruction::Select { .. }))
        );
    }

    #[test]
    fn reshape_dense_array_preserves_elements() {
        let arr = (0..6).map(|i| ScalarValue::F64(i as f64)).collect();
        let mut m = mk_module(vec![
            InstrResult {
                values: vec![(ValueId(0), f64_tensor(vec![6]))],
                instr: Instruction::Constant {
                    value: ConstantValue::DenseArray(arr),
                },
            },
            InstrResult {
                values: vec![(ValueId(1), f64_tensor(vec![2, 3]))],
                instr: Instruction::Reshape {
                    operand: ValueId(0),
                },
            },
            InstrResult {
                values: vec![],
                instr: Instruction::Return {
                    operands: vec![ValueId(1)],
                },
            },
        ]);
        fold_module(&mut m);
        // After folding the reshape becomes a DenseArray constant with the
        // same elements; the original is DCE'd.
        let body = &m.functions[0].body;
        match &body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseArray(arr),
            } => {
                assert_eq!(arr.len(), 6);
            }
            other => panic!("expected DenseArray constant, got {other:?}"),
        }
    }
}
