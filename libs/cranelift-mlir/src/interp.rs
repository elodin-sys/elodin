//! In-process StableHLO interpreter for display kernels.
//!
//! Used on wasm, where Cranelift JIT cannot emit executable pages. Native
//! `DisplayKernelExec` still compiles through Cranelift; this path is the
//! byte-buffer ABI for the same parsed IR.

use std::collections::HashMap;

use crate::ir::{
    CompareDirection, CompareType, ConstantValue, ElementType, InstrResult, Instruction, Module,
    ReduceOp, ScalarValue, TensorType, ValueId,
};
use crate::tensor_rt;

#[derive(Clone, Debug)]
struct Tensor {
    ty: TensorType,
    bytes: Vec<u8>,
}

impl Tensor {
    fn alloc(ty: TensorType) -> Self {
        let n = ty.byte_size();
        Self {
            ty,
            bytes: vec![0u8; n],
        }
    }

    fn from_bytes(ty: TensorType, bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() != ty.byte_size() {
            return Err(format!(
                "tensor is {} bytes, expected {}",
                bytes.len(),
                ty.byte_size()
            ));
        }
        Ok(Self { ty, bytes })
    }

    fn elem_count(&self) -> usize {
        self.ty.num_elements()
    }

    fn elem_size(&self) -> usize {
        self.ty.element_type.byte_size()
    }
}

pub fn interpret_main(
    module: &Module,
    inputs: &[&[u8]],
    outputs: &mut [&mut [u8]],
) -> Result<(), String> {
    let func = module
        .main_func()
        .ok_or_else(|| "compiled module is missing a callable main entrypoint".to_string())?;
    if inputs.len() != func.params.len() {
        return Err(format!(
            "expected {} inputs, got {}",
            func.params.len(),
            inputs.len()
        ));
    }
    if outputs.len() != func.result_types.len() {
        return Err(format!(
            "expected {} outputs, got {}",
            func.result_types.len(),
            outputs.len()
        ));
    }
    let mut env: HashMap<ValueId, Tensor> = HashMap::new();
    for ((id, ty), buf) in func.params.iter().zip(inputs) {
        env.insert(*id, Tensor::from_bytes(ty.clone(), buf.to_vec())?);
    }
    let returned = eval_body(module, &func.body, &mut env)?;
    if returned.len() != outputs.len() {
        return Err(format!(
            "kernel returned {} values, expected {}",
            returned.len(),
            outputs.len()
        ));
    }
    for (dst, src) in outputs.iter_mut().zip(returned) {
        if dst.len() != src.bytes.len() {
            return Err(format!(
                "output is {} bytes, expected {}",
                dst.len(),
                src.bytes.len()
            ));
        }
        dst.copy_from_slice(&src.bytes);
    }
    Ok(())
}

fn eval_body(
    module: &Module,
    body: &[InstrResult],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, String> {
    let mut last = Vec::new();
    for ir in body {
        match &ir.instr {
            Instruction::Return { operands } => {
                let mut out = Vec::with_capacity(operands.len());
                for id in operands {
                    out.push(lookup(env, *id)?);
                }
                return Ok(out);
            }
            _ => {
                let produced = eval_instr(module, &ir.instr, &ir.values, env)?;
                if produced.len() != ir.values.len() {
                    return Err(format!(
                        "instruction produced {} values, expected {}",
                        produced.len(),
                        ir.values.len()
                    ));
                }
                for ((id, ty), tensor) in ir.values.iter().zip(produced) {
                    // Some `chlo.*` and multiline `dot_general` printers
                    // record a scalar result type while the operand (and the
                    // function result) carry the real shape. The JIT follows
                    // the operand, so accept that wider tensor.
                    let shape_from_operand = ty.is_scalar()
                        && tensor.ty.element_type == ty.element_type
                        && tensor.elem_count() > ty.num_elements();
                    if tensor.ty != *ty && !shape_from_operand {
                        return Err(format!(
                            "value {id:?} has type {}, expected {ty}",
                            tensor.ty
                        ));
                    }
                    env.insert(*id, tensor);
                }
                last.clear();
                for (id, _) in &ir.values {
                    last.push(lookup(env, *id)?);
                }
            }
        }
    }
    Ok(last)
}

fn eval_instr(
    module: &Module,
    instr: &Instruction,
    results: &[(ValueId, TensorType)],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, String> {
    let result_ty = results
        .first()
        .map(|(_, ty)| ty.clone())
        .ok_or_else(|| "instruction has no result type".to_string())?;
    let out = match instr {
        Instruction::Constant { value } => materialize_constant(value, &result_ty)?,
        Instruction::Add { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Add)?,
        Instruction::Subtract { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Sub)?,
        Instruction::Multiply { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Mul)?,
        Instruction::Divide { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Div)?,
        Instruction::Remainder { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Rem)?,
        Instruction::Maximum { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Max)?,
        Instruction::Minimum { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Min)?,
        Instruction::Atan2 { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Atan2)?,
        Instruction::Power { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Pow)?,
        Instruction::Xor { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Xor)?,
        Instruction::Or { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Or)?,
        Instruction::And { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::And)?,
        Instruction::ShiftLeft { lhs, rhs } => binary(env, *lhs, *rhs, &result_ty, BinOp::Shl)?,
        Instruction::ShiftRightLogical { lhs, rhs } => {
            binary(env, *lhs, *rhs, &result_ty, BinOp::ShrL)?
        }
        Instruction::ShiftRightArithmetic { lhs, rhs } => {
            binary(env, *lhs, *rhs, &result_ty, BinOp::ShrA)?
        }
        Instruction::Negate { operand } => unary(env, *operand, &result_ty, UnOp::Neg)?,
        Instruction::Abs { operand } => unary(env, *operand, &result_ty, UnOp::Abs)?,
        Instruction::Sqrt { operand } => unary(env, *operand, &result_ty, UnOp::Sqrt)?,
        Instruction::Rsqrt { operand } => unary(env, *operand, &result_ty, UnOp::Rsqrt)?,
        Instruction::Floor { operand } => unary(env, *operand, &result_ty, UnOp::Floor)?,
        Instruction::Ceil { operand } => unary(env, *operand, &result_ty, UnOp::Ceil)?,
        Instruction::RoundNearestEven { operand } => {
            unary(env, *operand, &result_ty, UnOp::RoundNearestEven)?
        }
        Instruction::Sign { operand } => unary(env, *operand, &result_ty, UnOp::Sign)?,
        Instruction::Sine { operand } => unary(env, *operand, &result_ty, UnOp::Sin)?,
        Instruction::Cosine { operand } => unary(env, *operand, &result_ty, UnOp::Cos)?,
        Instruction::Tan { operand } => unary(env, *operand, &result_ty, UnOp::Tan)?,
        Instruction::Tanh { operand } => unary(env, *operand, &result_ty, UnOp::Tanh)?,
        Instruction::Sinh { operand } => unary(env, *operand, &result_ty, UnOp::Sinh)?,
        Instruction::Cosh { operand } => unary(env, *operand, &result_ty, UnOp::Cosh)?,
        Instruction::Asin { operand } => unary(env, *operand, &result_ty, UnOp::Asin)?,
        Instruction::Acos { operand } => unary(env, *operand, &result_ty, UnOp::Acos)?,
        Instruction::Atan { operand } => unary(env, *operand, &result_ty, UnOp::Atan)?,
        Instruction::Exponential { operand } => unary(env, *operand, &result_ty, UnOp::Exp)?,
        Instruction::Log { operand } => unary(env, *operand, &result_ty, UnOp::Log)?,
        Instruction::Log1p { operand } => unary(env, *operand, &result_ty, UnOp::Log1p)?,
        Instruction::Expm1 { operand } => unary(env, *operand, &result_ty, UnOp::Expm1)?,
        Instruction::Cbrt { operand } => unary(env, *operand, &result_ty, UnOp::Cbrt)?,
        Instruction::ErfInv { operand } => unary(env, *operand, &result_ty, UnOp::ErfInv)?,
        Instruction::Erfc { operand } => unary(env, *operand, &result_ty, UnOp::Erfc)?,
        Instruction::IsFinite { operand } => unary(env, *operand, &result_ty, UnOp::IsFinite)?,
        Instruction::Not { operand } => unary(env, *operand, &result_ty, UnOp::Not)?,
        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => compare(env, *lhs, *rhs, &result_ty, *direction, *compare_type)?,
        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => select(env, *cond, *on_true, *on_false, &result_ty)?,
        Instruction::Clamp { operand, min, max } => clamp(env, *operand, *min, *max, &result_ty)?,
        Instruction::Reshape { operand } => reshape(env, *operand, &result_ty)?,
        Instruction::BroadcastInDim {
            operand,
            broadcast_dims,
        } => broadcast(env, *operand, broadcast_dims, &result_ty)?,
        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => slice(env, *operand, start_indices, limit_indices, &result_ty)?,
        Instruction::Concatenate {
            operands,
            dimension,
        } => concatenate(env, operands, *dimension, &result_ty)?,
        Instruction::DotGeneral { lhs, rhs, dims } => {
            dot_general(env, *lhs, *rhs, dims, &result_ty)?
        }
        Instruction::Reduce {
            operand,
            init,
            op,
            dimensions,
        } => reduce(env, *operand, *init, *op, dimensions, &result_ty)?,
        Instruction::Convert { operand } => convert(env, *operand, &result_ty)?,
        Instruction::BitcastConvert { operand } => bitcast(env, *operand, &result_ty)?,
        Instruction::Iota { dimension } => iota(*dimension, &result_ty)?,
        Instruction::Transpose {
            operand,
            permutation,
        } => transpose(env, *operand, permutation, &result_ty)?,
        Instruction::Reverse {
            operand,
            dimensions,
        } => reverse(env, *operand, dimensions, &result_ty)?,
        Instruction::Pad {
            operand,
            padding_value,
            low,
            high: _,
            interior,
        } => pad(env, *operand, *padding_value, low, interior, &result_ty)?,
        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => dynamic_slice(env, *operand, start_indices, slice_sizes, &result_ty)?,
        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => dynamic_update_slice(env, *operand, *update, start_indices, &result_ty)?,
        Instruction::Gather {
            operand,
            indices,
            dims,
            slice_sizes,
        } => gather(env, *operand, *indices, dims, slice_sizes, &result_ty)?,
        Instruction::Call { callee, args } => {
            return call(module, callee, args, env);
        }
        Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
        } => {
            return eval_while(
                module,
                cond_body,
                loop_body,
                init_values,
                iter_arg_ids,
                results,
                env,
            );
        }
        Instruction::Case { index, branches } => {
            return eval_case(module, *index, branches, results, env);
        }
        Instruction::CustomCall { call_target, .. } => {
            return Err(format!("unsupported custom_call target: {call_target}"));
        }
        other => {
            return Err(format!(
                "display kernel interpreter does not implement {other:?}"
            ));
        }
    };
    Ok(vec![out])
}

fn call(
    module: &Module,
    callee: &str,
    args: &[ValueId],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, String> {
    let func = module
        .get_func(callee)
        .ok_or_else(|| format!("unknown callee {callee}"))?;
    if args.len() != func.params.len() {
        return Err(format!(
            "call {callee} expected {} args, got {}",
            func.params.len(),
            args.len()
        ));
    }
    let mut nested = HashMap::new();
    for ((id, ty), arg) in func.params.iter().zip(args) {
        let tensor = lookup(env, *arg)?;
        if tensor.ty != *ty {
            return Err(format!(
                "call {callee} arg type {} does not match {ty}",
                tensor.ty
            ));
        }
        nested.insert(*id, tensor);
    }
    eval_body(module, &func.body, &mut nested)
}

fn eval_while(
    module: &Module,
    cond_body: &[InstrResult],
    loop_body: &[InstrResult],
    init_values: &[ValueId],
    iter_arg_ids: &[ValueId],
    results: &[(ValueId, TensorType)],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, String> {
    if init_values.len() != iter_arg_ids.len() || iter_arg_ids.len() != results.len() {
        return Err("while loop arity mismatch".into());
    }
    let mut state: Vec<Tensor> = init_values
        .iter()
        .map(|id| lookup(env, *id))
        .collect::<Result<_, _>>()?;
    loop {
        let mut cond_env = env.clone();
        for (id, value) in iter_arg_ids.iter().zip(&state) {
            cond_env.insert(*id, value.clone());
        }
        let cond = eval_body(module, cond_body, &mut cond_env)?;
        let pred = cond
            .first()
            .ok_or_else(|| "while condition returned no value".to_string())?;
        if !is_true(pred)? {
            break;
        }
        let mut body_env = env.clone();
        for (id, value) in iter_arg_ids.iter().zip(&state) {
            body_env.insert(*id, value.clone());
        }
        state = eval_body(module, loop_body, &mut body_env)?;
        if state.len() != results.len() {
            return Err("while body arity mismatch".into());
        }
    }
    Ok(state)
}

fn eval_case(
    module: &Module,
    index: ValueId,
    branches: &[Vec<InstrResult>],
    results: &[(ValueId, TensorType)],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, String> {
    if branches.is_empty() {
        return Err("case has no branches".into());
    }
    let idx = lookup(env, index)?;
    let mut which = scalar_as_i64(&idx).unwrap_or(0);
    if which < 0 || which as usize >= branches.len() {
        which = (branches.len() - 1) as i64;
    }
    let mut nested = env.clone();
    let out = eval_body(module, &branches[which as usize], &mut nested)?;
    if out.len() != results.len() {
        return Err("case branch arity mismatch".into());
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Max,
    Min,
    Atan2,
    Pow,
    Xor,
    Or,
    And,
    Shl,
    ShrL,
    ShrA,
}

#[derive(Clone, Copy, Debug)]
enum UnOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Floor,
    Ceil,
    RoundNearestEven,
    Sign,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sinh,
    Cosh,
    Asin,
    Acos,
    Atan,
    Exp,
    Log,
    Log1p,
    Expm1,
    Cbrt,
    ErfInv,
    Erfc,
    IsFinite,
    Not,
}

fn binary(
    env: &HashMap<ValueId, Tensor>,
    lhs: ValueId,
    rhs: ValueId,
    result_ty: &TensorType,
    op: BinOp,
) -> Result<Tensor, String> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    if a.elem_count() != result_ty.num_elements() || b.elem_count() != result_ty.num_elements() {
        return Err("binary operands must already be broadcast to the result shape".into());
    }
    let mut out = Tensor::alloc(result_ty.clone());
    match result_ty.element_type {
        ElementType::F64 | ElementType::F32 => {
            if matches!(
                op,
                BinOp::Xor | BinOp::Or | BinOp::And | BinOp::Shl | BinOp::ShrL | BinOp::ShrA
            ) {
                return Err("integer bitwise op on float".into());
            }
            if result_ty.element_type == ElementType::F64 {
                zip_f64(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    BinOp::Rem => x % y,
                    BinOp::Max => x.max(y),
                    BinOp::Min => x.min(y),
                    BinOp::Atan2 => x.atan2(y),
                    BinOp::Pow => x.powf(y),
                    _ => x,
                });
            } else {
                zip_f32(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    BinOp::Rem => x % y,
                    BinOp::Max => x.max(y),
                    BinOp::Min => x.min(y),
                    BinOp::Atan2 => x.atan2(y),
                    BinOp::Pow => x.powf(y),
                    _ => x,
                });
            }
        }
        ElementType::I64 => zip_i64(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => {
                if y == 0 {
                    0
                } else {
                    x.wrapping_div(y)
                }
            }
            BinOp::Rem => {
                if y == 0 {
                    0
                } else {
                    x.wrapping_rem(y)
                }
            }
            BinOp::Max => x.max(y),
            BinOp::Min => x.min(y),
            BinOp::Xor => x ^ y,
            BinOp::Or => x | y,
            BinOp::And => x & y,
            BinOp::Shl => {
                if y >= 64 {
                    0
                } else {
                    x.wrapping_shl((y as u64).min(63) as u32)
                }
            }
            BinOp::ShrL => {
                if y >= 64 {
                    0
                } else {
                    ((x as u64).wrapping_shr((y as u64).min(63) as u32)) as i64
                }
            }
            BinOp::ShrA => {
                if y >= 64 {
                    x >> 63
                } else {
                    x.wrapping_shr((y as u64).min(63) as u32)
                }
            }
            BinOp::Atan2 | BinOp::Pow => x,
        }),
        ElementType::I32 => zip_i32(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => {
                if y == 0 {
                    0
                } else {
                    x.wrapping_div(y)
                }
            }
            BinOp::Rem => {
                if y == 0 {
                    0
                } else {
                    x.wrapping_rem(y)
                }
            }
            BinOp::Max => x.max(y),
            BinOp::Min => x.min(y),
            BinOp::Xor => x ^ y,
            BinOp::Or => x | y,
            BinOp::And => x & y,
            BinOp::Shl => {
                if y >= 32 {
                    0
                } else {
                    x.wrapping_shl((y as u32).min(31))
                }
            }
            BinOp::ShrL => {
                if y >= 32 {
                    0
                } else {
                    ((x as u32).wrapping_shr((y as u32).min(31))) as i32
                }
            }
            BinOp::ShrA => {
                if y >= 32 {
                    x >> 31
                } else {
                    x.wrapping_shr((y as u32).min(31))
                }
            }
            BinOp::Atan2 | BinOp::Pow => x,
        }),
        ElementType::UI64 => zip_u64(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => x.checked_div(y).unwrap_or(0),
            BinOp::Rem => x.checked_rem(y).unwrap_or(0),
            BinOp::Max => x.max(y),
            BinOp::Min => x.min(y),
            BinOp::Xor => x ^ y,
            BinOp::Or => x | y,
            BinOp::And => x & y,
            BinOp::Shl => {
                if y >= 64 {
                    0
                } else {
                    x.wrapping_shl(y as u32)
                }
            }
            BinOp::ShrL | BinOp::ShrA => {
                if y >= 64 {
                    0
                } else {
                    x.wrapping_shr(y as u32)
                }
            }
            BinOp::Atan2 | BinOp::Pow => x,
        }),
        ElementType::UI32 => zip_u32(&a.bytes, &b.bytes, &mut out.bytes, |x, y| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => x.checked_div(y).unwrap_or(0),
            BinOp::Rem => x.checked_rem(y).unwrap_or(0),
            BinOp::Max => x.max(y),
            BinOp::Min => x.min(y),
            BinOp::Xor => x ^ y,
            BinOp::Or => x | y,
            BinOp::And => x & y,
            BinOp::Shl => {
                if y >= 32 {
                    0
                } else {
                    x.wrapping_shl(y)
                }
            }
            BinOp::ShrL | BinOp::ShrA => {
                if y >= 32 {
                    0
                } else {
                    x.wrapping_shr(y)
                }
            }
            BinOp::Atan2 | BinOp::Pow => x,
        }),
        ElementType::I1 => {
            for (i, dst) in out.bytes.iter_mut().enumerate() {
                let x = a.bytes[i] != 0;
                let y = b.bytes[i] != 0;
                *dst = u8::from(match op {
                    BinOp::And => x && y,
                    BinOp::Or => x || y,
                    BinOp::Xor => x ^ y,
                    BinOp::Max => x || y,
                    BinOp::Min => x && y,
                    _ => return Err("unsupported i1 binary op".into()),
                });
            }
        }
    }
    Ok(out)
}

fn unary(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
    op: UnOp,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    // `chlo.*` text sometimes annotates the op as `tensor<f64> -> tensor<f64>`
    // while the operand (and the function result) is a vector. The JIT follows
    // the operand shape; do the same when the declared result is a scalar.
    let eval_ty = if result_ty.is_scalar()
        && !src.ty.is_scalar()
        && src.ty.element_type == result_ty.element_type
    {
        src.ty.clone()
    } else {
        result_ty.clone()
    };
    let mut out = Tensor::alloc(eval_ty.clone());
    match (op, eval_ty.element_type, src.ty.element_type) {
        (UnOp::IsFinite, ElementType::I1, ElementType::F64) => {
            for i in 0..out.bytes.len() {
                out.bytes[i] = u8::from(read_f64(&src.bytes, ElementType::F64, i).is_finite());
            }
        }
        (UnOp::IsFinite, ElementType::I1, ElementType::F32) => {
            for i in 0..out.bytes.len() {
                out.bytes[i] = u8::from(read_f64(&src.bytes, ElementType::F32, i).is_finite());
            }
        }
        (UnOp::Not, ElementType::I1, ElementType::I1) => {
            for (dst, &v) in out.bytes.iter_mut().zip(&src.bytes) {
                *dst = u8::from(v == 0);
            }
        }
        (UnOp::Not, ElementType::I64, ElementType::I64) => {
            zip_unary_i64(&src.bytes, &mut out.bytes, |x| !x);
        }
        (UnOp::Not, ElementType::I32, ElementType::I32) => {
            zip_unary_i32(&src.bytes, &mut out.bytes, |x| !x);
        }
        (_, ElementType::F64, ElementType::F64) => {
            if matches!(op, UnOp::ErfInv) {
                return Err("erfinv is not implemented in the display kernel interpreter".into());
            }
            zip_unary_f64(&src.bytes, &mut out.bytes, |a| float_unary(op, a));
        }
        (_, ElementType::F32, ElementType::F32) => {
            if matches!(op, UnOp::ErfInv) {
                return Err("erfinv is not implemented in the display kernel interpreter".into());
            }
            zip_unary_f32(&src.bytes, &mut out.bytes, |a| {
                float_unary(op, a as f64) as f32
            });
        }
        (UnOp::Neg, ElementType::I64, ElementType::I64) => {
            zip_unary_i64(&src.bytes, &mut out.bytes, |x| x.wrapping_neg());
        }
        (UnOp::Abs, ElementType::I64, ElementType::I64) => {
            zip_unary_i64(&src.bytes, &mut out.bytes, |x| x.wrapping_abs());
        }
        (UnOp::Neg, ElementType::I32, ElementType::I32) => {
            zip_unary_i32(&src.bytes, &mut out.bytes, |x| x.wrapping_neg());
        }
        (UnOp::Abs, ElementType::I32, ElementType::I32) => {
            zip_unary_i32(&src.bytes, &mut out.bytes, |x| x.wrapping_abs());
        }
        _ => {
            return Err(format!(
                "unary {op:?} is not implemented for {:?}",
                result_ty.element_type
            ));
        }
    }
    Ok(out)
}

fn float_unary(op: UnOp, a: f64) -> f64 {
    match op {
        UnOp::Neg => -a,
        UnOp::Abs => a.abs(),
        UnOp::Sqrt => a.sqrt(),
        UnOp::Rsqrt => 1.0 / a.sqrt(),
        UnOp::Floor => a.floor(),
        UnOp::Ceil => a.ceil(),
        UnOp::RoundNearestEven => a.round_ties_even(),
        UnOp::Sign => {
            if a == 0.0 {
                0.0
            } else {
                a.signum()
            }
        }
        UnOp::Sin => a.sin(),
        UnOp::Cos => a.cos(),
        UnOp::Tan => a.tan(),
        UnOp::Tanh => a.tanh(),
        UnOp::Sinh => a.sinh(),
        UnOp::Cosh => a.cosh(),
        UnOp::Asin => a.asin(),
        UnOp::Acos => a.acos(),
        UnOp::Atan => a.atan(),
        UnOp::Exp => a.exp(),
        UnOp::Log => a.ln(),
        UnOp::Log1p => a.ln_1p(),
        UnOp::Expm1 => a.exp_m1(),
        UnOp::Cbrt => a.cbrt(),
        UnOp::Erfc => tensor_rt::erfc_impl(a),
        UnOp::ErfInv | UnOp::IsFinite | UnOp::Not => a,
    }
}

fn compare(
    env: &HashMap<ValueId, Tensor>,
    lhs: ValueId,
    rhs: ValueId,
    result_ty: &TensorType,
    direction: CompareDirection,
    compare_type: CompareType,
) -> Result<Tensor, String> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    if a.ty.element_type != b.ty.element_type {
        return Err("compare operands have different element types".into());
    }
    if result_ty.element_type != ElementType::I1 {
        return Err("compare result must be i1".into());
    }
    let a_map = broadcast_map(&result_ty.shape, &a.ty.shape)?;
    let b_map = broadcast_map(&result_ty.shape, &b.ty.shape)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n = result_ty.num_elements();
    if out.bytes.len() != n {
        return Err("compare result must be i1".into());
    }
    match a.ty.element_type {
        ElementType::F64 | ElementType::F32 => {
            for i in 0..n {
                let x = read_f64(&a.bytes, a.ty.element_type, mapped_index(a, &a_map, i)?);
                let y = read_f64(&b.bytes, b.ty.element_type, mapped_index(b, &b_map, i)?);
                out.bytes[i] = u8::from(cmp_f64(x, y, direction, compare_type));
            }
        }
        ElementType::I64 | ElementType::I32 | ElementType::UI64 | ElementType::UI32 => {
            for i in 0..n {
                let x = read_int(&a.bytes, a.ty.element_type, mapped_index(a, &a_map, i)?);
                let y = read_int(&b.bytes, b.ty.element_type, mapped_index(b, &b_map, i)?);
                out.bytes[i] = u8::from(cmp_i64(x, y, direction));
            }
        }
        ElementType::I1 => {
            for i in 0..n {
                let x = i64::from(a.bytes[mapped_index(a, &a_map, i)?] != 0);
                let y = i64::from(b.bytes[mapped_index(b, &b_map, i)?] != 0);
                out.bytes[i] = u8::from(cmp_i64(x, y, direction));
            }
        }
    }
    Ok(out)
}

fn select(
    env: &HashMap<ValueId, Tensor>,
    cond: ValueId,
    on_true: ValueId,
    on_false: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let pred = lookup_ref(env, cond)?;
    let t = lookup_ref(env, on_true)?;
    let f = lookup_ref(env, on_false)?;
    if t.ty.element_type != result_ty.element_type || f.ty.element_type != result_ty.element_type {
        return Err("select operand element type does not match the result".into());
    }
    let pred_map = broadcast_map(&result_ty.shape, &pred.ty.shape)?;
    let true_map = broadcast_map(&result_ty.shape, &t.ty.shape)?;
    let false_map = broadcast_map(&result_ty.shape, &f.ty.shape)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let elem = result_ty.element_type.byte_size();
    for i in 0..result_ty.num_elements() {
        let (src, map) = if predicate_true(pred, pred_map.index(i))? {
            (t, &true_map)
        } else {
            (f, &false_map)
        };
        let src_off = map.index(i) * elem;
        let dst_off = i * elem;
        let end = src_off + elem;
        if end > src.bytes.len() {
            return Err("select operand is smaller than the result".into());
        }
        out.bytes[dst_off..dst_off + elem].copy_from_slice(&src.bytes[src_off..end]);
    }
    Ok(out)
}

fn predicate_true(pred: &Tensor, index: usize) -> Result<bool, String> {
    let elem = pred.elem_size();
    let off = index * elem;
    if off + elem > pred.bytes.len() {
        return Err("select predicate is out of range".into());
    }
    Ok(pred.bytes[off..off + elem].iter().any(|byte| *byte != 0))
}

struct BroadcastMap {
    dst_rank: usize,
    src_rank: usize,
    dst_strides: Vec<usize>,
    src_strides: Vec<usize>,
    src_shape: Vec<i64>,
    dst_shape: Vec<i64>,
}

fn broadcast_map(dst_shape: &[i64], src_shape: &[i64]) -> Result<BroadcastMap, String> {
    if src_shape.len() > dst_shape.len() {
        return Err("operand rank exceeds the result".into());
    }
    let offset = dst_shape.len() - src_shape.len();
    for (i, &dim) in src_shape.iter().enumerate() {
        let dst_dim = dst_shape[offset + i];
        if dim != dst_dim && dim != 1 {
            return Err(format!("cannot broadcast dimension {dim} onto {dst_dim}"));
        }
    }
    Ok(BroadcastMap {
        dst_rank: dst_shape.len(),
        src_rank: src_shape.len(),
        dst_strides: shape_strides(dst_shape),
        src_strides: shape_strides(src_shape),
        src_shape: src_shape.to_vec(),
        dst_shape: dst_shape.to_vec(),
    })
}

fn shape_strides(shape: &[i64]) -> Vec<usize> {
    let rank = shape.len();
    let mut out = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out[i] = out[i + 1].saturating_mul(shape[i + 1].max(0) as usize);
    }
    out
}

fn mapped_index(tensor: &Tensor, map: &BroadcastMap, flat: usize) -> Result<usize, String> {
    let index = map.index(flat);
    let size = tensor.elem_size();
    let off = index
        .checked_mul(size)
        .ok_or_else(|| "operand index is out of range".to_string())?;
    let end = off
        .checked_add(size)
        .ok_or_else(|| "operand index is out of range".to_string())?;
    if end > tensor.bytes.len() {
        return Err("operand index is out of range".into());
    }
    Ok(index)
}

impl BroadcastMap {
    fn index(&self, flat_dst: usize) -> usize {
        if self.dst_rank == 0 || self.src_rank == 0 {
            return 0;
        }
        let offset = self.dst_rank - self.src_rank;
        let mut rem = flat_dst;
        let mut src_index = 0usize;
        for d in 0..self.dst_rank {
            let stride = self.dst_strides[d].max(1);
            let coord = rem / stride;
            rem %= stride;
            if d >= offset {
                let s = d - offset;
                if self.src_shape[s] == self.dst_shape[d] {
                    src_index += coord * self.src_strides[s];
                }
            }
        }
        src_index
    }
}

fn clamp(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    min: ValueId,
    max: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let lo = lookup_ref(env, min)?;
    let hi = lookup_ref(env, max)?;
    if result_ty.element_type != ElementType::F64
        || src.ty.element_type != ElementType::F64
        || lo.ty.element_type != ElementType::F64
        || hi.ty.element_type != ElementType::F64
    {
        return Err("clamp is only implemented for f64".into());
    }
    let src_map = broadcast_map(&result_ty.shape, &src.ty.shape)?;
    let lo_map = broadcast_map(&result_ty.shape, &lo.ty.shape)?;
    let hi_map = broadcast_map(&result_ty.shape, &hi.ty.shape)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n = result_ty.num_elements();
    for i in 0..n {
        let v = read_f64(
            &src.bytes,
            ElementType::F64,
            mapped_index(src, &src_map, i)?,
        );
        let lo_v = read_f64(&lo.bytes, ElementType::F64, mapped_index(lo, &lo_map, i)?);
        let hi_v = read_f64(&hi.bytes, ElementType::F64, mapped_index(hi, &hi_map, i)?);
        write_f64(&mut out.bytes, ElementType::F64, i, v.clamp(lo_v, hi_v));
    }
    Ok(out)
}

fn reshape(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup(env, operand)?;
    if src.bytes.len() != result_ty.byte_size() {
        return Err("reshape changed the number of bytes".into());
    }
    Ok(Tensor {
        ty: result_ty.clone(),
        bytes: src.bytes,
    })
}

fn broadcast(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    broadcast_dims: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let dst_shape = result_ty.shape.clone();
    let src_shape = src.ty.shape.clone();
    tensor_rt::tensor_broadcast_nd_generic(
        out.bytes.as_mut_ptr(),
        src.bytes.as_ptr(),
        out.elem_count(),
        src.elem_count(),
        dst_shape.as_ptr(),
        dst_shape.len(),
        src_shape.as_ptr(),
        src_shape.len(),
        broadcast_dims.as_ptr(),
        src.elem_size(),
    );
    Ok(out)
}

fn slice(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    starts: &[i64],
    limits: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let shape = src.ty.shape.clone();
    tensor_rt::tensor_slice_generic(
        out.bytes.as_mut_ptr(),
        src.bytes.as_ptr(),
        out.elem_count(),
        src.elem_count(),
        shape.as_ptr(),
        shape.len(),
        starts.as_ptr(),
        limits.as_ptr(),
        src.elem_size(),
    );
    Ok(out)
}

fn concatenate(
    env: &HashMap<ValueId, Tensor>,
    operands: &[ValueId],
    dimension: i64,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    if operands.is_empty() {
        return Err("concatenate has no operands".into());
    }
    let mut acc = lookup(env, operands[0])?;
    for id in &operands[1..] {
        let next = lookup_ref(env, *id)?;
        let mut dst = Tensor::alloc(concat_ty(&acc.ty, &next.ty, dimension)?);
        let dim = dimension as usize;
        tensor_rt::tensor_concat_nd_f64(
            dst.bytes.as_mut_ptr(),
            dst.elem_count(),
            acc.bytes.as_ptr(),
            acc.elem_count(),
            next.bytes.as_ptr(),
            next.elem_count(),
            dst.ty.shape.as_ptr(),
            acc.ty.shape.as_ptr(),
            dst.ty.rank(),
            dim,
            acc.elem_size(),
        );
        acc = dst;
    }
    acc.ty = result_ty.clone();
    Ok(acc)
}

fn concat_ty(a: &TensorType, b: &TensorType, dim: i64) -> Result<TensorType, String> {
    if a.rank() != b.rank() || a.element_type != b.element_type {
        return Err("concatenate operand types disagree".into());
    }
    let d = dim as usize;
    let mut shape = a.shape.clone();
    shape[d] = a.shape[d] + b.shape[d];
    Ok(TensorType {
        shape,
        element_type: a.element_type,
    })
}

fn dot_general(
    env: &HashMap<ValueId, Tensor>,
    lhs: ValueId,
    rhs: ValueId,
    dims: &crate::ir::DotDims,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    expect_ty(a, ElementType::F64)?;
    expect_ty(b, ElementType::F64)?;
    let vals = dot_values(&a.ty, &a.bytes, &b.ty, &b.bytes, dims)?;
    // A multiline `stablehlo.dot_general` can leave the result type as a
    // scalar when the real shape is on the following line. Keep the declared
    // type when it already has the right number of elements.
    let out_ty =
        if result_ty.element_type == ElementType::F64 && result_ty.num_elements() == vals.len() {
            result_ty.clone()
        } else if vals.len() == 1 {
            TensorType::scalar(ElementType::F64)
        } else {
            TensorType {
                shape: vec![vals.len() as i64],
                element_type: ElementType::F64,
            }
        };
    let mut out = Tensor::alloc(out_ty);
    for (i, v) in vals.iter().enumerate() {
        write_f64(&mut out.bytes, ElementType::F64, i, *v);
    }
    Ok(out)
}

fn dot_values(
    l_ty: &TensorType,
    l: &[u8],
    r_ty: &TensorType,
    r: &[u8],
    dims: &crate::ir::DotDims,
) -> Result<Vec<f64>, String> {
    let at = |bytes: &[u8], i: usize| read_f64(bytes, ElementType::F64, i);
    if dims.lhs_batch == [0] && dims.rhs_batch == [0] && l_ty.rank() == 2 && r_ty.rank() == 2 {
        let batch = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        if r_ty.shape[0] as usize != batch || r_ty.shape[1] as usize != k {
            return Err("dot_general batch shapes do not match".into());
        }
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut acc = 0.0;
            for i in 0..k {
                acc += at(l, b * k + i) * at(r, b * k + i);
            }
            out.push(acc);
        }
        return Ok(out);
    }
    if l_ty.is_scalar() && r_ty.is_scalar() {
        return Ok(vec![at(l, 0) * at(r, 0)]);
    }
    // One in-range contracting axis. Empty or out-of-range dims fall through
    // to the not-implemented error instead of guessing a layout.
    let contract = |axes: &[i64], rank: usize| -> Option<usize> {
        match axes {
            [d] if *d >= 0 && (*d as usize) < rank => Some(*d as usize),
            _ => None,
        }
    };
    let unbatched = dims.lhs_batch.is_empty() && dims.rhs_batch.is_empty();
    if unbatched
        && l_ty.rank() == 1
        && r_ty.rank() == 1
        && contract(&dims.lhs_contracting, 1) == Some(0)
        && contract(&dims.rhs_contracting, 1) == Some(0)
    {
        let n = l_ty.shape[0] as usize;
        if r_ty.shape[0] as usize != n {
            return Err(InterpError::DotContractingSize);
        }
        let mut acc = 0.0;
        for i in 0..n {
            acc += at(l, i) * at(r, i);
        }
        return Ok(vec![acc]);
    }
    if unbatched
        && l_ty.rank() == 1
        && r_ty.rank() == 2
        && contract(&dims.lhs_contracting, 1) == Some(0)
        && let Some(rhs_contract_dim) = contract(&dims.rhs_contracting, 2)
    {
        let k = l_ty.shape[0] as usize;
        let rows = r_ty.shape[0] as usize;
        let cols = r_ty.shape[1] as usize;
        let (contracted, out_size) = if rhs_contract_dim == 1 {
            (cols, rows)
        } else {
            (rows, cols)
        };
        if contracted != k {
            return Err(InterpError::DotContractingSize);
        }
        let mut out = Vec::with_capacity(out_size);
        for i in 0..out_size {
            let mut acc = 0.0;
            for j in 0..k {
                let rv = if rhs_contract_dim == 1 {
                    at(r, i * cols + j)
                } else {
                    at(r, j * cols + i)
                };
                acc += at(l, j) * rv;
            }
            out.push(acc);
        }
        return Ok(out);
    }
    if unbatched
        && l_ty.rank() == 2
        && r_ty.rank() == 1
        && let (Some(lhs_contract_dim), Some(0)) = (
            contract(&dims.lhs_contracting, 2),
            contract(&dims.rhs_contracting, 1),
        )
    {
        let rows = l_ty.shape[0] as usize;
        let cols = l_ty.shape[1] as usize;
        let k = r_ty.shape[0] as usize;
        if lhs_contract_dim == 1 {
            if cols != k {
                return Err(InterpError::DotContractingSize);
            }
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                let mut acc = 0.0;
                for col in 0..cols {
                    acc += at(l, row * cols + col) * at(r, col);
                }
                out.push(acc);
            }
            return Ok(out);
        }
        if rows != k {
            return Err(InterpError::DotContractingSize);
        }
        let mut out = Vec::with_capacity(cols);
        for col in 0..cols {
            let mut acc = 0.0;
            for row in 0..rows {
                acc += at(l, row * cols + col) * at(r, row);
            }
            out.push(acc);
        }
        return Ok(out);
    }
    if l_ty.rank() == 2
        && r_ty.rank() == 2
        && dims.lhs_batch.is_empty()
        && dims.rhs_batch.is_empty()
        && (dims.lhs_contracting.is_empty() || dims.lhs_contracting == [1])
        && (dims.rhs_contracting.is_empty() || dims.rhs_contracting == [0])
    {
        let m = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        let n_cols = r_ty.shape[1] as usize;
        if r_ty.shape[0] as usize != k {
            return Err("dot_general contracting size does not match".into());
        }
        let mut out = vec![0.0; m * n_cols];
        for row in 0..m {
            for col in 0..n_cols {
                let mut acc = 0.0;
                for i in 0..k {
                    acc += at(l, row * k + i) * at(r, i * n_cols + col);
                }
                out[row * n_cols + col] = acc;
            }
        }
        return Ok(out);
    }
    Err("dot_general batch/contracting dims are not implemented in the interpreter".into())
}

fn reduce(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    init: ValueId,
    op: ReduceOp,
    dimensions: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let init_tensor = lookup_ref(env, init)?;
    if dimensions.len() != 1 {
        return Err("multi-dimension reduce is not implemented".into());
    }
    let dim = dimensions[0] as usize;
    if dim + 1 != src.ty.rank() {
        return Err("reduce is only implemented on the last axis".into());
    }
    let inner = src.ty.shape[dim] as usize;
    let outer = src.elem_count() / inner.max(1);
    match (op, src.ty.element_type) {
        (ReduceOp::Add, ElementType::F64) => {
            let mut out = Tensor::alloc(result_ty.clone());
            tensor_rt::tensor_reduce_sum_f64(
                out.bytes.as_mut_ptr().cast(),
                src.bytes.as_ptr().cast(),
                outer,
                inner,
            );
            Ok(out)
        }
        (ReduceOp::Maximum, ElementType::F64) => {
            let mut out = Tensor::alloc(result_ty.clone());
            tensor_rt::tensor_reduce_max_f64(
                out.bytes.as_mut_ptr().cast(),
                src.bytes.as_ptr().cast(),
                outer,
                inner,
            );
            Ok(out)
        }
        (ReduceOp::Minimum, ElementType::F64) => {
            let mut out = Tensor::alloc(result_ty.clone());
            tensor_rt::tensor_reduce_min_f64(
                out.bytes.as_mut_ptr().cast(),
                src.bytes.as_ptr().cast(),
                outer,
                inner,
            );
            Ok(out)
        }
        (
            ReduceOp::Add,
            ElementType::I64 | ElementType::I32 | ElementType::UI64 | ElementType::UI32,
        ) => {
            let init = read_int(&init_tensor.bytes, init_tensor.ty.element_type, 0);
            Ok(reduce_last_axis_int(src, init, result_ty, |acc, v| {
                acc.wrapping_add(v)
            }))
        }
        (ReduceOp::And, ElementType::I1) => {
            let init = read_int(&init_tensor.bytes, init_tensor.ty.element_type, 0);
            Ok(reduce_last_axis_int(src, init, result_ty, |acc, v| {
                i64::from(acc != 0 && v != 0)
            }))
        }
        (ReduceOp::Or, ElementType::I1) => {
            let init = read_int(&init_tensor.bytes, init_tensor.ty.element_type, 0);
            Ok(reduce_last_axis_int(src, init, result_ty, |acc, v| {
                i64::from(acc != 0 || v != 0)
            }))
        }
        _ => Err(format!(
            "reduce {op:?} is not implemented for {:?}",
            src.ty.element_type
        )),
    }
}

fn reduce_last_axis_int(
    src: &Tensor,
    init: i64,
    result_ty: &TensorType,
    op: impl Fn(i64, i64) -> i64,
) -> Tensor {
    let inner = src.ty.shape.last().copied().unwrap_or(1).max(1) as usize;
    let outer = src.elem_count() / inner;
    let mut out = Tensor::alloc(result_ty.clone());
    for group in 0..outer {
        let mut acc = init;
        for i in 0..inner {
            acc = op(
                acc,
                read_int(&src.bytes, src.ty.element_type, group * inner + i),
            );
        }
        write_int(&mut out.bytes, result_ty.element_type, group, acc);
    }
    out
}

fn convert(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n = result_ty.num_elements();
    for i in 0..n {
        let v = read_f64(&src.bytes, src.ty.element_type, i);
        write_f64(&mut out.bytes, result_ty.element_type, i, v);
    }
    Ok(out)
}

fn bitcast(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup(env, operand)?;
    if src.bytes.len() != result_ty.byte_size() {
        return Err("bitcast convert changed the number of bytes".into());
    }
    Ok(Tensor {
        ty: result_ty.clone(),
        bytes: src.bytes,
    })
}

fn iota(dimension: i64, result_ty: &TensorType) -> Result<Tensor, String> {
    let mut out = Tensor::alloc(result_ty.clone());
    match result_ty.element_type {
        ElementType::I64 => tensor_rt::tensor_iota_nd_i64(
            out.bytes.as_mut_ptr().cast(),
            out.elem_count(),
            result_ty.shape.as_ptr(),
            result_ty.rank(),
            dimension as usize,
        ),
        ElementType::F64 => tensor_rt::tensor_iota_nd_f64(
            out.bytes.as_mut_ptr().cast(),
            out.elem_count(),
            result_ty.shape.as_ptr(),
            result_ty.rank(),
            dimension as usize,
        ),
        ElementType::I32 | ElementType::UI32 => {
            let strides = shape_strides(&result_ty.shape);
            let dim = dimension as usize;
            let n = out.elem_count();
            for i in 0..n {
                let coord = if result_ty.shape.is_empty() {
                    0
                } else {
                    (i / strides[dim]) as i64 % result_ty.shape[dim]
                };
                write_int(&mut out.bytes, result_ty.element_type, i, coord);
            }
        }
        _ => return Err("iota is only implemented for i64, i32, ui32, and f64".into()),
    }
    Ok(out)
}

fn transpose(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    permutation: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let mut out = Tensor::alloc(result_ty.clone());
    tensor_rt::tensor_transpose_nd_generic(
        out.bytes.as_mut_ptr(),
        src.bytes.as_ptr(),
        src.elem_count(),
        src.ty.shape.as_ptr(),
        permutation.as_ptr(),
        src.ty.rank(),
        src.elem_size(),
    );
    Ok(out)
}

fn reverse(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    dimensions: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    expect_ty(src, ElementType::F64)?;
    let mut out = Tensor::alloc(result_ty.clone());
    tensor_rt::tensor_reverse_f64(
        out.bytes.as_mut_ptr().cast(),
        src.bytes.as_ptr().cast(),
        src.elem_count(),
        src.ty.shape.as_ptr(),
        src.ty.rank(),
        dimensions.as_ptr(),
        dimensions.len(),
    );
    Ok(out)
}

fn pad(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    padding_value: ValueId,
    low: &[i64],
    interior: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    if interior.iter().any(|&v| v != 0) {
        return Err("interior padding is not implemented".into());
    }
    let src = lookup_ref(env, operand)?;
    expect_ty(src, ElementType::F64)?;
    let pad = lookup_ref(env, padding_value)?;
    let pad_value = read_f64(&pad.bytes, ElementType::F64, 0);
    let mut out = Tensor::alloc(result_ty.clone());
    tensor_rt::tensor_pad_f64(
        out.bytes.as_mut_ptr().cast(),
        src.bytes.as_ptr().cast(),
        out.elem_count(),
        src.elem_count(),
        pad_value,
        result_ty.shape.as_ptr(),
        src.ty.shape.as_ptr(),
        src.ty.rank(),
        low.as_ptr(),
    );
    Ok(out)
}

fn dynamic_slice(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    start_indices: &[ValueId],
    slice_sizes: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let mut starts = Vec::with_capacity(start_indices.len());
    for id in start_indices {
        starts.push(scalar_as_i64(&lookup(env, *id)?)?);
    }
    let mut out = Tensor::alloc(result_ty.clone());
    tensor_rt::tensor_dynamic_slice_generic(
        out.bytes.as_mut_ptr(),
        src.bytes.as_ptr(),
        out.elem_count(),
        src.elem_count(),
        src.ty.shape.as_ptr(),
        src.ty.rank(),
        starts.as_ptr(),
        slice_sizes.as_ptr(),
        src.elem_size(),
    );
    Ok(out)
}

fn dynamic_update_slice(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    update: ValueId,
    start_indices: &[ValueId],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let upd = lookup_ref(env, update)?;
    let mut starts = Vec::with_capacity(start_indices.len());
    for id in start_indices {
        starts.push(scalar_as_i64(&lookup(env, *id)?)?);
    }
    let mut out = Tensor::alloc(result_ty.clone());
    tensor_rt::tensor_dynamic_update_slice_generic(
        out.bytes.as_mut_ptr(),
        src.bytes.as_ptr(),
        upd.bytes.as_ptr(),
        src.elem_count(),
        upd.elem_count(),
        src.ty.shape.as_ptr(),
        src.ty.rank(),
        starts.as_ptr(),
        upd.ty.shape.as_ptr(),
        src.elem_size(),
    );
    Ok(out)
}

fn gather(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    indices: ValueId,
    dims: &crate::ir::GatherDims,
    slice_sizes: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, String> {
    let src = lookup_ref(env, operand)?;
    let idx = lookup_ref(env, indices)?;
    let src_rank = src.ty.rank();
    let idx_rank = idx.ty.rank();
    let index_vector_dim = dims.index_vector_dim.max(0) as usize;
    let n_index = if index_vector_dim < idx_rank {
        idx.ty.shape[index_vector_dim] as usize
    } else {
        1
    };
    let batch_idx_dims: Vec<usize> = (0..idx_rank).filter(|&d| d != index_vector_dim).collect();
    let offset_dims: Vec<usize> = dims.offset_dims.iter().map(|&d| d as usize).collect();
    let collapsed: Vec<usize> = dims
        .collapsed_slice_dims
        .iter()
        .map(|&d| d as usize)
        .collect();
    let slice_dims_kept: Vec<usize> = (0..src_rank).filter(|d| !collapsed.contains(d)).collect();
    let out_strides = shape_strides(&result_ty.shape);
    let src_strides = shape_strides(&src.ty.shape);
    let idx_strides = shape_strides(&idx.ty.shape);
    let elem = src.elem_size();
    let mut out = Tensor::alloc(result_ty.clone());
    let n_out = result_ty.num_elements();
    let out_rank = result_ty.rank();
    for flat in 0..n_out {
        let mut coords = vec![0usize; out_rank];
        let mut rem = flat;
        for (d, coord) in coords.iter_mut().enumerate() {
            *coord = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let mut idx_coords = vec![0usize; idx_rank];
        let mut batch_i = 0;
        for (d, &coord) in coords.iter().enumerate() {
            if offset_dims.contains(&d) {
                continue;
            }
            if batch_i < batch_idx_dims.len() {
                idx_coords[batch_idx_dims[batch_i]] = coord;
                batch_i += 1;
            }
        }
        let mut start = vec![0i64; src_rank];
        for j in 0..n_index {
            if index_vector_dim < idx_rank {
                idx_coords[index_vector_dim] = j;
            }
            let flat_idx = if idx_rank == 0 {
                0
            } else {
                idx_coords
                    .iter()
                    .enumerate()
                    .map(|(d, &c)| c * idx_strides[d])
                    .sum()
            };
            let raw = read_int(&idx.bytes, idx.ty.element_type, flat_idx);
            let dim = dims.start_index_map.get(j).copied().unwrap_or(0) as usize;
            if dim < src_rank {
                let slice = slice_sizes.get(dim).copied().unwrap_or(1);
                let max_start = (src.ty.shape[dim] - slice).max(0);
                start[dim] = raw.clamp(0, max_start);
            }
        }
        let mut op_coord = start;
        for (k, &op_dim) in slice_dims_kept.iter().enumerate() {
            let off = offset_dims.get(k).map(|&d| coords[d] as i64).unwrap_or(0);
            op_coord[op_dim] += off;
        }
        let src_flat = if src_rank == 0 {
            0
        } else {
            (0..src_rank)
                .map(|d| op_coord[d].max(0) as usize * src_strides[d])
                .sum::<usize>()
                .min(src.elem_count().saturating_sub(1))
        };
        let db = flat * elem;
        let sb = src_flat * elem;
        out.bytes[db..db + elem].copy_from_slice(&src.bytes[sb..sb + elem]);
    }
    Ok(out)
}

fn materialize_constant(value: &ConstantValue, ty: &TensorType) -> Result<Tensor, String> {
    let mut out = Tensor::alloc(ty.clone());
    match value {
        ConstantValue::DenseScalar(scalar) => {
            write_scalar(&mut out.bytes, ty.element_type, 0, scalar)?;
        }
        ConstantValue::DenseSplat(scalar, splat_ty) => {
            let n = ty.num_elements();
            if splat_ty.num_elements() > n {
                return Err(format!(
                    "splat constant has {} elements but the result has {n}",
                    splat_ty.num_elements()
                ));
            }
            for i in 0..n {
                write_scalar(&mut out.bytes, ty.element_type, i, scalar)?;
            }
        }
        ConstantValue::DenseArray(values) => {
            if values.len() > ty.num_elements() {
                return Err("dense array is larger than its result type".into());
            }
            for (i, scalar) in values.iter().enumerate() {
                write_scalar(&mut out.bytes, ty.element_type, i, scalar)?;
            }
        }
        ConstantValue::DenseExternal {
            elem_type,
            byte_len,
            data,
        } => {
            if *elem_type != ty.element_type || *byte_len != out.bytes.len() {
                return Err("external constant type/size mismatch".into());
            }
            out.bytes.copy_from_slice(data.as_bytes());
        }
    }
    Ok(out)
}

fn lookup(env: &HashMap<ValueId, Tensor>, id: ValueId) -> Result<Tensor, String> {
    env.get(&id)
        .cloned()
        .ok_or_else(|| format!("unknown value {id:?}"))
}

fn lookup_ref(env: &HashMap<ValueId, Tensor>, id: ValueId) -> Result<&Tensor, String> {
    env.get(&id).ok_or_else(|| format!("unknown value {id:?}"))
}

fn expect_ty(tensor: &Tensor, ty: ElementType) -> Result<(), String> {
    if tensor.ty.element_type == ty {
        Ok(())
    } else {
        Err(format!("expected {ty:?}, got {:?}", tensor.ty.element_type))
    }
}

fn is_true(tensor: &Tensor) -> Result<bool, String> {
    if tensor.bytes.is_empty() {
        return Err("empty predicate".into());
    }
    Ok(match tensor.ty.element_type {
        ElementType::I1 => tensor.bytes[0] != 0,
        ElementType::F64 => read_f64(&tensor.bytes, ElementType::F64, 0) != 0.0,
        _ => read_int(&tensor.bytes, tensor.ty.element_type, 0) != 0,
    })
}

fn scalar_as_i64(tensor: &Tensor) -> Result<i64, String> {
    Ok(read_int(&tensor.bytes, tensor.ty.element_type, 0))
}

fn cmp_f64(a: f64, b: f64, direction: CompareDirection, compare_type: CompareType) -> bool {
    match compare_type {
        CompareType::TotalOrder => match direction {
            CompareDirection::Eq => a.to_bits() == b.to_bits(),
            CompareDirection::Ne => a.to_bits() != b.to_bits(),
            CompareDirection::Lt => a.total_cmp(&b).is_lt(),
            CompareDirection::Le => a.total_cmp(&b).is_le(),
            CompareDirection::Gt => a.total_cmp(&b).is_gt(),
            CompareDirection::Ge => a.total_cmp(&b).is_ge(),
        },
        _ => match direction {
            CompareDirection::Eq => a == b,
            CompareDirection::Ne => a != b,
            CompareDirection::Lt => a < b,
            CompareDirection::Le => a <= b,
            CompareDirection::Gt => a > b,
            CompareDirection::Ge => a >= b,
        },
    }
}

fn cmp_i64(a: i64, b: i64, direction: CompareDirection) -> bool {
    match direction {
        CompareDirection::Eq => a == b,
        CompareDirection::Ne => a != b,
        CompareDirection::Lt => a < b,
        CompareDirection::Le => a <= b,
        CompareDirection::Gt => a > b,
        CompareDirection::Ge => a >= b,
    }
}

fn write_scalar(
    bytes: &mut [u8],
    ty: ElementType,
    index: usize,
    scalar: &ScalarValue,
) -> Result<(), String> {
    write_f64(bytes, ty, index, scalar.as_f64());
    if matches!(ty, ElementType::I1) {
        bytes[index] = u8::from(scalar.as_i64() != 0);
    }
    Ok(())
}

fn write_int(bytes: &mut [u8], ty: ElementType, index: usize, value: i64) {
    match ty {
        ElementType::I64 | ElementType::UI64 => {
            let off = index * 8;
            bytes[off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
        ElementType::I32 | ElementType::UI32 => {
            let off = index * 4;
            bytes[off..off + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        ElementType::I1 => bytes[index] = u8::from(value != 0),
        ElementType::F64 | ElementType::F32 => write_f64(bytes, ty, index, value as f64),
    }
}

fn write_f64(bytes: &mut [u8], ty: ElementType, index: usize, value: f64) {
    match ty {
        ElementType::F64 => {
            let off = index * 8;
            bytes[off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
        ElementType::F32 => {
            let off = index * 4;
            bytes[off..off + 4].copy_from_slice(&(value as f32).to_le_bytes());
        }
        ElementType::I64 => {
            let off = index * 8;
            bytes[off..off + 8].copy_from_slice(&(value as i64).to_le_bytes());
        }
        ElementType::I32 => {
            let off = index * 4;
            bytes[off..off + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        ElementType::UI64 => {
            let off = index * 8;
            bytes[off..off + 8].copy_from_slice(&(value as u64).to_le_bytes());
        }
        ElementType::UI32 => {
            let off = index * 4;
            bytes[off..off + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        ElementType::I1 => bytes[index] = u8::from(value != 0.0),
    }
}

fn read_f64(bytes: &[u8], ty: ElementType, index: usize) -> f64 {
    match ty {
        ElementType::F64 => {
            let off = index * 8;
            f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
        }
        ElementType::F32 => {
            let off = index * 4;
            f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f64
        }
        ElementType::I64 | ElementType::UI64 | ElementType::I32 | ElementType::UI32 => {
            read_int(bytes, ty, index) as f64
        }
        ElementType::I1 => {
            if bytes[index] != 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn read_int(bytes: &[u8], ty: ElementType, index: usize) -> i64 {
    match ty {
        ElementType::I64 => {
            let off = index * 8;
            i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
        }
        ElementType::UI64 => {
            let off = index * 8;
            u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as i64
        }
        ElementType::I32 => {
            let off = index * 4;
            i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as i64
        }
        ElementType::UI32 => {
            let off = index * 4;
            u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as i64
        }
        ElementType::I1 => i64::from(bytes[index] != 0),
        ElementType::F64 => read_f64(bytes, ty, index) as i64,
        ElementType::F32 => read_f64(bytes, ty, index) as i64,
    }
}

fn zip_f64(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(f64, f64) -> f64) {
    let n = dst.len() / 8;
    for i in 0..n {
        write_f64(
            dst,
            ElementType::F64,
            i,
            op(
                read_f64(a, ElementType::F64, i),
                read_f64(b, ElementType::F64, i),
            ),
        );
    }
}

fn zip_f32(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(f32, f32) -> f32) {
    let n = dst.len() / 4;
    for i in 0..n {
        write_f64(
            dst,
            ElementType::F32,
            i,
            op(
                read_f64(a, ElementType::F32, i) as f32,
                read_f64(b, ElementType::F32, i) as f32,
            ) as f64,
        );
    }
}

fn zip_unary_f64(src: &[u8], dst: &mut [u8], op: impl Fn(f64) -> f64) {
    let n = dst.len() / 8;
    for i in 0..n {
        write_f64(
            dst,
            ElementType::F64,
            i,
            op(read_f64(src, ElementType::F64, i)),
        );
    }
}

fn zip_unary_f32(src: &[u8], dst: &mut [u8], op: impl Fn(f32) -> f32) {
    let n = dst.len() / 4;
    for i in 0..n {
        write_f64(
            dst,
            ElementType::F32,
            i,
            op(read_f64(src, ElementType::F32, i) as f32) as f64,
        );
    }
}

fn zip_i64(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(i64, i64) -> i64) {
    let n = dst.len() / 8;
    for i in 0..n {
        let x = read_int(a, ElementType::I64, i);
        let y = read_int(b, ElementType::I64, i);
        let off = i * 8;
        dst[off..off + 8].copy_from_slice(&op(x, y).to_le_bytes());
    }
}

fn zip_i32(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(i32, i32) -> i32) {
    let n = dst.len() / 4;
    for i in 0..n {
        let x = read_int(a, ElementType::I32, i) as i32;
        let y = read_int(b, ElementType::I32, i) as i32;
        let off = i * 4;
        dst[off..off + 4].copy_from_slice(&op(x, y).to_le_bytes());
    }
}

fn zip_u64(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(u64, u64) -> u64) {
    let n = dst.len() / 8;
    for i in 0..n {
        let x = read_int(a, ElementType::UI64, i) as u64;
        let y = read_int(b, ElementType::UI64, i) as u64;
        let off = i * 8;
        dst[off..off + 8].copy_from_slice(&op(x, y).to_le_bytes());
    }
}

fn zip_u32(a: &[u8], b: &[u8], dst: &mut [u8], op: impl Fn(u32, u32) -> u32) {
    let n = dst.len() / 4;
    for i in 0..n {
        let x = read_int(a, ElementType::UI32, i) as u32;
        let y = read_int(b, ElementType::UI32, i) as u32;
        let off = i * 4;
        dst[off..off + 4].copy_from_slice(&op(x, y).to_le_bytes());
    }
}

fn zip_unary_i64(src: &[u8], dst: &mut [u8], op: impl Fn(i64) -> i64) {
    let n = dst.len() / 8;
    for i in 0..n {
        let x = read_int(src, ElementType::I64, i);
        let off = i * 8;
        dst[off..off + 8].copy_from_slice(&op(x).to_le_bytes());
    }
}

fn zip_unary_i32(src: &[u8], dst: &mut [u8], op: impl Fn(i32) -> i32) {
    let n = dst.len() / 4;
    for i in 0..n {
        let x = read_int(src, ElementType::I32, i) as i32;
        let off = i * 4;
        dst[off..off + 4].copy_from_slice(&op(x).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DotDims;

    fn ty(shape: &[i64], element_type: ElementType) -> TensorType {
        TensorType {
            shape: shape.to_vec(),
            element_type,
        }
    }

    fn f64s(shape: &[i64], vals: &[f64]) -> Tensor {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            ty: ty(shape, ElementType::F64),
            bytes,
        }
    }

    fn f32s(shape: &[i64], vals: &[f32]) -> Tensor {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor {
            ty: ty(shape, ElementType::F32),
            bytes,
        }
    }

    fn read_f64s(tensor: &Tensor) -> Vec<f64> {
        (0..tensor.elem_count())
            .map(|i| read_f64(&tensor.bytes, tensor.ty.element_type, i))
            .collect()
    }

    #[test]
    fn f32_sqrt_is_implemented() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f32s(&[2], &[4.0, 9.0]));
        let out = unary(&env, ValueId(0), &ty(&[2], ElementType::F32), UnOp::Sqrt).unwrap();
        let vals = read_f64s(&out);
        assert!((vals[0] - 2.0).abs() < 1e-6);
        assert!((vals[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn splat_larger_than_result_is_an_error() {
        let err = materialize_constant(
            &ConstantValue::DenseSplat(ScalarValue::F64(1.0), ty(&[8], ElementType::F64)),
            &ty(&[4], ElementType::F64),
        )
        .unwrap_err();
        assert!(err.contains("splat"));
    }

    #[test]
    fn splat_fills_the_result() {
        let out = materialize_constant(
            &ConstantValue::DenseSplat(ScalarValue::F64(1.5), ty(&[], ElementType::F64)),
            &ty(&[3], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![1.5, 1.5, 1.5]);
    }

    #[test]
    fn clamp_broadcasts_scalar_bounds() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f64s(&[3], &[-5.0, 0.5, 3.0]));
        env.insert(ValueId(1), f64s(&[], &[-1.0]));
        env.insert(ValueId(2), f64s(&[], &[1.0]));
        let out = clamp(
            &env,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            &ty(&[3], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_broadcasts_unit_axes() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f64s(&[2, 2], &[0.0, 2.0, -3.0, 0.5]));
        env.insert(ValueId(1), f64s(&[2, 1], &[-1.0, -0.5]));
        env.insert(ValueId(2), f64s(&[1, 2], &[1.0, 1.5]));
        let out = clamp(
            &env,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            &ty(&[2, 2], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![0.0, 1.5, -0.5, 0.5]);
    }

    #[test]
    fn clamp_short_operand_is_an_error() {
        let mut env = HashMap::new();
        env.insert(
            ValueId(0),
            Tensor {
                ty: ty(&[2], ElementType::F64),
                bytes: 1.0_f64.to_le_bytes().to_vec(),
            },
        );
        env.insert(ValueId(1), f64s(&[], &[-1.0]));
        env.insert(ValueId(2), f64s(&[], &[1.0]));
        let err = clamp(
            &env,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            &ty(&[2], ElementType::F64),
        )
        .unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn clamp_rejects_unbroadcastable_bounds() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f64s(&[3], &[-5.0, 0.5, 3.0]));
        env.insert(ValueId(1), f64s(&[2], &[-1.0, 0.0]));
        env.insert(ValueId(2), f64s(&[], &[1.0]));
        let err = clamp(
            &env,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            &ty(&[3], ElementType::F64),
        )
        .unwrap_err();
        assert!(err.contains("broadcast"));
    }

    #[test]
    fn compare_broadcasts_scalar_operand() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f64s(&[3], &[-1.0, 0.0, 2.0]));
        env.insert(ValueId(1), f64s(&[], &[0.0]));
        let out = compare(
            &env,
            ValueId(0),
            ValueId(1),
            &ty(&[3], ElementType::I1),
            CompareDirection::Gt,
            CompareType::Float,
        )
        .unwrap();
        assert_eq!(out.bytes, vec![0, 0, 1]);
    }

    #[test]
    fn compare_broadcasts_integer_and_predicate_scalars() {
        let mut env = HashMap::new();
        env.insert(
            ValueId(0),
            Tensor {
                ty: ty(&[2], ElementType::I64),
                bytes: {
                    let mut bytes = Vec::new();
                    bytes.extend_from_slice(&1_i64.to_le_bytes());
                    bytes.extend_from_slice(&4_i64.to_le_bytes());
                    bytes
                },
            },
        );
        env.insert(
            ValueId(1),
            Tensor {
                ty: ty(&[], ElementType::I64),
                bytes: 2_i64.to_le_bytes().to_vec(),
            },
        );
        let ints = compare(
            &env,
            ValueId(0),
            ValueId(1),
            &ty(&[2], ElementType::I1),
            CompareDirection::Lt,
            CompareType::Signed,
        )
        .unwrap();
        assert_eq!(ints.bytes, vec![1, 0]);

        env.insert(
            ValueId(2),
            Tensor {
                ty: ty(&[2], ElementType::I1),
                bytes: vec![1, 0],
            },
        );
        env.insert(
            ValueId(3),
            Tensor {
                ty: ty(&[], ElementType::I1),
                bytes: vec![1],
            },
        );
        let preds = compare(
            &env,
            ValueId(2),
            ValueId(3),
            &ty(&[2], ElementType::I1),
            CompareDirection::Eq,
            CompareType::Signed,
        )
        .unwrap();
        assert_eq!(preds.bytes, vec![1, 0]);
    }

    #[test]
    fn compare_rejects_unbroadcastable_operand() {
        let mut env = HashMap::new();
        env.insert(ValueId(0), f64s(&[3], &[-1.0, 0.0, 2.0]));
        env.insert(ValueId(1), f64s(&[2], &[0.0, 1.0]));
        let err = compare(
            &env,
            ValueId(0),
            ValueId(1),
            &ty(&[3], ElementType::I1),
            CompareDirection::Gt,
            CompareType::Float,
        )
        .unwrap_err();
        assert!(err.contains("broadcast"));
    }

    #[test]
    fn select_broadcasts_scalar_arms() {
        let mut env = HashMap::new();
        env.insert(
            ValueId(0),
            Tensor {
                ty: ty(&[2], ElementType::I1),
                bytes: vec![1, 0],
            },
        );
        env.insert(ValueId(1), f64s(&[], &[5.0]));
        env.insert(ValueId(2), f64s(&[], &[1.0]));
        let out = select(
            &env,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            &ty(&[2], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![5.0, 1.0]);
    }

    #[test]
    fn vector_dot_is_a_scalar() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[3], &[1.0, 2.0, 3.0]));
        env.insert(ValueId(2), f64s(&[3], &[4.0, 5.0, 6.0]));
        let dims = DotDims {
            lhs_contracting: vec![0],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let out = dot_general(
            &env,
            ValueId(1),
            ValueId(2),
            &dims,
            &ty(&[], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![32.0]);
    }

    #[test]
    fn matrix_vector_product_keeps_the_free_dimension() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        env.insert(ValueId(2), f64s(&[3], &[1.0, 1.0, 1.0]));
        let dims = DotDims {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let out = dot_general(
            &env,
            ValueId(1),
            ValueId(2),
            &dims,
            &ty(&[2], ElementType::F64),
        )
        .unwrap();
        assert_eq!(read_f64s(&out), vec![6.0, 15.0]);
    }
}
