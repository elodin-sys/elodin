//! In-process StableHLO interpreter for display kernels.
//!
//! Used on wasm, where Cranelift JIT cannot emit executable pages. Native
//! `DisplayKernelExec` still compiles through Cranelift; this path is the
//! byte-buffer ABI for the same parsed IR.

use std::collections::HashMap;

use crate::ir::{
    CompareDirection, CompareType, ConstantValue, ElementType, InstrResult, Instruction, Module,
    ReduceOp, ScalarValue, ScatterCombiner, ScatterDims, TensorType, ValueId,
};
use crate::tensor_rt;
use thiserror::Error;

/// Failure from the display-kernel interpreter.
#[derive(Debug, Error)]
pub enum InterpError {
    #[error("tensor is {actual} bytes, expected {expected}")]
    TensorSize { actual: usize, expected: usize },

    #[error("compiled module is missing a callable main entrypoint")]
    MissingMain,

    #[error("expected {expected} inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },

    #[error("expected {expected} outputs, got {actual}")]
    OutputCount { expected: usize, actual: usize },

    #[error("kernel returned {actual} values, expected {expected}")]
    ReturnCount { actual: usize, expected: usize },

    #[error("output is {actual} bytes, expected {expected}")]
    OutputByteSize { actual: usize, expected: usize },

    #[error("instruction produced {actual} values, expected {expected}")]
    ProducedCount { actual: usize, expected: usize },

    #[error("value {id:?} has type {actual}, expected {expected}")]
    ValueType {
        id: ValueId,
        actual: TensorType,
        expected: TensorType,
    },

    #[error("instruction has no result type")]
    MissingResultType,

    #[error("unsupported custom_call target: {0}")]
    UnsupportedCustomCall(String),

    #[error("display kernel interpreter does not implement {0:?}")]
    UnimplementedInstruction(Box<Instruction>),

    #[error("unknown callee {0}")]
    UnknownCallee(String),

    #[error("call {callee} expected {expected} args, got {actual}")]
    CallArity {
        callee: String,
        expected: usize,
        actual: usize,
    },

    #[error("call {callee} arg type {actual} does not match {expected}")]
    CallArgType {
        callee: String,
        actual: TensorType,
        expected: TensorType,
    },

    #[error("while loop arity mismatch")]
    WhileArityMismatch,

    #[error("while condition returned no value")]
    WhileConditionEmpty,

    #[error("while body arity mismatch")]
    WhileBodyArityMismatch,

    #[error("case has no branches")]
    CaseEmpty,

    #[error("case branch arity mismatch")]
    CaseArityMismatch,

    #[error("binary operands must already be broadcast to the result shape")]
    BinaryNotBroadcast,

    #[error("integer bitwise op on float")]
    BitwiseOnFloat,

    #[error("unsupported i1 binary op")]
    UnsupportedI1Binary,

    #[error("erfinv is not implemented in the display kernel interpreter")]
    ErfInvNotImplemented,

    #[error("unary {op} is not implemented for {element_type:?}")]
    UnaryNotImplemented {
        op: &'static str,
        element_type: ElementType,
    },

    #[error("compare operands have different element types")]
    CompareElementTypes,

    #[error("compare result must be i1")]
    CompareResultNotI1,

    #[error("select operand element type does not match the result")]
    SelectElementType,

    #[error("select operand is smaller than the result")]
    SelectOperandTooSmall,

    #[error("select predicate is out of range")]
    SelectPredicateOutOfRange,

    #[error("operand rank exceeds the result")]
    BroadcastRank,

    #[error("cannot broadcast dimension {dim} onto {dst_dim}")]
    BroadcastDimension { dim: i64, dst_dim: i64 },

    #[error("operand index is out of range")]
    OperandIndexOutOfRange,

    #[error("reshape changed the number of bytes")]
    ReshapeByteSize,

    #[error("concatenate has no operands")]
    ConcatEmpty,

    #[error("concatenate operand types disagree")]
    ConcatTypeMismatch,

    #[error("dot_general batch shapes do not match")]
    DotBatchShape,

    #[error("dot_general contracting size does not match")]
    DotContractingSize,

    #[error("argmin/argmax over {0} dimensions is not implemented")]
    ArgminmaxDims(usize),

    #[error("reduce {op:?} is not implemented for {element_type:?}")]
    ReduceNotImplemented {
        op: ReduceOp,
        element_type: ElementType,
    },

    #[error("bitcast convert changed the number of bytes")]
    BitcastByteSize,

    #[error("splat constant has {actual} elements but the result has {expected}")]
    SplatTooLarge { actual: usize, expected: usize },

    #[error("dense array is larger than its result type")]
    DenseArrayTooLarge,

    #[error("external constant type/size mismatch")]
    ExternalConstantMismatch,

    #[error("unknown value {0:?}")]
    UnknownValue(ValueId),

    #[error("expected {expected:?}, got {actual:?}")]
    ExpectedElementType {
        expected: ElementType,
        actual: ElementType,
    },

    #[error("empty predicate")]
    EmptyPredicate,
}

impl InterpError {
    /// True when the JIT may still execute the op and the interpreter should be skipped.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedCustomCall(_)
                | Self::UnimplementedInstruction(_)
                | Self::ErfInvNotImplemented
                | Self::UnaryNotImplemented { .. }
                | Self::ReduceNotImplemented { .. }
                | Self::ArgminmaxDims(_)
        )
    }
}

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

    fn from_bytes(ty: TensorType, bytes: Vec<u8>) -> Result<Self, InterpError> {
        if bytes.len() != ty.byte_size() {
            return Err(InterpError::TensorSize {
                actual: bytes.len(),
                expected: ty.byte_size(),
            });
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
) -> Result<(), InterpError> {
    let func = module.main_func().ok_or(InterpError::MissingMain)?;
    if inputs.len() != func.params.len() {
        return Err(InterpError::InputCount {
            expected: func.params.len(),
            actual: inputs.len(),
        });
    }
    if outputs.len() != func.result_types.len() {
        return Err(InterpError::OutputCount {
            expected: func.result_types.len(),
            actual: outputs.len(),
        });
    }
    let mut env: HashMap<ValueId, Tensor> = HashMap::new();
    for ((id, ty), buf) in func.params.iter().zip(inputs) {
        env.insert(*id, Tensor::from_bytes(ty.clone(), buf.to_vec())?);
    }
    let returned = eval_body(module, &func.body, &mut env)?;
    if returned.len() != outputs.len() {
        return Err(InterpError::ReturnCount {
            actual: returned.len(),
            expected: outputs.len(),
        });
    }
    for (dst, src) in outputs.iter_mut().zip(returned) {
        if dst.len() != src.bytes.len() {
            return Err(InterpError::OutputByteSize {
                actual: dst.len(),
                expected: src.bytes.len(),
            });
        }
        dst.copy_from_slice(&src.bytes);
    }
    Ok(())
}

fn eval_body(
    module: &Module,
    body: &[InstrResult],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, InterpError> {
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
                    return Err(InterpError::ProducedCount {
                        actual: produced.len(),
                        expected: ir.values.len(),
                    });
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
                        return Err(InterpError::ValueType {
                            id: *id,
                            actual: tensor.ty.clone(),
                            expected: ty.clone(),
                        });
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
) -> Result<Vec<Tensor>, InterpError> {
    let result_ty = results
        .first()
        .map(|(_, ty)| ty.clone())
        .ok_or(InterpError::MissingResultType)?;
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
        Instruction::DotGeneral { lhs, rhs, dims } => dot_general(env, *lhs, *rhs, dims)?,
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
        Instruction::Scatter {
            operand,
            indices,
            updates,
            dims,
            combiner,
        } => scatter(env, *operand, *indices, *updates, dims, *combiner)?,
        Instruction::ReduceArgminmax {
            values,
            indices,
            dimensions,
            is_min,
        } => return argminmax(env, *values, *indices, dimensions, *is_min, results),
        Instruction::Sort {
            inputs,
            dimension,
            comparator,
            comparator_params,
            ..
        } => {
            return sort(
                module,
                env,
                inputs,
                *dimension,
                comparator,
                comparator_params,
            );
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
        } => match (operands.as_slice(), init_values.as_slice()) {
            ([operand], [init]) => reduce_window(
                module,
                env,
                *operand,
                *init,
                (body, body_params),
                [
                    window_dimensions,
                    window_strides,
                    base_dilations,
                    window_dilations,
                ],
                padding,
                &result_ty,
            )?,
            _ => {
                return Err(InterpError::UnimplementedInstruction(Box::new(
                    instr.clone(),
                )));
            }
        },
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
        } => select_and_scatter(
            module,
            env,
            [*operand, *source, *init_value],
            (select_body, select_params),
            (scatter_body, scatter_params),
            window_dimensions,
            window_strides,
            padding,
            &result_ty,
        )?,
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
            return Err(InterpError::UnsupportedCustomCall(call_target.clone()));
        }
        other => {
            return Err(InterpError::UnimplementedInstruction(Box::new(
                other.clone(),
            )));
        }
    };
    Ok(vec![out])
}

fn call(
    module: &Module,
    callee: &str,
    args: &[ValueId],
    env: &mut HashMap<ValueId, Tensor>,
) -> Result<Vec<Tensor>, InterpError> {
    let func = module
        .get_func(callee)
        .ok_or_else(|| InterpError::UnknownCallee(callee.to_string()))?;
    if args.len() != func.params.len() {
        return Err(InterpError::CallArity {
            callee: callee.to_string(),
            expected: func.params.len(),
            actual: args.len(),
        });
    }
    let mut nested = HashMap::new();
    for ((id, ty), arg) in func.params.iter().zip(args) {
        let tensor = lookup(env, *arg)?;
        if tensor.ty != *ty {
            return Err(InterpError::CallArgType {
                callee: callee.to_string(),
                actual: tensor.ty.clone(),
                expected: ty.clone(),
            });
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
) -> Result<Vec<Tensor>, InterpError> {
    if init_values.len() != iter_arg_ids.len() || iter_arg_ids.len() != results.len() {
        return Err(InterpError::WhileArityMismatch);
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
        let pred = cond.first().ok_or(InterpError::WhileConditionEmpty)?;
        if !is_true(pred)? {
            break;
        }
        let mut body_env = env.clone();
        for (id, value) in iter_arg_ids.iter().zip(&state) {
            body_env.insert(*id, value.clone());
        }
        state = eval_body(module, loop_body, &mut body_env)?;
        if state.len() != results.len() {
            return Err(InterpError::WhileBodyArityMismatch);
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
) -> Result<Vec<Tensor>, InterpError> {
    if branches.is_empty() {
        return Err(InterpError::CaseEmpty);
    }
    let idx = lookup(env, index)?;
    let mut which = scalar_as_i64(&idx).unwrap_or(0);
    if which < 0 || which as usize >= branches.len() {
        which = (branches.len() - 1) as i64;
    }
    let mut nested = env.clone();
    let out = eval_body(module, &branches[which as usize], &mut nested)?;
    if out.len() != results.len() {
        return Err(InterpError::CaseArityMismatch);
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

impl UnOp {
    fn label(self) -> &'static str {
        match self {
            Self::Neg => "Neg",
            Self::Abs => "Abs",
            Self::Sqrt => "Sqrt",
            Self::Rsqrt => "Rsqrt",
            Self::Floor => "Floor",
            Self::Ceil => "Ceil",
            Self::RoundNearestEven => "RoundNearestEven",
            Self::Sign => "Sign",
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Tan => "Tan",
            Self::Tanh => "Tanh",
            Self::Sinh => "Sinh",
            Self::Cosh => "Cosh",
            Self::Asin => "Asin",
            Self::Acos => "Acos",
            Self::Atan => "Atan",
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Log1p => "Log1p",
            Self::Expm1 => "Expm1",
            Self::Cbrt => "Cbrt",
            Self::ErfInv => "ErfInv",
            Self::Erfc => "Erfc",
            Self::IsFinite => "IsFinite",
            Self::Not => "Not",
        }
    }
}

fn binary(
    env: &HashMap<ValueId, Tensor>,
    lhs: ValueId,
    rhs: ValueId,
    result_ty: &TensorType,
    op: BinOp,
) -> Result<Tensor, InterpError> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    if a.elem_count() != result_ty.num_elements() || b.elem_count() != result_ty.num_elements() {
        return Err(InterpError::BinaryNotBroadcast);
    }
    let mut out = Tensor::alloc(result_ty.clone());
    match result_ty.element_type {
        ElementType::F64 | ElementType::F32 => {
            if matches!(
                op,
                BinOp::Xor | BinOp::Or | BinOp::And | BinOp::Shl | BinOp::ShrL | BinOp::ShrA
            ) {
                return Err(InterpError::BitwiseOnFloat);
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
                    _ => return Err(InterpError::UnsupportedI1Binary),
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
) -> Result<Tensor, InterpError> {
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
                return Err(InterpError::ErfInvNotImplemented);
            }
            zip_unary_f64(&src.bytes, &mut out.bytes, |a| float_unary(op, a));
        }
        (_, ElementType::F32, ElementType::F32) => {
            if matches!(op, UnOp::ErfInv) {
                return Err(InterpError::ErfInvNotImplemented);
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
            return Err(InterpError::UnaryNotImplemented {
                op: op.label(),
                element_type: result_ty.element_type,
            });
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
        UnOp::Erfc => libm::erfc(a),
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
) -> Result<Tensor, InterpError> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    if a.ty.element_type != b.ty.element_type {
        return Err(InterpError::CompareElementTypes);
    }
    if result_ty.element_type != ElementType::I1 {
        return Err(InterpError::CompareResultNotI1);
    }
    let a_map = broadcast_map(&result_ty.shape, &a.ty.shape)?;
    let b_map = broadcast_map(&result_ty.shape, &b.ty.shape)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n = result_ty.num_elements();
    if out.bytes.len() != n {
        return Err(InterpError::CompareResultNotI1);
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
            let unsigned = compare_type == CompareType::Unsigned
                || matches!(a.ty.element_type, ElementType::UI64 | ElementType::UI32);
            for i in 0..n {
                let pred = if unsigned {
                    let x =
                        read_int(&a.bytes, a.ty.element_type, mapped_index(a, &a_map, i)?) as u64;
                    let y =
                        read_int(&b.bytes, b.ty.element_type, mapped_index(b, &b_map, i)?) as u64;
                    cmp_ord(x, y, direction)
                } else {
                    let x = read_int(&a.bytes, a.ty.element_type, mapped_index(a, &a_map, i)?);
                    let y = read_int(&b.bytes, b.ty.element_type, mapped_index(b, &b_map, i)?);
                    cmp_ord(x, y, direction)
                };
                out.bytes[i] = u8::from(pred);
            }
        }
        ElementType::I1 => {
            for i in 0..n {
                let x = i64::from(a.bytes[mapped_index(a, &a_map, i)?] != 0);
                let y = i64::from(b.bytes[mapped_index(b, &b_map, i)?] != 0);
                out.bytes[i] = u8::from(cmp_ord(x, y, direction));
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
) -> Result<Tensor, InterpError> {
    let pred = lookup_ref(env, cond)?;
    let t = lookup_ref(env, on_true)?;
    let f = lookup_ref(env, on_false)?;
    if t.ty.element_type != result_ty.element_type || f.ty.element_type != result_ty.element_type {
        return Err(InterpError::SelectElementType);
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
            return Err(InterpError::SelectOperandTooSmall);
        }
        out.bytes[dst_off..dst_off + elem].copy_from_slice(&src.bytes[src_off..end]);
    }
    Ok(out)
}

fn predicate_true(pred: &Tensor, index: usize) -> Result<bool, InterpError> {
    let elem = pred.elem_size();
    let off = index * elem;
    if off + elem > pred.bytes.len() {
        return Err(InterpError::SelectPredicateOutOfRange);
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

fn broadcast_map(dst_shape: &[i64], src_shape: &[i64]) -> Result<BroadcastMap, InterpError> {
    if src_shape.len() > dst_shape.len() {
        return Err(InterpError::BroadcastRank);
    }
    let offset = dst_shape.len() - src_shape.len();
    for (i, &dim) in src_shape.iter().enumerate() {
        let dst_dim = dst_shape[offset + i];
        if dim != dst_dim && dim != 1 {
            return Err(InterpError::BroadcastDimension { dim, dst_dim });
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

fn mapped_index(tensor: &Tensor, map: &BroadcastMap, flat: usize) -> Result<usize, InterpError> {
    let index = map.index(flat);
    let size = tensor.elem_size();
    let off = index
        .checked_mul(size)
        .ok_or(InterpError::OperandIndexOutOfRange)?;
    let end = off
        .checked_add(size)
        .ok_or(InterpError::OperandIndexOutOfRange)?;
    if end > tensor.bytes.len() {
        return Err(InterpError::OperandIndexOutOfRange);
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
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let lo = lookup_ref(env, min)?;
    let hi = lookup_ref(env, max)?;
    let et = result_ty.element_type;
    let src_map = broadcast_map(&result_ty.shape, &src.ty.shape)?;
    let lo_map = broadcast_map(&result_ty.shape, &lo.ty.shape)?;
    let hi_map = broadcast_map(&result_ty.shape, &hi.ty.shape)?;
    let mut out = Tensor::alloc(result_ty.clone());
    for i in 0..result_ty.num_elements() {
        let s = mapped_index(src, &src_map, i)?;
        let l = mapped_index(lo, &lo_map, i)?;
        let h = mapped_index(hi, &hi_map, i)?;
        if is_real(et) {
            let v = read_f64(&src.bytes, et, s);
            let (l, h) = (read_f64(&lo.bytes, et, l), read_f64(&hi.bytes, et, h));
            let r = if v.is_nan() || l.is_nan() || h.is_nan() {
                f64::NAN
            } else {
                v.max(l).min(h)
            };
            write_f64(&mut out.bytes, et, i, r);
        } else {
            let v = read_int(&src.bytes, et, s);
            let (l, h) = (read_int(&lo.bytes, et, l), read_int(&hi.bytes, et, h));
            let r = if et == ElementType::UI64 {
                (v as u64).max(l as u64).min(h as u64) as i64
            } else {
                v.max(l).min(h)
            };
            write_int(&mut out.bytes, et, i, r);
        }
    }
    Ok(out)
}

fn reshape(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup(env, operand)?;
    if src.bytes.len() != result_ty.byte_size() {
        return Err(InterpError::ReshapeByteSize);
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
) -> Result<Tensor, InterpError> {
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
) -> Result<Tensor, InterpError> {
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
) -> Result<Tensor, InterpError> {
    if operands.is_empty() {
        return Err(InterpError::ConcatEmpty);
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

fn concat_ty(a: &TensorType, b: &TensorType, dim: i64) -> Result<TensorType, InterpError> {
    if a.rank() != b.rank() || a.element_type != b.element_type {
        return Err(InterpError::ConcatTypeMismatch);
    }
    let d = dim as usize;
    let mut shape = a.shape.clone();
    shape[d] = a.shape[d] + b.shape[d];
    Ok(TensorType {
        shape,
        element_type: a.element_type,
    })
}

/// Any batch, free and contracting axes, in the operands' own arithmetic (integers wrap).
/// The output axes are batch, lhs free, then rhs free.
fn dot_general(
    env: &HashMap<ValueId, Tensor>,
    lhs: ValueId,
    rhs: ValueId,
    dims: &crate::ir::DotDims,
) -> Result<Tensor, InterpError> {
    let a = lookup_ref(env, lhs)?;
    let b = lookup_ref(env, rhs)?;
    let et = a.ty.element_type;
    expect_ty(b, et)?;
    let (l_st, r_st) = (shape_strides(&a.ty.shape), shape_strides(&b.ty.shape));
    // Axes as (size, lhs stride, rhs stride).
    let paired = |l_axes: &[i64], r_axes: &[i64]| -> Option<Vec<(i64, usize, usize)>> {
        if l_axes.len() != r_axes.len() {
            return None;
        }
        l_axes
            .iter()
            .zip(r_axes)
            .map(|(&l, &r)| {
                let (l, r) = (usize::try_from(l).ok()?, usize::try_from(r).ok()?);
                let size = *a.ty.shape.get(l)?;
                (b.ty.shape.get(r) == Some(&size)).then(|| (size, l_st[l], r_st[r]))
            })
            .collect()
    };
    let mut out_axes =
        paired(&dims.lhs_batch, &dims.rhs_batch).ok_or(InterpError::DotBatchShape)?;
    let k_axes = paired(&dims.lhs_contracting, &dims.rhs_contracting)
        .ok_or(InterpError::DotContractingSize)?;
    let free = |rank: usize, batch: &[i64], contracting: &[i64]| -> Vec<usize> {
        (0..rank)
            .filter(|&d| !batch.contains(&(d as i64)) && !contracting.contains(&(d as i64)))
            .collect()
    };
    for d in free(a.ty.rank(), &dims.lhs_batch, &dims.lhs_contracting) {
        out_axes.push((a.ty.shape[d], l_st[d], 0));
    }
    for d in free(b.ty.rank(), &dims.rhs_batch, &dims.rhs_contracting) {
        out_axes.push((b.ty.shape[d], 0, r_st[d]));
    }
    let shape = out_axes.iter().map(|axis| axis.0).collect();
    let mut out = Tensor::alloc(TensorType {
        shape,
        element_type: et,
    });
    let offsets = |axes: &[(i64, usize, usize)], mut flat: usize| {
        axes.iter().rev().fold((0, 0), |(l, r), &(size, ls, rs)| {
            let c = flat % size as usize;
            flat /= size as usize;
            (l + c * ls, r + c * rs)
        })
    };
    let n_k = k_axes.iter().map(|axis| axis.0).product::<i64>() as usize;
    for o in 0..out.elem_count() {
        let (lo, ro) = offsets(&out_axes, o);
        let (mut real, mut single, mut int) = (0.0, 0.0f32, 0i64);
        for k in 0..n_k {
            let (lk, rk) = offsets(&k_axes, k);
            let (x, y) = ((&a.bytes, lo + lk), (&b.bytes, ro + rk));
            match et {
                ElementType::F64 => real += read_f64(x.0, et, x.1) * read_f64(y.0, et, y.1),
                ElementType::F32 => {
                    single += read_f64(x.0, et, x.1) as f32 * read_f64(y.0, et, y.1) as f32;
                }
                _ => {
                    let p = read_int(x.0, et, x.1).wrapping_mul(read_int(y.0, et, y.1));
                    int = int.wrapping_add(p);
                }
            }
        }
        match et {
            ElementType::F64 => write_f64(&mut out.bytes, et, o, real),
            ElementType::F32 => write_f64(&mut out.bytes, et, o, single as f64),
            _ => write_int(&mut out.bytes, et, o, int),
        }
    }
    Ok(out)
}

fn reduce(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    init: ValueId,
    op: ReduceOp,
    dimensions: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let init_tensor = lookup_ref(env, init)?;
    let et = src.ty.element_type;
    let not_implemented = || InterpError::ReduceNotImplemented {
        op,
        element_type: et,
    };
    let shape: Vec<usize> = src.ty.shape.iter().map(|&d| d as usize).collect();
    let kept: Vec<usize> = (0..shape.len())
        .filter(|&a| !dimensions.contains(&(a as i64)))
        .collect();
    let out_index = |flat: usize| {
        let coord = |a: usize| flat / shape[a + 1..].iter().product::<usize>() % shape[a];
        kept.iter().fold(0, |o, &a| o * shape[a] + coord(a))
    };
    let n_out = result_ty.num_elements();
    let mut out = Tensor::alloc(result_ty.clone());
    if is_real(et) {
        let fold: fn(f64, f64) -> f64 = match op {
            ReduceOp::Add => |a, b| a + b,
            ReduceOp::Multiply => |a, b| a * b,
            ReduceOp::Maximum => f64::max,
            ReduceOp::Minimum => f64::min,
            ReduceOp::And | ReduceOp::Or => return Err(not_implemented()),
        };
        let init = read_f64(&init_tensor.bytes, init_tensor.ty.element_type, 0);
        let mut acc = vec![init; n_out];
        for flat in 0..src.elem_count() {
            let o = out_index(flat);
            acc[o] = fold(acc[o], read_f64(&src.bytes, et, flat));
        }
        for (o, v) in acc.into_iter().enumerate() {
            write_f64(&mut out.bytes, result_ty.element_type, o, v);
        }
    } else {
        let fold: fn(i64, i64) -> i64 = match (op, et) {
            (ReduceOp::And, ElementType::I1) => |a, b| i64::from(a != 0 && b != 0),
            (ReduceOp::Or, ElementType::I1) => |a, b| i64::from(a != 0 || b != 0),
            (ReduceOp::Add, _) => i64::wrapping_add,
            (ReduceOp::Multiply, _) => i64::wrapping_mul,
            (ReduceOp::Maximum, ElementType::UI64) => |a, b| (a as u64).max(b as u64) as i64,
            (ReduceOp::Minimum, ElementType::UI64) => |a, b| (a as u64).min(b as u64) as i64,
            (ReduceOp::Maximum, _) => i64::max,
            (ReduceOp::Minimum, _) => i64::min,
            (ReduceOp::And | ReduceOp::Or, _) => return Err(not_implemented()),
        };
        let init = read_int(&init_tensor.bytes, init_tensor.ty.element_type, 0);
        let mut acc = vec![init; n_out];
        for flat in 0..src.elem_count() {
            let o = out_index(flat);
            acc[o] = fold(acc[o], read_int(&src.bytes, et, flat));
        }
        for (o, v) in acc.into_iter().enumerate() {
            write_int(&mut out.bytes, result_ty.element_type, o, v);
        }
    }
    Ok(out)
}

fn is_integral(ty: ElementType) -> bool {
    matches!(
        ty,
        ElementType::I1
            | ElementType::I32
            | ElementType::I64
            | ElementType::UI32
            | ElementType::UI64
    )
}

fn is_real(ty: ElementType) -> bool {
    matches!(ty, ElementType::F64 | ElementType::F32)
}

fn convert(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n = result_ty.num_elements();
    let src_ty = src.ty.element_type;
    let dst_ty = result_ty.element_type;
    for i in 0..n {
        match (src_ty, dst_ty) {
            (ElementType::UI64, ElementType::F64 | ElementType::F32) => {
                let v = read_int(&src.bytes, src_ty, i) as u64 as f64;
                write_f64(&mut out.bytes, dst_ty, i, v);
            }
            (src_ty, dst_ty) if is_integral(src_ty) && is_integral(dst_ty) => {
                write_int(&mut out.bytes, dst_ty, i, read_int(&src.bytes, src_ty, i));
            }
            (src_ty, dst_ty) if is_integral(src_ty) && is_real(dst_ty) => {
                let v = read_int(&src.bytes, src_ty, i) as f64;
                write_f64(&mut out.bytes, dst_ty, i, v);
            }
            (src_ty, dst_ty) if is_real(src_ty) && is_integral(dst_ty) => {
                let v = read_f64(&src.bytes, src_ty, i);
                let bits = match dst_ty {
                    ElementType::I1 => i64::from(v != 0.0),
                    ElementType::UI64 => (v as u64) as i64,
                    ElementType::UI32 => i64::from(v as u32),
                    ElementType::I32 => i64::from(v as i32),
                    ElementType::I64 => v as i64,
                    ElementType::F64 | ElementType::F32 => unreachable!("dst is integral"),
                };
                write_int(&mut out.bytes, dst_ty, i, bits);
            }
            _ => {
                let v = read_f64(&src.bytes, src_ty, i);
                write_f64(&mut out.bytes, dst_ty, i, v);
            }
        }
    }
    Ok(out)
}

fn bitcast(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup(env, operand)?;
    if src.bytes.len() != result_ty.byte_size() {
        return Err(InterpError::BitcastByteSize);
    }
    Ok(Tensor {
        ty: result_ty.clone(),
        bytes: src.bytes,
    })
}

fn iota(dimension: i64, result_ty: &TensorType) -> Result<Tensor, InterpError> {
    let mut out = Tensor::alloc(result_ty.clone());
    let et = result_ty.element_type;
    let strides = shape_strides(&result_ty.shape);
    let dim = dimension as usize;
    for i in 0..out.elem_count() {
        let coord = if result_ty.shape.is_empty() {
            0
        } else {
            (i / strides[dim]) as i64 % result_ty.shape[dim]
        };
        if is_real(et) {
            write_f64(&mut out.bytes, et, i, coord as f64);
        } else {
            write_int(&mut out.bytes, et, i, coord);
        }
    }
    Ok(out)
}

fn transpose(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    permutation: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
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
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let shape = &src.ty.shape;
    let strides = shape_strides(shape);
    let e = src.elem_size();
    let mut out = Tensor::alloc(result_ty.clone());
    for i in 0..src.elem_count() {
        let j: usize = (0..shape.len())
            .map(|d| {
                let c = i / strides[d] % shape[d] as usize;
                let c = if dimensions.contains(&(d as i64)) {
                    shape[d] as usize - 1 - c
                } else {
                    c
                };
                c * strides[d]
            })
            .sum();
        out.bytes[i * e..(i + 1) * e].copy_from_slice(&src.bytes[j * e..(j + 1) * e]);
    }
    Ok(out)
}

fn pad(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    padding_value: ValueId,
    low: &[i64],
    interior: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let pad = lookup_ref(env, padding_value)?;
    let e = src.elem_size();
    let mut out = Tensor::alloc(result_ty.clone());
    for slot in out.bytes.chunks_exact_mut(e) {
        slot.copy_from_slice(&pad.bytes[..e]);
    }
    let (src_strides, out_strides) = (
        shape_strides(&src.ty.shape),
        shape_strides(&result_ty.shape),
    );
    'elements: for i in 0..src.elem_count() {
        let mut o = 0;
        for d in 0..src.ty.rank() {
            let c = (i / src_strides[d]) as i64 % src.ty.shape[d];
            let step = 1 + interior.get(d).copied().unwrap_or(0);
            let p = low.get(d).copied().unwrap_or(0) + c * step;
            if p < 0 || p >= result_ty.shape[d] {
                continue 'elements;
            }
            o += p as usize * out_strides[d];
        }
        out.bytes[o * e..(o + 1) * e].copy_from_slice(&src.bytes[i * e..(i + 1) * e]);
    }
    Ok(out)
}

fn dynamic_slice(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    start_indices: &[ValueId],
    slice_sizes: &[i64],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
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
) -> Result<Tensor, InterpError> {
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
) -> Result<Tensor, InterpError> {
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

fn scatter(
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    indices: ValueId,
    updates: ValueId,
    dims: &ScatterDims,
    combiner: ScatterCombiner,
) -> Result<Tensor, InterpError> {
    let idx = lookup_ref(env, indices)?;
    let upd = lookup_ref(env, updates)?;
    let mut out = lookup_ref(env, operand)?.clone();
    let shape = out.ty.shape.clone();
    let et = out.ty.element_type;
    let ivd = dims.index_vector_dim as usize;
    let uwd: Vec<usize> = dims
        .update_window_dims
        .iter()
        .map(|&d| d as usize)
        .collect();
    let windowed: Vec<usize> = (0..shape.len())
        .filter(|&d| {
            let d = d as i64;
            !dims.inserted_window_dims.contains(&d) && !dims.input_batching_dims.contains(&d)
        })
        .collect();
    let (upd_strides, idx_strides, out_strides) = (
        shape_strides(&upd.ty.shape),
        shape_strides(&idx.ty.shape),
        shape_strides(&shape),
    );
    for u in 0..upd.elem_count() {
        let uc: Vec<usize> = (0..upd.ty.rank())
            .map(|d| u / upd_strides[d] % upd.ty.shape[d] as usize)
            .collect();
        let scatter_index: Vec<usize> = (0..uc.len())
            .filter(|d| !uwd.contains(d))
            .map(|d| uc[d])
            .collect();
        let mut full = vec![0i64; shape.len()];
        for (k, &d) in dims.scatter_dims_to_operand_dims.iter().enumerate() {
            let mut ic = scatter_index.clone();
            if ivd < idx.ty.rank() {
                ic.insert(ivd, k);
            }
            let pos = ic.iter().zip(&idx_strides).map(|(c, s)| c * s).sum();
            full[d as usize] = read_int(&idx.bytes, idx.ty.element_type, pos);
        }
        for (&d, &j) in dims
            .input_batching_dims
            .iter()
            .zip(&dims.scatter_indices_batching_dims)
        {
            let j = j as usize;
            full[d as usize] = scatter_index[if j < ivd { j } else { j - 1 }] as i64;
        }
        let window_size = |d: usize| {
            windowed
                .iter()
                .position(|&w| w == d)
                .map_or(1, |k| upd.ty.shape[uwd[k]])
        };
        if (0..shape.len()).any(|d| full[d] < 0 || full[d] + window_size(d) > shape[d]) {
            continue;
        }
        for (k, &d) in windowed.iter().enumerate() {
            full[d] += uc[uwd[k]] as i64;
        }
        let o: usize = (0..shape.len())
            .map(|d| full[d] as usize * out_strides[d])
            .sum();
        let e = et.byte_size();
        if combiner == ScatterCombiner::Set {
            out.bytes[o * e..(o + 1) * e].copy_from_slice(&upd.bytes[u * e..(u + 1) * e]);
        } else if is_real(et) {
            let (a, b) = (read_f64(&out.bytes, et, o), read_f64(&upd.bytes, et, u));
            let r = match combiner {
                ScatterCombiner::Add => a + b,
                ScatterCombiner::Multiply => a * b,
                ScatterCombiner::Minimum => a.min(b),
                _ => a.max(b),
            };
            write_f64(&mut out.bytes, et, o, r);
        } else {
            let (a, b) = (read_int(&out.bytes, et, o), read_int(&upd.bytes, et, u));
            let r = match combiner {
                ScatterCombiner::Add => a.wrapping_add(b),
                ScatterCombiner::Multiply => a.wrapping_mul(b),
                ScatterCombiner::Minimum => a.min(b),
                _ => a.max(b),
            };
            write_int(&mut out.bytes, et, o, r);
        }
    }
    Ok(out)
}

fn argminmax(
    env: &HashMap<ValueId, Tensor>,
    values: ValueId,
    indices: ValueId,
    dimensions: &[i64],
    is_min: bool,
    results: &[(ValueId, TensorType)],
) -> Result<Vec<Tensor>, InterpError> {
    let vals = lookup_ref(env, values)?;
    let idxs = lookup_ref(env, indices)?;
    let &[axis] = dimensions else {
        return Err(InterpError::ArgminmaxDims(dimensions.len()));
    };
    let (shape, vt, it) = (&vals.ty.shape, vals.ty.element_type, idxs.ty.element_type);
    let axis = axis as usize;
    let outer = shape[..axis].iter().product::<i64>() as usize;
    let inner = shape[axis] as usize;
    let after = shape[axis + 1..].iter().product::<i64>() as usize;
    let int_value = |p: usize| match vt {
        ElementType::UI64 => read_int(&vals.bytes, vt, p) as u64 as i128,
        _ => read_int(&vals.bytes, vt, p) as i128,
    };
    let (mut out_v, mut out_i) = (
        Tensor::alloc(results[0].1.clone()),
        Tensor::alloc(results[1].1.clone()),
    );
    let (ev, ei) = (vals.elem_size(), idxs.elem_size());
    for o in 0..outer {
        for a in 0..after {
            let mut best = o * inner * after + a;
            for k in 1..inner {
                let p = (o * inner + k) * after + a;
                let smaller_index = read_int(&idxs.bytes, it, p) < read_int(&idxs.bytes, it, best);
                let take = if is_real(vt) {
                    let (b, v) = (
                        read_f64(&vals.bytes, vt, best),
                        read_f64(&vals.bytes, vt, p),
                    );
                    let better = if is_min { v < b } else { v > b };
                    !b.is_nan() && (v.is_nan() || better || (v == b && smaller_index))
                } else {
                    let (b, v) = (int_value(best), int_value(p));
                    (if is_min { v < b } else { v > b }) || (v == b && smaller_index)
                };
                if take {
                    best = p;
                }
            }
            let q = o * after + a;
            out_v.bytes[q * ev..(q + 1) * ev]
                .copy_from_slice(&vals.bytes[best * ev..(best + 1) * ev]);
            out_i.bytes[q * ei..(q + 1) * ei]
                .copy_from_slice(&idxs.bytes[best * ei..(best + 1) * ei]);
        }
    }
    Ok(vec![out_v, out_i])
}

fn scalar_at(t: &Tensor, p: usize) -> Tensor {
    let e = t.elem_size();
    Tensor {
        ty: TensorType::scalar(t.ty.element_type),
        bytes: t.bytes[p * e..(p + 1) * e].to_vec(),
    }
}

/// Evaluates a region with two scalar block arguments.
fn apply_region(
    module: &Module,
    env: &mut HashMap<ValueId, Tensor>,
    (body, params): (&[InstrResult], &[ValueId]),
    a: Tensor,
    b: Tensor,
) -> Result<Tensor, InterpError> {
    env.insert(params[0], a);
    env.insert(params[1], b);
    Ok(eval_body(module, body, env)?.swap_remove(0))
}

/// Flat operand index of window element `w` of window `o`, or `None` in padding or in a
/// base-dilation hole.
fn window_source(
    [o, w]: [usize; 2],
    [out_shape, src_shape, dims]: [&[i64]; 3],
    [strides, base_dil, win_dil]: [&[i64]; 3],
    padding: &[(i64, i64)],
) -> Option<usize> {
    let (out_st, src_st, win_st) = (
        shape_strides(out_shape),
        shape_strides(src_shape),
        shape_strides(dims),
    );
    let mut flat = 0;
    for d in 0..src_shape.len() {
        let oc = (o / out_st[d]) as i64 % out_shape[d];
        let wc = (w / win_st[d]) as i64 % dims[d];
        let p = oc * strides[d] + wc * win_dil[d] - padding[d].0;
        if p < 0 || p % base_dil[d] != 0 || p / base_dil[d] >= src_shape[d] {
            return None;
        }
        flat += (p / base_dil[d]) as usize * src_st[d];
    }
    Some(flat)
}

/// Padding and base-dilation holes take the init value, as the StableHLO spec defines.
#[allow(clippy::too_many_arguments)]
fn reduce_window(
    module: &Module,
    env: &HashMap<ValueId, Tensor>,
    operand: ValueId,
    init: ValueId,
    region: (&[InstrResult], &[ValueId]),
    [dims, strides, base_dil, win_dil]: [&[i64]; 4],
    padding: &[(i64, i64)],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let src = lookup_ref(env, operand)?;
    let init = lookup(env, init)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let n_win = dims.iter().product::<i64>() as usize;
    let mut local = env.clone();
    let e = src.elem_size();
    for o in 0..result_ty.num_elements() {
        let mut acc = init.clone();
        for w in 0..n_win {
            let value = window_source(
                [o, w],
                [&result_ty.shape, &src.ty.shape, dims],
                [strides, base_dil, win_dil],
                padding,
            )
            .map_or_else(|| init.clone(), |f| scalar_at(src, f));
            acc = apply_region(module, &mut local, region, acc, value)?;
        }
        out.bytes[o * e..(o + 1) * e].copy_from_slice(&acc.bytes);
    }
    Ok(out)
}

/// Each source element scatters onto the element its window selects; like XLA, padding
/// is never selected and `select(current, candidate)` true keeps the current element.
#[allow(clippy::too_many_arguments)]
fn select_and_scatter(
    module: &Module,
    env: &HashMap<ValueId, Tensor>,
    [operand, source, init]: [ValueId; 3],
    select: (&[InstrResult], &[ValueId]),
    scatter: (&[InstrResult], &[ValueId]),
    dims: &[i64],
    strides: &[i64],
    padding: &[(i64, i64)],
    result_ty: &TensorType,
) -> Result<Tensor, InterpError> {
    let op = lookup_ref(env, operand)?;
    let src = lookup_ref(env, source)?;
    let init = lookup(env, init)?;
    let mut out = Tensor::alloc(result_ty.clone());
    let e = op.elem_size();
    for slot in out.bytes.chunks_exact_mut(e) {
        slot.copy_from_slice(&init.bytes);
    }
    let ones = vec![1; dims.len()];
    let n_win = dims.iter().product::<i64>() as usize;
    let mut local = env.clone();
    for s in 0..src.elem_count() {
        let mut selected: Option<usize> = None;
        for w in 0..n_win {
            let Some(f) = window_source(
                [s, w],
                [&src.ty.shape, &op.ty.shape, dims],
                [strides, &ones, &ones],
                padding,
            ) else {
                continue;
            };
            let keep = match selected {
                Some(cur) => {
                    let (a, b) = (scalar_at(op, cur), scalar_at(op, f));
                    apply_region(module, &mut local, select, a, b)?.bytes[0] != 0
                }
                None => false,
            };
            if !keep {
                selected = Some(f);
            }
        }
        if let Some(f) = selected {
            let (a, b) = (scalar_at(&out, f), scalar_at(src, s));
            let updated = apply_region(module, &mut local, scatter, a, b)?;
            out.bytes[f * e..(f + 1) * e].copy_from_slice(&updated.bytes);
        }
    }
    Ok(out)
}

/// Stable sort along one axis, ordered by evaluating the comparator region on each pair.
fn sort(
    module: &Module,
    env: &HashMap<ValueId, Tensor>,
    inputs: &[ValueId],
    dimension: i64,
    comparator: &[InstrResult],
    params: &[ValueId],
) -> Result<Vec<Tensor>, InterpError> {
    let ops = inputs
        .iter()
        .map(|v| lookup(env, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let shape = ops[0].ty.shape.clone();
    let rank = shape.len() as i64;
    let dim = (if dimension < 0 {
        dimension + rank
    } else {
        dimension
    }) as usize;
    let outer = shape[..dim].iter().product::<i64>() as usize;
    let len = shape[dim] as usize;
    let after = shape[dim + 1..].iter().product::<i64>() as usize;
    let mut outs = ops.clone();
    for o in 0..outer {
        for a in 0..after {
            let at = |k: usize| (o * len + k) * after + a;
            let less = |x: usize, y: usize| -> Result<bool, InterpError> {
                let mut local = env.clone();
                for (i, t) in ops.iter().enumerate() {
                    local.insert(params[2 * i], scalar_at(t, at(x)));
                    local.insert(params[2 * i + 1], scalar_at(t, at(y)));
                }
                Ok(eval_body(module, comparator, &mut local)?[0].bytes[0] != 0)
            };
            let mut err = None;
            let mut perm: Vec<usize> = (0..len).collect();
            perm.sort_by(|&x, &y| match (less(x, y), less(y, x)) {
                (Ok(true), _) => std::cmp::Ordering::Less,
                (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
                (Ok(_), Ok(_)) => std::cmp::Ordering::Equal,
                (Err(e), _) | (_, Err(e)) => {
                    err.get_or_insert(e);
                    std::cmp::Ordering::Equal
                }
            });
            if let Some(e) = err {
                return Err(e);
            }
            for (out, op) in outs.iter_mut().zip(&ops) {
                let e = op.elem_size();
                for (k, &src) in perm.iter().enumerate() {
                    out.bytes[at(k) * e..(at(k) + 1) * e]
                        .copy_from_slice(&op.bytes[at(src) * e..(at(src) + 1) * e]);
                }
            }
        }
    }
    Ok(outs)
}

fn materialize_constant(value: &ConstantValue, ty: &TensorType) -> Result<Tensor, InterpError> {
    let mut out = Tensor::alloc(ty.clone());
    match value {
        ConstantValue::DenseScalar(scalar) => {
            write_scalar(&mut out.bytes, ty.element_type, 0, scalar)?;
        }
        ConstantValue::DenseSplat(scalar, splat_ty) => {
            let n = ty.num_elements();
            if splat_ty.num_elements() > n {
                return Err(InterpError::SplatTooLarge {
                    actual: splat_ty.num_elements(),
                    expected: n,
                });
            }
            for i in 0..n {
                write_scalar(&mut out.bytes, ty.element_type, i, scalar)?;
            }
        }
        ConstantValue::DenseArray(values) => {
            if values.len() > ty.num_elements() {
                return Err(InterpError::DenseArrayTooLarge);
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
                return Err(InterpError::ExternalConstantMismatch);
            }
            out.bytes.copy_from_slice(data.as_bytes());
        }
    }
    Ok(out)
}

fn lookup(env: &HashMap<ValueId, Tensor>, id: ValueId) -> Result<Tensor, InterpError> {
    env.get(&id).cloned().ok_or(InterpError::UnknownValue(id))
}

fn lookup_ref(env: &HashMap<ValueId, Tensor>, id: ValueId) -> Result<&Tensor, InterpError> {
    env.get(&id).ok_or(InterpError::UnknownValue(id))
}

fn expect_ty(tensor: &Tensor, ty: ElementType) -> Result<(), InterpError> {
    if tensor.ty.element_type == ty {
        Ok(())
    } else {
        Err(InterpError::ExpectedElementType {
            expected: ty,
            actual: tensor.ty.element_type,
        })
    }
}

fn is_true(tensor: &Tensor) -> Result<bool, InterpError> {
    if tensor.bytes.is_empty() {
        return Err(InterpError::EmptyPredicate);
    }
    Ok(match tensor.ty.element_type {
        ElementType::I1 => tensor.bytes[0] != 0,
        ElementType::F64 => read_f64(&tensor.bytes, ElementType::F64, 0) != 0.0,
        _ => read_int(&tensor.bytes, tensor.ty.element_type, 0) != 0,
    })
}

fn scalar_as_i64(tensor: &Tensor) -> Result<i64, InterpError> {
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

fn cmp_ord<T: PartialOrd>(a: T, b: T, direction: CompareDirection) -> bool {
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
) -> Result<(), InterpError> {
    if is_integral(ty) {
        write_int(bytes, ty, index, scalar.as_i64());
    } else {
        write_f64(bytes, ty, index, scalar.as_f64());
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
        ElementType::UI64 => read_int(bytes, ty, index) as u64 as f64,
        ElementType::I64 | ElementType::I32 | ElementType::UI32 => {
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
        assert!(matches!(err, InterpError::SplatTooLarge { .. }));
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
        assert!(matches!(err, InterpError::OperandIndexOutOfRange));
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
        assert!(matches!(err, InterpError::BroadcastDimension { .. }));
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
        assert!(matches!(err, InterpError::BroadcastDimension { .. }));
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
        let out = dot_general(&env, ValueId(1), ValueId(2), &dims).unwrap();
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
        let out = dot_general(&env, ValueId(1), ValueId(2), &dims).unwrap();
        assert_eq!(read_f64s(&out), vec![6.0, 15.0]);
    }

    fn dot_dims(lhs: &[i64], rhs: &[i64]) -> DotDims {
        DotDims {
            lhs_contracting: lhs.to_vec(),
            rhs_contracting: rhs.to_vec(),
            lhs_batch: vec![],
            rhs_batch: vec![],
        }
    }

    #[test]
    fn vector_dot_rejects_unequal_lengths() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2], &[1.0, 2.0]));
        env.insert(ValueId(2), f64s(&[3], &[4.0, 5.0, 6.0]));
        let err = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[0], &[0])).unwrap_err();
        assert!(matches!(err, InterpError::DotContractingSize), "{err}");
    }

    #[test]
    fn vector_dot_without_contracting_dims_is_an_outer_product() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2], &[1.0, 2.0]));
        env.insert(ValueId(2), f64s(&[2], &[3.0, 4.0]));
        let out = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[], &[])).unwrap();
        assert_eq!(out.ty.shape, [2, 2]);
        assert_eq!(read_f64s(&out), vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn matrix_vector_contracts_the_named_lhs_dimension() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        env.insert(ValueId(2), f64s(&[2], &[10.0, 20.0]));
        let out = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[0], &[0])).unwrap();
        assert_eq!(read_f64s(&out), vec![90.0, 120.0, 150.0]);
    }

    #[test]
    fn matrix_vector_rejects_mismatched_contract_extent() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        env.insert(ValueId(2), f64s(&[2], &[10.0, 20.0]));
        let err = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[1], &[0])).unwrap_err();
        assert!(matches!(err, InterpError::DotContractingSize), "{err}");
    }

    #[test]
    fn vector_matrix_contracts_the_named_rhs_dimension() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2], &[10.0, 20.0]));
        env.insert(ValueId(2), f64s(&[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        let out = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[0], &[0])).unwrap();
        assert_eq!(read_f64s(&out), vec![180.0, 210.0, 240.0]);
    }

    #[test]
    fn vector_matrix_rejects_an_empty_contracting_dim() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[2], &[1.0, 2.0]));
        env.insert(ValueId(2), f64s(&[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        let err = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[0], &[])).unwrap_err();
        assert!(matches!(err, InterpError::DotContractingSize), "{err}");
    }

    #[test]
    fn vector_matrix_rejects_mismatched_contract_extent() {
        let mut env = HashMap::new();
        env.insert(ValueId(1), f64s(&[4], &[1.0, 2.0, 3.0, 4.0]));
        env.insert(ValueId(2), f64s(&[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        let err = dot_general(&env, ValueId(1), ValueId(2), &dot_dims(&[0], &[1])).unwrap_err();
        assert!(matches!(err, InterpError::DotContractingSize), "{err}");
    }

    fn ints(element_type: ElementType, shape: &[i64], words: &[i64]) -> Tensor {
        let mut bytes = Vec::new();
        for &word in words {
            match element_type {
                ElementType::I64 | ElementType::UI64 => {
                    bytes.extend_from_slice(&word.to_le_bytes())
                }
                ElementType::I32 | ElementType::UI32 => {
                    bytes.extend_from_slice(&(word as i32).to_le_bytes());
                }
                ElementType::I1 => bytes.push(u8::from(word != 0)),
                ElementType::F64 | ElementType::F32 => panic!("ints() is for integer tensors"),
            }
        }
        Tensor {
            ty: ty(shape, element_type),
            bytes,
        }
    }

    #[test]
    fn convert_i64_to_ui64_is_exact_above_2_53() {
        let mut env = HashMap::new();
        let value = 9_007_199_254_740_993_i64;
        env.insert(ValueId(0), ints(ElementType::I64, &[1], &[value]));
        let out = convert(&env, ValueId(0), &ty(&[1], ElementType::UI64)).unwrap();
        let bits = u64::from_le_bytes(out.bytes[..8].try_into().unwrap());
        assert_eq!(bits, value as u64);
    }

    #[test]
    fn convert_ui64_to_f64_is_unsigned() {
        let mut env = HashMap::new();
        env.insert(
            ValueId(0),
            ints(ElementType::UI64, &[1], &[u64::MAX as i64]),
        );
        let out = convert(&env, ValueId(0), &ty(&[1], ElementType::F64)).unwrap();
        let value = f64::from_le_bytes(out.bytes[..8].try_into().unwrap());
        assert_eq!(value, u64::MAX as f64);
    }

    #[test]
    fn compare_ui64_above_i64_max_is_unsigned() {
        let mut env = HashMap::new();
        env.insert(
            ValueId(0),
            ints(ElementType::UI64, &[1], &[u64::MAX as i64]),
        );
        env.insert(ValueId(1), ints(ElementType::UI64, &[1], &[1]));
        let out = compare(
            &env,
            ValueId(0),
            ValueId(1),
            &ty(&[1], ElementType::I1),
            CompareDirection::Gt,
            CompareType::Unsigned,
        )
        .unwrap();
        assert_eq!(out.bytes, vec![1]);

        env.insert(
            ValueId(2),
            ints(ElementType::UI32, &[1], &[i64::from(u32::MAX)]),
        );
        env.insert(ValueId(3), ints(ElementType::UI32, &[1], &[1]));
        let out = compare(
            &env,
            ValueId(2),
            ValueId(3),
            &ty(&[1], ElementType::I1),
            CompareDirection::Gt,
            CompareType::Signed,
        )
        .unwrap();
        assert_eq!(out.bytes, vec![1]);
    }

    #[test]
    fn integer_constant_above_2_53_is_exact() {
        let value = (1_i64 << 53) + 1;
        let out = materialize_constant(
            &ConstantValue::DenseScalar(ScalarValue::I64(value)),
            &ty(&[1], ElementType::I64),
        )
        .unwrap();
        let bits = i64::from_le_bytes(out.bytes[..8].try_into().unwrap());
        assert_eq!(bits, value);
    }
}
