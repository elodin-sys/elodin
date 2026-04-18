use std::collections::HashMap;

use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{alt, opt};
use winnow::prelude::*;
use winnow::token::{any, take_till, take_while};

use crate::ir::*;

type PResult<T> = winnow::Result<T>;
type Stream<'a> = &'a str;

pub fn parse_module(input: &str) -> Result<Module, String> {
    let mut s = input;
    match module_parser(&mut s) {
        Ok(mut m) => {
            attach_source_lines(&mut m, input);
            Ok(m)
        }
        Err(_e) => {
            let remaining = &s[..s.len().min(200)];
            let consumed = input.len() - s.len();
            let line_num = input[..consumed].matches('\n').count() + 1;
            Err(format!(
                "Parse error at line {line_num} (byte {consumed}): near: {remaining:?}"
            ))
        }
    }
}

/// After `module_parser` returns, walk the raw input once to locate
/// each `func.func @name` declaration and populate the matching
/// `FuncDef`'s `source_line`. Used by the profiler's Tracy zones so
/// the GUI can jump back to the StableHLO source. Linear-time: one
/// pass builds a `name → line` map, one pass back-fills `FuncDef`.
fn attach_source_lines(module: &mut Module, input: &str) {
    let mut name_to_line: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (i, line) in input.lines().enumerate() {
        // Match on `func.func` with an `@name` somewhere on the same
        // line. Handles both `func.func @foo(` and `func.func private
        // @foo(` forms. Line number is 1-indexed to match typical
        // editor conventions.
        if let Some(pos) = line.find("func.func") {
            let rest = &line[pos..];
            if let Some(at_off) = rest.find('@') {
                let after_at = &rest[at_off + 1..];
                let end = after_at
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .unwrap_or(after_at.len());
                if end > 0 {
                    let name = &after_at[..end];
                    name_to_line.insert(name.to_string(), (i + 1) as u32);
                }
            }
        }
    }
    for f in &mut module.functions {
        if let Some(line) = name_to_line.get(&f.name) {
            f.source_line = Some(*line);
        }
    }
}

// ---------------------------------------------------------------------------
// Module & function structure
// ---------------------------------------------------------------------------

fn module_parser(input: &mut Stream<'_>) -> PResult<Module> {
    ws(input)?;
    skip_until_brace(input)?;
    let _ = '{'.parse_next(input)?;
    ws(input)?;

    let mut functions = Vec::new();
    while !input.starts_with('}') && !input.is_empty() {
        if input.starts_with("func.func") {
            let f = func_def(input)?;
            functions.push(f);
        } else {
            skip_line(input)?;
        }
        ws(input)?;
    }
    let _ = opt('}').parse_next(input)?;
    Ok(Module::new(functions))
}

fn func_def(input: &mut Stream<'_>) -> PResult<FuncDef> {
    let _ = "func.func".parse_next(input)?;
    ws(input)?;
    let is_public = opt("public").parse_next(input)?.is_some();
    if is_public {
        ws(input)?;
    }
    let is_private = opt("private").parse_next(input)?.is_some();
    if is_private {
        ws(input)?;
    }
    let _ = '@'.parse_next(input)?;
    let name = ident(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;

    let mut ctx = ValueCtx::new();
    let mut params = Vec::new();

    if !input.starts_with(')') {
        loop {
            ws(input)?;
            let vn = value_name(input)?;
            ws(input)?;
            let _ = ':'.parse_next(input)?;
            ws(input)?;
            let ty = tensor_type(input)?;
            let vid = ctx.get_or_create(&vn);
            params.push((vid, ty));
            ws(input)?;
            if input.starts_with(')') {
                break;
            }
            let _ = ','.parse_next(input)?;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let ret_types = if input.starts_with("->") {
        let _ = "->".parse_next(input)?;
        ws(input)?;
        let rt = result_types(input)?;
        ws(input)?;
        rt
    } else {
        Vec::new()
    };

    if input.starts_with("attributes") {
        let _ = take_till(0.., '{').parse_next(input)?;
    }

    let _ = '{'.parse_next(input)?;
    ws(input)?;

    let body = parse_body(input, &mut ctx)?;

    ws(input)?;
    let _ = '}'.parse_next(input)?;
    ws(input)?;

    Ok(FuncDef {
        name,
        is_public,
        params,
        result_types: ret_types,
        body,
        // Populated by `parse_module`'s post-pass via a scan for
        // `func.func @name` occurrences. Leave `None` here so this
        // inner parser stays line-agnostic.
        source_line: None,
    })
}

// ---------------------------------------------------------------------------
// Body & instruction dispatch
// ---------------------------------------------------------------------------

fn parse_body(input: &mut Stream<'_>, ctx: &mut ValueCtx) -> PResult<Vec<InstrResult>> {
    let mut instrs = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with('}')
            || input.starts_with("stablehlo.return")
            || input.starts_with("\"stablehlo.return\"")
            || input.starts_with("return")
            || input.is_empty()
        {
            break;
        }
        if let Some(ir) = parse_instruction(input, ctx)? {
            instrs.push(ir);
        }
    }

    if input.starts_with("stablehlo.return")
        || input.starts_with("\"stablehlo.return\"")
        || input.starts_with("return")
    {
        let ret = parse_return(input, ctx)?;
        instrs.push(ret);
    }

    Ok(instrs)
}

fn parse_instruction(input: &mut Stream<'_>, ctx: &mut ValueCtx) -> PResult<Option<InstrResult>> {
    ws(input)?;

    if input.starts_with('}') || input.is_empty() {
        return Ok(None);
    }

    let checkpoint = *input;

    let mut result_names: Vec<String> = Vec::new();
    if input.starts_with('%') {
        let vn = value_name(input)?;
        ws(input)?;

        if input.starts_with(':') && !input.starts_with(": ") {
            // Multi-result: %name:N = op ...
            let _ = ':'.parse_next(input)?;
            let n = integer(input)?;
            result_names.push(vn.clone());
            // Alias "name#0" to the same ValueId as "name" so both
            // %name and %name#0 resolve to the first result.
            ctx.alias(&format!("{}#0", vn), &vn);
            for i in 1..n {
                result_names.push(format!("{}#{}", vn, i));
            }
            ws(input)?;
            let _ = '='.parse_next(input)?;
            ws(input)?;
        } else if input.starts_with('=') {
            result_names.push(vn);
            let _ = '='.parse_next(input)?;
            ws(input)?;
        } else {
            *input = checkpoint;
            skip_to_newline(input)?;
            return Ok(None);
        }
    }

    let instr_result = parse_op(input, ctx, &result_names)?;
    Ok(instr_result)
}

fn parse_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    ws(input)?;

    if input.starts_with("stablehlo.constant") {
        return parse_constant_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.add ") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.add", |l, r| {
            Instruction::Add { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.subtract") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.subtract", |l, r| {
            Instruction::Subtract { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.multiply") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.multiply", |l, r| {
            Instruction::Multiply { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.divide") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.divide", |l, r| {
            Instruction::Divide { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.maximum") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.maximum", |l, r| {
            Instruction::Maximum { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.xor") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.xor", |l, r| {
            Instruction::Xor { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.or ") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.or", |l, r| {
            Instruction::Or { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.and") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.and", |l, r| {
            Instruction::And { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.shift_left") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.shift_left", |l, r| {
            Instruction::ShiftLeft { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.shift_right_logical") {
        return parse_binary_op(
            input,
            ctx,
            result_names,
            "stablehlo.shift_right_logical",
            |l, r| Instruction::ShiftRightLogical { lhs: l, rhs: r },
        );
    }
    if input.starts_with("stablehlo.shift_right_arithmetic") {
        return parse_binary_op(
            input,
            ctx,
            result_names,
            "stablehlo.shift_right_arithmetic",
            |l, r| Instruction::ShiftRightArithmetic { lhs: l, rhs: r },
        );
    }
    if input.starts_with("stablehlo.negate") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.negate", |o| {
            Instruction::Negate { operand: o }
        });
    }
    if input.starts_with("stablehlo.sqrt") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.sqrt", |o| {
            Instruction::Sqrt { operand: o }
        });
    }
    if input.starts_with("stablehlo.rsqrt") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.rsqrt", |o| {
            Instruction::Rsqrt { operand: o }
        });
    }
    if input.starts_with("stablehlo.log_plus_one") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.log_plus_one", |o| {
            Instruction::Log1p { operand: o }
        });
    }
    if input.starts_with("stablehlo.is_finite") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.is_finite", |o| {
            Instruction::IsFinite { operand: o }
        });
    }
    if input.starts_with("stablehlo.not ") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.not", |o| {
            Instruction::Not { operand: o }
        });
    }
    if input.starts_with("stablehlo.ceil") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.ceil", |o| {
            Instruction::Ceil { operand: o }
        });
    }
    if input.starts_with("stablehlo.convert") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.convert", |o| {
            Instruction::Convert { operand: o }
        });
    }
    if input.starts_with("stablehlo.bitcast_convert") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.bitcast_convert", |o| {
            Instruction::BitcastConvert { operand: o }
        });
    }
    if input.starts_with("stablehlo.reshape") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.reshape", |o| {
            Instruction::Reshape { operand: o }
        });
    }
    if input.starts_with("stablehlo.broadcast_in_dim") {
        return parse_broadcast_in_dim(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.slice") {
        return parse_slice_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.concatenate") {
        return parse_concatenate_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.dot_general") {
        return parse_dot_general_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.reduce(") || input.starts_with("stablehlo.reduce (") {
        return parse_reduce_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.iota") {
        return parse_iota_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.compare") {
        return parse_compare_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.select") {
        return parse_select_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.gather\"") || input.starts_with("stablehlo.gather") {
        return parse_gather_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.transpose") {
        return parse_transpose_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.dynamic_update_slice") {
        return parse_dynamic_update_slice_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.dynamic_slice") {
        return parse_dynamic_slice_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.sort\"") || input.starts_with("stablehlo.sort") {
        return parse_sort_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.batch_norm_inference\"") {
        return parse_batch_norm_inference_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.real_dynamic_slice\"") {
        return parse_real_dynamic_slice_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.map\"") || input.starts_with("stablehlo.map") {
        return parse_map_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.reduce_window\"") {
        return parse_reduce_window_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.select_and_scatter\"") {
        return parse_select_and_scatter_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.convolution\"") || input.starts_with("stablehlo.convolution")
    {
        return parse_convolution_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.cholesky") {
        return parse_cholesky_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.triangular_solve\"")
        || input.starts_with("stablehlo.triangular_solve")
    {
        return parse_triangular_solve_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.fft") {
        return parse_fft_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.rng\"") || input.starts_with("stablehlo.rng") {
        return parse_rng_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.while") {
        return parse_while_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.case\"") || input.starts_with("stablehlo.case") {
        return parse_case_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.sine") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.sine", |o| {
            Instruction::Sine { operand: o }
        });
    }
    if input.starts_with("stablehlo.cosine") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.cosine", |o| {
            Instruction::Cosine { operand: o }
        });
    }
    if input.starts_with("stablehlo.atan2") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.atan2", |l, r| {
            Instruction::Atan2 { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.abs") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.abs", |o| {
            Instruction::Abs { operand: o }
        });
    }
    if input.starts_with("stablehlo.minimum") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.minimum", |l, r| {
            Instruction::Minimum { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.sign") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.sign", |o| {
            Instruction::Sign { operand: o }
        });
    }
    if input.starts_with("stablehlo.remainder") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.remainder", |l, r| {
            Instruction::Remainder { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.exponential_minus_one") {
        return parse_unary_op(
            input,
            ctx,
            result_names,
            "stablehlo.exponential_minus_one",
            |o| Instruction::Expm1 { operand: o },
        );
    }
    if input.starts_with("stablehlo.exponential") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.exponential", |o| {
            Instruction::Exponential { operand: o }
        });
    }
    if input.starts_with("stablehlo.log ") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.log", |o| {
            Instruction::Log { operand: o }
        });
    }
    if input.starts_with("stablehlo.clamp") {
        return parse_clamp_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.power") {
        return parse_binary_op(input, ctx, result_names, "stablehlo.power", |l, r| {
            Instruction::Power { lhs: l, rhs: r }
        });
    }
    if input.starts_with("stablehlo.reverse") {
        return parse_reverse_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.tanh") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.tanh", |o| {
            Instruction::Tanh { operand: o }
        });
    }
    if input.starts_with("stablehlo.tan ") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.tan", |o| {
            Instruction::Tan { operand: o }
        });
    }
    if input.starts_with("stablehlo.floor") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.floor", |o| {
            Instruction::Floor { operand: o }
        });
    }
    if input.starts_with("stablehlo.round_nearest_even") {
        return parse_unary_op(
            input,
            ctx,
            result_names,
            "stablehlo.round_nearest_even",
            |o| Instruction::RoundNearestEven { operand: o },
        );
    }
    if input.starts_with("chlo.erf_inv") {
        return parse_unary_op(input, ctx, result_names, "chlo.erf_inv", |o| {
            Instruction::ErfInv { operand: o }
        });
    }
    if input.starts_with("stablehlo.pad") {
        return parse_pad_op(input, ctx, result_names);
    }
    if input.starts_with("\"stablehlo.scatter\"") || input.starts_with("stablehlo.scatter") {
        return parse_scatter_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.custom_call") {
        return parse_custom_call_op(input, ctx, result_names);
    }
    if input.starts_with("stablehlo.output_operand_alias") {
        skip_to_newline(input)?;
        return Ok(None);
    }
    if input.starts_with("chlo.acos") {
        return parse_unary_op(input, ctx, result_names, "chlo.acos", |o| {
            Instruction::Acos { operand: o }
        });
    }
    if input.starts_with("chlo.asin") {
        return parse_unary_op(input, ctx, result_names, "chlo.asin", |o| {
            Instruction::Asin { operand: o }
        });
    }
    if input.starts_with("chlo.atan ") {
        return parse_unary_op(input, ctx, result_names, "chlo.atan", |o| {
            Instruction::Atan { operand: o }
        });
    }
    if input.starts_with("chlo.sinh") {
        return parse_unary_op(input, ctx, result_names, "chlo.sinh", |o| {
            Instruction::Sinh { operand: o }
        });
    }
    if input.starts_with("chlo.cosh") {
        return parse_unary_op(input, ctx, result_names, "chlo.cosh", |o| {
            Instruction::Cosh { operand: o }
        });
    }
    if input.starts_with("chlo.erfc") {
        return parse_unary_op(input, ctx, result_names, "chlo.erfc", |o| {
            Instruction::Erfc { operand: o }
        });
    }
    if input.starts_with("chlo.square") {
        return parse_unary_op(input, ctx, result_names, "chlo.square", |o| {
            Instruction::Multiply { lhs: o, rhs: o }
        });
    }
    if input.starts_with("stablehlo.expm1") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.expm1", |o| {
            Instruction::Expm1 { operand: o }
        });
    }
    if input.starts_with("stablehlo.cbrt") {
        return parse_unary_op(input, ctx, result_names, "stablehlo.cbrt", |o| {
            Instruction::Cbrt { operand: o }
        });
    }
    if input.starts_with("call @") || input.starts_with("func.call @") {
        return parse_call_op(input, ctx, result_names);
    }
    if input.starts_with("return") || input.starts_with("stablehlo.return") {
        return Ok(None);
    }

    let line: &str = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let op_name = line.split_whitespace().next().unwrap_or("unknown");
    eprintln!("warning: skipping unrecognized op: {}", op_name);
    Ok(None)
}

// ---------------------------------------------------------------------------
// Constant
// ---------------------------------------------------------------------------

fn parse_constant_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.constant".parse_next(input)?;
    ws(input)?;
    let _ = "dense<".parse_next(input)?;

    let value = parse_dense_value(input)?;
    let _ = '>'.parse_next(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let ty = tensor_type(input)?;

    skip_to_newline(input)?;

    let value = match value {
        ConstantValue::DenseScalar(sv) if !ty.is_scalar() => {
            ConstantValue::DenseSplat(sv, ty.clone())
        }
        other => other,
    };

    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Constant { value },
    }))
}

fn parse_dense_value(input: &mut Stream<'_>) -> PResult<ConstantValue> {
    if input.starts_with('"') {
        // Hex blob: dense<"0xDEADBEEF...">
        let _ = '"'.parse_next(input)?;
        let hex_str: &str = take_till(0.., '"').parse_next(input)?;
        let _ = '"'.parse_next(input)?;
        let hex_data = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        let mut bytes = Vec::new();
        let mut i = 0;
        while i + 1 < hex_data.len() {
            if let Ok(b) = u8::from_str_radix(&hex_data[i..i + 2], 16) {
                bytes.push(b);
            }
            i += 2;
        }
        let mut values = Vec::new();
        let mut j = 0;
        while j + 7 < bytes.len() {
            let arr: [u8; 8] = bytes[j..j + 8].try_into().unwrap();
            values.push(ScalarValue::F64(f64::from_le_bytes(arr)));
            j += 8;
        }
        return Ok(ConstantValue::DenseArray(values));
    }
    if input.starts_with('[') {
        let _ = '['.parse_next(input)?;
        let mut vals = Vec::new();
        loop {
            ws(input)?;
            if input.starts_with(']') {
                break;
            }
            if input.starts_with('[') {
                let inner = parse_dense_value(input)?;
                match inner {
                    ConstantValue::DenseArray(arr) => vals.extend(arr),
                    ConstantValue::DenseScalar(s) => vals.push(s),
                    _ => {}
                }
            } else {
                let sv = parse_scalar_value(input)?;
                vals.push(sv);
            }
            ws(input)?;
            let _ = opt(',').parse_next(input)?;
        }
        let _ = ']'.parse_next(input)?;
        Ok(ConstantValue::DenseArray(vals))
    } else {
        let sv = parse_scalar_value(input)?;
        Ok(ConstantValue::DenseScalar(sv))
    }
}

fn parse_scalar_value(input: &mut Stream<'_>) -> PResult<ScalarValue> {
    if input.starts_with("true") {
        let _ = "true".parse_next(input)?;
        return Ok(ScalarValue::I1(true));
    }
    if input.starts_with("false") {
        let _ = "false".parse_next(input)?;
        return Ok(ScalarValue::I1(false));
    }
    if input.starts_with("0x") {
        let _ = "0x".parse_next(input)?;
        let hex: &str = take_while(1.., |c: char| c.is_ascii_hexdigit()).parse_next(input)?;
        let bits = u64::from_str_radix(hex, 16).unwrap();
        return Ok(ScalarValue::F64(f64::from_bits(bits)));
    }

    let checkpoint = *input;
    let neg = opt('-').parse_next(input)?;
    let digits: &str = digit1.parse_next(input)?;

    if input.starts_with('.') || input.starts_with('E') || input.starts_with('e') {
        *input = checkpoint;
        let v = float_literal(input)?;
        return Ok(ScalarValue::F64(v));
    }

    if neg.is_some() {
        let full = format!("-{digits}");
        let v: i64 = full
            .parse()
            .expect("negative integer literal too large for i64");
        Ok(ScalarValue::I64(v))
    } else if let Ok(v) = digits.parse::<i64>() {
        Ok(ScalarValue::I64(v))
    } else {
        let v: u64 = digits.parse().expect("integer literal too large for u64");
        Ok(ScalarValue::UI64(v))
    }
}

// ---------------------------------------------------------------------------
// Binary & unary op helpers
// ---------------------------------------------------------------------------

fn parse_binary_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
    mut op_name: &str,
    make_instr: impl FnOnce(ValueId, ValueId) -> Instruction,
) -> PResult<Option<InstrResult>> {
    let _ = op_name.parse_next(input)?;
    ws(input)?;
    let lhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;

    let rest_of_line = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_result_type_from_str(rest_of_line);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: make_instr(lhs, rhs),
    }))
}

fn parse_unary_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
    mut op_name: &str,
    make_instr: impl FnOnce(ValueId) -> Instruction,
) -> PResult<Option<InstrResult>> {
    let _ = op_name.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_result_type_from_str(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: make_instr(operand),
    }))
}

// ---------------------------------------------------------------------------
// Specific op parsers
// ---------------------------------------------------------------------------

fn parse_broadcast_in_dim(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.broadcast_in_dim".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let _ = "dims = [".parse_next(input)?;
    let dims_str: &str = take_till(0.., ']').parse_next(input)?;
    let _ = ']'.parse_next(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let dims: Vec<i64> = dims_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::BroadcastInDim {
            operand,
            broadcast_dims: dims,
        },
    }))
}

fn parse_slice_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.slice".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = '['.parse_next(input)?;
    let slice_str: &str = take_till(0.., ']').parse_next(input)?;
    let _ = ']'.parse_next(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut starts = Vec::new();
    let mut limits = Vec::new();
    for part in slice_str.split(',') {
        let part = part.trim();
        let parts: Vec<&str> = part.split(':').collect();
        if parts.len() >= 2 {
            starts.push(parts[0].trim().parse::<i64>().unwrap_or(0));
            limits.push(parts[1].trim().parse::<i64>().unwrap_or(0));
        }
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Slice {
            operand,
            start_indices: starts,
            limit_indices: limits,
        },
    }))
}

fn parse_concatenate_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.concatenate".parse_next(input)?;
    ws(input)?;
    let operands = parse_value_list(input, ctx)?;
    ws(input)?;
    let _ = "dim = ".parse_next(input)?;
    let dim = integer(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Concatenate {
            operands,
            dimension: dim,
        },
    }))
}

fn parse_dot_general_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.dot_general".parse_next(input)?;
    ws(input)?;
    let lhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut lhs_batch = Vec::new();
    let mut rhs_batch = Vec::new();
    let mut lhs_contracting = Vec::new();
    let mut rhs_contracting = Vec::new();

    // Handle "contracting_dims = [X] x [Y]" format
    if let Some(idx) = rest.find("contracting_dims = [") {
        let after = &rest[idx + 20..];
        if let Some(x_idx) = after.find("] x [") {
            lhs_contracting = parse_int_list(&after[..x_idx]);
            let after2 = &after[x_idx + 5..];
            if let Some(end) = after2.find(']') {
                rhs_contracting = parse_int_list(&after2[..end]);
            }
        }
    }
    if let Some(idx) = rest.find("batching_dims = [") {
        let after = &rest[idx + 17..];
        if let Some(x_idx) = after.find("] x [") {
            lhs_batch = parse_int_list(&after[..x_idx]);
            let after2 = &after[x_idx + 5..];
            if let Some(end) = after2.find(']') {
                rhs_batch = parse_int_list(&after2[..end]);
            }
        }
    }

    // Handle "#stablehlo.dot<...>" attribute format
    if rest.contains("lhs_contracting_dimensions = [") {
        lhs_contracting = extract_bracket_ints(rest, "lhs_contracting_dimensions = [");
    }
    if rest.contains("rhs_contracting_dimensions = [") {
        rhs_contracting = extract_bracket_ints(rest, "rhs_contracting_dimensions = [");
    }
    if rest.contains("lhs_batching_dimensions = [") {
        lhs_batch = extract_bracket_ints(rest, "lhs_batching_dimensions = [");
    }
    if rest.contains("rhs_batching_dimensions = [") {
        rhs_batch = extract_bracket_ints(rest, "rhs_batching_dimensions = [");
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::DotGeneral {
            lhs,
            rhs,
            dims: DotDims {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            },
        },
    }))
}

fn parse_reduce_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.reduce".parse_next(input)?;
    let _ = opt(' ').parse_next(input)?;
    let _ = '('.parse_next(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = "init:".parse_next(input)?;
    ws(input)?;
    let init = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Multi-operand reduce: (%arg0 init: %cst), (%idx init: %c) across ...
    let mut second_operand = None;
    if input.starts_with(',') {
        let _ = ','.parse_next(input)?;
        ws(input)?;
        let _ = '('.parse_next(input)?;
        let op2 = parse_value_ref(input, ctx)?;
        ws(input)?;
        let _ = "init:".parse_next(input)?;
        ws(input)?;
        let init2 = parse_value_ref(input, ctx)?;
        ws(input)?;
        let _ = ')'.parse_next(input)?;
        ws(input)?;
        second_operand = Some((op2, init2));
    }

    let first_line = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut full_text = first_line.to_string();

    // The reducer body can appear inline or on the next line(s).
    let has_inline_body = first_line.trim().ends_with('{');
    let has_nextline_body = {
        let trimmed = input.trim_start();
        trimmed.starts_with("reducer(") || trimmed.starts_with("reducer (")
    };

    if has_inline_body || has_nextline_body {
        if has_nextline_body && !has_inline_body {
            ws(input)?;
        }
        let pre_brace = take_till(0.., '{').parse_next(input)?;
        full_text.push_str(pre_brace);
        let _ = '{'.parse_next(input)?;
        full_text.push('{');
        let mut depth = 1u32;
        while depth > 0 && !input.is_empty() {
            let ch = any.parse_next(input)?;
            full_text.push(ch);
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
            }
        }
    }

    let mut dimensions = Vec::new();
    if let Some(idx) = full_text.find("dimensions = [") {
        let after = &full_text[idx + 14..];
        if let Some(end) = after.find(']') {
            dimensions = parse_int_list(&after[..end]);
        }
    }

    if let Some((indices, _init2)) = second_operand {
        let has_lt = full_text.contains("LT,") || full_text.contains("LE,");
        let has_gt = full_text.contains("GT,") || full_text.contains("GE,");
        let is_min = !has_gt || has_lt;
        let mut types = Vec::new();
        if let Some(arrow) = first_line.rfind("-> ") {
            let after = strip_outer_parens(first_line[arrow + 3..].trim());
            for part in split_tensor_types(after) {
                types.push(parse_tensor_type_from_str(part.trim()));
            }
        }
        if types.len() < 2 {
            types = vec![
                TensorType::scalar(ElementType::F64),
                TensorType::scalar(ElementType::I64),
            ];
        }
        let values = make_values(ctx, result_names, types);
        return Ok(Some(InstrResult {
            values,
            instr: Instruction::ReduceArgminmax {
                values: operand,
                indices,
                dimensions,
                is_min,
            },
        }));
    }

    let op = if full_text.contains("stablehlo.add") {
        ReduceOp::Add
    } else if full_text.contains("stablehlo.minimum") {
        ReduceOp::Minimum
    } else if full_text.contains("stablehlo.maximum") {
        ReduceOp::Maximum
    } else if full_text.contains("stablehlo.and") {
        ReduceOp::And
    } else if full_text.contains("stablehlo.or") {
        ReduceOp::Or
    } else {
        ReduceOp::Add
    };

    let ty = parse_final_type(first_line);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Reduce {
            operand,
            init,
            op,
            dimensions,
        },
    }))
}

fn parse_iota_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.iota".parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut dimension: i64 = 0;
    if let Some(idx) = rest.find("dim = ") {
        let after = &rest[idx + 6..];
        let end = after
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(after.len());
        if let Ok(v) = after[..end].parse::<i64>() {
            dimension = v;
        }
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Iota { dimension },
    }))
}

fn parse_compare_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.compare".parse_next(input)?;
    ws(input)?;

    if input.starts_with('%') {
        // Attribute format: operands first, then comparison_direction/compare_type attrs
        let lhs = parse_value_ref(input, ctx)?;
        ws(input)?;
        let _ = ','.parse_next(input)?;
        ws(input)?;
        let rhs = parse_value_ref(input, ctx)?;
        ws(input)?;

        let rest = take_till(0.., '\n').parse_next(input)?;
        let _ = opt('\n').parse_next(input)?;

        let direction = extract_compare_direction(rest);
        let compare_type = extract_compare_type(rest);
        let ty = parse_final_type(rest);

        let values = make_values(ctx, result_names, vec![ty]);
        Ok(Some(InstrResult {
            values,
            instr: Instruction::Compare {
                lhs,
                rhs,
                direction,
                compare_type,
            },
        }))
    } else {
        // Concise format: direction first, e.g. "LT, %a, %b, FLOAT"
        let dir_str = take_while(1.., |c: char| c.is_ascii_uppercase()).parse_next(input)?;
        let direction = match dir_str {
            "EQ" => CompareDirection::Eq,
            "NE" => CompareDirection::Ne,
            "LT" => CompareDirection::Lt,
            "LE" => CompareDirection::Le,
            "GT" => CompareDirection::Gt,
            "GE" => CompareDirection::Ge,
            _ => CompareDirection::Eq,
        };
        ws(input)?;
        let _ = ','.parse_next(input)?;
        ws(input)?;
        let lhs = parse_value_ref(input, ctx)?;
        ws(input)?;
        let _ = ','.parse_next(input)?;
        ws(input)?;
        let rhs = parse_value_ref(input, ctx)?;
        ws(input)?;
        let _ = ','.parse_next(input)?;
        ws(input)?;
        let ctype_str = take_while(1.., |c: char| c.is_ascii_uppercase()).parse_next(input)?;
        let compare_type = match ctype_str {
            "FLOAT" => CompareType::Float,
            "SIGNED" => CompareType::Signed,
            "UNSIGNED" => CompareType::Unsigned,
            "TOTALORDER" => CompareType::TotalOrder,
            _ => CompareType::Float,
        };
        ws(input)?;
        let _ = ':'.parse_next(input)?;
        ws(input)?;

        let rest = take_till(0.., '\n').parse_next(input)?;
        let _ = opt('\n').parse_next(input)?;

        let ty = parse_final_type(rest);
        let values = make_values(ctx, result_names, vec![ty]);
        Ok(Some(InstrResult {
            values,
            instr: Instruction::Compare {
                lhs,
                rhs,
                direction,
                compare_type,
            },
        }))
    }
}

fn parse_select_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.select".parse_next(input)?;
    ws(input)?;
    let cond = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let on_true = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let on_false = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Select {
            cond,
            on_true,
            on_false,
        },
    }))
}

fn parse_gather_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
    }
    let _ = "stablehlo.gather".parse_next(input)?;
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let indices = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let offset_dims = extract_bracket_ints(rest, "offset_dims = [");
    let collapsed_slice_dims = extract_bracket_ints(rest, "collapsed_slice_dims = [");
    let start_index_map = extract_bracket_ints(rest, "start_index_map = [");

    let mut index_vector_dim: i64 = 1;
    if let Some(idx) = rest.find("index_vector_dim = ") {
        let after = &rest[idx + 19..];
        let end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        if let Ok(v) = after[..end].parse::<i64>() {
            index_vector_dim = v;
        }
    }

    // slice_sizes can be "slice_sizes = [...]" or "slice_sizes = array<i64: ...>"
    let mut slice_sizes = extract_bracket_ints(rest, "slice_sizes = [");
    if slice_sizes.is_empty()
        && let Some(idx) = rest.find("slice_sizes = array<i64: ")
    {
        let after = &rest[idx + 24..];
        if let Some(end) = after.find('>') {
            slice_sizes = parse_int_list(&after[..end]);
        }
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Gather {
            operand,
            indices,
            dims: GatherDims {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
            },
            slice_sizes,
        },
    }))
}

fn parse_transpose_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.transpose".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let permutation = extract_bracket_ints(rest, "dims = [");
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Transpose {
            operand,
            permutation,
        },
    }))
}

fn parse_clamp_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.clamp".parse_next(input)?;
    ws(input)?;
    let min = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let max = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Clamp { operand, min, max },
    }))
}

fn parse_reverse_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.reverse".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let dimensions = extract_bracket_ints(rest, "dims = [");
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Reverse {
            operand,
            dimensions,
        },
    }))
}

fn parse_pad_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.pad".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let padding_value = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let low = extract_bracket_ints(rest, "low = [");
    let high = extract_bracket_ints(rest, "high = [");
    let interior = extract_bracket_ints(rest, "interior = [");
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Pad {
            operand,
            padding_value,
            low,
            high,
            interior,
        },
    }))
}

fn parse_scatter_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with("\"stablehlo.scatter\"") {
        let _ = "\"stablehlo.scatter\"".parse_next(input)?;
    } else {
        let _ = "stablehlo.scatter".parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let indices = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let updates = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Skip everything until the line starting with "}) :" which ends the scatter
    loop {
        let line: &str = take_till(0.., '\n').parse_next(input)?;
        let _ = opt('\n').parse_next(input)?;
        let trimmed = line.trim();
        if trimmed.starts_with("}) :") {
            let ty = parse_final_type(trimmed);
            let values = make_values(ctx, result_names, vec![ty]);
            return Ok(Some(InstrResult {
                values,
                instr: Instruction::Scatter {
                    operand,
                    indices,
                    updates,
                },
            }));
        }
        if input.is_empty() {
            break;
        }
    }

    let values = make_values(
        ctx,
        result_names,
        vec![TensorType::scalar(ElementType::F64)],
    );
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Scatter {
            operand,
            indices,
            updates,
        },
    }))
}

fn parse_custom_call_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.custom_call".parse_next(input)?;
    ws(input)?;
    let _ = '@'.parse_next(input)?;
    let call_target = ident(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;

    let mut operands = Vec::new();
    if !input.starts_with(')') {
        loop {
            ws(input)?;
            if input.starts_with(')') {
                break;
            }
            let v = parse_value_ref(input, ctx)?;
            operands.push(v);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            } else {
                break;
            }
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Capture rest of line, extract backend_config and result types
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty_str = rest.trim();

    let mut backend_config = std::collections::HashMap::new();
    if let Some(bc_start) = ty_str.find("mhlo.backend_config = {") {
        let after = &ty_str[bc_start + "mhlo.backend_config = {".len()..];
        if let Some(bc_end) = after.find('}') {
            let bc_inner = &after[..bc_end];
            for kv in bc_inner.split(',') {
                let kv = kv.trim();
                if kv.is_empty() {
                    continue;
                }
                if let Some((key, val_part)) = kv.split_once('=') {
                    let key = key.trim().to_string();
                    let val_str = val_part.trim().split(':').next().unwrap_or("").trim();
                    if let Ok(v) = val_str.parse::<i64>() {
                        backend_config.insert(key, v);
                    }
                }
            }
        }
    }

    let mut types = Vec::new();
    if let Some(after_arrow) = ty_str.rfind("-> ") {
        let result_part = &ty_str[after_arrow + 3..];
        if result_part.starts_with('(') {
            let inner = result_part.trim_start_matches('(').trim_end_matches(')');
            for part in inner.split(", tensor<") {
                let part = if part.starts_with("tensor<") {
                    part.to_string()
                } else {
                    format!("tensor<{part}")
                };
                types.push(parse_tensor_type_from_str(&part));
            }
        } else {
            types.push(parse_tensor_type_from_str(result_part));
        }
    }

    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::CustomCall {
            call_target,
            operands,
            backend_config,
        },
    }))
}

fn parse_dynamic_slice_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.dynamic_slice".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;

    let mut start_indices = Vec::new();
    loop {
        if input.starts_with("sizes") || input.starts_with(':') {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        start_indices.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
            ws(input)?;
        }
    }

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut slice_sizes = extract_bracket_ints(rest, "sizes = [");
    if slice_sizes.is_empty() {
        slice_sizes = extract_bracket_ints(rest, "sizes = array<i64: ");
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        },
    }))
}

fn parse_dynamic_update_slice_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.dynamic_update_slice".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let update = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;

    let mut start_indices = Vec::new();
    loop {
        if input.starts_with(':') || input.starts_with('\n') || input.is_empty() {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        start_indices.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
            ws(input)?;
        }
    }

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        },
    }))
}

fn parse_call_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with("func.call") {
        let _ = "func.call".parse_next(input)?;
        ws(input)?;
    } else {
        let _ = "call".parse_next(input)?;
        ws(input)?;
    }
    let _ = '@'.parse_next(input)?;
    let callee = ident(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;
    let args = parse_value_list(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut ret_types = Vec::new();
    if let Some(idx) = rest.find("-> ") {
        let after = rest[idx + 3..].trim();
        if after.starts_with('(') {
            let inner = &after[1..after.rfind(')').unwrap_or(after.len())];
            for part in split_tensor_types(inner) {
                ret_types.push(parse_tensor_type_from_str(part.trim()));
            }
        } else {
            ret_types.push(parse_tensor_type_from_str(after));
        }
    }
    if ret_types.is_empty() {
        ret_types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, ret_types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Call { callee, args },
    }))
}

fn parse_while_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.while".parse_next(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;

    let mut init_names = Vec::new();
    let mut init_values = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let vn = value_name(input)?;
        ws(input)?;
        let _ = '='.parse_next(input)?;
        ws(input)?;
        let init_v = parse_value_ref(input, ctx)?;
        init_values.push(init_v);
        init_names.push(vn);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let rest_of_type_line = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut iter_types = Vec::new();
    if let Some(idx) = rest_of_type_line.find(": ") {
        let type_str = &rest_of_type_line[idx + 2..];
        for part in split_tensor_types(type_str.trim()) {
            iter_types.push(parse_tensor_type_from_str(part.trim()));
        }
    }

    let mut cond_ctx = ctx.child();
    let mut cond_iter_ids = Vec::new();
    for name in &init_names {
        cond_iter_ids.push(cond_ctx.get_or_create(name));
    }

    ws(input)?;
    let _ = "cond".parse_next(input)?;
    ws(input)?;
    let _ = '{'.parse_next(input)?;
    ws(input)?;
    let cond_body = parse_body(input, &mut cond_ctx)?;
    ws(input)?;
    let _ = '}'.parse_next(input)?;
    ws(input)?;

    let mut body_ctx = ctx.child();
    let mut body_iter_ids = Vec::new();
    for name in &init_names {
        body_iter_ids.push(body_ctx.get_or_create(name));
    }

    let _ = "do".parse_next(input)?;
    ws(input)?;
    let _ = '{'.parse_next(input)?;
    ws(input)?;
    let loop_body = parse_body(input, &mut body_ctx)?;
    ws(input)?;
    let _ = '}'.parse_next(input)?;

    let values = if !result_names.is_empty() && !iter_types.is_empty() {
        make_values(ctx, result_names, iter_types)
    } else {
        vec![]
    };

    Ok(Some(InstrResult {
        values,
        instr: Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids: body_iter_ids,
        },
    }))
}

fn parse_sort_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.sort".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.sort".parse_next(input)?;
    }
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let mut operands = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        operands.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        } else {
            break;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let mut dimension: i64 = 0;
    let mut is_stable = false;

    // Parse optional <{dimension = N, is_stable = true}>
    if input.starts_with('<') {
        let _ = '<'.parse_next(input)?;
        let attrs: &str = take_till(0.., '>').parse_next(input)?;
        let _ = '>'.parse_next(input)?;
        ws(input)?;
        for segment in attrs.split(',') {
            let seg = segment.trim().trim_start_matches('{').trim_end_matches('}');
            if let Some(rest) = seg.strip_prefix("dimension")
                && let Some(val_str) = rest.split('=').nth(1)
            {
                let val_str = val_str.trim().split(':').next().unwrap_or("0").trim();
                dimension = val_str.parse::<i64>().unwrap_or(0);
            }
            if seg.contains("is_stable") && seg.contains("true") {
                is_stable = true;
            }
        }
    }

    // Also handle {dimension = N, is_stable = true} inline attrs
    if input.starts_with('{') {
        let _ = '{'.parse_next(input)?;
        let attrs: &str = take_till(0.., '}').parse_next(input)?;
        let _ = '}'.parse_next(input)?;
        ws(input)?;
        for segment in attrs.split(',') {
            let seg = segment.trim();
            if let Some(rest) = seg.strip_prefix("dimension")
                && let Some(val_str) = rest.split('=').nth(1)
            {
                let val_str = val_str.trim().split(':').next().unwrap_or("0").trim();
                dimension = val_str.parse::<i64>().unwrap_or(0);
            }
            if seg.contains("is_stable") && seg.contains("true") {
                is_stable = true;
            }
        }
    }

    // Skip to comparator region opening '('
    while !input.is_empty() && !input.starts_with('(') {
        let _ = any.parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let _ = '{'.parse_next(input)?;
    ws(input)?;

    // Parse ^bb0 header with typed block arguments
    let mut comp_ctx = ctx.child();
    let mut comp_params = Vec::new();
    if input.starts_with('^') {
        let _ = take_till(0.., '(').parse_next(input)?;
        let _ = '('.parse_next(input)?;
        loop {
            ws(input)?;
            if input.starts_with(')') {
                break;
            }
            let name = value_name(input)?;
            ws(input)?;
            let _ = ':'.parse_next(input)?;
            ws(input)?;
            let _ty_str: &str = take_till(0.., |c: char| c == ',' || c == ')').parse_next(input)?;
            let pid = comp_ctx.get_or_create(&name);
            comp_params.push(pid);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            }
        }
        let _ = ')'.parse_next(input)?;
        ws(input)?;
        if input.starts_with(':') {
            let _ = ':'.parse_next(input)?;
            let _ = take_till(0.., '\n').parse_next(input)?;
            let _ = opt('\n').parse_next(input)?;
        }
    }

    let comp_body = parse_body(input, &mut comp_ctx)?;
    ws(input)?;
    let _ = '}'.parse_next(input)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Parse result types from the trailing `: (...) -> (...)` or `: ... -> ...`
    let rest_line = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let mut types = Vec::new();
    if let Some(arrow_pos) = rest_line.rfind("-> ") {
        let after = rest_line[arrow_pos + 3..].trim();
        let after = strip_outer_parens(after);
        for part in split_tensor_types(after) {
            types.push(parse_tensor_type_from_str(part.trim()));
        }
    }
    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Sort {
            inputs: operands,
            dimension,
            is_stable,
            comparator: comp_body,
            comparator_params: comp_params,
        },
    }))
}

fn parse_cholesky_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.cholesky".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let lower = rest.contains("lower = true");
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::CholeskyOp { operand, lower },
    }))
}

fn parse_triangular_solve_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.triangular_solve".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.triangular_solve".parse_next(input)?;
    }
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let a = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let b = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let left_side = rest.contains("left_side = true");
    let lower = rest.contains("lower = true");
    let unit_diagonal = rest.contains("unit_diagonal = true");
    let transpose_a = rest.contains("TRANSPOSE") && !rest.contains("NO_TRANSPOSE");
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::TriangularSolve {
            a,
            b,
            left_side,
            lower,
            unit_diagonal,
            transpose_a,
        },
    }))
}

fn parse_fft_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = "stablehlo.fft".parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let fft_type = if rest.contains("type = IRFFT") || rest.contains("fft_type = IRFFT") {
        FftType::Irfft
    } else if rest.contains("type = RFFT") || rest.contains("fft_type = RFFT") {
        FftType::Rfft
    } else if rest.contains("type = IFFT") || rest.contains("fft_type = IFFT") {
        FftType::Ifft
    } else {
        FftType::Fft
    };

    let mut fft_length = Vec::new();
    if let Some(idx) = rest.find("length") {
        let after = &rest[idx..];
        if let Some(s) = after.find('[')
            && let Some(e) = after.find(']')
        {
            fft_length = parse_int_list(&after[s + 1..e]);
        }
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Fft {
            operand,
            fft_type,
            fft_length,
        },
    }))
}

fn parse_rng_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.rng".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.rng".parse_next(input)?;
    }
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let mut operands = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        operands.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        } else {
            break;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let rng_distribution = if rest.contains("NORMAL") {
        RngDistribution::Normal
    } else {
        RngDistribution::Uniform
    };

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Rng {
            operands,
            rng_distribution,
        },
    }))
}

fn parse_batch_norm_inference_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = '"'.parse_next(input)?;
    let _ = "stablehlo.batch_norm_inference".parse_next(input)?;
    let _ = '"'.parse_next(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let scale = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let offset = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let mean = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let variance = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut epsilon = 0.0f64;
    let mut feature_index = 0i64;
    if let Some(idx) = rest.find("epsilon") {
        let after = &rest[idx..];
        if let Some(eq) = after.find('=') {
            let val_str = after[eq + 1..]
                .trim()
                .split(':')
                .next()
                .unwrap_or("0")
                .trim()
                .trim_end_matches(',');
            epsilon = val_str.parse::<f64>().unwrap_or(0.0);
        }
    }
    if let Some(idx) = rest.find("feature_index") {
        let after = &rest[idx..];
        if let Some(eq) = after.find('=') {
            let val_str = after[eq + 1..]
                .trim()
                .split(':')
                .next()
                .unwrap_or("0")
                .trim()
                .trim_end_matches(',');
            feature_index = val_str.parse::<i64>().unwrap_or(0);
        }
    }

    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::BatchNormInference {
            operand,
            scale,
            offset,
            mean,
            variance,
            epsilon,
            feature_index,
        },
    }))
}

fn parse_real_dynamic_slice_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = '"'.parse_next(input)?;
    let _ = "stablehlo.real_dynamic_slice".parse_next(input)?;
    let _ = '"'.parse_next(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let start_indices = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let limit_indices = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let strides = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let ty = parse_final_type(rest);
    let values = make_values(ctx, result_names, vec![ty]);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::RealDynamicSlice {
            operand,
            start_indices,
            limit_indices,
            strides,
        },
    }))
}

fn parse_region_body_with_block_args(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
) -> PResult<(Vec<InstrResult>, Vec<ValueId>)> {
    let _ = '{'.parse_next(input)?;
    ws(input)?;
    let mut params = Vec::new();
    let mut body_ctx = ctx.child();
    if input.starts_with('^') {
        let _ = take_till(0.., '(').parse_next(input)?;
        let _ = '('.parse_next(input)?;
        loop {
            ws(input)?;
            if input.starts_with(')') {
                break;
            }
            let name = value_name(input)?;
            ws(input)?;
            let _ = ':'.parse_next(input)?;
            ws(input)?;
            let _ty: &str = take_till(0.., |c: char| c == ',' || c == ')').parse_next(input)?;
            let pid = body_ctx.get_or_create(&name);
            params.push(pid);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            }
        }
        let _ = ')'.parse_next(input)?;
        ws(input)?;
        if input.starts_with(':') {
            let _ = ':'.parse_next(input)?;
            let _ = take_till(0.., '\n').parse_next(input)?;
            let _ = opt('\n').parse_next(input)?;
        }
    }
    let body = parse_body(input, &mut body_ctx)?;
    ws(input)?;
    let _ = '}'.parse_next(input)?;
    Ok((body, params))
}

fn parse_map_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.map".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.map".parse_next(input)?;
    }
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let mut operands = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        operands.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        } else {
            break;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Skip to region opening '('
    while !input.is_empty() && !input.starts_with('(') {
        let _ = any.parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let (body, body_params) = parse_region_body_with_block_args(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut dimensions = Vec::new();
    if let Some(idx) = rest.find("dimensions") {
        let after = &rest[idx..];
        if let Some(s) = after.find('[')
            && let Some(e) = after.find(']')
        {
            dimensions = parse_int_list(&after[s + 1..e]);
        }
        if dimensions.is_empty()
            && let Some(s) = after.find("array<i64:")
        {
            let after2 = &after[s + 10..];
            if let Some(e) = after2.find('>') {
                dimensions = parse_int_list(&after2[..e]);
            }
        }
    }

    let mut types = Vec::new();
    if let Some(arrow) = rest.rfind("-> ") {
        let after = strip_outer_parens(rest[arrow + 3..].trim());
        for part in split_tensor_types(after) {
            types.push(parse_tensor_type_from_str(part.trim()));
        }
    }
    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Map {
            inputs: operands,
            dimensions,
            body,
            body_params,
        },
    }))
}

fn parse_reduce_window_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = '"'.parse_next(input)?;
    let _ = "stablehlo.reduce_window".parse_next(input)?;
    let _ = '"'.parse_next(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let mut all_ops = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let v = parse_value_ref(input, ctx)?;
        all_ops.push(v);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        } else {
            break;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let half = all_ops.len() / 2;
    let operands = all_ops[..half].to_vec();
    let init_values = all_ops[half..].to_vec();

    // Skip to region opening '('
    while !input.is_empty() && !input.starts_with('(') {
        let _ = any.parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let (body, body_params) = parse_region_body_with_block_args(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Collect remaining text for attributes + type
    let mut attr_text = String::new();
    if input.starts_with('{') {
        let _ = '{'.parse_next(input)?;
        let s: &str = take_till(0.., '}').parse_next(input)?;
        attr_text = s.to_string();
        let _ = '}'.parse_next(input)?;
        ws(input)?;
    }
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let full = format!("{attr_text} {rest}");

    let window_dimensions = extract_array_attr(&full, "window_dimensions");
    let window_strides = extract_array_attr(&full, "window_strides");
    let base_dilations = extract_array_attr(&full, "base_dilations");
    let window_dilations = extract_array_attr(&full, "window_dilations");
    let padding = extract_padding_attr(&full);

    let mut types = Vec::new();
    if let Some(arrow) = rest.rfind("-> ") {
        let after = strip_outer_parens(rest[arrow + 3..].trim());
        for part in split_tensor_types(after) {
            types.push(parse_tensor_type_from_str(part.trim()));
        }
    }
    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::ReduceWindow {
            operands,
            init_values,
            body,
            body_params,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
        },
    }))
}

fn parse_select_and_scatter_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    let _ = '"'.parse_next(input)?;
    let _ = "stablehlo.select_and_scatter".parse_next(input)?;
    let _ = '"'.parse_next(input)?;
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let operand = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let source = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let init_value = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Skip to the opening '(' that wraps the two regions
    while !input.is_empty() && !input.starts_with('(') {
        let _ = any.parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let (select_body, select_params) = parse_region_body_with_block_args(input, ctx)?;
    ws(input)?;
    if input.starts_with(',') {
        let _ = ','.parse_next(input)?;
        ws(input)?;
    }
    let (scatter_body, scatter_params) = parse_region_body_with_block_args(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Collect attributes
    let mut attr_text = String::new();
    if input.starts_with('{') {
        let _ = '{'.parse_next(input)?;
        let s: &str = take_till(0.., '}').parse_next(input)?;
        attr_text = s.to_string();
        let _ = '}'.parse_next(input)?;
        ws(input)?;
    }
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    let full = format!("{attr_text} {rest}");

    let window_dimensions = extract_array_attr(&full, "window_dimensions");
    let window_strides = extract_array_attr(&full, "window_strides");
    let padding = extract_padding_attr(&full);

    let mut types = Vec::new();
    if let Some(arrow) = rest.rfind("-> ") {
        let after = strip_outer_parens(rest[arrow + 3..].trim());
        for part in split_tensor_types(after) {
            types.push(parse_tensor_type_from_str(part.trim()));
        }
    }
    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::SelectAndScatter {
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
        },
    }))
}

fn parse_convolution_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.convolution".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.convolution".parse_next(input)?;
    }
    ws(input)?;
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let lhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ','.parse_next(input)?;
    ws(input)?;
    let rhs = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    // Collect all remaining text (attributes + type) up through newline(s)
    // Attributes may span multiple lines until we hit the result type
    let mut full_text = String::new();
    let mut depth = 0i32;
    loop {
        if input.is_empty() {
            break;
        }
        let ch = any.parse_next(input)?;
        full_text.push(ch);
        if ch == '{' {
            depth += 1;
        }
        if ch == '}' {
            depth -= 1;
        }
        if ch == '\n' && depth <= 0 {
            break;
        }
    }

    let window_strides = extract_array_attr(&full_text, "window_strides");
    let lhs_dilation = extract_array_attr(&full_text, "lhs_dilation");
    let rhs_dilation = extract_array_attr(&full_text, "rhs_dilation");
    let padding = extract_padding_attr(&full_text);
    let feature_group_count = extract_i64_attr(&full_text, "feature_group_count").unwrap_or(1);
    let batch_group_count = extract_i64_attr(&full_text, "batch_group_count").unwrap_or(1);
    let dimension_numbers = parse_conv_dimension_numbers(&full_text);

    let mut types = Vec::new();
    if let Some(arrow) = full_text.rfind("-> ") {
        let after = strip_outer_parens(full_text[arrow + 3..].trim());
        for part in split_tensor_types(after) {
            types.push(parse_tensor_type_from_str(part.trim()));
        }
    }
    if types.is_empty() {
        types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Convolution {
            lhs,
            rhs,
            dimension_numbers,
            window_strides,
            padding,
            lhs_dilation,
            rhs_dilation,
            feature_group_count,
            batch_group_count,
        },
    }))
}

fn extract_array_attr(text: &str, name: &str) -> Vec<i64> {
    if let Some(idx) = text.find(name) {
        let after = &text[idx..];
        if let Some(s) = after.find('[')
            && let Some(e) = after[s..].find(']')
        {
            return parse_int_list(&after[s + 1..s + e]);
        }
        if let Some(s) = after.find("array<i64:")
            && let Some(e) = after[s + 10..].find('>')
        {
            return parse_int_list(&after[s + 10..s + 10 + e]);
        }
        if let Some(s) = after.find("dense<")
            && let Some(e) = after[s + 6..].find('>')
        {
            let inner = &after[s + 6..s + 6 + e];
            if let Some(b) = inner.find('[')
                && let Some(be) = inner.find(']')
            {
                return parse_int_list(&inner[b + 1..be]);
            }
            return parse_int_list(inner);
        }
    }
    Vec::new()
}

fn extract_padding_attr(text: &str) -> Vec<(i64, i64)> {
    if let Some(idx) = text.find("padding") {
        let after = &text[idx..];
        if let Some(d) = after.find("dense<")
            && let Some(e) = after[d + 6..].find('>')
        {
            let after2 = &after[d + 6..];
            let inner = &after2[..e].trim();
            // Handle scalar splat like `dense<0>`
            if !inner.contains('[')
                && let Ok(v) = inner.parse::<i64>()
            {
                let type_part = &after[d + 6 + e..];
                let n_pairs = if let Some(ts) = type_part.find("tensor<") {
                    let tafter = &type_part[ts + 7..];
                    if let Some(x) = tafter.find('x') {
                        tafter[..x].parse::<usize>().unwrap_or(1)
                    } else {
                        1
                    }
                } else {
                    1
                };
                return (0..n_pairs).map(|_| (v, v)).collect();
            }
            let mut pairs = Vec::new();
            for seg in inner.split('[') {
                let seg = seg.trim().trim_end_matches(']').trim_end_matches(',');
                if seg.is_empty() {
                    continue;
                }
                let nums: Vec<i64> = seg
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if nums.len() == 2 {
                    pairs.push((nums[0], nums[1]));
                }
            }
            return pairs;
        }
    }
    Vec::new()
}

fn extract_i64_attr(text: &str, name: &str) -> Option<i64> {
    if let Some(idx) = text.find(name) {
        let after = &text[idx..];
        if let Some(eq) = after.find('=') {
            let val = after[eq + 1..]
                .trim()
                .split(':')
                .next()?
                .trim()
                .trim_end_matches(',');
            return val.parse().ok();
        }
    }
    None
}

fn parse_conv_dimension_numbers(text: &str) -> ConvDimensionNumbers {
    fn parse_group(s: &str, labels: &[&str]) -> (Vec<(String, usize)>,) {
        let s = s.trim().trim_start_matches('[').trim_end_matches(']');
        let items: Vec<(String, usize)> = s
            .split(',')
            .map(|t| t.trim().to_string())
            .enumerate()
            .map(|(pos, tok)| (tok, pos))
            .collect();
        (items,)
    }

    let mut dn = ConvDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: vec![],
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        kernel_spatial_dimensions: vec![],
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: vec![],
    };

    if let Some(start) = text.find("#stablehlo.conv<") {
        let after = &text[start + 16..];
        if let Some(end_pos) = after.find('>') {
            let conv_str = &after[..end_pos];
            let parts: Vec<&str> = conv_str.split("->").collect();
            if parts.len() == 2 {
                let lhs_parts: Vec<&str> = parts[0].splitn(2, 'x').collect();
                if lhs_parts.len() >= 2 {
                    // Input: b=batch, f=feature, numbers=spatial
                    let input_s = lhs_parts[0]
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']');
                    for (pos, tok) in input_s.split(',').map(|t| t.trim()).enumerate() {
                        match tok {
                            "b" => dn.input_batch_dimension = pos as i64,
                            "f" => dn.input_feature_dimension = pos as i64,
                            _ => dn.input_spatial_dimensions.push(pos as i64),
                        }
                    }
                    // Kernel: i=input_feature, o=output_feature, numbers=spatial
                    let kernel_s = lhs_parts[1]
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']');
                    for (pos, tok) in kernel_s.split(',').map(|t| t.trim()).enumerate() {
                        match tok {
                            "i" => dn.kernel_input_feature_dimension = pos as i64,
                            "o" => dn.kernel_output_feature_dimension = pos as i64,
                            _ => dn.kernel_spatial_dimensions.push(pos as i64),
                        }
                    }
                }
                // Output: b=batch, f=feature, numbers=spatial
                let output_s = parts[1]
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']');
                for (pos, tok) in output_s.split(',').map(|t| t.trim()).enumerate() {
                    match tok {
                        "b" => dn.output_batch_dimension = pos as i64,
                        "f" => dn.output_feature_dimension = pos as i64,
                        _ => dn.output_spatial_dimensions.push(pos as i64),
                    }
                }
            }
        }
    }
    dn
}

fn parse_case_op(
    input: &mut Stream<'_>,
    ctx: &mut ValueCtx,
    result_names: &[String],
) -> PResult<Option<InstrResult>> {
    if input.starts_with('"') {
        let _ = '"'.parse_next(input)?;
        let _ = "stablehlo.case".parse_next(input)?;
        let _ = '"'.parse_next(input)?;
    } else {
        let _ = "stablehlo.case".parse_next(input)?;
    }
    let _ = '('.parse_next(input)?;
    ws(input)?;
    let index = parse_value_ref(input, ctx)?;
    ws(input)?;
    let _ = ')'.parse_next(input)?;
    ws(input)?;

    let _ = '('.parse_next(input)?;

    let mut branches = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with(')') {
            break;
        }
        let _ = '{'.parse_next(input)?;
        ws(input)?;
        let mut branch_ctx = ctx.child();
        let body = parse_body(input, &mut branch_ctx)?;
        ws(input)?;
        let _ = '}'.parse_next(input)?;
        branches.push(body);
        ws(input)?;
        if input.starts_with(',') {
            let _ = ','.parse_next(input)?;
        }
    }
    let _ = ')'.parse_next(input)?;
    ws(input)?;
    let _ = ':'.parse_next(input)?;
    ws(input)?;
    let rest = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;

    let mut ret_types = Vec::new();
    if let Some(idx) = rest.find("-> ") {
        let after = strip_outer_parens(rest[idx + 3..].trim());
        for part in split_tensor_types(after) {
            ret_types.push(parse_tensor_type_from_str(part.trim()));
        }
    } else {
        for part in split_tensor_types(rest.trim()) {
            let p = part.trim().trim_start_matches('(').trim_end_matches(')');
            if p.starts_with("tensor<") {
                ret_types.push(parse_tensor_type_from_str(p));
            }
        }
    }
    if ret_types.is_empty() {
        ret_types.push(TensorType::scalar(ElementType::F64));
    }

    let values = make_values(ctx, result_names, ret_types);
    Ok(Some(InstrResult {
        values,
        instr: Instruction::Case { index, branches },
    }))
}

fn parse_return(input: &mut Stream<'_>, ctx: &mut ValueCtx) -> PResult<InstrResult> {
    if input.starts_with("\"stablehlo.return\"") {
        let _ = "\"stablehlo.return\"".parse_next(input)?;
    } else if input.starts_with("stablehlo.return") {
        let _ = "stablehlo.return".parse_next(input)?;
    } else {
        let _ = "return".parse_next(input)?;
    }
    ws(input)?;

    let mut operands = Vec::new();

    if !input.starts_with(':') && !input.starts_with('\n') && !input.starts_with('}') {
        loop {
            ws(input)?;
            if input.starts_with(':') || input.starts_with('\n') || input.is_empty() {
                break;
            }
            let v = parse_value_ref(input, ctx)?;
            operands.push(v);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            } else {
                break;
            }
        }
    }

    skip_to_newline(input)?;

    Ok(InstrResult {
        values: vec![],
        instr: Instruction::Return { operands },
    })
}

// ---------------------------------------------------------------------------
// Primitive parsers
// ---------------------------------------------------------------------------

fn ws(input: &mut Stream<'_>) -> PResult<()> {
    let _ = multispace0.parse_next(input)?;
    while input.starts_with("//") {
        skip_line(input)?;
        let _ = multispace0.parse_next(input)?;
    }
    Ok(())
}

fn ident(input: &mut Stream<'_>) -> PResult<String> {
    let s = take_while(1.., |c: char| {
        c.is_alphanumeric() || c == '_' || c == '.' || c == '-'
    })
    .parse_next(input)?;
    Ok(s.to_string())
}

fn value_name(input: &mut Stream<'_>) -> PResult<String> {
    let _ = '%'.parse_next(input)?;
    let name = take_while(1.., |c: char| c.is_alphanumeric() || c == '_').parse_next(input)?;
    Ok(name.to_string())
}

fn integer(input: &mut Stream<'_>) -> PResult<i64> {
    let neg = opt('-').parse_next(input)?;
    let digits: &str = digit1.parse_next(input)?;
    let val: i64 = digits.parse().unwrap();
    Ok(if neg.is_some() { -val } else { val })
}

fn float_literal(input: &mut Stream<'_>) -> PResult<f64> {
    if input.starts_with("0x") {
        let _ = "0x".parse_next(input)?;
        let hex: &str = take_while(1.., |c: char| c.is_ascii_hexdigit()).parse_next(input)?;
        let bits = u64::from_str_radix(hex, 16).unwrap();
        return Ok(f64::from_bits(bits));
    }

    let start = *input;
    let _ = opt('-').parse_next(input)?;
    let _ = digit1.parse_next(input)?;
    let _ = opt(('.', digit1)).parse_next(input)?;

    if input.starts_with('E')
        || input.starts_with('e')
        || input.starts_with("E-")
        || input.starts_with("e-")
        || input.starts_with("E+")
        || input.starts_with("e+")
    {
        let _ = any.parse_next(input)?;
        let _ = opt(alt(('+', '-'))).parse_next(input)?;
        let _ = digit1.parse_next(input)?;
    }

    let consumed = &start[..start.len() - input.len()];
    Ok(consumed.parse::<f64>().unwrap())
}

fn element_type(input: &mut Stream<'_>) -> PResult<ElementType> {
    alt((
        "f64".map(|_| ElementType::F64),
        "f32".map(|_| ElementType::F32),
        "ui64".map(|_| ElementType::UI64),
        "ui32".map(|_| ElementType::UI32),
        "i1".map(|_| ElementType::I1),
        "i32".map(|_| ElementType::I32),
        "i64".map(|_| ElementType::I64),
    ))
    .parse_next(input)
}

fn tensor_type(input: &mut Stream<'_>) -> PResult<TensorType> {
    let _ = "tensor<".parse_next(input)?;
    let mut dims = Vec::new();

    loop {
        if let Ok(et) = element_type.parse_next(input) {
            let _ = '>'.parse_next(input)?;
            return Ok(TensorType {
                shape: dims,
                element_type: et,
            });
        }
        let d = integer(input)?;
        dims.push(d);
        let _ = 'x'.parse_next(input)?;
    }
}

fn result_types(input: &mut Stream<'_>) -> PResult<Vec<TensorType>> {
    if input.starts_with('(') {
        let _ = '('.parse_next(input)?;
        ws(input)?;
        let mut types = Vec::new();
        loop {
            ws(input)?;
            if input.starts_with(')') {
                break;
            }
            let t = result_type_with_attrs(input)?;
            types.push(t);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            }
        }
        let _ = ')'.parse_next(input)?;
        Ok(types)
    } else {
        let t = tensor_type(input)?;
        Ok(vec![t])
    }
}

fn result_type_with_attrs(input: &mut Stream<'_>) -> PResult<TensorType> {
    let t = tensor_type(input)?;
    ws(input)?;
    if input.starts_with('{') {
        skip_braces(input)?;
    }
    Ok(t)
}

fn parse_value_ref(input: &mut Stream<'_>, ctx: &mut ValueCtx) -> PResult<ValueId> {
    let vn = value_name(input)?;
    if input.starts_with('#') {
        let _ = '#'.parse_next(input)?;
        let idx: &str = digit1.parse_next(input)?;
        let full = format!("{}#{}", vn, idx);
        Ok(ctx.get_or_create(&full))
    } else {
        Ok(ctx.get_or_create(&vn))
    }
}

fn parse_value_list(input: &mut Stream<'_>, ctx: &mut ValueCtx) -> PResult<Vec<ValueId>> {
    let mut vals = Vec::new();
    loop {
        ws(input)?;
        if input.starts_with('%') {
            let v = parse_value_ref(input, ctx)?;
            vals.push(v);
            ws(input)?;
            if input.starts_with(',') {
                let _ = ','.parse_next(input)?;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    Ok(vals)
}

// ---------------------------------------------------------------------------
// Type extraction from rest-of-line strings
// ---------------------------------------------------------------------------

fn parse_result_type_from_str(s: &str) -> TensorType {
    let s = s.trim();
    let s = if let Some(idx) = s.rfind("-> ") {
        &s[idx + 3..]
    } else {
        let parts: Vec<&str> = s.split(')').collect();
        if parts.len() > 1 {
            let last = parts.last().unwrap().trim();
            if let Some(stripped) = last.strip_prefix("-> ") {
                stripped
            } else {
                s
            }
        } else {
            s
        }
    };

    let s = s.trim();
    if let Some(rest) = s.strip_prefix("tensor<") {
        parse_tensor_type_inner(rest)
    } else if s.starts_with('(') {
        let inner = &s[1..s.rfind(')').unwrap_or(s.len())];
        parse_tensor_type_from_str(inner.trim())
    } else {
        TensorType::scalar(ElementType::F64)
    }
}

fn parse_final_type(s: &str) -> TensorType {
    let s = s.trim();
    if let Some(idx) = s.rfind("-> ") {
        let after = s[idx + 3..].trim();
        let after = after
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(after);
        if let Some(rest) = after.strip_prefix("tensor<") {
            return parse_tensor_type_inner(rest);
        }
    }
    if let Some(idx) = s.rfind("tensor<") {
        return parse_tensor_type_inner(&s[idx + 7..]);
    }
    TensorType::scalar(ElementType::F64)
}

fn parse_tensor_type_from_str(s: &str) -> TensorType {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("tensor<") {
        parse_tensor_type_inner(rest)
    } else {
        TensorType::scalar(ElementType::F64)
    }
}

fn parse_tensor_type_inner(s: &str) -> TensorType {
    let s = match s.find('>') {
        Some(idx) => &s[..idx],
        None => s,
    };
    let s = s.trim();
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() == 1 {
        let et = str_to_element_type(parts[0]);
        TensorType::scalar(et)
    } else {
        let et = str_to_element_type(parts.last().unwrap().trim());
        let dims: Vec<i64> = parts[..parts.len() - 1]
            .iter()
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        TensorType {
            shape: dims,
            element_type: et,
        }
    }
}

fn str_to_element_type(s: &str) -> ElementType {
    match s.trim() {
        "f64" => ElementType::F64,
        "f32" => ElementType::F32,
        "i1" => ElementType::I1,
        "i8" | "si8" => ElementType::I32,
        "i16" | "si16" => ElementType::I32,
        "i32" => ElementType::I32,
        "i64" => ElementType::I64,
        "ui8" => ElementType::UI32,
        "ui16" => ElementType::UI32,
        "ui32" => ElementType::UI32,
        "ui64" => ElementType::UI64,
        _ => {
            eprintln!("warning: unrecognized tensor element type '{s}', defaulting to f64");
            ElementType::F64
        }
    }
}

fn strip_outer_parens(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('(') {
        &s[1..s.rfind(')').unwrap_or(s.len())]
    } else {
        s
    }
}

fn split_tensor_types(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

// ---------------------------------------------------------------------------
// Attribute extraction helpers
// ---------------------------------------------------------------------------

fn extract_bracket_ints(s: &str, prefix: &str) -> Vec<i64> {
    if let Some(idx) = s.find(prefix) {
        let after = &s[idx + prefix.len()..];
        if let Some(end) = after.find(']') {
            return parse_int_list(&after[..end]);
        }
    }
    Vec::new()
}

fn parse_int_list(s: &str) -> Vec<i64> {
    s.split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn extract_compare_direction(s: &str) -> CompareDirection {
    if let Some(idx) = s.find("comparison_direction") {
        let after = &s[idx..];
        // Match direction keywords (check longer ones first to avoid prefix collisions)
        for (kw, dir) in [
            ("GE", CompareDirection::Ge),
            ("GT", CompareDirection::Gt),
            ("LE", CompareDirection::Le),
            ("LT", CompareDirection::Lt),
            ("NE", CompareDirection::Ne),
            ("EQ", CompareDirection::Eq),
        ] {
            if after.contains(kw) {
                return dir;
            }
        }
    }
    CompareDirection::Eq
}

fn extract_compare_type(s: &str) -> CompareType {
    if let Some(idx) = s.find("compare_type") {
        let after = &s[idx..];
        if after.contains("TOTALORDER") {
            return CompareType::TotalOrder;
        }
        if after.contains("UNSIGNED") {
            return CompareType::Unsigned;
        }
        if after.contains("SIGNED") {
            return CompareType::Signed;
        }
        if after.contains("FLOAT") {
            return CompareType::Float;
        }
    }
    CompareType::Float
}

// ---------------------------------------------------------------------------
// Utility parsers
// ---------------------------------------------------------------------------

fn skip_line(input: &mut Stream<'_>) -> PResult<()> {
    let _ = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    Ok(())
}

fn skip_to_newline(input: &mut Stream<'_>) -> PResult<()> {
    let _ = take_till(0.., '\n').parse_next(input)?;
    let _ = opt('\n').parse_next(input)?;
    Ok(())
}

fn skip_until_brace(input: &mut Stream<'_>) -> PResult<()> {
    let _ = take_till(0.., '{').parse_next(input)?;
    Ok(())
}

fn skip_braces(input: &mut Stream<'_>) -> PResult<()> {
    let _ = '{'.parse_next(input)?;
    let mut depth = 1u32;
    while depth > 0 {
        let c = any.parse_next(input)?;
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SSA value context
// ---------------------------------------------------------------------------

struct ValueCtx {
    next_id: u32,
    name_to_id: HashMap<String, ValueId>,
}

impl ValueCtx {
    fn new() -> Self {
        Self {
            next_id: 0,
            name_to_id: HashMap::new(),
        }
    }

    fn get_or_create(&mut self, name: &str) -> ValueId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = ValueId(self.next_id);
        self.next_id += 1;
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    fn fresh(&mut self) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        id
    }

    fn alias(&mut self, alias: &str, target: &str) {
        let id = self.get_or_create(target);
        self.name_to_id.insert(alias.to_string(), id);
    }

    fn child(&self) -> Self {
        Self {
            next_id: self.next_id,
            name_to_id: self.name_to_id.clone(),
        }
    }
}

fn make_values(
    ctx: &mut ValueCtx,
    result_names: &[String],
    types: Vec<TensorType>,
) -> Vec<(ValueId, TensorType)> {
    if types.len() == 1 {
        let vid = if result_names.is_empty() {
            ctx.fresh()
        } else {
            let vid = ctx.get_or_create(&result_names[0]);
            if result_names.len() > 1 {
                let alias = format!("{}#0", result_names[0]);
                ctx.name_to_id.insert(alias, vid);
            }
            vid
        };
        return vec![(vid, types.into_iter().next().unwrap())];
    }

    let base_name = if !result_names.is_empty() {
        result_names[0].clone()
    } else {
        String::new()
    };

    types
        .into_iter()
        .enumerate()
        .map(|(i, ty)| {
            let name = if !base_name.is_empty() {
                format!("{}#{}", base_name, i)
            } else {
                format!("__anon_{}", ctx.next_id + i as u32)
            };
            let vid = ctx.get_or_create(&name);
            if i == 0 && !base_name.is_empty() {
                ctx.name_to_id.insert(base_name.clone(), vid);
            }
            (vid, ty)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tensor_type() {
        let mut s = "tensor<3xf64>";
        let t = tensor_type(&mut s).unwrap();
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.element_type, ElementType::F64);
    }

    #[test]
    fn test_parse_scalar_tensor_type() {
        let mut s = "tensor<f64>";
        let t = tensor_type(&mut s).unwrap();
        assert!(t.shape.is_empty());
        assert_eq!(t.element_type, ElementType::F64);
    }

    #[test]
    fn test_parse_2d_tensor_type() {
        let mut s = "tensor<4x3xf64>";
        let t = tensor_type(&mut s).unwrap();
        assert_eq!(t.shape, vec![4, 3]);
        assert_eq!(t.element_type, ElementType::F64);
    }

    #[test]
    fn test_parse_small_module() {
        let mlir = r#"
module @module attributes {mhlo.num_partitions = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.add %0, %arg1 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}
"#;
        let module = parse_module(mlir).unwrap();
        assert_eq!(module.functions.len(), 1);
        let f = &module.functions[0];
        assert_eq!(f.name, "main");
        assert!(f.is_public);
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.body.len(), 3);
    }

    #[test]
    fn test_parse_constant_splat() {
        let mlir = r#"
module @test {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
    %0 = stablehlo.add %cst, %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
        let module = parse_module(mlir).unwrap();
        let f = &module.functions[0];
        assert_eq!(f.body.len(), 3);
        match &f.body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseSplat(_, ty),
            } => {
                assert_eq!(ty.shape, vec![3]);
            }
            other => panic!("expected DenseSplat, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_multi_func_module() {
        let mlir = r#"
module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> (tensor<6xf64>, tensor<3xf64>) {
    %0 = call @inner(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %1 = stablehlo.concatenate %0, %arg1, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    return %1, %0 : tensor<6xf64>, tensor<3xf64>
  }
  func.func private @inner(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.negate %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
        let module = parse_module(mlir).unwrap();
        assert_eq!(module.functions.len(), 2);
        assert!(module.get_func("main").is_some());
        assert!(module.get_func("inner").is_some());
        let main = module.main_func().unwrap();
        assert!(main.is_public);
        assert_eq!(main.params.len(), 2);
        assert_eq!(main.result_types.len(), 2);
    }

    #[test]
    fn test_parse_hex_constant() {
        let mlir = r#"
module @test {
  func.func public @main() -> tensor<f64> {
    %cst = stablehlo.constant dense<0x3FF0000000000000> : tensor<f64>
    return %cst : tensor<f64>
  }
}
"#;
        let module = parse_module(mlir).unwrap();
        let f = &module.functions[0];
        match &f.body[0].instr {
            Instruction::Constant {
                value: ConstantValue::DenseScalar(ScalarValue::F64(v)),
            } => {
                assert!((v - 1.0).abs() < 1e-15);
            }
            other => panic!("expected DenseScalar(F64(1.0)), got {:?}", other),
        }
    }
}
