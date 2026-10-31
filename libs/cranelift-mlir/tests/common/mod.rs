// Shared test helpers; not every test binary that includes this module
// consumes every helper.
#![allow(dead_code)]

use std::borrow::Cow;

use cranelift_mlir::const_fold::fold_module;
use cranelift_mlir::ir::ElementType;
use cranelift_mlir::lower::{CompileConfig, compile_module_with_config};
use cranelift_mlir::parser::parse_module;

pub type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

pub fn run_mlir(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    run_mlir_with(mlir, inputs, output_sizes, CompileConfig::from_env(), false)
}

pub fn run_mlir_mem(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let config = CompileConfig {
        force_pointer_abi_main: true,
        ..CompileConfig::from_env()
    };
    run_mlir_with(mlir, inputs, output_sizes, config, false)
}

/// `run_mlir` (`run_mlir_mem` when `pointer_abi`) with `fold_module` run
/// before compiling, as `nox-py` does.
pub fn run_mlir_folded(
    mlir: &str,
    inputs: &[&[u8]],
    output_sizes: &[usize],
    pointer_abi: bool,
) -> Vec<Vec<u8>> {
    let config = CompileConfig {
        force_pointer_abi_main: pointer_abi,
        ..CompileConfig::from_env()
    };
    run_mlir_with(mlir, inputs, output_sizes, config, true)
}

/// Parse once, run the interpreter and the JIT, and require their outputs to
/// match. Returns the JIT bytes so existing golden assertions stay on the
/// compiled path. Ops the interpreter does not implement are skipped (logged)
/// unless `ELODIN_CRANELIFT_INTERP_STRICT=1`. With `fold`, the JIT compiles the
/// folded module while the interpreter runs the module as parsed.
fn run_mlir_with(
    mlir: &str,
    inputs: &[&[u8]],
    output_sizes: &[usize],
    config: CompileConfig,
    fold: bool,
) -> Vec<Vec<u8>> {
    let mut module = parse_module(mlir).expect("parse failed");
    let result_meta: Vec<(ElementType, usize)> = module
        .main_func()
        .expect("module is missing a callable main entrypoint")
        .result_types
        .iter()
        .map(|ty| (ty.element_type, ty.byte_size()))
        .collect();

    // Exact result sizes: some tests pass oversized JIT buffers, and the
    // interpreter rejects a length that is not the tensor's byte size.
    let mut interp_bufs: Vec<Vec<u8>> = result_meta.iter().map(|(_, n)| vec![0u8; *n]).collect();
    let interp_result = {
        let mut outs: Vec<&mut [u8]> = interp_bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        cranelift_mlir::interp::interpret_main(&module, inputs, &mut outs)
    };

    if fold {
        fold_module(&mut module);
    }
    let mem_path = config.force_pointer_abi_main;
    let compiled = compile_module_with_config(&module, config).unwrap_or_else(|err| {
        panic!(
            "compile failed{}: {err}",
            if mem_path { " (mem path)" } else { "" }
        )
    });
    // Drop the parsed module before the JIT call so imported constants must
    // outlive it via CompiledModule, not the IR.
    drop(module);

    let fn_ptr = compiled.get_main_fn();
    let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    match interp_result {
        Ok(()) => {
            assert_eq!(
                output_bufs.len(),
                interp_bufs.len(),
                "JIT returned {} outputs, interpreter returned {}",
                output_bufs.len(),
                interp_bufs.len()
            );
            for (index, ((jit, interp), (elem, _))) in output_bufs
                .iter()
                .zip(&interp_bufs)
                .zip(&result_meta)
                .enumerate()
            {
                compare_output(index, *elem, jit, interp);
            }
        }
        Err(err) if err.is_unsupported() => {
            if interp_strict() {
                panic!(
                    "interpreter does not implement an op under ELODIN_CRANELIFT_INTERP_STRICT: {err}"
                );
            }
            eprintln!("[interp skipped] {err}");
        }
        Err(err) => panic!("interpreter failed: {err}"),
    }

    output_bufs
}

fn interp_strict() -> bool {
    std::env::var("ELODIN_CRANELIFT_INTERP_STRICT")
        .ok()
        .as_deref()
        == Some("1")
}

fn compare_output(index: usize, elem: ElementType, jit: &[u8], interp: &[u8]) {
    assert!(
        jit.len() >= interp.len(),
        "output {index}: JIT buffer is {} bytes, result type is {} bytes",
        jit.len(),
        interp.len()
    );
    let jit = &jit[..interp.len()];
    match elem {
        ElementType::F64 => compare_floats(index, elem, jit, interp, 8, 1e-12),
        ElementType::F32 => compare_floats(index, elem, jit, interp, 4, 1e-6),
        ElementType::I1
        | ElementType::I32
        | ElementType::I64
        | ElementType::UI32
        | ElementType::UI64 => compare_ints(index, elem, jit, interp),
    }
}

fn compare_floats(
    index: usize,
    elem: ElementType,
    jit: &[u8],
    interp: &[u8],
    width: usize,
    tol_scale: f64,
) {
    let n = interp.len() / width;
    for i in 0..n {
        let jit_val = read_float(elem, &jit[i * width..(i + 1) * width]);
        let interp_val = read_float(elem, &interp[i * width..(i + 1) * width]);
        if floats_match(jit_val, interp_val, tol_scale) {
            continue;
        }
        panic!(
            "output {index} element {i} ({elem}): JIT {jit_val:.17} vs interpreter {interp_val:.17}"
        );
    }
}

fn floats_match(jit: f64, interp: f64, tol_scale: f64) -> bool {
    if jit.is_nan() && interp.is_nan() {
        return true;
    }
    if jit == interp {
        return true;
    }
    (jit - interp).abs() <= tol_scale * (1.0 + jit.abs().max(interp.abs()))
}

fn read_float(elem: ElementType, bytes: &[u8]) -> f64 {
    match elem {
        ElementType::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
        ElementType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
        _ => unreachable!("read_float on {elem}"),
    }
}

fn compare_ints(index: usize, elem: ElementType, jit: &[u8], interp: &[u8]) {
    if jit == interp {
        return;
    }
    let width = elem.byte_size();
    let n = interp.len() / width;
    for i in 0..n {
        let a = &jit[i * width..(i + 1) * width];
        let b = &interp[i * width..(i + 1) * width];
        if a != b {
            panic!(
                "output {index} element {i} ({elem}): JIT {} vs interpreter {}",
                decode_int(elem, a),
                decode_int(elem, b)
            );
        }
    }
}

fn decode_int(elem: ElementType, bytes: &[u8]) -> String {
    match elem {
        ElementType::I1 => format!("{}", bytes[0]),
        ElementType::I32 => format!("{}", i32::from_le_bytes(bytes.try_into().unwrap())),
        ElementType::I64 => format!("{}", i64::from_le_bytes(bytes.try_into().unwrap())),
        ElementType::UI32 => format!("{}", u32::from_le_bytes(bytes.try_into().unwrap())),
        ElementType::UI64 => format!("{}", u64::from_le_bytes(bytes.try_into().unwrap())),
        ElementType::F32 | ElementType::F64 => unreachable!("decode_int on {elem}"),
    }
}

pub fn f64_buf(vals: &[f64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn i64_buf(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn i32_buf(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn u32_buf(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn u64_buf(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn read_f64s(buf: &[u8]) -> Cow<'_, [f64]> {
    // Buffers are little-endian. A native cast is only valid on little-endian
    // hosts, and only when the allocation happens to be f64-aligned.
    if cfg!(target_endian = "little")
        && let Ok(vals) = bytemuck::try_cast_slice(buf)
    {
        return Cow::Borrowed(vals);
    }
    Cow::Owned(
        buf.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
    )
}

pub fn read_u32s(buf: &[u8]) -> Vec<u32> {
    buf.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

pub fn read_u64s(buf: &[u8]) -> Vec<u64> {
    buf.as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect()
}

pub fn read_i64s(buf: &[u8]) -> Vec<i64> {
    buf.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

pub fn read_i32s(buf: &[u8]) -> Vec<i32> {
    buf.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

pub fn assert_f64_close(actual: f64, expected: f64) {
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1e-15);
    assert!(
        diff / denom < 1e-10,
        "expected {expected}, got {actual} (relative error: {:.2e})",
        diff / denom
    );
}

pub fn assert_f64s_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let denom = e.abs().max(1e-15);
        assert!(
            diff / denom < 1e-10,
            "element {i}: expected {e}, got {a} (relative error: {:.2e})",
            diff / denom
        );
    }
}
