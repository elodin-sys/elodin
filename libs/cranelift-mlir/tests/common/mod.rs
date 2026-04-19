// Shared test helpers; not every test binary that includes this module
// consumes every helper.
#![allow(dead_code)]

use cranelift_mlir::lower::{CompileConfig, compile_module, compile_module_with_config};
use cranelift_mlir::parser::parse_module;

pub type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

pub fn run_mlir(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let module = parse_module(mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    output_bufs
}

pub fn run_mlir_mem(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let module = parse_module(mlir).expect("parse failed");
    let config = CompileConfig {
        force_pointer_abi_main: true,
        ..CompileConfig::from_env()
    };
    let compiled = compile_module_with_config(&module, config).expect("compile failed (mem path)");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    output_bufs
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

pub fn read_f64s(buf: &[u8]) -> Vec<f64> {
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_u32s(buf: &[u8]) -> Vec<u32> {
    buf.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_u64s(buf: &[u8]) -> Vec<u64> {
    buf.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_i64s(buf: &[u8]) -> Vec<i64> {
    buf.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn read_i32s(buf: &[u8]) -> Vec<i32> {
    buf.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
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
