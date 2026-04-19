mod common;
use common::*;

use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

/// Extract the @threefry2x32 and @closed_call functions from the ball MLIR
/// and wrap them in a main function for testing.
#[test]
fn test_threefry2x32_with_known_inputs() {
    let ball_mlir = include_str!("../testdata/ball.stablehlo.mlir");

    let threefry_start = ball_mlir
        .find("func.func private @threefry2x32(")
        .expect("threefry2x32 not found");

    let closed_call_start = ball_mlir
        .find("func.func private @closed_call(")
        .expect("closed_call not found");

    let closed_call_end = ball_mlir[closed_call_start..]
        .find("\n  }")
        .map(|i| closed_call_start + i + 4)
        .unwrap();

    let threefry_end = ball_mlir[threefry_start..]
        .find("\n  }")
        .map(|i| threefry_start + i + 4)
        .unwrap();

    let threefry_fn = &ball_mlir[threefry_start..threefry_end];
    let closed_call_fn = &ball_mlir[closed_call_start..closed_call_end];

    let test_mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<3xui32>, %arg3: tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>) {{
    %0:2 = call @threefry2x32(%arg0, %arg1, %arg2, %arg3) : (tensor<ui32>, tensor<ui32>, tensor<3xui32>, tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>)
    return %0#0, %0#1 : tensor<3xui32>, tensor<3xui32>
  }}
  {threefry_fn}
  {closed_call_fn}
}}"#
    );

    eprintln!("Test MLIR length: {} bytes", test_mlir.len());

    let module = parse_module(&test_mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: unsafe extern "C" fn(*const *const u8, *mut *mut u8) =
        unsafe { std::mem::transmute(fn_ptr) };

    // Inputs matching the ball simulation seed=0 PRNG:
    // key0 = 0, key1 = 0
    // counter_hi = [0, 0, 0], counter_lo = [0, 1, 2]
    let key0: u32 = 0;
    let key1: u32 = 0;
    let counter_hi: Vec<u32> = vec![0, 0, 0];
    let counter_lo: Vec<u32> = vec![0, 1, 2];

    let inputs: Vec<Vec<u8>> = vec![
        u32_buf(&[key0]),
        u32_buf(&[key1]),
        u32_buf(&counter_hi),
        u32_buf(&counter_lo),
    ];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    let mut out_hi = vec![0u8; 12];
    let mut out_lo = vec![0u8; 12];
    let mut output_ptrs = vec![out_hi.as_mut_ptr(), out_lo.as_mut_ptr()];

    unsafe {
        tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr());
    }

    let result_hi = read_u32s(&out_hi);
    let result_lo = read_u32s(&out_lo);

    eprintln!("threefry2x32 output hi: {:?}", result_hi);
    eprintln!("threefry2x32 output lo: {:?}", result_lo);
    eprintln!(
        "hi hex: {}",
        result_hi
            .iter()
            .map(|v| format!("{v:#010X}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "lo hex: {}",
        result_lo
            .iter()
            .map(|v| format!("{v:#010X}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // The expected output from JAX random.bits(key(0), shape=(3,), dtype=uint32):
    // bits_lo = [0xf29a4fa7, 0xfa843692, 0x55110e28] = [4070199207, 4202968722, 1427181096]
    //
    // The _uniform function reconstructs uint64 as: (hi << 32) | lo
    // From JAX: bits_u64 = [0x6b20015999ba4efe, 0x375f238fcddb151d, 0xf71f4ea9a20e4081]
    // So hi = [0x6b200159, 0x375f238f, 0xf71f4ea9]
    //    lo = [0x99ba4efe, 0xcddb151d, 0xa20e4081]
    let expected_hi: Vec<u32> = vec![0x6b200159, 0x375f238f, 0xf71f4ea9];
    let expected_lo: Vec<u32> = vec![0x99ba4efe, 0xcddb151d, 0xa20e4081];

    // Note: threefry2x32 returns (result_hi, result_lo) where the _uniform function
    // reconstructs u64 as (convert(result_hi) << 32) | convert(result_lo)
    // The first result of threefry2x32 is "hi" in the ball MLIR calling convention.
    for i in 0..3 {
        assert_eq!(
            result_hi[i], expected_hi[i],
            "hi[{i}]: got {:#010X}, expected {:#010X}",
            result_hi[i], expected_hi[i]
        );
        assert_eq!(
            result_lo[i], expected_lo[i],
            "lo[{i}]: got {:#010X}, expected {:#010X}",
            result_lo[i], expected_lo[i]
        );
    }
}
