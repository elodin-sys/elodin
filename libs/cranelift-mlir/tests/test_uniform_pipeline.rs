mod common;
use common::*;

use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

#[test]
fn test_ui32_to_ui64_shift_left_32() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<ui64> {
    %0 = stablehlo.convert %arg0 : (tensor<ui32>) -> tensor<ui64>
    %c = stablehlo.constant dense<32> : tensor<ui64>
    %1 = stablehlo.shift_left %0, %c : tensor<ui64>
    return %1 : tensor<ui64>
  }
}
"#;
    let val: u32 = 0xDEADBEEF;
    let out = run_mlir(mlir, &[&u32_buf(&[val])], &[8]);
    let result = read_u64s(&out[0])[0];
    let expected = (val as u64) << 32;
    assert_eq!(
        result, expected,
        "got {result:#018X}, expected {expected:#018X}"
    );
}

#[test]
fn test_uniform_float_construction() {
    // Reproduce the _uniform pipeline:
    // 1. Take two ui32 values (hi, lo from threefry)
    // 2. Convert each to ui64
    // 3. Shift hi left by 32, OR with lo -> ui64
    // 4. Shift right by 12
    // 5. OR with 0x3FF0000000000000 (1.0 exponent)
    // 6. Bitcast to f64
    // 7. Subtract 1.0
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<f64> {
    %0 = stablehlo.convert %arg0 : (tensor<ui32>) -> tensor<ui64>
    %1 = stablehlo.convert %arg1 : (tensor<ui32>) -> tensor<ui64>
    %c32 = stablehlo.constant dense<32> : tensor<ui64>
    %2 = stablehlo.shift_left %0, %c32 : tensor<ui64>
    %3 = stablehlo.or %2, %1 : tensor<ui64>
    %c12 = stablehlo.constant dense<12> : tensor<ui64>
    %4 = stablehlo.shift_right_logical %3, %c12 : tensor<ui64>
    %c_exp = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %5 = stablehlo.or %4, %c_exp : tensor<ui64>
    %6 = stablehlo.bitcast_convert %5 : (tensor<ui64>) -> tensor<f64>
    %c_one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %7 = stablehlo.subtract %6, %c_one : tensor<f64>
    return %7 : tensor<f64>
  }
}
"#;
    // Use known threefry output bits: hi=0xABCD1234, lo=0x56789ABC
    let hi: u32 = 0xABCD1234;
    let lo: u32 = 0x56789ABC;
    let out = run_mlir(mlir, &[&u32_buf(&[hi]), &u32_buf(&[lo])], &[8]);
    let result = read_f64s(&out[0])[0];

    // Compute reference in Rust
    let combined: u64 = ((hi as u64) << 32) | (lo as u64);
    let shifted = combined >> 12;
    let with_exp = shifted | 0x3FF0000000000000u64;
    let reference = f64::from_bits(with_exp) - 1.0;

    assert!(
        (result - reference).abs() < 1e-15,
        "got {result}, expected {reference}"
    );
}

#[test]
fn test_full_prng_pipeline_seed_zero() {
    // Run the full ball MLIR and extract just the @inner function result
    // by building a wrapper that calls @inner with seed=0
    // But @inner is private and called from @main, so let's just run @main
    // with seed=0 and check the wind output.

    let ball_mlir = include_str!("../testdata/ball.stablehlo.mlir");
    let module = parse_module(ball_mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: unsafe extern "C" fn(*const *const u8, *mut *mut u8) =
        unsafe { std::mem::transmute(fn_ptr) };

    // main(%arg0: tensor<i64>,       -- tick (Globals.tick)
    //      %arg1: tensor<i64>,       -- seed (ball.seed)
    //      %arg2: tensor<3xf64>,     -- wind (ball.wind)
    //      %arg3: tensor<7xf64>,     -- world_pos (ball.world_pos)
    //      %arg4: tensor<6xf64>,     -- world_vel (ball.world_vel)
    //      %arg5: tensor<6xf64>,     -- force (ball.force)
    //      %arg6: tensor<7xf64>,     -- world_accel (ball.world_accel) [probably inertia]
    //      %arg7: tensor<6xf64>,     -- inertia? or accel
    //      %arg8: tensor<f64>)       -- sim_time_step

    // Build inputs matching the ball initial state
    let tick: i64 = 0;
    let seed: i64 = 0;
    let wind = [0.0f64, 0.0, 0.0];
    // world_pos: quaternion [0,0,0,1] + linear [0,0,6]
    let world_pos = [0.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0];
    // world_vel: angular [0,0,0] + linear [0,0,0]
    let world_vel = [0.0f64; 6];
    // force: angular [0,0,0] + linear [0,0,0]
    let force = [0.0f64; 6];
    // world_accel/inertia: [0,0,0,1, 1, 0, 0] (quat + mass + inertia_diag?)
    let inertia = [0.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    // another 6xf64 (world_accel?)
    let accel = [0.0f64; 6];
    let time_step: f64 = 1.0 / 120.0;

    let tick_buf = tick.to_le_bytes().to_vec();
    let seed_buf = seed.to_le_bytes().to_vec();
    let wind_buf: Vec<u8> = wind.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pos_buf: Vec<u8> = world_pos.iter().flat_map(|v| v.to_le_bytes()).collect();
    let vel_buf: Vec<u8> = world_vel.iter().flat_map(|v| v.to_le_bytes()).collect();
    let force_buf: Vec<u8> = force.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inertia_buf: Vec<u8> = inertia.iter().flat_map(|v| v.to_le_bytes()).collect();
    let accel_buf: Vec<u8> = accel.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ts_buf = time_step.to_le_bytes().to_vec();

    let inputs: Vec<&[u8]> = vec![
        &tick_buf,
        &seed_buf,
        &wind_buf,
        &pos_buf,
        &vel_buf,
        &force_buf,
        &inertia_buf,
        &accel_buf,
        &ts_buf,
    ];

    // main returns 9 outputs:
    // tensor<6xf64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<i64>,
    // tensor<6xf64>, tensor<7xf64>, tensor<7xf64>, tensor<6xf64>
    let output_sizes = vec![48, 8, 8, 24, 8, 48, 56, 56, 48];
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe {
        tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr());
    }

    // Output index 3 should be the new wind (tensor<3xf64>)
    let new_wind = read_f64s(&output_bufs[3]);
    eprintln!("Cranelift wind: {:?}", new_wind);
    eprintln!("Expected wind:  [-0.2058421394796434, -0.7847657764467411, 1.8160866726679836]");

    // Check if wind matches JAX reference
    let expected = [
        -0.2058421394796434f64,
        -0.7847657764467411,
        1.8160866726679836,
    ];
    for (i, (&got, &exp)) in new_wind.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        let rel = diff / exp.abs().max(1e-15);
        eprintln!(
            "  wind[{i}]: got={got:.16}, exp={exp:.16}, abs_diff={diff:.3e}, rel_diff={rel:.3e}"
        );
    }

    // For now, just report the values -- we expect this to fail until the PRNG is fixed
    for (i, (&got, &exp)) in new_wind.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "wind[{i}] diverged: got {got}, expected {exp}"
        );
    }
}
