mod common;
use common::*;

#[test]
fn test_one_threefry_round() {
    // One round of the threefry quarter-round, inlined as a standalone main function.
    // Uses the rotation constant 13 (first element of [13, 15, 26, 6]).
    // Input: x=[0,0,0], y=[0,0,0], rotation=13
    // x_new = x + y = [0,0,0]
    // y_rotated = rotate_left(y, 13) = [0,0,0]
    // y_new = x_new ^ y_rotated = [0,0,0]
    // With all-zero inputs, the output is all zeros.
    // Let's use non-zero inputs instead.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xui32>, %arg1: tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>) {
    %c = stablehlo.constant dense<13> : tensor<ui32>
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xui32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %2 = stablehlo.shift_left %arg1, %1 : tensor<3xui32>
    %c32 = stablehlo.constant dense<32> : tensor<ui32>
    %3 = stablehlo.subtract %c32, %c : tensor<ui32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %5 = stablehlo.shift_right_logical %arg1, %4 : tensor<3xui32>
    %6 = stablehlo.or %2, %5 : tensor<3xui32>
    %7 = stablehlo.xor %0, %6 : tensor<3xui32>
    return %0, %7 : tensor<3xui32>, tensor<3xui32>
  }
}
"#;
    let x: Vec<u32> = vec![100, 200, 300];
    let y: Vec<u32> = vec![0xDEAD, 0xBEEF, 0xCAFE];

    let out = run_mlir(mlir, &[&u32_buf(&x), &u32_buf(&y)], &[12, 12]);
    let x_new = read_u32s(&out[0]);
    let y_new = read_u32s(&out[1]);

    // Reference computation in Rust
    let mut x_exp = Vec::new();
    let mut y_exp = Vec::new();
    for i in 0..3 {
        let xn = x[i].wrapping_add(y[i]);
        let rot = y[i].rotate_left(13);
        let yn = xn ^ rot;
        x_exp.push(xn);
        y_exp.push(yn);
    }

    assert_eq!(x_new, x_exp, "x mismatch");
    assert_eq!(y_new, y_exp, "y mismatch");
}

#[test]
fn test_closed_call_standalone() {
    // Copy the closed_call function body but make it public main
    let mlir = include_str!("closed_call_test.mlir");

    let out = run_mlir(
        mlir,
        &[
            &i64_buf(&[0]),              // arg0: tensor<i64> (counter)
            &u32_buf(&[0, 0, 0]),        // arg1: tensor<3xui32> (x)
            &u32_buf(&[0, 0, 0]),        // arg2: tensor<3xui32> (y)
            &u32_buf(&[0]),              // arg3: tensor<ui32> (key1)
            &u32_buf(&[0]),              // arg4: tensor<ui32> (key2)
            &u32_buf(&[0]),              // arg5: tensor<ui32> (key_xor_const)
            &u32_buf(&[13, 15, 26, 6]),  // arg6: tensor<4xui32> (rotations1)
            &u32_buf(&[17, 29, 16, 24]), // arg7: tensor<4xui32> (rotations2)
        ],
        &[8, 12, 12, 4, 4, 4, 16, 16],
    );

    let new_counter = read_i64s(&out[0]);
    let out_x = read_u32s(&out[1]);
    let out_y = read_u32s(&out[2]);

    eprintln!("counter: {:?}", new_counter);
    eprintln!("out_x: {:?}", out_x);
    eprintln!("out_y: {:?}", out_y);

    assert_eq!(new_counter[0], 1, "counter should increment");

    // With all-zero inputs and rotations [13,15,26,6], the output should
    // be deterministic. Let's just verify it's not all zeros (which would
    // mean the key injection at the end added something).
    // Actually with all-zero keys the output x/y should be all zeros because:
    // x = x + y = 0, rotate(y=0, any) = 0, y_new = x ^ rot = 0, repeated 4 times
    // Then: x_final = x + key1_broadcast = 0 + [0,0,0] = [0,0,0]
    //        y_temp = y + key2_broadcast = 0 + [0,0,0] = [0,0,0]
    //        y_final = y_temp + convert(counter+1=1)_broadcast = [1,1,1]
    assert_eq!(out_x, vec![0, 0, 0], "x should be zero with zero inputs");
    assert_eq!(
        out_y,
        vec![1, 1, 1],
        "y should be [1,1,1] from counter injection"
    );
}
