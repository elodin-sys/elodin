mod common;
use common::*;

#[test]
fn test_dynamic_slice_3body_pattern() {
    // Exact three-body while loop body pattern:
    // dynamic_slice of tensor<2x3x7xf64> at (loop_index, 0, 0) with sizes [1, 3, 7]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x7xf64>, %arg1: tensor<i64>) -> tensor<1x3x7xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, %c1, sizes = [1, 3, 7] : (tensor<2x3x7xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x7xf64>
    return %0 : tensor<1x3x7xf64>
  }
}
"#;
    // 2x3x7 = 42 elements
    let data: Vec<f64> = (0..42).map(|i| (i + 1) as f64).collect();
    let in0 = f64_buf(&data);

    // Slice at index 0 -> first 21 elements
    let idx0 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&in0, &idx0], &[168]);
    let result = read_f64s(&out[0]);
    let expected0: Vec<f64> = (1..22).map(|i| i as f64).collect();
    assert_eq!(result, expected0, "dynamic_slice at index 0");

    // Slice at index 1 -> last 21 elements
    let idx1 = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&in0, &idx1], &[168]);
    let result = read_f64s(&out[0]);
    let expected1: Vec<f64> = (22..43).map(|i| i as f64).collect();
    assert_eq!(result, expected1, "dynamic_slice at index 1");
}

#[test]
fn test_dynamic_update_slice_3body_pattern() {
    // Exact pattern: update tensor<2xi64> at a dynamic index with tensor<1xi64>
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xi64>, %arg1: tensor<1xi64>, %arg2: tensor<i64>) -> tensor<2xi64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
}
"#;
    let base = i64_buf(&[100, 200]);
    let update = i64_buf(&[999]);

    // Update at index 0
    let idx0 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&base, &update, &idx0], &[16]);
    assert_eq!(read_i64s(&out[0]), &[999, 200]);

    // Update at index 1
    let idx1 = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&base, &update, &idx1], &[16]);
    assert_eq!(read_i64s(&out[0]), &[100, 999]);
}

#[test]
fn test_broadcast_in_dim_6_to_3x6() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<6xf64>) -> tensor<3x6xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<6xf64>) -> tensor<3x6xf64>
    return %0 : tensor<3x6xf64>
  }
}
"#;
    let input = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&input], &[144]);
    let result = read_f64s(&out[0]);
    // Each row should be [1,2,3,4,5,6]
    assert_eq!(result.len(), 18);
    for r in 0..3 {
        for c in 0..6 {
            assert!(
                (result[r * 6 + c] - (c + 1) as f64).abs() < 1e-10,
                "at [{r}][{c}]: got {}, expected {}",
                result[r * 6 + c],
                c + 1
            );
        }
    }
}

#[test]
fn test_broadcast_in_dim_3x1_to_3x3() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x1xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
  }
}
"#;
    let input = f64_buf(&[10.0, 20.0, 30.0]);
    let out = run_mlir(mlir, &[&input], &[72]);
    let result = read_f64s(&out[0]);
    // Each row is replicated: [[10,10,10],[20,20,20],[30,30,30]]
    assert_eq!(
        result,
        &[10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0]
    );
}

#[test]
fn test_transpose_3body_pattern() {
    // Exact three-body pattern: transpose 3x2x7 with dims=[1,0,2] -> 2x3x7
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x2x7xf64>) -> tensor<2x3x7xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<3x2x7xf64>) -> tensor<2x3x7xf64>
    return %0 : tensor<2x3x7xf64>
  }
}
"#;
    // 3x2x7 = 42 elements
    let data: Vec<f64> = (0..42).map(|i| i as f64).collect();
    let in0 = f64_buf(&data);
    let out = run_mlir(mlir, &[&in0], &[336]);
    let result = read_f64s(&out[0]);

    // Verify: output[j][i][k] = input[i][j][k]
    for j in 0..2 {
        for i in 0..3 {
            for k in 0..7 {
                let input_flat = i * 2 * 7 + j * 7 + k;
                let output_flat = j * 3 * 7 + i * 7 + k;
                let expected = input_flat as f64;
                let got = result[output_flat];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "at [{j}][{i}][{k}]: got {got}, expected {expected}"
                );
            }
        }
    }
}
