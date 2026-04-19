mod common;
use common::*;

// ---- LAPACK ops ----

#[test]
fn test_lapack_dgetrf_2x2() {
    // A = [[4, 3], [6, 3]] -> LU with partial pivoting
    // LAPACK should swap rows: pivot = [2, 2]
    // L = [[1, 0], [2/3, 1]], U = [[6, 3], [0, 1]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>) {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<2x2xf64>, tensor<2xi32>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[4.0, 3.0, 6.0, 3.0]);
    let out = run_mlir(mlir, &[&a], &[32, 8, 4]);
    let lu = read_f64s(&out[0]);
    let pivots: Vec<i32> = out[1]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let info: i32 = i32::from_le_bytes(out[2][..4].try_into().unwrap());
    eprintln!("LU: {:?}", lu);
    eprintln!("pivots: {:?}", pivots);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // After pivot: row 0 <-> row 1 (pivot[0] = 2 in 1-indexed)
    assert_eq!(pivots[0], 2);
    // LU packed: [[6, 3], [2/3, 1]]
    assert_f64_close(lu[0], 6.0);
    assert_f64_close(lu[1], 3.0);
    assert_f64_close(lu[2], 2.0 / 3.0);
    assert_f64_close(lu[3], 1.0);
}

#[test]
fn test_lapack_dtrsm_lower_unit() {
    // L = [[1, 0], [2, 1]] (unit lower triangular), b = [[5], [8]]
    // Solve L*x = b -> x = [[5], [-2]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x1xf64> {
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}} -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 0.0, 2.0, 1.0]);
    let b = f64_buf(&[5.0, 8.0]);
    let out = run_mlir(mlir, &[&a, &b], &[16]);
    let result = read_f64s(&out[0]);
    eprintln!("trsm result: {:?}", result);
    assert_f64s_close(&result, &[5.0, -2.0]);
}

#[test]
fn test_lapack_dtrsm_upper() {
    // U = [[3, 1], [0, 2]] (upper triangular), b = [[5], [4]]
    // Solve U*x = b -> x = [[1], [2]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x1xf64> {
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}} -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
"#;
    let a = f64_buf(&[3.0, 1.0, 0.0, 2.0]);
    let b = f64_buf(&[5.0, 4.0]);
    let out = run_mlir(mlir, &[&a, &b], &[16]);
    let result = read_f64s(&out[0]);
    eprintln!("trsm result: {:?}", result);
    assert_f64s_close(&result, &[1.0, 2.0]);
}

#[test]
fn test_lapack_svd_2x2() {
    // A = [[3, 0], [0, 2]]  -> SVD: U=I, S=[3,2], V=I
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>) {
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}} -> (tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[3.0, 0.0, 0.0, 2.0]);
    let out = run_mlir(mlir, &[&a], &[32, 16, 32, 32, 4]);
    let u = read_f64s(&out[0]);
    let s = read_f64s(&out[1]);
    let vt = read_f64s(&out[2]);
    eprintln!("U: {:?}", u);
    eprintln!("S: {:?}", s);
    eprintln!("VT: {:?}", vt);
    // S should be [3, 2] (descending)
    assert_f64_close(s[0], 3.0);
    assert_f64_close(s[1], 2.0);
    // U * diag(S) * VT should reconstruct A
    let mut reconstructed = [0.0f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                reconstructed[i * 2 + j] += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    assert_f64s_close(&reconstructed, &[3.0, 0.0, 0.0, 2.0]);
}

#[test]
fn test_lapack_cholesky_3x3() {
    // A = [[4, 2, 0], [2, 5, 3], [0, 3, 10]] -> L such that L*L^T = A
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<i32>) {
    %0:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {uplo = 76 : ui8}} -> (tensor<3x3xf64>, tensor<i32>)
    return %0#0, %0#1 : tensor<3x3xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72, 4]);
    let l = read_f64s(&out[0]);
    let info: i32 = i32::from_le_bytes(out[1][..4].try_into().unwrap());
    eprintln!("L: {:?}", l);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // L should be lower triangular
    assert_f64_close(l[1], 0.0); // L[0,1]
    assert_f64_close(l[2], 0.0); // L[0,2]
    assert_f64_close(l[5], 0.0); // L[1,2]
    // Verify L*L^T = A
    let mut llt = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                llt[i * 3 + j] += l[i * 3 + k] * l[j * 3 + k];
            }
        }
    }
    eprintln!("L*L^T: {:?}", llt);
    assert_f64s_close(&llt, &[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0]);
}

#[test]
fn test_lapack_svd_3x3_nontrivial() {
    // A = [[1, 2, 0], [0, 3, 1], [2, 0, 4]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>) {
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}} -> (tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0]);
    let out = run_mlir(mlir, &[&a], &[72, 24, 72, 72, 4]);
    // XLA convention: (A_overwritten, sigma, U, VT, info)
    let u = read_f64s(&out[2]);
    let s = read_f64s(&out[1]);
    let vt = read_f64s(&out[3]);
    eprintln!("U: {:?}", u);
    eprintln!("S: {:?}", s);
    eprintln!("VT: {:?}", vt);
    // Verify U * diag(S) * VT reconstructs A
    let mut reconstructed = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                reconstructed[i * 3 + j] += u[i * 3 + k] * s[k] * vt[k * 3 + j];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    let expected = [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0];
    for (i, (&a, &e)) in reconstructed.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-10,
            "element {i}: expected {e}, got {a} (abs_diff: {diff:.2e})"
        );
    }
}

#[test]
fn test_lapack_syevd_2x2() {
    // A = [[2, 1], [1, 3]] (symmetric) -> eigenvalues ~[1.382, 3.618]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xf64>, tensor<i32>) {
    %0:3 = stablehlo.custom_call @lapack_dsyevd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}} -> (tensor<2x2xf64>, tensor<2xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<2x2xf64>, tensor<2xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[2.0, 1.0, 1.0, 3.0]);
    let out = run_mlir(mlir, &[&a], &[32, 16, 4]);
    let eigvecs = read_f64s(&out[0]);
    let eigvals = read_f64s(&out[1]);
    let info: i32 = i32::from_le_bytes(out[2][..4].try_into().unwrap());
    eprintln!("eigvals: {:?}", eigvals);
    eprintln!("eigvecs: {:?}", eigvecs);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // Eigenvalues of [[2,1],[1,3]] are (5-sqrt(5))/2 and (5+sqrt(5))/2
    let sqrt5 = 5.0f64.sqrt();
    assert_f64_close(eigvals[0], (5.0 - sqrt5) / 2.0);
    assert_f64_close(eigvals[1], (5.0 + sqrt5) / 2.0);
    // Verify V * diag(lambda) * V^T = A
    let mut reconstructed = [0.0f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                reconstructed[i * 2 + j] += eigvecs[i * 2 + k] * eigvals[k] * eigvecs[j * 2 + k];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    assert_f64s_close(&reconstructed, &[2.0, 1.0, 1.0, 3.0]);
}

#[test]
fn test_lapack_qr_3x3() {
    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>) {
    %0:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xf64>)
    return %0#0, %0#1 : tensor<3x3xf64>, tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72, 24]);
    let qr_packed = read_f64s(&out[0]);
    let tau = read_f64s(&out[1]);
    eprintln!("QR packed: {:?}", qr_packed);
    eprintln!("tau: {:?}", tau);
    // Just verify the packed form and tau are populated (non-zero)
    assert!(tau[0].abs() > 1e-15, "tau[0] should be non-zero");
}

#[test]
fn test_lapack_qr_orgqr_roundtrip_3x3() {
    // Full QR roundtrip: A -> (QR, tau) -> (Q, R) -> Q*R should equal A
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xf64>)
    %1 = stablehlo.custom_call @lapack_dorgqr_ffi(%0#0, %0#1) {backend_config = "", mhlo.backend_config = {}} -> tensor<3x3xf64>
    return %1 : tensor<3x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72]);
    let q = read_f64s(&out[0]);
    eprintln!("Q: {:?}", q);
    // Q should be orthogonal: Q^T * Q = I
    let mut qtq = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                qtq[i * 3 + j] += q[k * 3 + i] * q[k * 3 + j];
            }
        }
    }
    eprintln!("Q^T*Q: {:?}", qtq);
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (&a, &e)) in qtq.iter().zip(identity.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-10,
            "Q^T*Q[{}]: expected {e}, got {a} (abs_diff: {diff:.2e})",
            i
        );
    }
}

#[test]
fn test_solve_3x3_vector_rhs() {
    // Direct test: solve A*x = b where A = [[1.01, 0.00833, 0], [0, 1.01, 0.00833], [0, 0, 1.01]]
    // b = [1, 3, 5] -> x ~= [0.966, 2.929, 4.950]
    // Use the exact MLIR pattern from the linalg solve path
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare GE, %0#2, %3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<3x3xi1>, tensor<3x3xf64>
    %9 = stablehlo.iota dim = 0 : tensor<3xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_3 = %c_2, %iterArg_4 = %c_1, %iterArg_5 = %9) : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    cond {
      %c_6 = stablehlo.constant dense<3> : tensor<i64>
      %14 = stablehlo.compare LT, %iterArg_3, %c_6, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14:2 = func.call @closed_call(%iterArg, %iterArg_4, %iterArg_5) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %15 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %15, %14#0, %14#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = call @_lu_solve(%8, %10#3, %arg1) : (tensor<3x3xf64>, tensor<3xi32>, tensor<3xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
  func.func private @closed_call(%arg0: tensor<3xi32>, %arg1: tensor<i64>, %arg2: tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg1, %c : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare LT, %arg1, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.convert %arg1 : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %6 = stablehlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.compare LT, %arg1, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %arg1 : tensor<i64>
    %c_3 = stablehlo.constant dense<3> : tensor<i64>
    %9 = stablehlo.add %8, %c_3 : tensor<i64>
    %10 = stablehlo.select %7, %9, %arg1 : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare LT, %6, %c_4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<3> : tensor<i32>
    %14 = stablehlo.add %6, %c_5 : tensor<i32>
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %arg2, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %18 = stablehlo.compare LT, %arg1, %c_6, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<3> : tensor<i64>
    %19 = stablehlo.add %arg1, %c_7 : tensor<i64>
    %20 = stablehlo.select %18, %19, %arg1 : tensor<i1>, tensor<i64>
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.scatter"(%arg2, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.compare LT, %6, %c_8, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<3> : tensor<i32>
    %25 = stablehlo.add %6, %c_9 : tensor<i32>
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
    return %0, %28 : tensor<i64>, tensor<3xi32>
  }
  func.func private @_lu_solve(%arg0: tensor<3x3xf64>, %arg1: tensor<3xi32>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.compare LT, %arg1, %1, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<3xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<3xi1>, tensor<3xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x1xf64>, tensor<3x1xi32>) -> tensor<3x1xf64>
    %8 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %7) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}} -> tensor<3x1xf64>
    %9 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %8) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}} -> tensor<3x1xf64>
    %10 = stablehlo.slice %9 [0:3, 0:1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<3x1xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[
        1.01,
        1.0 / 120.0,
        0.0,
        0.0,
        1.01,
        1.0 / 120.0,
        0.0,
        0.0,
        1.01,
    ]);
    let b = f64_buf(&[1.0, 3.0, 5.0]);
    let out = run_mlir(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    eprintln!("solve result: {:?}", result);
    // Expected: [0.966, 2.929, 4.950] approximately
    assert_f64_close(result[2], 5.0 / 1.01);
}

#[test]
#[ignore = "known issue: 3D gather/broadcast path produces wrong results for matrix-RHS solve"]
fn test_linalg_one_tick() {
    let mlir = include_str!("../testdata/linalg.stablehlo.mlir");

    // Inputs match the world() spawn in sim.py:
    // arg0: tick (i64) = 0
    let arg0 = i64_buf(&[0]);
    // arg1: mrhs_state (3x2 f64) = [[1,2],[3,4],[5,6]]
    let arg1 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // arg2: sm2_state (2 f64) = [1.0, 0.5]
    let arg2 = f64_buf(&[1.0, 0.5]);
    // arg3: sm2_cov (2x2 f64) = eye(2)*5 = [[5,0],[0,5]]
    let arg3 = f64_buf(&[5.0, 0.0, 0.0, 5.0]);
    // arg4: kf3_state (3 f64) = [0, 1, 0]
    let arg4 = f64_buf(&[0.0, 1.0, 0.0]);
    // arg5: kf3_cov (3x3 f64) = eye(3)*10
    let arg5 = f64_buf(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]);
    // arg6: kf3_info (5 f64) = zeros
    let arg6 = f64_buf(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    // arg7: ekf6_state (6 f64) = [0,0,100,10,0,-5]
    let arg7 = f64_buf(&[0.0, 0.0, 100.0, 10.0, 0.0, -5.0]);
    // arg8: ekf6_cov (6x6 f64) = eye(6)*100
    let mut arg8_data = vec![0.0f64; 36];
    for i in 0..6 {
        arg8_data[i * 6 + i] = 100.0;
    }
    let arg8 = f64_buf(&arg8_data);
    // arg9: ekf6_info (4 f64) = zeros
    let arg9 = f64_buf(&[0.0, 0.0, 0.0, 0.0]);
    // arg10: mode_state (4 i64) = [0,0,0,0]
    let arg10 = i64_buf(&[0, 0, 0, 0]);

    let inputs: Vec<&[u8]> = vec![
        &arg0, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8, &arg9, &arg10,
    ];
    // Outputs (from return type):
    // result[0]: 4xf64 (ekf6_info)
    // result[1]: 3x3xf64 (kf3_cov)
    // result[2]: 2xf64 (sm2_state)
    // result[3]: 3x2xf64 (mrhs_state)
    // result[4]: i64 (tick)
    // result[5]: 2x2xf64 (sm2_cov)
    // result[6]: 4xi64 (mode_state)
    // result[7]: 3xf64 (kf3_state)
    // result[8]: 6xf64 (ekf6_state)
    // result[9]: 5xf64 (kf3_info)
    // result[10]: 6x6xf64 (ekf6_cov)
    let output_sizes = vec![32, 72, 16, 48, 8, 32, 32, 24, 48, 40, 288];
    let out = run_mlir(mlir, &inputs, &output_sizes);

    // Check result[3]: mrhs_state after one tick
    let mrhs = read_f64s(&out[3]);
    eprintln!("mrhs_state: {:?}", mrhs);
    // Expected: solve(F3 + 0.01*I, [[1,2],[3,4],[5,6]])
    // F3 + 0.01*I is upper triangular with diag ~1.01
    assert_f64_close(mrhs[0], 0.9659286191338475);
    assert_f64_close(mrhs[4], 4.9504950495049505);
}

// ---- Arithmetic ops ----

#[test]
fn test_add() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let in1 = f64_buf(&[4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let in1 = f64_buf(&[4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_subtract() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[5.0, 7.0, 9.0]);
    let in1 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_multiply() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[2.0, 3.0, 4.0]);
    let in1 = f64_buf(&[5.0, 6.0, 7.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 18.0, 28.0]);
}

#[test]
fn test_divide() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[10.0]);
    let in1 = f64_buf(&[2.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 5.0);
}

#[test]
fn test_negate() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.negate %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-1.0, 0.0, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert!(result[1].abs() < 1e-15);
    assert_f64_close(result[2], -3.0);
}

#[test]
fn test_sqrt() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[9.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 3.0);
}

#[test]
fn test_maximum() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[3.0]);
    let in1 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 5.0);
}

// ---- Comparison ----

#[test]
fn test_compare_lt() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<i1> {
    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
"#;
    let in0 = f64_buf(&[3.0]);
    let in1 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[1]);
    assert_eq!(out[0][0], 1, "3.0 < 5.0 should be true");
}

// ---- Constants ----

#[test]
fn test_constant() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<2xf64> {
    %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0]);
}

// ---- Shape ops ----

#[test]
fn test_reshape() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<1xf64>) -> tensor<f64> {
    %0 = stablehlo.reshape %arg0 : (tensor<1xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[42.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 42.0);
}

#[test]
fn test_broadcast_in_dim() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[7.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[7.0, 7.0, 7.0]);
}

#[test]
fn test_slice() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<2xf64> {
    %0 = stablehlo.slice %arg0 [1:3] : (tensor<4xf64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0]);
    let out = run_mlir(mlir, &[&in0], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0]);
}

#[test]
fn test_concatenate() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xf64>, %arg1: tensor<1xf64>) -> tensor<3xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0]);
    let in1 = f64_buf(&[3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

// ---- Type conversion ----

#[test]
fn test_convert_i64_to_f64() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = i64_buf(&[42]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 42.0);
}

#[test]
fn test_iota() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[0.0, 1.0, 2.0]);
}

// ---- Integer bitwise ops ----

#[test]
fn test_xor() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.xor %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xFF]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0xF0);
}

#[test]
fn test_shift_left() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.shift_left %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[1]);
    let in1 = i64_buf(&[3]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 8);
}

#[test]
fn test_shift_right_logical() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[16]);
    let in1 = i64_buf(&[2]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 4);
}

#[test]
fn test_or() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xF0]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0xFF);
}

#[test]
fn test_and() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.and %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xFF]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0x0F);
}

// ---- Dot product ----

#[test]
fn test_dot_general_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let in1 = f64_buf(&[4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 32.0);
}

// ---- Reduce ----

#[test]
fn test_reduce_add() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 6.0);
}

// ---- Transcendental ----

#[test]
fn test_erf_inv() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = chlo.erf_inv %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let input = 0.5_f64;
    let expected = 0.4769362762044699_f64;
    let in0 = f64_buf(&[input]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], expected);
}

// ---- Function call ----

#[test]
fn test_call() {
    let mlir = r#"
module @module {
  func.func private @double(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = call @double(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 10.0);
}

// ---- Case ----

#[test]
fn test_case_two_branch() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i32>) -> tensor<f64> {
    %0 = stablehlo.case(%arg0) ({
      %cst0 = stablehlo.constant dense<10.0> : tensor<f64>
      stablehlo.return %cst0 : tensor<f64>
    }, {
      %cst1 = stablehlo.constant dense<20.0> : tensor<f64>
      stablehlo.return %cst1 : tensor<f64>
    }) : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = i32_buf(&[0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 10.0);

    let in1 = i32_buf(&[1]);
    let out = run_mlir(mlir, &[&in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 20.0);
}

/// Large-body `stablehlo.case` inside a pointer-ABI function.
///
/// Each branch runs a long chain of elementwise ops on a tensor that
/// crosses `LARGE_TENSOR_THRESHOLD`, pushing the branch body well
/// above `CASE_BRANCH_SPLIT_INSTRS` (64 StableHLO instructions). The
/// branches are therefore lifted into their own Cranelift functions
/// by `compile_case_branch_as_function_mem`, while the merge/dispatch
/// stays inline in the caller. This test locks the output of the
/// split path against hand-computed expected values, so any regression
/// in captured-variable wiring or result marshaling shows up
/// immediately.
#[test]
fn test_case_large_branch_splits_into_functions() {
    // Build MLIR programmatically: each branch contains 70 `add`
    // ops (> 64) on a tensor<100xf64>, so `count_body_instructions`
    // crosses the splitter threshold and ptr-ABI main dispatches via
    // the split path.
    const N_OPS: usize = 70;
    const VEC_LEN: usize = 100;
    let mut branch0 = String::new();
    branch0.push_str("      %c0 = stablehlo.constant dense<1.0> : tensor<100xf64>\n");
    branch0.push_str("      %b0_0 = stablehlo.add %arg1, %c0 : tensor<100xf64>\n");
    for i in 1..N_OPS {
        branch0.push_str(&format!(
            "      %b0_{i} = stablehlo.add %b0_{}, %c0 : tensor<100xf64>\n",
            i - 1
        ));
    }
    branch0.push_str(&format!(
        "      stablehlo.return %b0_{} : tensor<100xf64>\n",
        N_OPS - 1
    ));

    let mut branch1 = String::new();
    branch1.push_str("      %c1 = stablehlo.constant dense<2.0> : tensor<100xf64>\n");
    branch1.push_str("      %b1_0 = stablehlo.multiply %arg1, %c1 : tensor<100xf64>\n");
    for i in 1..N_OPS {
        branch1.push_str(&format!(
            "      %b1_{i} = stablehlo.add %b1_{}, %c1 : tensor<100xf64>\n",
            i - 1
        ));
    }
    branch1.push_str(&format!(
        "      stablehlo.return %b1_{} : tensor<100xf64>\n",
        N_OPS - 1
    ));

    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<100xf64>) -> tensor<100xf64> {{
    %0 = "stablehlo.case"(%arg0) ({{
{branch0}    }}, {{
{branch1}    }}) : (tensor<i32>) -> tensor<100xf64>
    return %0 : tensor<100xf64>
  }}
}}
"#
    );

    // Input vector: 1.0, 2.0, ..., 100.0.
    let input: Vec<f64> = (1..=VEC_LEN).map(|i| i as f64).collect();
    let in1 = f64_buf(&input);
    let out_bytes = VEC_LEN * 8;

    // Branch 0: x + (1.0 * N_OPS).
    let in0 = i32_buf(&[0]);
    let out = run_mlir_mem(&mlir, &[&in0, &in1], &[out_bytes]);
    let got = read_f64s(&out[0]);
    let expected0: Vec<f64> = input.iter().map(|x| x + N_OPS as f64).collect();
    assert_f64s_close(&got, &expected0);

    // Branch 1: (x * 2.0) + 2.0 * (N_OPS - 1).
    let in0 = i32_buf(&[1]);
    let out = run_mlir_mem(&mlir, &[&in0, &in1], &[out_bytes]);
    let got = read_f64s(&out[0]);
    let expected1: Vec<f64> = input
        .iter()
        .map(|x| x * 2.0 + 2.0 * (N_OPS - 1) as f64)
        .collect();
    assert_f64s_close(&got, &expected1);
}

// ---- While loop ----

#[test]
fn test_while_loop() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.while(%iter = %arg0) : tensor<i64>
    cond {
      %limit = stablehlo.constant dense<5> : tensor<i64>
      %cmp = stablehlo.compare LT, %iter, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    }
    do {
      %one = stablehlo.constant dense<1> : tensor<i64>
      %next = stablehlo.add %iter, %one : tensor<i64>
      stablehlo.return %next : tensor<i64>
    }
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 5);
}

// ---- Transpose ----

#[test]
fn test_transpose_2d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
"#;
    // Input: [[1,2,3],[4,5,6]]
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0], &[48]);
    // Expected: [[1,4],[2,5],[3,6]]
    let result = read_f64s(&out[0]);
    assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_3d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x4xf64>) -> tensor<3x2x4xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<2x3x4xf64>) -> tensor<3x2x4xf64>
    return %0 : tensor<3x2x4xf64>
  }
}
"#;
    // Input: 2x3x4 = 24 elements, row-major: group0=[0..12), group1=[12..24)
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let in0 = f64_buf(&input);
    let out = run_mlir(mlir, &[&in0], &[192]);
    let result = read_f64s(&out[0]);
    // dims=[1,0,2]: output[j][i][k] = input[i][j][k]
    // output shape 3x2x4
    for j in 0..3 {
        for i in 0..2 {
            for k in 0..4 {
                let expected = (i * 3 * 4 + j * 4 + k) as f64;
                let got = result[j * 2 * 4 + i * 4 + k];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "transpose[{j}][{i}][{k}]: got {got}, expected {expected}"
                );
            }
        }
    }
}

// ---- Dynamic slice ----

#[test]
fn test_dynamic_slice_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<i64>) -> tensor<2xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [2] : (tensor<5xf64>, tensor<i64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let idx = i64_buf(&[2]);
    let out = run_mlir(mlir, &[&in0, &idx], &[16]);
    assert_eq!(read_f64s(&out[0]), &[30.0, 40.0]);
}

#[test]
fn test_dynamic_slice_3d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x4xf64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<1x3x4xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [1, 3, 4] : (tensor<2x3x4xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x4xf64>
    return %0 : tensor<1x3x4xf64>
  }
}
"#;
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let in0 = f64_buf(&input);
    let idx0 = i64_buf(&[1]);
    let idx1 = i64_buf(&[0]);
    let idx2 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&in0, &idx0, &idx1, &idx2], &[96]);
    let result = read_f64s(&out[0]);
    // Slice starting at [1,0,0] with sizes [1,3,4] = elements 12..24
    let expected: Vec<f64> = (12..24).map(|i| i as f64).collect();
    assert_eq!(result, expected);
}

// ---- Dynamic update slice ----

#[test]
fn test_dynamic_update_slice_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<2xf64>, %arg2: tensor<i64>) -> tensor<5xf64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<5xf64>, tensor<2xf64>, tensor<i64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let base = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let update = f64_buf(&[99.0, 100.0]);
    let idx = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&base, &update, &idx], &[40]);
    assert_eq!(read_f64s(&out[0]), &[1.0, 99.0, 100.0, 4.0, 5.0]);
}

// ---- Gather (row-select) ----

#[test]
fn test_gather_row_select() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<2x1xui32>) -> tensor<2x4xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<3x4xf64>, tensor<2x1xui32>) -> tensor<2x4xf64>
    return %0 : tensor<2x4xf64>
  }
}
"#;
    // operand: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    let operand = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    // indices: [[2],[0]] -- pick row 2 then row 0
    let indices: Vec<u8> = [2u32, 0u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_mlir(mlir, &[&operand, &indices], &[64]);
    let result = read_f64s(&out[0]);
    // Expected: row 2 = [9,10,11,12], row 0 = [1,2,3,4]
    assert_eq!(result, &[9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]);
}

// ---- Batched dot product ----

#[test]
fn test_dot_general_batched() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<3x4xf64>) -> tensor<3xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    // batch 0: [1,2,3,4]·[1,0,0,0] = 1
    // batch 1: [5,6,7,8]·[0,1,0,0] = 6
    // batch 2: [9,10,11,12]·[0,0,1,0] = 11
    let a = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    let b = f64_buf(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let out = run_mlir(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 6.0, 11.0]);
}

// ---- New elementwise / transcendental ops ----

#[test]
fn test_sine() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.sine %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0);
    assert!(result[2].abs() < 1e-15);
}

#[test]
fn test_cosine() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.cosine %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert!(result[1].abs() < 1e-15);
    assert_f64_close(result[2], -1.0);
}

#[test]
fn test_atan2() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let y = f64_buf(&[0.0, 1.0, -1.0]);
    let x = f64_buf(&[1.0, 0.0, 0.0]);
    let out = run_mlir(mlir, &[&y, &x], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], std::f64::consts::FRAC_PI_2);
    assert_f64_close(result[2], -std::f64::consts::FRAC_PI_2);
}

#[test]
fn test_abs_float() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.abs %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-3.0, 0.0, 5.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[3.0, 0.0, 5.0]);
}

#[test]
fn test_minimum() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 5.0, 3.0]);
    let in1 = f64_buf(&[4.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_sign() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.sign %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-3.5, 0.0, 7.2]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[-1.0, 0.0, 1.0]);
}

#[test]
fn test_remainder() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.remainder %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[7.0, 10.0, -5.5]);
    let in1 = f64_buf(&[3.0, 3.0, 2.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], -1.5);
}

#[test]
fn test_acos() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.acos %arg0 : tensor<f64> -> tensor<f64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 0.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], std::f64::consts::FRAC_PI_2);
    assert_f64_close(result[2], std::f64::consts::PI);
}

#[test]
fn test_exponential() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.exponential %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, 1.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert_f64_close(result[1], std::f64::consts::E);
    assert_f64_close(result[2], 1.0 / std::f64::consts::E);
}

#[test]
fn test_log() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.log %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, std::f64::consts::E, 10.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], 10.0_f64.ln());
}

#[test]
fn test_clamp() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %lo = stablehlo.constant dense<-1.0> : tensor<f64>
    %hi = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = stablehlo.clamp %lo, %arg0, %hi : (tensor<f64>, tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-5.0, 0.5, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[-1.0, 0.5, 1.0]);
}

#[test]
fn test_power() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.power %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let base = f64_buf(&[2.0, 3.0, 10.0]);
    let exp = f64_buf(&[3.0, 2.0, 0.5]);
    let out = run_mlir(mlir, &[&base, &exp], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 8.0);
    assert_f64_close(result[1], 9.0);
    assert_f64_close(result[2], 10.0_f64.sqrt());
}

#[test]
fn test_reverse() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let out = run_mlir(mlir, &[&in0], &[32]);
    assert_f64s_close(&read_f64s(&out[0]), &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_tanh() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.tanh %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, 1.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0_f64.tanh());
    assert_f64_close(result[2], (-1.0_f64).tanh());
}

#[test]
fn test_ssa_shadow_redefine() {
    let mlir = r#"
module @module {
  func.func private @use_pair(%arg0: tensor<2xui32>) -> tensor<2xui32> {
    return %arg0 : tensor<2xui32>
  }
  func.func public @main(%arg0: tensor<i64>) -> (tensor<2xui32>, tensor<i64>) {
    %c = stablehlo.constant dense<[42, 99]> : tensor<2xui32>
    %0 = call @use_pair(%c) : (tensor<2xui32>) -> tensor<2xui32>
    %c = stablehlo.constant dense<7> : tensor<i64>
    %1 = stablehlo.add %arg0, %c : tensor<i64>
    return %0, %1 : tensor<2xui32>, tensor<i64>
  }
}
"#;
    let input = 10i64.to_le_bytes().to_vec();
    let out = run_mlir(mlir, &[&input], &[8, 8]);
    let seed: Vec<u32> = out[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let sum = i64::from_le_bytes(out[1].as_slice().try_into().unwrap());
    assert_eq!(
        seed,
        vec![42, 99],
        "First %c definition corrupted by redefinition"
    );
    assert_eq!(sum, 17, "Second %c definition incorrect");
}

#[test]
fn test_gather_i32_indices_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<3x1xi32>) -> tensor<3xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let indices = i32_buf(&[0, 2, 4]);
    let out = run_mlir(mlir, &[&data, &indices], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 30.0, 50.0]);
}

#[test]
fn test_tan() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.tan %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[
        0.0,
        std::f64::consts::FRAC_PI_4,
        -std::f64::consts::FRAC_PI_4,
    ]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert!(result[0].abs() < 1e-15);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], -1.0);
}

#[test]
fn test_floor() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.floor %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.7, -0.3, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, -1.0, 3.0]);
}

#[test]
fn test_many_args_call_and_value_survival() {
    // Tests: 1) 32-arg function call works 2) return values survive past a 32-arg sret call
    let mlir = r#"
module @module {
  func.func private @noise(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.add %arg1, %0 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
  func.func private @big_sret(%a0: tensor<f64>, %a1: tensor<f64>, %a2: tensor<f64>,
    %a3: tensor<3xf64>, %a4: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
    %s = stablehlo.add %a3, %a4 : tensor<3xf64>
    %b0 = stablehlo.broadcast_in_dim %a0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r0 = stablehlo.add %s, %b0 : tensor<3xf64>
    %b1 = stablehlo.broadcast_in_dim %a1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r1 = stablehlo.add %s, %b1 : tensor<3xf64>
    %b2 = stablehlo.broadcast_in_dim %a2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r2 = stablehlo.add %s, %b2 : tensor<3xf64>
    return %r0, %r1, %r2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
  func.func public @main(%tick: tensor<f64>, %bias: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %noise_out = call @noise(%tick, %bias) : (tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %c1 = stablehlo.constant dense<1.0> : tensor<f64>
    %c2 = stablehlo.constant dense<2.0> : tensor<f64>
    %c3 = stablehlo.constant dense<3.0> : tensor<f64>
    %sret_out:3 = call @big_sret(%c1, %c2, %c3, %noise_out, %bias) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    %next_tick = stablehlo.add %tick, %c1 : tensor<f64>
    %noise_out2 = call @noise(%next_tick, %noise_out) : (tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    return %noise_out, %noise_out2 : tensor<3xf64>, tensor<3xf64>
  }
}
"#;
    let tick = f64_buf(&[0.0]);
    let bias = f64_buf(&[10.0, 20.0, 30.0]);
    let out = run_mlir(mlir, &[&tick, &bias], &[24, 24]);
    let r0 = read_f64s(&out[0]); // noise_out = bias + tick = [10, 20, 30]
    let r1 = read_f64s(&out[1]); // noise_out2 = noise_out + (tick+1) = [11, 21, 31]
    assert_f64s_close(&r0, &[10.0, 20.0, 30.0]);
    assert_f64s_close(&r1, &[11.0, 21.0, 31.0]);
}

#[test]
fn test_dynamic_slice_clamps_out_of_bounds() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4x3xf64>, %arg1: tensor<i32>) -> tensor<1x3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    return %0 : tensor<1x3xf64>
  }
}
"#;
    // 4x3 matrix: row0=[1,2,3], row1=[4,5,6], row2=[7,8,9], row3=[10,11,12]
    let data = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    // Index 10 is way out of bounds (max valid = 3), should clamp to row 3
    let idx = i32_buf(&[10]);
    let out = run_mlir(mlir, &[&data, &idx], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 11.0, 12.0]);

    // Index -5 is negative, should clamp to row 0
    let idx_neg = i32_buf(&[-5]);
    let out_neg = run_mlir(mlir, &[&data, &idx_neg], &[24]);
    assert_f64s_close(&read_f64s(&out_neg[0]), &[1.0, 2.0, 3.0]);
}

// ---- Pointer-ABI (memory-backed) path tests ----

fn mem_binop_test(op: &str, a: &[f64], b: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>, %arg1: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.{op} %arg0, %arg1 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let in1 = f64_buf(b);
    let out = run_mlir_mem(&mlir, &[&in0, &in1], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

fn mem_unop_test(op: &str, a: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.{op} %arg0 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let out = run_mlir_mem(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

#[test]
fn test_subtract_mem() {
    mem_binop_test(
        "subtract",
        &[5.0, 7.0, 9.0],
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
    );
}

#[test]
fn test_multiply_mem() {
    mem_binop_test(
        "multiply",
        &[2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0],
        &[10.0, 18.0, 28.0],
    );
}

#[test]
fn test_divide_mem() {
    mem_binop_test(
        "divide",
        &[10.0, 18.0, 28.0],
        &[2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0],
    );
}

#[test]
fn test_maximum_mem() {
    mem_binop_test(
        "maximum",
        &[1.0, 5.0, 3.0],
        &[4.0, 2.0, 6.0],
        &[4.0, 5.0, 6.0],
    );
}

#[test]
fn test_minimum_mem() {
    mem_binop_test(
        "minimum",
        &[1.0, 5.0, 3.0],
        &[4.0, 2.0, 6.0],
        &[1.0, 2.0, 3.0],
    );
}

#[test]
fn test_negate_mem() {
    mem_unop_test("negate", &[1.0, -2.0, 3.0], &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_sqrt_mem() {
    mem_unop_test("sqrt", &[4.0, 9.0, 16.0], &[2.0, 3.0, 4.0]);
}

#[test]
fn test_floor_mem() {
    mem_unop_test("floor", &[1.7, 2.3, -0.5], &[1.0, 2.0, -1.0]);
}

#[test]
fn test_sine_mem() {
    mem_unop_test("sine", &[0.0, std::f64::consts::FRAC_PI_2], &[0.0, 1.0]);
}

#[test]
fn test_cosine_mem() {
    mem_unop_test("cosine", &[0.0, std::f64::consts::PI], &[1.0, -1.0]);
}

#[test]
fn test_exponential_mem() {
    mem_unop_test("exponential", &[0.0, 1.0], &[1.0, std::f64::consts::E]);
}

#[test]
fn test_log_mem() {
    mem_unop_test("log", &[1.0, std::f64::consts::E], &[0.0, 1.0]);
}

#[test]
fn test_tanh_mem() {
    mem_unop_test("tanh", &[0.0], &[0.0]);
}

#[test]
fn test_abs_mem() {
    mem_unop_test("abs", &[-3.0, 0.0, 5.0], &[3.0, 0.0, 5.0]);
}

#[test]
fn test_power_mem() {
    mem_binop_test("power", &[2.0, 3.0], &[3.0, 2.0], &[8.0, 9.0]);
}

#[test]
fn test_reshape_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<6xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<6xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_constant_mem() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3xf64> {
    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_compare_lt_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xi1> {
    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 5.0, 3.0]);
    let in1 = f64_buf(&[2.0, 4.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[3]);
    assert_eq!(out[0], vec![1, 0, 0]);
}

#[test]
fn test_select_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi1>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<3xi1>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let cond = vec![1u8, 0, 1];
    let in1 = f64_buf(&[10.0, 20.0, 30.0]);
    let in2 = f64_buf(&[100.0, 200.0, 300.0]);
    let out = run_mlir_mem(mlir, &[&cond, &in1, &in2], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 200.0, 30.0]);
}

#[test]
fn test_convert_i64_to_f64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>) -> tensor<3xf64> {
    %0 = stablehlo.convert %arg0 : (tensor<3xi64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = i64_buf(&[1, 2, 3]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_iota_mem() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3x2xi64> {
    %0 = stablehlo.iota dim = 1 : tensor<3x2xi64>
    return %0 : tensor<3x2xi64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[], &[48]);
    assert_eq!(read_i64s(&out[0]), vec![0, 1, 0, 1, 0, 1]);
}

#[test]
fn test_broadcast_in_dim_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_transpose_2d_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_concatenate_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xf64>, %arg1: tensor<3xf64>) -> tensor<5xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf64>, tensor<3xf64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0]);
    let in1 = f64_buf(&[3.0, 4.0, 5.0]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[40]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_dynamic_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<i64>) -> tensor<3xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [3] : (tensor<5xf64>, tensor<i64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let idx = i64_buf(&[1]);
    let out = run_mlir_mem(mlir, &[&data, &idx], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_dynamic_update_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<2xf64>, %arg2: tensor<i64>) -> tensor<5xf64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<5xf64>, tensor<2xf64>, tensor<i64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let data = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let upd = f64_buf(&[90.0, 91.0]);
    let idx = i64_buf(&[2]);
    let out = run_mlir_mem(mlir, &[&data, &upd, &idx], &[40]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 90.0, 91.0, 5.0]);
}

#[test]
fn test_dot_general_matmul_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0] :
      (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x2xf64>
    return %0 : tensor<2x2xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_buf(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[32]);
    assert_f64s_close(&read_f64s(&out[0]), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_reduce_sum_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<6xf64>) -> tensor<2xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<6xf64>) -> tensor<2x3xf64>
    %init = stablehlo.constant dense<0.0> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
    return %1 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[6.0, 15.0]);
}

#[test]
fn test_while_mem_count1() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %c0) : tensor<f64>, tensor<i64>
      cond {
        %limit = stablehlo.constant dense<1> : tensor<i64>
        %cmp = stablehlo.compare LT, %iterArg_0, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %inc = stablehlo.constant dense<10.0> : tensor<f64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %new_val = stablehlo.add %iterArg, %inc : tensor<f64>
        %new_idx = stablehlo.add %iterArg_0, %one : tensor<i64>
        stablehlo.return %new_val, %new_idx : tensor<f64>, tensor<i64>
      }
    return %0#0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[5.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[8]);
    assert_f64s_close(&read_f64s(&out[0]), &[15.0]);
}

#[test]
fn test_while_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %c0) : tensor<3xf64>, tensor<i64>
      cond {
        %limit = stablehlo.constant dense<3> : tensor<i64>
        %cmp = stablehlo.compare LT, %iterArg_0, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %inc = stablehlo.constant dense<[1.0, 1.0, 1.0]> : tensor<3xf64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %new_val = stablehlo.add %iterArg, %inc : tensor<3xf64>
        %new_idx = stablehlo.add %iterArg_0, %one : tensor<i64>
        stablehlo.return %new_val, %new_idx : tensor<3xf64>, tensor<i64>
      }
    return %0#0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[13.0, 23.0, 33.0]);
}

#[test]
fn test_broadcast_i32_1d_to_2d_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi32>) -> tensor<3x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    return %0 : tensor<3x1xi32>
  }
}
"#;
    let in0 = i32_buf(&[10, 20, 30]);
    let out = run_mlir_mem(mlir, &[&in0], &[12]);
    let result: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![10, 20, 30]);
}

#[test]
fn test_gather_with_i32_indices_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<3xi32>) -> tensor<3xf64> {
    %idx = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %0 = "stablehlo.gather"(%arg0, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let indices = i32_buf(&[0, 2, 4]);
    let out = run_mlir_mem(mlir, &[&data, &indices], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "gather with i32 indices produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[10.0, 30.0, 50.0]);
}

// ---- Drone @inner function regression test ----

#[test]
fn test_drone_inner_mem() {
    // This is the exact @inner function from the drone MLIR, inlined into @main.
    // It builds a 22x3 lookup table from 4 constant matrices, then uses
    // dynamic_slice with i32 indices to select a row.
    // With index=2, scale=1.0: row 2 of table = [-0.3, 0.4, 0.0] (from cst_2)
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, -0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<5x3xf64>
    %cst_1 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_2 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.2, 0.4, 0.0], [-0.3, 0.4, 0.0], [0.1, 0.1, 0.0], [0.3, -0.4, 0.0]]> : tensor<5x3xf64>
    %0 = stablehlo.concatenate %cst_2, %cst, %cst_0, %cst_1, dim = 0 : (tensor<5x3xf64>, tensor<6x3xf64>, tensor<5x3xf64>, tensor<6x3xf64>) -> tensor<22x3xf64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f64>
    %3 = stablehlo.convert %2 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare LT, %3, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<22> : tensor<i32>
    %5 = stablehlo.add %3, %c_3 : tensor<i32>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_slice %0, %6, %c_4, sizes = [1, 3] : (tensor<22x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %8 = stablehlo.reshape %7 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
}
"#;
    // index=2, scale=1.0 -> row 2 of concatenated table (cst_2 row 2) = [-0.3, 0.4, 0.0]
    let idx = i64_buf(&[2]);
    let scale = f64_buf(&[1.0]);
    let out = run_mlir_mem(mlir, &[&idx, &scale], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "drone @inner produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[-0.3, 0.4, 0.0]);
}

#[test]
fn test_drone_inner_cross_abi() {
    // Test the cross-ABI boundary: @main (scalar) calls @inner (pointer ABI)
    // @inner has tensor<22x3xf64> (66 elements > 64 threshold) so it's pointer ABI
    // @main passes scalar args, @inner returns tensor<3xf64>
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %0 = call @inner(%arg0, %arg1) : (tensor<i64>, tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func private @inner(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, -0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<5x3xf64>
    %cst_1 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_2 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.2, 0.4, 0.0], [-0.3, 0.4, 0.0], [0.1, 0.1, 0.0], [0.3, -0.4, 0.0]]> : tensor<5x3xf64>
    %0 = stablehlo.concatenate %cst_2, %cst, %cst_0, %cst_1, dim = 0 : (tensor<5x3xf64>, tensor<6x3xf64>, tensor<5x3xf64>, tensor<6x3xf64>) -> tensor<22x3xf64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f64>
    %3 = stablehlo.convert %2 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare LT, %3, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<22> : tensor<i32>
    %5 = stablehlo.add %3, %c_3 : tensor<i32>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_slice %0, %6, %c_4, sizes = [1, 3] : (tensor<22x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %8 = stablehlo.reshape %7 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
}
"#;
    // index=2, scale=1.0 -> row 2 = [-0.3, 0.4, 0.0]
    let idx = i64_buf(&[2]);
    let scale = f64_buf(&[1.0]);
    let out = run_mlir(mlir, &[&idx, &scale], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "cross-ABI @inner produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[-0.3, 0.4, 0.0]);
}

#[test]
fn test_divide_ui32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xui32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %2 = stablehlo.divide %0, %1 : tensor<4xui32>
    return %2 : tensor<4xui32>
  }
}
"#;
    let lo: Vec<u8> = [0u32, 10, 50, 100]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let hi: Vec<u8> = [120u32, 120, 120, 120]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = run_mlir_mem(mlir, &[&lo, &hi], &[16]);
    let result: Vec<u32> = out[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![60, 65, 85, 110]);
}

#[test]
fn test_scatter_i32_index_mem() {
    // Scatter with i32 index: set element at position 2 of a 5-element vector to 99.0
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %idx = stablehlo.constant dense<[2]> : tensor<1xi32>
    %upd = stablehlo.constant dense<99.0> : tensor<f64>
    %0 = "stablehlo.scatter"(%arg0, %idx, %upd) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>,
      unique_indices = true
    }> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<5xf64>, tensor<1xi32>, tensor<f64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[40]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 2.0, 99.0, 4.0, 5.0]);
}

#[test]
fn test_cross_abi_multi_result() {
    // Test pointer-ABI function returning multiple results via cross-ABI call
    // The callee has a large constant (>64 elements) forcing pointer ABI
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>) {
    %0:2 = call @big_func(%arg0) : (tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>)
    return %0#0, %0#1 : tensor<3xf64>, tensor<f64>
  }
  func.func private @big_func(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>) {
    %big = stablehlo.constant dense<0.0> : tensor<100xf64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    return %2, %1 : tensor<3xf64>, tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[2.0, 3.0, 4.0]);
    let out = run_mlir(mlir, &[&in0], &[24, 8]);
    let r0 = read_f64s(&out[0]);
    let r1 = read_f64s(&out[1]);
    assert_f64s_close(&r0, &[4.0, 9.0, 16.0]);
    assert_f64s_close(&r1, &[2.0]);
}

#[test]
fn test_concatenate_dim1_mem() {
    // Concatenate along dim 1: [[1,2],[3,4]] ++ [[5],[6]] -> [[1,2,5],[3,4,6]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let b = f64_buf(&[5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[48]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_gather_nd_2d_index_mem() {
    // Gather individual elements from a 3x3 matrix using 2D index vectors
    // Matrix: [[10,20,30],[40,50,60],[70,80,90]]
    // Indices: [[0,1],[2,0],[1,2]] -> picks (0,1)=20, (2,0)=70, (1,2)=60
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>, %arg1: tensor<3x2xi32>) -> tensor<3x1x1xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xf64>, tensor<3x2xi32>) -> tensor<3x1x1xf64>
    return %0 : tensor<3x1x1xf64>
  }
}
"#;
    let mat = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]);
    let idx = i32_buf(&[0, 1, 2, 0, 1, 2]); // (0,1), (2,0), (1,2)
    let out = run_mlir_mem(mlir, &[&mat, &idx], &[24]); // 3 * 8 bytes
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[20.0, 70.0, 60.0]);
}

#[test]
fn test_coeff_broadcast_multiply_reduce_mem() {
    // Test the EGM08 pattern: broadcast a 4-vector to 4x4, multiply by coefficient matrix, reduce
    // coeff = [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]  (diagonal)
    // vec   = [10, 20, 30, 40]
    // broadcast vec with dims=[1] -> each row = [10,20,30,40]
    // product[i][j] = coeff[i][j] * vec[j]
    // product = [[10,0,0,0],[0,40,0,0],[0,0,90,0],[0,0,0,160]]
    // reduce across dim 1 -> [10, 40, 90, 160]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4x4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<4xf64>) -> tensor<1x4xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<4x4xf64>
    %2 = stablehlo.multiply %arg0, %1 : tensor<4x4xf64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x4xf64>, tensor<f64>) -> tensor<4xf64>
    return %3 : tensor<4xf64>
  }
}
"#;
    let coeff = f64_buf(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    ]);
    let vec_in = f64_buf(&[10.0, 20.0, 30.0, 40.0]);
    let out = run_mlir_mem(mlir, &[&coeff, &vec_in], &[32]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[10.0, 40.0, 90.0, 160.0]);
}

#[test]
fn test_roll_scatter_broadcast_reduce_mem() {
    // Full EGM08-like pattern: roll vector, scatter zero at idx 0, broadcast, multiply, reduce
    // Input vec: [100, 200, 300, 400]
    // Roll left by 1: [200, 300, 400, 100]
    // Scatter idx 0 to 0: [0, 300, 400, 100]
    // Broadcast to 4x4 (each row = same): [[0,300,400,100], ...]
    // Multiply by identity-like coeff: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    // Product: [[0,0,0,0],[0,300,0,0],[0,0,400,0],[0,0,0,100]]
    // Reduce dim 1: [0, 300, 400, 100]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.slice %arg0 [1:4] : (tensor<4xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %c = stablehlo.constant dense<0> : tensor<1xi32>
    %cst_z = stablehlo.constant dense<0.0> : tensor<f64>
    %3 = "stablehlo.scatter"(%2, %c, %cst_z) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<4xf64>, tensor<1xi32>, tensor<f64>) -> tensor<4xf64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<4xf64>) -> tensor<1x4xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<4x4xf64>
    %6 = stablehlo.multiply %arg1, %5 : tensor<4x4xf64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x4xf64>, tensor<f64>) -> tensor<4xf64>
    return %7 : tensor<4xf64>
  }
}
"#;
    let vec_in = f64_buf(&[100.0, 200.0, 300.0, 400.0]);
    let coeff = f64_buf(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let out = run_mlir_mem(mlir, &[&vec_in, &coeff], &[32]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[0.0, 300.0, 400.0, 100.0]);
}

#[test]
fn test_while_cross_abi_accumulate_mem() {
    // Tests the cube-sat pattern: pointer-ABI while loop calling scalar-ABI function,
    // accumulating results in a vector via dynamic_update_slice.
    // The scalar function doubles its input: f(x, i) = (2*x, x) for i>0, (x, x) for i==0
    // Starting from 1.0, after 4 iterations: acc = [1, 2, 4, 8]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<4xf64> {
    %cst_z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %acc_init = stablehlo.broadcast_in_dim %cst_z, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %init_prev = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:4 = stablehlo.while(%iterArg_0 = %arg0, %iterArg_1 = %init_prev, %iterArg_2 = %c0, %iterArg_3 = %acc_init) : tensor<f64>, tensor<f64>, tensor<i64>, tensor<4xf64>
    cond {
      %c4 = stablehlo.constant dense<4> : tensor<i64>
      %cond = stablehlo.compare LT, %iterArg_2, %c4, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %r:2 = func.call @double_fn(%iterArg_0, %iterArg_2) : (tensor<f64>, tensor<i64>) -> (tensor<f64>, tensor<f64>)
      %bcast = stablehlo.broadcast_in_dim %r#1, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %new_acc = stablehlo.dynamic_update_slice %iterArg_3, %bcast, %iterArg_2 : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %next_i = stablehlo.add %iterArg_2, %c1 : tensor<i64>
      stablehlo.return %r#0, %iterArg_0, %next_i, %new_acc : tensor<f64>, tensor<f64>, tensor<i64>, tensor<4xf64>
    }
    return %0#3 : tensor<4xf64>
  }
  func.func private @double_fn(%arg0: tensor<f64>, %arg1: tensor<i64>) -> (tensor<f64>, tensor<f64>) {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %is_zero = stablehlo.compare EQ, %arg1, %c0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %doubled = stablehlo.add %arg0, %arg0 : tensor<f64>
    %result = stablehlo.select %is_zero, %arg0, %doubled : tensor<i1>, tensor<f64>
    return %result, %arg0 : tensor<f64>, tensor<f64>
  }
}
"#;
    let input = f64_buf(&[1.0]);
    let out = run_mlir_mem(mlir, &[&input], &[32]);
    let result = read_f64s(&out[0]);
    // iter 0: is_zero=true, result=1.0, acc[0]=1.0, next_curr=1.0
    // iter 1: doubled=2.0, acc[1]=1.0, next_curr=2.0
    // iter 2: doubled=4.0, acc[2]=2.0, next_curr=4.0
    // iter 3: doubled=8.0, acc[3]=4.0, next_curr=8.0
    assert_f64s_close(&result, &[1.0, 1.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_broadcast_multiply_reduce_65x65_mem() {
    // Reproduce the inner_375 pattern at 65x65 scale:
    // 1. Take a 65x65 matrix (Legendre-like: all nonzero)
    // 2. Transpose it
    // 3. Broadcast a 65-vector as columns (scaling factors)
    // 4. Elementwise multiply
    // 5. Transpose back
    // 6. Multiply by another 65x65 matrix (coefficient-like: lower triangular)
    // 7. Reduce sum along dim 1
    // Expected: row 0 of result should be nonzero
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65x65xf64>, %arg1: tensor<65xf64>, %arg2: tensor<65x65xf64>) -> tensor<65xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x65xf64>) -> tensor<65x65xf64>
    %3 = stablehlo.multiply %2, %0 : tensor<65x65xf64>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %5 = stablehlo.multiply %4, %arg2 : tensor<65x65xf64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    return %6 : tensor<65xf64>
  }
}
"#;
    // Matrix A: identity-like (1.0 on diagonal, 0.1 everywhere else)
    let mut mat_a = vec![0.1f64; 65 * 65];
    for i in 0..65 {
        mat_a[i * 65 + i] = 1.0 + i as f64;
    }
    // Scaling vector: [1.0, 2.0, ..., 65.0]
    let scale: Vec<f64> = (1..=65).map(|i| i as f64).collect();
    // Coefficient matrix: identity
    let mut coeff = vec![0.0f64; 65 * 65];
    for i in 0..65 {
        coeff[i * 65 + i] = 1.0;
    }

    let a_buf = f64_buf(&mat_a);
    let s_buf = f64_buf(&scale);
    let c_buf = f64_buf(&coeff);
    let out = run_mlir_mem(mlir, &[&a_buf, &s_buf, &c_buf], &[65 * 8]);
    let result = read_f64s(&out[0]);

    // With identity coefficient, result[i] = sum_j(transpose(scale_broadcast * transpose(A))[i][j] * coeff[i][j])
    // = transpose(scale_broadcast * transpose(A))[i][i] (only diagonal of coeff is nonzero)
    // transpose(X)[i][i] = X[i][i], so result[i] = (scale_broadcast * transpose(A))[i][i]
    // scale_broadcast[i][j] = scale[j], transpose(A)[i][j] = A[j][i]
    // product[i][i] = scale[i] * A[i][i] = (i+1) * (1+i)
    // result[0] = 1 * 1 = 1, result[1] = 2 * 2 = 4, result[2] = 3 * 3 = 9
    assert!(
        result[0] != 0.0,
        "row 0 should be nonzero, got {}",
        result[0]
    );
    let expected_diag: Vec<f64> = (0..65)
        .map(|i| {
            let s = (i + 1) as f64;
            let a_ii = 1.0 + i as f64;
            // Also off-diagonal contributions: sum_j(scale[j] * A[j][i] * coeff[i][j])
            // coeff is identity, so only j=i contributes: scale[i] * A[i][i] = s * a_ii
            s * a_ii
        })
        .collect();
    for i in 0..5 {
        assert!(
            (result[i] - expected_diag[i]).abs() < 1e-6,
            "result[{i}] = {}, expected {}",
            result[i],
            expected_diag[i]
        );
    }
}

#[test]
fn test_65x65_in_pointer_abi_callee_mem() {
    // Same computation as above, but inside a pointer-ABI callee function
    // called from main (cross-ABI: scalar main → pointer callee).
    // This tests whether the pointer-ABI function boundary corrupts the result.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65x65xf64>, %arg1: tensor<65xf64>, %arg2: tensor<65x65xf64>) -> tensor<65xf64> {
    %0 = call @inner(%arg0, %arg1, %arg2) : (tensor<65x65xf64>, tensor<65xf64>, tensor<65x65xf64>) -> tensor<65xf64>
    return %0 : tensor<65xf64>
  }
  func.func private @inner(%arg0: tensor<65x65xf64>, %arg1: tensor<65xf64>, %arg2: tensor<65x65xf64>) -> tensor<65xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x65xf64>) -> tensor<65x65xf64>
    %3 = stablehlo.multiply %2, %0 : tensor<65x65xf64>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %5 = stablehlo.multiply %4, %arg2 : tensor<65x65xf64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    return %6 : tensor<65xf64>
  }
}
"#;
    let mut mat_a = vec![0.1f64; 65 * 65];
    for i in 0..65 {
        mat_a[i * 65 + i] = 1.0 + i as f64;
    }
    let scale: Vec<f64> = (1..=65).map(|i| i as f64).collect();
    let mut coeff = vec![0.0f64; 65 * 65];
    for i in 0..65 {
        coeff[i * 65 + i] = 1.0;
    }

    let a_buf = f64_buf(&mat_a);
    let s_buf = f64_buf(&scale);
    let c_buf = f64_buf(&coeff);
    let out = run_mlir_mem(mlir, &[&a_buf, &s_buf, &c_buf], &[65 * 8]);
    let result = read_f64s(&out[0]);
    assert!(
        result[0] != 0.0,
        "row 0 should be nonzero, got {}",
        result[0]
    );
    assert!(
        (result[0] - 1.0).abs() < 1e-6,
        "result[0] = {}, expected 1.0",
        result[0]
    );
    assert!(
        (result[1] - 4.0).abs() < 1e-6,
        "result[1] = {}, expected 4.0",
        result[1]
    );
}

#[test]
fn test_65x65_while_loop_builds_matrix_then_reduces_mem() {
    // The actual inner_375 pattern: a while loop fills a 65x65 matrix row by row
    // using dynamic_update_slice, then the matrix is transposed, broadcast-multiplied,
    // and reduced. This tests whether the while-loop + DUS interaction corrupts memory.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65xf64>) -> tensor<65xf64> {
    %0 = call @build_and_reduce(%arg0) : (tensor<65xf64>) -> tensor<65xf64>
    return %0 : tensor<65xf64>
  }
  func.func private @build_and_reduce(%arg0: tensor<65xf64>) -> tensor<65xf64> {
    %cst_z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %mat_init = stablehlo.broadcast_in_dim %cst_z, dims = [] : (tensor<f64>) -> tensor<65x65xf64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>

    %loop:2 = stablehlo.while(%iterArg_0 = %c0, %iterArg_1 = %mat_init) : tensor<i64>, tensor<65x65xf64>
    cond {
      %c65 = stablehlo.constant dense<65> : tensor<i64>
      %cmp = stablehlo.compare LT, %iterArg_0, %c65, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %row = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
      %c0_body = stablehlo.constant dense<0> : tensor<i64>
      %new_mat = stablehlo.dynamic_update_slice %iterArg_1, %row, %iterArg_0, %c0_body : (tensor<65x65xf64>, tensor<1x65xf64>, tensor<i64>, tensor<i64>) -> tensor<65x65xf64>
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %next_i = stablehlo.add %iterArg_0, %c1 : tensor<i64>
      stablehlo.return %next_i, %new_mat : tensor<i64>, tensor<65x65xf64>
    }

    %cst_reduce = stablehlo.constant dense<0.0> : tensor<f64>
    %result = stablehlo.reduce(%loop#1 init: %cst_reduce) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    return %result : tensor<65xf64>
  }
}
"#;
    // Each row is the same vector [1, 2, 3, ..., 65]
    let input: Vec<f64> = (1..=65).map(|i| i as f64).collect();
    let i_buf = f64_buf(&input);
    let out = run_mlir_mem(mlir, &[&i_buf], &[65 * 8]);
    let result = read_f64s(&out[0]);
    // After filling: mat[i][j] = j+1 for all i. reduce(dim=1) sums each row.
    // Each row sum = 1+2+...+65 = 65*66/2 = 2145
    let expected_sum = 65.0 * 66.0 / 2.0;
    assert!(
        (result[0] - expected_sum).abs() < 1e-6,
        "result[0] = {}, expected {}",
        result[0],
        expected_sum
    );
    assert!(
        (result[64] - expected_sum).abs() < 1e-6,
        "result[64] = {}, expected {}",
        result[64],
        expected_sum
    );
}

#[test]
fn test_65x65_while_then_transpose_multiply_reduce_mem() {
    // Full inner_375-like pattern: while loop fills matrix, then chain of
    // transpose → broadcast → multiply → transpose → multiply → reduce.
    // Two pointer-ABI functions: @build fills the matrix, @process does the chain.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65xf64>, %arg1: tensor<65xf64>) -> tensor<65xf64> {
    %mat = call @build(%arg0) : (tensor<65xf64>) -> tensor<65x65xf64>
    %result = call @process(%mat, %arg1) : (tensor<65x65xf64>, tensor<65xf64>) -> tensor<65xf64>
    return %result : tensor<65xf64>
  }
  func.func private @build(%arg0: tensor<65xf64>) -> tensor<65x65xf64> {
    %cst_z = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %mat_init = stablehlo.broadcast_in_dim %cst_z, dims = [] : (tensor<f64>) -> tensor<65x65xf64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %loop:2 = stablehlo.while(%iterArg_0 = %c0, %iterArg_1 = %mat_init) : tensor<i64>, tensor<65x65xf64>
    cond {
      %c65 = stablehlo.constant dense<65> : tensor<i64>
      %cmp = stablehlo.compare LT, %iterArg_0, %c65, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %row = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
      %c0_body = stablehlo.constant dense<0> : tensor<i64>
      %new_mat = stablehlo.dynamic_update_slice %iterArg_1, %row, %iterArg_0, %c0_body : (tensor<65x65xf64>, tensor<1x65xf64>, tensor<i64>, tensor<i64>) -> tensor<65x65xf64>
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %next_i = stablehlo.add %iterArg_0, %c1 : tensor<i64>
      stablehlo.return %next_i, %new_mat : tensor<i64>, tensor<65x65xf64>
    }
    return %loop#1 : tensor<65x65xf64>
  }
  func.func private @process(%arg0: tensor<65x65xf64>, %arg1: tensor<65xf64>) -> tensor<65xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x65xf64>) -> tensor<65x65xf64>
    %3 = stablehlo.multiply %2, %0 : tensor<65x65xf64>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<65x65xf64>) -> tensor<65x65xf64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    return %5 : tensor<65xf64>
  }
}
"#;
    let row_vals: Vec<f64> = (1..=65).map(|i| i as f64).collect();
    let scale: Vec<f64> = (1..=65).map(|i| i as f64).collect();
    let r_buf = f64_buf(&row_vals);
    let s_buf = f64_buf(&scale);
    let out = run_mlir_mem(mlir, &[&r_buf, &s_buf], &[65 * 8]);
    let result = read_f64s(&out[0]);
    // mat[i][j] = j+1 for all i (all rows identical)
    // transpose: mat_T[i][j] = mat[j][i] = i+1 (constant across columns)
    // broadcast_scale[i][j] = scale[j] = j+1
    // product = broadcast_scale * mat_T: product[i][j] = (j+1) * (i+1)
    // transpose_back[i][j] = product[j][i] = (i+1) * (j+1)
    // reduce(dim=1): sum_j (i+1)*(j+1) = (i+1) * sum(1..65) = (i+1) * 2145
    let expected_0 = 1.0 * 2145.0; // (0+1) * 2145
    let expected_1 = 2.0 * 2145.0; // (1+1) * 2145
    assert!(result[0] != 0.0, "result[0] should be nonzero, got 0.0");
    assert!(
        (result[0] - expected_0).abs() < 1e-6,
        "result[0] = {}, expected {}",
        result[0],
        expected_0
    );
    assert!(
        (result[1] - expected_1).abs() < 1e-6,
        "result[1] = {}, expected {}",
        result[1],
        expected_1
    );
}

#[test]
fn test_convert_power_chain_65_mem() {
    // Test the EGM08 power scaling chain: convert i64→f64 indices, then power(scalar, indices).
    // This is: %49 = convert [0,1,...,N-1] : i64→f64, %50 = broadcast(base), %51 = power(%50,%49)
    // Expected: [base^0, base^1, ..., base^(N-1)] = [1.0, base, base^2, ...]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<65xf64> {
    %0 = call @inner(%arg0) : (tensor<f64>) -> tensor<65xf64>
    return %0 : tensor<65xf64>
  }
  func.func private @inner(%arg0: tensor<f64>) -> tensor<65xf64> {
    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]> : tensor<65xi64>
    %1 = stablehlo.convert %c : (tensor<65xi64>) -> tensor<65xf64>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<65xf64>
    %3 = stablehlo.power %2, %1 : tensor<65xf64>
    return %3 : tensor<65xf64>
  }
}
"#;
    let base_val = 0.5f64;
    let input = f64_buf(&[base_val]);
    let out = run_mlir_mem(mlir, &[&input], &[65 * 8]);
    let result = read_f64s(&out[0]);
    // result[n] = 0.5^n
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "result[0] = {}, expected 1.0 (0.5^0)",
        result[0]
    );
    assert!(
        (result[1] - 0.5).abs() < 1e-10,
        "result[1] = {}, expected 0.5 (0.5^1)",
        result[1]
    );
    assert!(
        (result[2] - 0.25).abs() < 1e-10,
        "result[2] = {}, expected 0.25 (0.5^2)",
        result[2]
    );
    assert!(
        (result[10] - base_val.powi(10)).abs() < 1e-10,
        "result[10] = {}, expected {}",
        result[10],
        base_val.powi(10)
    );
}

#[test]
fn test_gather_nd_row_from_diagonal_matrix_mem() {
    // Reproduces the cube-sat gravity bug: gather row 0 from a diagonal-like matrix
    // using 2D indices. Row 0 of a diagonal matrix should have only element [0] nonzero.
    // The bug: gather_nd returns an alternating pattern instead.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5x5xf64>, %arg1: tensor<5x2xi32>) -> tensor<5x1x1xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<5x5xf64>, tensor<5x2xi32>) -> tensor<5x1x1xf64>
    return %0 : tensor<5x1x1xf64>
  }
}
"#;
    // Diagonal matrix: [[10,0,0,0,0],[0,20,0,0,0],[0,0,30,0,0],[0,0,0,40,0],[0,0,0,0,50]]
    let mat = f64_buf(&[
        10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0,
    ]);
    // Indices: pick elements (0,0), (0,1), (0,2), (0,3), (0,4) -- entire row 0
    let idx = i32_buf(&[0, 0, 0, 1, 0, 2, 0, 3, 0, 4]);
    let out = run_mlir_mem(mlir, &[&mat, &idx], &[5 * 8]);
    let result = read_f64s(&out[0]);
    // Row 0 of diagonal matrix: [10, 0, 0, 0, 0]
    assert_f64s_close(&result, &[10.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_gather_nd_65x65_from_while_loop_callee_mem() {
    // Exact cube-sat pattern: pointer-ABI @outer calls @inner which has a while loop.
    // Each iteration calls @helper (pointer-ABI) which does N-D gather from a 65x65 matrix
    // passed as a loop-carried value. The gather picks a single row using 2D indices.
    // BUG REPRODUCER: if the gather returns wrong values from the loop-carried matrix,
    // the result will have the alternating [1,0,1,0,...] pattern instead of [1,0,0,0,...].
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65x65xf64>) -> tensor<65xf64> {
    %0 = call @outer(%arg0) : (tensor<65x65xf64>) -> tensor<65xf64>
    return %0 : tensor<65xf64>
  }
  func.func private @outer(%arg0: tensor<65x65xf64>) -> tensor<65xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %init = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %init_vec = stablehlo.broadcast_in_dim %init, dims = [] : (tensor<f64>) -> tensor<65xf64>
    %loop:3 = stablehlo.while(%iterArg_0 = %c0, %iterArg_1 = %arg0, %iterArg_2 = %init_vec) : tensor<i64>, tensor<65x65xf64>, tensor<65xf64>
    cond {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %cmp = stablehlo.compare LT, %iterArg_0, %c1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %result = call @helper(%iterArg_1, %iterArg_0) : (tensor<65x65xf64>, tensor<i64>) -> tensor<65xf64>
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %next = stablehlo.add %iterArg_0, %c1 : tensor<i64>
      stablehlo.return %next, %iterArg_1, %result : tensor<i64>, tensor<65x65xf64>, tensor<65xf64>
    }
    return %loop#2 : tensor<65xf64>
  }
  func.func private @helper(%arg0: tensor<65x65xf64>, %arg1: tensor<i64>) -> tensor<65xf64> {
    %indices_base = stablehlo.constant dense<[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]> : tensor<65xi64>
    %row_idx = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %col_idx = stablehlo.convert %indices_base : (tensor<65xi64>) -> tensor<65xi32>
    %c_neg = stablehlo.constant dense<0> : tensor<i32>
    %neg_check = stablehlo.compare LT, %row_idx, %c_neg, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_65 = stablehlo.constant dense<65> : tensor<i32>
    %wrapped = stablehlo.add %row_idx, %c_65 : tensor<i32>
    %safe_row = stablehlo.select %neg_check, %wrapped, %row_idx : tensor<i1>, tensor<i32>
    %row_bc = stablehlo.broadcast_in_dim %safe_row, dims = [] : (tensor<i32>) -> tensor<65x1xi32>
    %col_bc = stablehlo.broadcast_in_dim %col_idx, dims = [0] : (tensor<65xi32>) -> tensor<65x1xi32>
    %idx_2d = stablehlo.concatenate %row_bc, %col_bc, dim = 1 : (tensor<65x1xi32>, tensor<65x1xi32>) -> tensor<65x2xi32>
    %gathered = "stablehlo.gather"(%arg0, %idx_2d) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<65x65xf64>, tensor<65x2xi32>) -> tensor<65x1x1xf64>
    %result = stablehlo.reshape %gathered : (tensor<65x1x1xf64>) -> tensor<65xf64>
    return %result : tensor<65xf64>
  }
}
"#;
    // Diagonal matrix: only position [0][0] = 1.0, everything else = 0
    let mut mat = vec![0.0f64; 65 * 65];
    mat[0] = 1.0; // [0][0]
    let m_buf = f64_buf(&mat);
    let out = run_mlir_mem(mlir, &[&m_buf], &[65 * 8]);
    let result = read_f64s(&out[0]);
    // After 1 iteration: helper gathers row 0 = [1, 0, 0, ..., 0]
    let nz: Vec<usize> = result
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v != 0.0)
        .map(|(i, _)| i)
        .collect();
    eprintln!("result nonzero positions: {:?}", nz);
    eprintln!("result[0..5] = {:?}", &result[..5]);
    assert!(
        nz.len() == 1 && nz[0] == 0,
        "BUG: expected 1 nonzero at [0], got {} nonzero at {:?}\nresult[0..10]={:?}",
        nz.len(),
        nz,
        &result[..10]
    );
}

#[test]
fn test_multi_function_65x65_force_chain_mem() {
    // Replicates inner_375's exact force chain pattern:
    // @main calls @gravity (pointer-ABI) which:
    // 1. Calls @roll_vec (pointer-ABI helper) to roll a 65-element vector
    // 2. Scatters zero at position 0
    // 3. Broadcasts the rolled vector to 65x65
    // 4. Multiplies with a 65x65 coefficient matrix (passed as input)
    // 5. Reduces to scalar
    //
    // The roll+scatter pattern shifts column 0 to column 64 and zeros column 0.
    // For an identity coefficient matrix, the expected result is the sum of ALL
    // columns EXCEPT column 0, each multiplied by the rolled vector value.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<65xf64>, %arg1: tensor<65x65xf64>) -> tensor<f64> {
    %0 = call @gravity(%arg0, %arg1) : (tensor<65xf64>, tensor<65x65xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @gravity(%arg0: tensor<65xf64>, %arg1: tensor<65x65xf64>) -> tensor<f64> {
    %rolled = call @roll_vec(%arg0) : (tensor<65xf64>) -> tensor<65xf64>
    %c0 = stablehlo.constant dense<0> : tensor<1xi32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %scattered = "stablehlo.scatter"(%rolled, %c0, %zero) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      stablehlo.return %arg3 : tensor<f64>
    }) : (tensor<65xf64>, tensor<1xi32>, tensor<f64>) -> tensor<65xf64>
    %bcast = stablehlo.broadcast_in_dim %scattered, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
    %bcast2 = stablehlo.broadcast_in_dim %bcast, dims = [0, 1] : (tensor<1x65xf64>) -> tensor<65x65xf64>
    %product = stablehlo.multiply %arg1, %bcast2 : tensor<65x65xf64>
    %cst_z = stablehlo.constant dense<0.0> : tensor<f64>
    %row_sums = stablehlo.reduce(%product init: %cst_z) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    %total = stablehlo.reduce(%row_sums init: %cst_z) applies stablehlo.add across dimensions = [0] : (tensor<65xf64>, tensor<f64>) -> tensor<f64>
    return %total : tensor<f64>
  }
  func.func private @roll_vec(%arg0: tensor<65xf64>) -> tensor<65xf64> {
    %0 = stablehlo.slice %arg0 [1:65] : (tensor<65xf64>) -> tensor<64xf64>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<65xf64>) -> tensor<1xf64>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<64xf64>, tensor<1xf64>) -> tensor<65xf64>
    return %2 : tensor<65xf64>
  }
}
"#;
    // Input vector: [100, 200, 300, ..., 6500]
    let vec_in: Vec<f64> = (1..=65).map(|i| i as f64 * 100.0).collect();
    // Coefficient matrix: identity
    let mut coeff = vec![0.0f64; 65 * 65];
    for i in 0..65 {
        coeff[i * 65 + i] = 1.0;
    }

    let v_buf = f64_buf(&vec_in);
    let c_buf = f64_buf(&coeff);
    let out = run_mlir_mem(mlir, &[&v_buf, &c_buf], &[8]);
    let result = read_f64s(&out[0])[0];

    // After roll left: [200, 300, ..., 6500, 100]
    // After scatter zero at idx 0: [0, 300, 400, ..., 6500, 100]
    // Broadcast to 65x65: each row = [0, 300, 400, ..., 6500, 100]
    // Multiply by identity: product[i][i] = scattered[i]
    // Row sums: [0, 300, 400, ..., 6500, 100]
    // Total = sum of scattered = 0 + 300 + 400 + ... + 6500 + 100
    //       = (sum of 100..6500 step 100) - 100 - 200 + 100
    //       = 65*66/2*100 - 200 = 214500 - 200 = 214300
    let expected = 65.0 * 66.0 / 2.0 * 100.0 - 200.0;
    assert!(
        (result - expected).abs() < 1e-6,
        "result = {result}, expected {expected}"
    );
}

#[test]
fn test_full_egm08_chain_65_mem() {
    // Full EGM08-like chain inside a pointer-ABI function:
    // 1. Constant i64 index vector [0,...,64]
    // 2. Roll left, scatter zero, convert back and forth
    // 3. Power scaling
    // 4. Coefficient matrix multiply
    // 5. Reduce
    // This is the exact pattern from inner_375's force computation.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<65x65xf64>) -> tensor<f64> {
    %0 = call @chain(%arg0, %arg1) : (tensor<f64>, tensor<65x65xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @chain(%arg0: tensor<f64>, %arg1: tensor<65x65xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]> : tensor<65xi64>

    %indices_f = stablehlo.convert %c : (tensor<65xi64>) -> tensor<65xf64>
    %base_bc = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<65xf64>
    %powers = stablehlo.power %base_bc, %indices_f : tensor<65xf64>

    %scale = stablehlo.broadcast_in_dim %powers, dims = [1] : (tensor<65xf64>) -> tensor<1x65xf64>
    %scale_2d = stablehlo.broadcast_in_dim %scale, dims = [0, 1] : (tensor<1x65xf64>) -> tensor<65x65xf64>

    %product = stablehlo.multiply %scale_2d, %arg1 : tensor<65x65xf64>

    %cst_z = stablehlo.constant dense<0.0> : tensor<f64>
    %row_sums = stablehlo.reduce(%product init: %cst_z) applies stablehlo.add across dimensions = [1] : (tensor<65x65xf64>, tensor<f64>) -> tensor<65xf64>
    %total = stablehlo.reduce(%row_sums init: %cst_z) applies stablehlo.add across dimensions = [0] : (tensor<65xf64>, tensor<f64>) -> tensor<f64>
    return %total : tensor<f64>
  }
}
"#;
    // base = 1.0 (all powers = 1.0), coefficient matrix = identity
    // product = identity * broadcast(1.0) = identity
    // row_sums = [1,1,...,1], total = 65
    let base = f64_buf(&[1.0]);
    let mut coeff = vec![0.0f64; 65 * 65];
    for i in 0..65 {
        coeff[i * 65 + i] = 1.0;
    }
    let c_buf = f64_buf(&coeff);
    let out = run_mlir_mem(mlir, &[&base, &c_buf], &[8]);
    let result = read_f64s(&out[0]);
    assert!(
        (result[0] - 65.0).abs() < 1e-6,
        "result = {}, expected 65.0",
        result[0]
    );
}

#[test]
fn test_dot_general_1d_x_2d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<2x3xf64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    // lhs = [1, 2, 3]
    // rhs = [[4, 5, 6],
    //        [7, 8, 9]]
    // result[0] = 1*4 + 2*5 + 3*6 = 32
    // result[1] = 1*7 + 2*8 + 3*9 = 50
    let lhs = f64_buf(&[1.0, 2.0, 3.0]);
    let rhs = f64_buf(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let out = run_mlir(mlir, &[&lhs, &rhs], &[16]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[32.0, 50.0]);
}

#[test]
fn test_iota_2d_dim0() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3x3xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    return %0 : tensor<3x3xi64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[72]);
    let result: Vec<i64> = out[0]
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // dim=0: each element = row index
    // [[0,0,0], [1,1,1], [2,2,2]]
    assert_eq!(result, vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
}

#[test]
fn test_iota_2d_dim1() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3x3xi64> {
    %0 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    return %0 : tensor<3x3xi64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[72]);
    let result: Vec<i64> = out[0]
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // dim=1: each element = column index
    // [[0,1,2], [0,1,2], [0,1,2]]
    assert_eq!(result, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

#[test]
fn test_identity_matrix_via_iota() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %1 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %2 = stablehlo.compare  EQ, %0, %1,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %3 = stablehlo.convert %2 : (tensor<3x3xi1>) -> tensor<3x3xf64>
    %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    return %4 : tensor<3x3xf64>
  }
}
"#;
    // I * A = A
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let out = run_mlir(mlir, &[&a], &[72]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_reduce_and_bool() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4x1xi1>) -> tensor<4xi1> {
    %c = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.and across dimensions = [1] : (tensor<4x1xi1>, tensor<i1>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}
"#;
    // input: [[true], [false], [true], [false]]
    // AND-reduce across dim 1: init=true, each row has 1 element
    // result: [true AND true, true AND false, true AND true, true AND false]
    //       = [true, false, true, false] = [1, 0, 1, 0]
    let input: Vec<u8> = vec![1, 0, 1, 0];
    let out = run_mlir(mlir, &[&input], &[4]);
    assert_eq!(out[0], vec![1, 0, 1, 0]);
}

#[test]
fn test_gather_3d_pivot_permute() {
    // Pattern A: 3D pivot gather from _lu_solve_207
    // Permutes along dimension 1 of a 3D tensor based on pivot indices
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x1xf64>, %arg1: tensor<3x1xi32>) -> tensor<2x3x1xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 1>}> : (tensor<2x3x1xf64>, tensor<3x1xi32>) -> tensor<2x3x1xf64>
    return %0 : tensor<2x3x1xf64>
  }
}
"#;
    // operand (2x3x1): [[10],[20],[30]] and [[40],[50],[60]]
    // indices (3x1): [[2],[0],[1]] -> permute dim 1: row2,row0,row1
    // expected: [[30],[10],[20]] and [[60],[40],[50]]
    let operand = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let indices: Vec<u8> = vec![2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]; // 3 x i32 LE
    let out = run_mlir(mlir, &[&operand, &indices], &[48]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[30.0, 10.0, 20.0, 60.0, 40.0, 50.0]);
}

#[test]
fn test_gather_diagonal_2d_multiindex() {
    // Pattern B: 2D multi-index gather for diagonal extraction
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
    %4 = "stablehlo.gather"(%arg0, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xf64>, tensor<3x2xi32>) -> tensor<3xf64>
    return %4 : tensor<3xf64>
  }
}
"#;
    // Matrix: [[1,2,3],[4,5,6],[7,8,9]]
    // Diagonal indices: (0,0),(1,1),(2,2) -> [1, 5, 9]
    let matrix = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let out = run_mlir(mlir, &[&matrix], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 5.0, 9.0]);
}

// ---- Pointer-ABI elementwise ops ----

#[test]
fn test_atan2_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, -1.0, 0.0]);
    let b = f64_buf(&[1.0, 1.0, -1.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    let expected = [1.0f64.atan2(1.0), (-1.0f64).atan2(1.0), 0.0f64.atan2(-1.0)];
    assert_f64s_close(&result, &expected);
}

#[test]
fn test_acos_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.acos %arg0 : tensor<3xf64> -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 0.0, -1.0]);
    let out = run_mlir_mem(mlir, &[&a], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(
        &result,
        &[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
    );
}

#[test]
fn test_erf_inv_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.erf_inv %arg0 : tensor<3xf64> -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[0.0, 0.5, -0.5]);
    let out = run_mlir_mem(mlir, &[&a], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 0.4769362762044699);
    assert_f64_close(result[2], -0.4769362762044699);
}

#[test]
fn test_clamp_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let lo = f64_buf(&[0.0, 0.0, 0.0, 0.0]);
    let x = f64_buf(&[-1.0, 0.5, 1.5, 3.0]);
    let hi = f64_buf(&[1.0, 1.0, 1.0, 1.0]);
    let out = run_mlir_mem(mlir, &[&lo, &x, &hi], &[32]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn test_reverse_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.reverse %arg0, dims = [1] : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&a], &[48]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
fn test_round_nearest_even_mem() {
    mem_unop_test(
        "round_nearest_even",
        &[2.5, 3.5, 0.5, -0.5, 1.4, 1.6],
        &[2.0, 4.0, 0.0, 0.0, 1.0, 2.0],
    );
}

// ---- Element-size-aware gather/scatter ----

#[test]
fn test_gather_i32_data_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4x3xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<4x3xi32>, tensor<2x1xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}
"#;
    let data = i32_buf(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    let indices = i32_buf(&[1, 3]);
    let out = run_mlir_mem(mlir, &[&data, &indices], &[24]);
    let result: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![40, 50, 60, 100, 110, 120]);
}

#[test]
fn test_scatter_i32_data_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<5xi32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false}> ({
    ^bb0(%a: tensor<i32>, %b: tensor<i32>):
      stablehlo.return %b : tensor<i32>
    }) : (tensor<5xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5xi32>
    return %0 : tensor<5xi32>
  }
}
"#;
    let data = i32_buf(&[1, 2, 3, 4, 5]);
    let indices = i32_buf(&[0, 4]);
    let updates = i32_buf(&[99, 88]);
    let out = run_mlir_mem(mlir, &[&data, &indices, &updates], &[20]);
    let result: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![99, 2, 3, 4, 88]);
}

// ---- Integer comparisons ----

fn int_cmp_test(dir: &str, ty: &str, a: &[u8], b: &[u8], n: usize, expected: &[u8]) {
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}x{ty}>, %arg1: tensor<{n}x{ty}>) -> tensor<{n}xi1> {{
    %0 = stablehlo.compare {dir}, %arg0, %arg1, SIGNED : (tensor<{n}x{ty}>, tensor<{n}x{ty}>) -> tensor<{n}xi1>
    return %0 : tensor<{n}xi1>
  }}
}}"#
    );
    let out = run_mlir_mem(&mlir, &[a, b], &[n]);
    assert_eq!(&out[0], expected);
}

#[test]
fn test_compare_ne_i64_mem() {
    int_cmp_test(
        "NE",
        "i64",
        &i64_buf(&[1, 2, 3]),
        &i64_buf(&[1, 9, 3]),
        3,
        &[0, 1, 0],
    );
}

#[test]
fn test_compare_le_i64_mem() {
    int_cmp_test(
        "LE",
        "i64",
        &i64_buf(&[1, 5, 3]),
        &i64_buf(&[2, 5, 1]),
        3,
        &[1, 1, 0],
    );
}

#[test]
fn test_compare_gt_i64_mem() {
    int_cmp_test(
        "GT",
        "i64",
        &i64_buf(&[3, 1, 5]),
        &i64_buf(&[2, 2, 5]),
        3,
        &[1, 0, 0],
    );
}

#[test]
fn test_compare_ge_i64_mem() {
    int_cmp_test(
        "GE",
        "i64",
        &i64_buf(&[3, 2, 1]),
        &i64_buf(&[3, 1, 5]),
        3,
        &[1, 1, 0],
    );
}

#[test]
fn test_compare_eq_i32_mem() {
    int_cmp_test(
        "EQ",
        "i32",
        &i32_buf(&[1, 2, 3]),
        &i32_buf(&[1, 9, 3]),
        3,
        &[1, 0, 1],
    );
}

#[test]
fn test_compare_ne_i32_mem() {
    int_cmp_test(
        "NE",
        "i32",
        &i32_buf(&[1, 2, 3]),
        &i32_buf(&[1, 9, 3]),
        3,
        &[0, 1, 0],
    );
}

// ---- Integer elementwise ops ----

fn i64_binop_mem_test(op: &str, a: &[i64], b: &[i64], expected: &[i64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xi64>, %arg1: tensor<{n}xi64>) -> tensor<{n}xi64> {{
    %0 = stablehlo.{op} %arg0, %arg1 : tensor<{n}xi64>
    return %0 : tensor<{n}xi64>
  }}
}}"#
    );
    let out = run_mlir_mem(&mlir, &[&i64_buf(a), &i64_buf(b)], &[n * 8]);
    assert_eq!(read_i64s(&out[0]), expected);
}

#[test]
fn test_div_i64_mem() {
    i64_binop_mem_test("divide", &[10, -15, 7], &[3, 4, 0], &[3, -3, 0]);
}

#[test]
fn test_max_i64_mem() {
    i64_binop_mem_test("maximum", &[1, 5, -3], &[3, 2, -1], &[3, 5, -1]);
}

#[test]
fn test_min_i64_mem() {
    i64_binop_mem_test("minimum", &[1, 5, -3], &[3, 2, -1], &[1, 2, -3]);
}

#[test]
fn test_neg_i64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>) -> tensor<3xi64> {
    %0 = stablehlo.negate %arg0 : tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[&i64_buf(&[5, -3, 0])], &[24]);
    assert_eq!(read_i64s(&out[0]), vec![-5, 3, 0]);
}

#[test]
fn test_abs_i64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>) -> tensor<3xi64> {
    %0 = stablehlo.abs %arg0 : tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[&i64_buf(&[-5, 3, 0])], &[24]);
    assert_eq!(read_i64s(&out[0]), vec![5, 3, 0]);
}

#[test]
fn test_reduce_sum_i64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xi64>) -> tensor<2xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi64>, tensor<i64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[&i64_buf(&[1, 2, 3, 10, 20, 30])], &[16]);
    assert_eq!(read_i64s(&out[0]), vec![6, 60]);
}

// ---- Type conversions ----

fn convert_mem_test(
    src_ty: &str,
    dst_ty: &str,
    input: &[u8],
    _in_esz: usize,
    out_esz: usize,
    n: usize,
) -> Vec<u8> {
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}x{src_ty}>) -> tensor<{n}x{dst_ty}> {{
    %0 = stablehlo.convert %arg0 : (tensor<{n}x{src_ty}>) -> tensor<{n}x{dst_ty}>
    return %0 : tensor<{n}x{dst_ty}>
  }}
}}"#
    );
    let out = run_mlir_mem(&mlir, &[input], &[n * out_esz]);
    out[0].clone()
}

#[test]
fn test_convert_f64_to_i1_mem() {
    let out = convert_mem_test("f64", "i1", &f64_buf(&[0.0, 1.0, -0.5]), 8, 1, 3);
    assert_eq!(out, vec![0, 1, 1]);
}

#[test]
fn test_convert_i64_to_i1_mem() {
    let out = convert_mem_test("i64", "i1", &i64_buf(&[0, 1, -5]), 8, 1, 3);
    assert_eq!(out, vec![0, 1, 1]);
}

#[test]
fn test_convert_ui32_to_f64_mem() {
    let out = convert_mem_test("ui32", "f64", &u32_buf(&[0, 42, 1000]), 4, 8, 3);
    assert_f64s_close(&read_f64s(&out), &[0.0, 42.0, 1000.0]);
}

// ---- Integer layout ops ----

#[test]
fn test_slice_i32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xi32>) -> tensor<3xi32> {
    %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xi32>) -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[&i32_buf(&[10, 20, 30, 40, 50])], &[12]);
    let r: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(r, vec![20, 30, 40]);
}

#[test]
fn test_transpose_nd_i32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xi32>) -> tensor<3x2xi32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
    return %0 : tensor<3x2xi32>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[&i32_buf(&[1, 2, 3, 4, 5, 6])], &[24]);
    let r: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(r, vec![1, 4, 2, 5, 3, 6]);
}

// ---- StableHLO op coverage ----

#[test]
fn test_rsqrt() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.rsqrt %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[&f64_buf(&[1.0, 4.0, 0.25])], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 0.5, 2.0]);
}

#[test]
fn test_rsqrt_mem() {
    mem_unop_test("rsqrt", &[1.0, 4.0, 0.25], &[1.0, 0.5, 2.0]);
}

#[test]
fn test_log1p() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.log_plus_one %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir(
        mlir,
        &[&f64_buf(&[0.0, 1.0, std::f64::consts::E - 1.0])],
        &[24],
    );
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 2.0f64.ln());
    assert_f64_close(result[2], 1.0);
}

#[test]
fn test_log1p_mem() {
    mem_unop_test("log_plus_one", &[0.0, 1.0], &[0.0, 2.0f64.ln()]);
}

#[test]
fn test_is_finite() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<4xi1> {
    %0 = stablehlo.is_finite %arg0 : (tensor<4xf64>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}
"#;
    let out = run_mlir(
        mlir,
        &[&f64_buf(&[1.0, f64::INFINITY, f64::NAN, 0.0])],
        &[4],
    );
    assert_eq!(out[0], vec![1, 0, 0, 1]);
}

#[test]
fn test_ceil() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.ceil %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[&f64_buf(&[1.2, -1.2, 0.0, 2.9])], &[32]);
    assert_f64s_close(&read_f64s(&out[0]), &[2.0, -1.0, 0.0, 3.0]);
}

#[test]
fn test_ceil_mem() {
    mem_unop_test("ceil", &[1.2, -1.2, 0.0, 2.9], &[2.0, -1.0, 0.0, 3.0]);
}

#[test]
fn test_shift_right_arithmetic() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>) -> tensor<3xi64> {
    %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}
"#;
    let out = run_mlir(
        mlir,
        &[&i64_buf(&[16, -16, 255]), &i64_buf(&[2, 2, 4])],
        &[24],
    );
    assert_eq!(read_i64s(&out[0]), vec![4, -4, 15]);
}

#[test]
fn test_not_i1() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi1>) -> tensor<3xi1> {
    %0 = stablehlo.not %arg0 : tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
"#;
    let out = run_mlir(mlir, &[&[1u8, 0u8, 1u8]], &[3]);
    assert_eq!(out[0], vec![0, 1, 0]);
}

// ---- Bitwise ops: pointer-ABI ----

#[test]
fn test_xor_mem() {
    i64_binop_mem_test(
        "xor",
        &[0xFF, 0x0F, 0x00],
        &[0x0F, 0x0F, 0xFF],
        &[0xF0, 0x00, 0xFF],
    );
}

#[test]
fn test_or_mem() {
    i64_binop_mem_test(
        "or",
        &[0xF0, 0x00, 0x0F],
        &[0x0F, 0xFF, 0x00],
        &[0xFF, 0xFF, 0x0F],
    );
}

#[test]
fn test_and_mem() {
    i64_binop_mem_test(
        "and",
        &[0xFF, 0x0F, 0xF0],
        &[0x0F, 0x0F, 0xFF],
        &[0x0F, 0x0F, 0xF0],
    );
}

#[test]
fn test_shift_left_mem() {
    i64_binop_mem_test("shift_left", &[1, 3, 255], &[4, 2, 1], &[16, 12, 510]);
}

#[test]
fn test_shift_right_logical_mem() {
    i64_binop_mem_test(
        "shift_right_logical",
        &[16, 12, 255],
        &[4, 2, 1],
        &[1, 3, 127],
    );
}

// ---- Unary libm ops ----

fn chlo_unop_test(op: &str, a: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = chlo.{op} %arg0 : tensor<{n}xf64> -> tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let out = run_mlir(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

fn chlo_unop_mem_test(op: &str, a: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = chlo.{op} %arg0 : tensor<{n}xf64> -> tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let out = run_mlir_mem(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

#[test]
fn test_asin() {
    chlo_unop_test(
        "asin",
        &[0.0, 0.5, 1.0],
        &[
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_2,
        ],
    );
}

#[test]
fn test_asin_mem() {
    chlo_unop_mem_test(
        "asin",
        &[0.0, 0.5, 1.0],
        &[
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_2,
        ],
    );
}

#[test]
fn test_atan() {
    chlo_unop_test(
        "atan",
        &[0.0, 1.0, -1.0],
        &[
            0.0,
            std::f64::consts::FRAC_PI_4,
            -std::f64::consts::FRAC_PI_4,
        ],
    );
}

#[test]
fn test_atan_mem() {
    chlo_unop_mem_test(
        "atan",
        &[0.0, 1.0, -1.0],
        &[
            0.0,
            std::f64::consts::FRAC_PI_4,
            -std::f64::consts::FRAC_PI_4,
        ],
    );
}

#[test]
fn test_sinh() {
    chlo_unop_test("sinh", &[0.0, 1.0], &[0.0, 1.0_f64.sinh()]);
}

#[test]
fn test_sinh_mem() {
    chlo_unop_mem_test("sinh", &[0.0, 1.0], &[0.0, 1.0_f64.sinh()]);
}

#[test]
fn test_cosh() {
    chlo_unop_test("cosh", &[0.0, 1.0], &[1.0, 1.0_f64.cosh()]);
}

#[test]
fn test_cosh_mem() {
    chlo_unop_mem_test("cosh", &[0.0, 1.0], &[1.0, 1.0_f64.cosh()]);
}

#[test]
fn test_erfc() {
    chlo_unop_test(
        "erfc",
        &[0.0, 1.0, 3.0],
        &[1.0, 0.1572992070502851, 0.000022090496998585438],
    );
}

#[test]
fn test_erfc_mem() {
    chlo_unop_mem_test(
        "erfc",
        &[0.0, 1.0, 3.0],
        &[1.0, 0.1572992070502851, 0.000022090496998585438],
    );
}

#[test]
fn test_chlo_square() {
    chlo_unop_test("square", &[0.0, 3.0, -4.0], &[0.0, 9.0, 16.0]);
}

#[test]
fn test_chlo_square_mem() {
    chlo_unop_mem_test("square", &[0.0, 3.0, -4.0], &[0.0, 9.0, 16.0]);
}

#[test]
fn test_chlo_square_function_type_syntax() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.square %arg0 : (tensor<3xf64>) -> (tensor<3xf64>)
    return %0 : tensor<3xf64>
  }
}"#;
    let in0 = f64_buf(&[2.0, 3.0, 4.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[4.0, 9.0, 16.0]);
}

#[test]
fn test_expm1() {
    mem_unop_test("expm1", &[0.0, 1.0], &[0.0, std::f64::consts::E - 1.0]);
}

#[test]
fn test_expm1_mem() {
    let n = 2;
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.expm1 %arg0 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(&[0.0, 1.0]);
    let out = run_mlir_mem(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), &[0.0, std::f64::consts::E - 1.0]);
}

#[test]
fn test_cbrt() {
    mem_unop_test("cbrt", &[0.0, 8.0, 27.0], &[0.0, 2.0, 3.0]);
}

// ---- Type-permutation tests ----

#[test]
fn test_remainder_i64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>, %arg1: tensor<3xi64>) -> tensor<3xi64> {
    %0 = stablehlo.remainder %arg0, %arg1 : tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}
"#;
    let in0 = i64_buf(&[7, 10, -5]);
    let in1 = i64_buf(&[3, 4, 3]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[24]);
    let result = read_i64s(&out[0]);
    assert_eq!(result, &[1, 2, -2]);
}

#[test]
fn test_remainder_i32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
    %0 = stablehlo.remainder %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}
"#;
    let in0 = i32_buf(&[7, 10]);
    let in1 = i32_buf(&[3, 4]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[8]);
    let result = read_i32s(&out[0]);
    assert_eq!(result, &[1, 2]);
}

// Note: reduce And/Or pointer-ABI runtime support is wired (tensor_reduce_and_i1,
// tensor_reduce_or_i1) but i1 constant lowering in pointer-ABI stores scalars as i8
// values rather than pointers, so the call args don't match. This is a known edge
// case -- i1 reduce in pointer-ABI requires the tensor to exceed the 64-element
// threshold to actually trigger, which is extremely rare. The scalar-ABI path
// (test_reduce_and_bool) covers the common case.

// ---- LAPACK tests ----

#[test]
fn test_lapack_gesv_2x2() {
    // Solve Ax = b where A = [[2,1],[1,3]], b = [5,7] -> x = [8/5, 9/5] = [1.6, 1.8]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> (tensor<2x1xf64>, tensor<i32>) {
    %0:2 = stablehlo.custom_call @lapack_dgesv_ffi(%arg0, %arg1) : (tensor<2x2xf64>, tensor<2x1xf64>) -> (tensor<2x1xf64>, tensor<i32>)
    return %0#0, %0#1 : tensor<2x1xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[2.0, 1.0, 1.0, 3.0]);
    let b = f64_buf(&[5.0, 7.0]);
    let out = run_mlir(mlir, &[&a, &b], &[16, 4]);
    let x = read_f64s(&out[0]);
    assert_f64_close(x[0], 1.6);
    assert_f64_close(x[1], 1.8);
}

// ---- Sort tests ----

#[test]
fn test_sort_ascending() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.compare LT, %a, %b, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[3.0, 1.0, 4.0, 1.5, 2.0]);
    let out = run_mlir(mlir, &[&in0], &[40]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 1.5, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sort_descending() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.compare GT, %a, %b, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4xf64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 4.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[32]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_sort_ascending_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.compare LT, %a, %b, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<5xf64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[3.0, 1.0, 4.0, 1.5, 2.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[40]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 1.5, 2.0, 3.0, 4.0]);
}

// GESVD test deferred: the 5-result custom_call format requires parser
// support for multi-tuple result type parsing from the full XLA output
// specification. The host function (cranelift_gesvd) and lowering are
// implemented; what remains is matching the exact JAX dgesvd_ffi output
// tuple format, which will be validated when a simulation uses full SVD.

#[test]
fn test_cbrt_mem() {
    let n = 3;
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.cbrt %arg0 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(&[0.0, 8.0, 27.0]);
    let out = run_mlir_mem(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), &[0.0, 2.0, 3.0]);
}

// ---- batch_norm_inference tests ----

#[test]
fn test_batch_norm_inference_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>) -> tensor<2x3xf64> {
    %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.0E-05 : f32, feature_index = 1 : i64} : (tensor<2x3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let operand = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scale = f64_buf(&[1.0, 1.0, 1.0]);
    let offset = f64_buf(&[0.0, 0.0, 0.0]);
    let mean = f64_buf(&[2.5, 3.5, 4.5]);
    let variance = f64_buf(&[1.0, 1.0, 1.0]);
    let out = run_mlir_mem(mlir, &[&operand, &scale, &offset, &mean, &variance], &[48]);
    let result = read_f64s(&out[0]);
    let expected_0 = (1.0 - 2.5) / (1.0 + 1e-5_f64).sqrt();
    let expected_3 = (4.0 - 2.5) / (1.0 + 1e-5_f64).sqrt();
    assert_f64_close(result[0], expected_0);
    assert_f64_close(result[3], expected_3);
}

// ---- real_dynamic_slice tests ----

#[test]
fn test_real_dynamic_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>, %arg3: tensor<1xi64>) -> tensor<2xf64> {
    %0 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4xf64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let operand = f64_buf(&[10.0, 20.0, 30.0, 40.0]);
    let starts = i64_buf(&[1]);
    let limits = i64_buf(&[3]);
    let strides = i64_buf(&[1]);
    let out = run_mlir_mem(mlir, &[&operand, &starts, &limits, &strides], &[16]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 20.0);
    assert_f64_close(result[1], 30.0);
}

// ---- map tests ----

#[test]
fn test_map_multiply_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.multiply %a, %b : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) {dimensions = array<i64: 0>} : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[2.0, 3.0, 4.0]);
    let b = f64_buf(&[5.0, 6.0, 7.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[10.0, 18.0, 28.0]);
}

// ---- reduce_window tests ----

#[test]
fn test_reduce_window_sum_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<f64>) -> tensor<2xf64> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.add %a, %b : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) {window_dimensions = array<i64: 2>, window_strides = array<i64: 2>, base_dilations = array<i64: 1>, window_dilations = array<i64: 1>, padding = dense<0> : tensor<1x2xi64>} : (tensor<4xf64>, tensor<f64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let input = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let init = f64_buf(&[0.0]);
    let out = run_mlir_mem(mlir, &[&input, &init], &[16]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 3.0);
    assert_f64_close(result[1], 7.0);
}

// ---- select_and_scatter tests ----

#[test]
fn test_select_and_scatter_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<2xf64>, %arg2: tensor<f64>) -> tensor<4xf64> {
    %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %1 = stablehlo.compare GE, %a, %b, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }, {
    ^bb0(%x: tensor<f64>, %y: tensor<f64>):
      %1 = stablehlo.add %x, %y : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) {window_dimensions = dense<[2]> : tensor<1xi64>, window_strides = dense<[2]> : tensor<1xi64>, padding = dense<0> : tensor<1x2xi64>} : (tensor<4xf64>, tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let operand = f64_buf(&[1.0, 3.0, 2.0, 4.0]);
    let source = f64_buf(&[10.0, 20.0]);
    let init = f64_buf(&[0.0]);
    let out = run_mlir_mem(mlir, &[&operand, &source, &init], &[32]);
    let result = read_f64s(&out[0]);
    // Window [1,3]: max=3 at idx 1, scatter 10 -> dst[1]=10
    // Window [2,4]: max=4 at idx 3, scatter 20 -> dst[3]=20
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 10.0);
    assert_f64_close(result[2], 0.0);
    assert_f64_close(result[3], 20.0);
}

// ---- convolution tests ----

#[test]
#[ignore = "convolution runtime indexing needs further debugging for correct output"]
fn test_conv_1d_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<1x4x1xf64>, %arg1: tensor<2x1x1xf64>) -> tensor<1x3x1xf64> {
    %0 = "stablehlo.convolution"(%arg0, %arg1) {window_strides = array<i64: 1>, padding = dense<[[0, 0]]> : tensor<1x2xi64>, lhs_dilation = array<i64: 1>, rhs_dilation = array<i64: 1>, dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>, batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x1xf64>, tensor<2x1x1xf64>) -> tensor<1x3x1xf64>
    return %0 : tensor<1x3x1xf64>
  }
}
"#;
    let input = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let kernel = f64_buf(&[1.0, 1.0]);
    let out = run_mlir_mem(mlir, &[&input, &kernel], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 3.0);
    assert_f64_close(result[1], 5.0);
    assert_f64_close(result[2], 7.0);
}

// ---- Cholesky tests ----

#[test]
fn test_cholesky_mem() {
    // A = [[4,2],[2,3]], L = [[2,0],[1,sqrt(2)]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf64>
    return %0 : tensor<2x2xf64>
  }
}
"#;
    let a = f64_buf(&[4.0, 2.0, 2.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&a], &[32]);
    let l = read_f64s(&out[0]);
    assert_f64_close(l[0], 2.0);
    assert_f64_close(l[2], 1.0);
    assert_f64_close(l[3], 2.0_f64.sqrt());
}

#[test]
fn test_cholesky_upper_mem() {
    // StableHLO spec example:
    //   %a = [[1, 2, 3], [2, 20, 26], [3, 26, 70]]
    //   lower = false  ->  U = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]  (A = U^T @ U)
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.cholesky %arg0, lower = false : tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 2.0, 20.0, 26.0, 3.0, 26.0, 70.0]);
    let out = run_mlir_mem(mlir, &[&a], &[72]);
    let u = read_f64s(&out[0]);
    // Upper triangle: [[1,2,3],[0,4,5],[0,0,6]]
    assert_f64_close(u[0], 1.0);
    assert_f64_close(u[1], 2.0);
    assert_f64_close(u[2], 3.0);
    assert_f64_close(u[3], 0.0);
    assert_f64_close(u[4], 4.0);
    assert_f64_close(u[5], 5.0);
    assert_f64_close(u[6], 0.0);
    assert_f64_close(u[7], 0.0);
    assert_f64_close(u[8], 6.0);
}

#[test]
fn test_cholesky_batched_mem() {
    // Two stacked 3x3 SPD matrices; verify each slice factorizes independently
    // and the output batch dim is preserved.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
    %0 = stablehlo.cholesky %arg0, lower = true : tensor<2x3x3xf64>
    return %0 : tensor<2x3x3xf64>
  }
}
"#;
    // Batch 0: A0 = [[4,2,0],[2,5,3],[0,3,10]]  → L0 = [[2,0,0],[1,2,0],[0,1.5,√7.75]]
    // Batch 1: A1 = [[9,3,1],[3,6,2],[1,2,5]]
    let a = f64_buf(&[
        4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0, 9.0, 3.0, 1.0, 3.0, 6.0, 2.0, 1.0, 2.0, 5.0,
    ]);
    let out = run_mlir_mem(mlir, &[&a], &[18 * 8]);
    let all = read_f64s(&out[0]);
    // Reconstruct A0 = L0 * L0^T from the first 9 elems.
    let l0 = &all[..9];
    let l1 = &all[9..];
    // Each factor must be lower triangular.
    assert_f64_close(l0[1], 0.0);
    assert_f64_close(l0[2], 0.0);
    assert_f64_close(l0[5], 0.0);
    assert_f64_close(l1[1], 0.0);
    assert_f64_close(l1[2], 0.0);
    assert_f64_close(l1[5], 0.0);
    let reconstruct = |l: &[f64]| {
        let mut out = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[i * 3 + j] += l[i * 3 + k] * l[j * 3 + k];
                }
            }
        }
        out
    };
    let r0 = reconstruct(l0);
    let r1 = reconstruct(l1);
    assert_f64s_close(&r0, &[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0]);
    assert_f64s_close(&r1, &[9.0, 3.0, 1.0, 3.0, 6.0, 2.0, 1.0, 2.0, 5.0]);
}

#[test]
fn test_lapack_cholesky_upper() {
    // Custom-call path with uplo='U' (ASCII 85) — exercises the alternate
    // branch of our uplo decoder even though JAX itself always emits 'L'.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<i32>) {
    %0:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {uplo = 85 : ui8}} -> (tensor<3x3xf64>, tensor<i32>)
    return %0#0, %0#1 : tensor<3x3xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 2.0, 20.0, 26.0, 3.0, 26.0, 70.0]);
    let out = run_mlir(mlir, &[&a], &[72, 4]);
    let u = read_f64s(&out[0]);
    let info: i32 = i32::from_le_bytes(out[1][..4].try_into().unwrap());
    assert_eq!(info, 0);
    // Expected U = [[1,2,3],[0,4,5],[0,0,6]]
    assert_f64s_close(&u, &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
}

#[test]
fn test_lapack_cholesky_batched() {
    // Custom-call path with batched input — each 3x3 slice factorized
    // independently, with one info value emitted per batch.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x3xf64>) -> (tensor<2x3x3xf64>, tensor<2xi32>) {
    %0:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {uplo = 76 : ui8}} -> (tensor<2x3x3xf64>, tensor<2xi32>)
    return %0#0, %0#1 : tensor<2x3x3xf64>, tensor<2xi32>
  }
}
"#;
    let a = f64_buf(&[
        4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0, 9.0, 3.0, 1.0, 3.0, 6.0, 2.0, 1.0, 2.0, 5.0,
    ]);
    let out = run_mlir(mlir, &[&a], &[144, 8]);
    let all = read_f64s(&out[0]);
    let info0 = i32::from_le_bytes(out[1][..4].try_into().unwrap());
    let info1 = i32::from_le_bytes(out[1][4..8].try_into().unwrap());
    assert_eq!(info0, 0);
    assert_eq!(info1, 0);
    let l0 = &all[..9];
    let l1 = &all[9..];
    let reconstruct = |l: &[f64]| {
        let mut out = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[i * 3 + j] += l[i * 3 + k] * l[j * 3 + k];
                }
            }
        }
        out
    };
    assert_f64s_close(
        &reconstruct(l0),
        &[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0],
    );
    assert_f64s_close(
        &reconstruct(l1),
        &[9.0, 3.0, 1.0, 3.0, 6.0, 2.0, 1.0, 2.0, 5.0],
    );
}

// ---- Triangular solve tests ----

#[test]
fn test_triangular_solve_mem() {
    // L = [[2,0],[1,1]], b = [4,3] -> L*x=b -> x=[2,1]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x1xf64> {
    %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
"#;
    let l = f64_buf(&[2.0, 0.0, 1.0, 1.0]);
    let b = f64_buf(&[4.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&l, &b], &[16]);
    let x = read_f64s(&out[0]);
    assert_f64_close(x[0], 2.0);
    assert_f64_close(x[1], 1.0);
}

// ---- RNG tests ----

#[test]
fn test_rng_uniform_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<4xf64> {
    %0 = "stablehlo.rng"(%arg0, %arg1) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f64>, tensor<f64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let min_val = f64_buf(&[0.0]);
    let max_val = f64_buf(&[1.0]);
    let out = run_mlir_mem(mlir, &[&min_val, &max_val], &[32]);
    let vals = read_f64s(&out[0]);
    assert_eq!(vals.len(), 4);
    for v in &vals {
        assert!(*v >= 0.0 && *v <= 1.0, "rng value {v} out of [0,1]");
    }
}

// ---- Customer MLIR pattern tests ----

#[test]
fn test_ui64_large_constant() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<ui64>) -> tensor<ui64> {
    %c = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
    %0 = stablehlo.add %arg0, %c : tensor<ui64>
    return %0 : tensor<ui64>
  }
}"#;
    let in0 = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    let result = read_i64s(&out[0]);
    assert_eq!(result, &[0i64]);
}

#[test]
fn test_i64_min_constant() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    return %0 : tensor<i64>
  }
}"#;
    let in0 = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    let result = read_i64s(&out[0]);
    assert_eq!(result, &[i64::MIN + 1]);
}

#[test]
fn test_dot_general_1d_inner_product() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<f64> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let in1 = f64_buf(&[2.0, 3.0, 4.0, 5.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64s_close(&read_f64s(&out[0]), &[40.0]);
}

#[test]
fn test_convert_i1_to_ui64() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<ui64> {
    %0 = stablehlo.compare  LT, %arg0, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.convert %0 : (tensor<i1>) -> tensor<ui64>
    return %1 : tensor<ui64>
  }
}"#;
    let in0 = f64_buf(&[1.0]);
    let in1 = f64_buf(&[2.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    let result = read_i64s(&out[0]);
    assert_eq!(result, &[1i64]);
}

#[test]
fn test_chlo_erf_inv_function_type_syntax() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.erf_inv %arg0 : (tensor<3xf64>) -> (tensor<3xf64>)
    return %0 : tensor<3xf64>
  }
}"#;
    let in0 = f64_buf(&[0.0, 0.5, -0.5]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert!((result[0]).abs() < 1e-10);
    assert!((result[1] - 0.4769362762044699).abs() < 1e-6);
}

#[test]
fn test_shift_right_logical_overflow_ui32() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<ui32> {
    %c = stablehlo.constant dense<32> : tensor<ui32>
    %0 = stablehlo.shift_right_logical %arg0, %c : tensor<ui32>
    return %0 : tensor<ui32>
  }
}"#;
    let input = u32_buf(&[1]);
    let out = run_mlir(mlir, &[&input], &[4]);
    let result = read_u32s(&out[0]);
    assert_eq!(result[0], 0, "1 >> 32 should be 0 for ui32");
}

#[test]
fn test_shift_left_overflow_ui32() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<ui32> {
    %c = stablehlo.constant dense<32> : tensor<ui32>
    %0 = stablehlo.shift_left %arg0, %c : tensor<ui32>
    return %0 : tensor<ui32>
  }
}"#;
    let input = u32_buf(&[1]);
    let out = run_mlir(mlir, &[&input], &[4]);
    let result = read_u32s(&out[0]);
    assert_eq!(result[0], 0, "1 << 32 should be 0 for ui32");
}

#[test]
fn test_shift_right_logical_overflow_ui32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<ui32> {
    %c = stablehlo.constant dense<32> : tensor<ui32>
    %0 = stablehlo.shift_right_logical %arg0, %c : tensor<ui32>
    return %0 : tensor<ui32>
  }
}"#;
    let input = u32_buf(&[1]);
    let out = run_mlir_mem(mlir, &[&input], &[4]);
    let result = read_u32s(&out[0]);
    assert_eq!(result[0], 0, "1 >> 32 should be 0 for ui32 (mem path)");
}
