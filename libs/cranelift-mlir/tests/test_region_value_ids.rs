//! `ValueId`s must stay unique within a function across nested regions.
//!
//! Each region gets its own SSA name scope, but const folding, DCE and
//! lowering key on `ValueId` across nested bodies. When every region
//! restarted id numbering from its parent's counter, its ids were reused by
//! sibling regions and by the values defined after it, so folding JAX's
//! `%4 = stablehlo.convert %3 : tensor<f64>` into `%3` also rewrote a use
//! inside the `lax.cond` true branch, which then returned `-inf`.

mod common;
use common::*;

/// `max(x) > 0`: `lax.cond` takes region 1 (the true branch).
const X_POS: [f64; 4] = [0.3, -0.7, 0.2, 0.9];
/// `max(x) <= 0`: `lax.cond` takes region 0.
const X_NEG: [f64; 4] = [-0.3, -0.7, -0.2, -0.9];
/// `%arg1`, a scalar the regions read from the enclosing scope.
const A: f64 = 1.5;
/// Scalar ABI, then pointer ABI.
const BOTH_ABIS: &[bool] = &[false, true];

/// Runs `mlir` on each listed ABI (`true` = pointer ABI), unfolded and
/// folded as `nox-py` compiles it, and compares the f64 output.
fn check(mlir: &str, pointer_abi: &[bool], inputs: &[Vec<u8>], expected: &[f64]) {
    let inputs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    let sizes = [expected.len() * 8];
    for &mem in pointer_abi {
        for fold in [false, true] {
            let out = match (mem, fold) {
                (false, false) => run_mlir(mlir, &inputs, &sizes),
                (true, false) => run_mlir_mem(mlir, &inputs, &sizes),
                (_, true) => run_mlir_folded(mlir, &inputs, &sizes, mem),
            };
            let got = read_f64s(&out[0]);
            let close = |(g, e): (&f64, &f64)| (g - e).abs() <= 1e-12 * e.abs().max(1.0);
            let ok = got.iter().zip(expected).all(close);
            assert!(ok, "mem={mem} fold={fold}: {got:?} != {expected:?}");
        }
    }
}

/// `x * case(%2, regions)` as JAX lowers `lax.cond` / `lax.switch`, where
/// `prelude` defines the index `%2`. JAX adds the same-type `convert` when
/// region 0 returns a weakly typed constant.
fn case_mlir(prelude: &str, regions: &[&str]) -> String {
    let regions: Vec<String> = regions.iter().map(|r| format!("{{\n{r}\n    }}")).collect();
    let regions = regions.join(", ");
    format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<4xf64>, %arg1: tensor<f64>, %arg2: tensor<i32>) -> tensor<4xf64> {{
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
{prelude}
    %3 = "stablehlo.case"(%2) ({regions}) : (tensor<i32>) -> tensor<f64>
    %4 = stablehlo.convert %3 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %6 = stablehlo.multiply %arg0, %5 : tensor<4xf64>
    return %6 : tensor<4xf64>
  }}
}}"#
    )
}

/// Checks that `case_mlir` returns `x * k` for index argument `i`.
fn check_case(mlir: &str, abis: &[bool], x: [f64; 4], i: i32, k: f64) {
    let inputs = [f64_buf(&x), f64_buf(&[A]), i32_buf(&[i])];
    check(mlir, abis, &inputs, &x.map(|v| v * k));
}

const COND: &str = "    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.compare GT, %0, %cst_0, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %2 = stablehlo.convert %1 : (tensor<i1>) -> tensor<i32>";
const ONE: &str = "      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      stablehlo.return %cst_1 : tensor<f64>";
/// `lambda _: jnp.max(x)`, re-traced inside the branch.
const MAX: &str = "      %cst_1 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
      %7 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
      stablehlo.return %7 : tensor<f64>";

fn max(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// `lambda _: k * s` for a captured scalar `s`.
fn scale(s: &str, k: &str) -> String {
    format!(
        "      %cst_1 = stablehlo.constant dense<{k}> : tensor<f64>
      %7 = stablehlo.multiply {s}, %cst_1 : tensor<f64>
      stablehlo.return %7 : tensor<f64>"
    )
}

/// JAX's graph for `x * lax.cond(max(x) > 0, lambda _: max(x), lambda _: 1.0, None)`.
#[test]
fn cond_region1_reduce() {
    let mlir = case_mlir(COND, &[ONE, MAX]);
    check_case(&mlir, BOTH_ABIS, X_POS, 0, max(&X_POS));
    check_case(&mlir, BOTH_ABIS, X_NEG, 0, 1.0);
}

#[test]
fn cond_region0_reduce() {
    let mlir = case_mlir(COND, &[MAX, ONE]);
    check_case(&mlir, BOTH_ABIS, X_POS, 0, 1.0);
    check_case(&mlir, BOTH_ABIS, X_NEG, 0, max(&X_NEG));
}

/// The true branch scales an outer scalar made by a reduce, an argument,
/// arithmetic or a dot product.
#[test]
fn cond_region1_captures_scalar() {
    let prelude = format!(
        "{COND}
    %mul = stablehlo.multiply %arg1, %arg1 : tensor<f64>
    %dot = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>"
    );
    let (m, dot) = (max(&X_POS), X_POS.iter().map(|v| v * v).sum());
    for (s, v) in [("%0", m), ("%arg1", A), ("%mul", A * A), ("%dot", dot)] {
        let mlir = case_mlir(&prelude, &[ONE, &scale(s, "2.0")]);
        check_case(&mlir, BOTH_ABIS, X_POS, 0, 2.0 * v);
    }
}

/// `lax.switch(i, [lambda: 1.0, lambda: 2 * max(x), lambda: 3 * a])` for an
/// in-range `i`. JAX's index `clamp` is left out: the pointer-ABI `clamp`
/// only handles f64.
#[test]
fn switch_three_regions() {
    let prelude = "    %2 = stablehlo.convert %arg2 : tensor<i32>";
    let mlir = case_mlir(prelude, &[ONE, &scale("%0", "2.0"), &scale("%arg1", "3.0")]);
    for (i, k) in [(0, 1.0), (1, 2.0 * max(&X_POS)), (2, 3.0 * A)] {
        check_case(&mlir, BOTH_ABIS, X_POS, i, k);
    }
}

/// `1` if `max(x) <= 0`, else `2 * max(x)` if `max(x) <= 0.5`, else `3 * max(x)`.
#[test]
fn nested_case() {
    let inner = "      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
      %7 = stablehlo.compare GT, %0, %cst_1, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %8 = stablehlo.convert %7 : (tensor<i1>) -> tensor<i32>
      %9 = \"stablehlo.case\"(%8) ({
        %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
        %10 = stablehlo.multiply %0, %cst_2 : tensor<f64>
        stablehlo.return %10 : tensor<f64>
      }, {
        %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
        %10 = stablehlo.multiply %0, %cst_2 : tensor<f64>
        stablehlo.return %10 : tensor<f64>
      }) : (tensor<i32>) -> tensor<f64>
      %11 = stablehlo.convert %9 : tensor<f64>
      stablehlo.return %11 : tensor<f64>";
    let mlir = case_mlir(COND, &[ONE, inner]);
    let x_mid = [0.3, -0.7, 0.2, 0.4];
    check_case(&mlir, BOTH_ABIS, X_POS, 0, 3.0 * max(&X_POS));
    check_case(&mlir, BOTH_ABIS, x_mid, 0, 2.0 * max(&x_mid));
    check_case(&mlir, BOTH_ABIS, X_NEG, 0, 1.0);
}

/// A branch over `CASE_BRANCH_SPLIT_INSTRS`, which the pointer-ABI path
/// compiles as its own function (`compile_case_branch_as_function_mem`)
/// taking the captured `%0` and `%arg1` as arguments.
#[test]
fn case_split_branch_captures() {
    let mut big = String::from("%b0 = stablehlo.add %0, %arg1 : tensor<f64>\n");
    for i in 0..69 {
        let n = i + 1;
        big += &format!("%b{n} = stablehlo.add %b{i}, %arg1 : tensor<f64>\n");
    }
    big += "stablehlo.return %b69 : tensor<f64>";
    let mlir = case_mlir(COND, &[ONE, &big]);
    let k = (0..70).fold(max(&X_POS), |b, _| b + A);
    check_case(&mlir, &[true], X_POS, 0, k);
    check_case(&mlir, &[true], X_NEG, 0, 1.0);
}

/// `fori_loop(0, 3, lambda i, acc: acc * s, x)` closing over `s = a * a`,
/// which JAX carries as an iteration argument. The loop results and the
/// `convert` after the loop used to reuse the cond and body ids.
#[test]
fn while_carries_outer_scalar() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %s = stablehlo.multiply %arg1, %arg1 : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %s, %iterArg_0 = %c, %iterArg_1 = %arg0) : tensor<f64>, tensor<i64>, tensor<f64>
    cond {
      %c_2 = stablehlo.constant dense<3> : tensor<i64>
      %3 = stablehlo.compare LT, %iterArg_0, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.multiply %iterArg_1, %iterArg : tensor<f64>
      %c_2 = stablehlo.constant dense<1> : tensor<i64>
      %4 = stablehlo.add %iterArg_0, %c_2 : tensor<i64>
      stablehlo.return %iterArg, %4, %3 : tensor<f64>, tensor<i64>, tensor<f64>
    }
    %1 = stablehlo.convert %0#2 : tensor<f64>
    return %1 : tensor<f64>
  }
}"#;
    let inputs = [f64_buf(&[0.5]), f64_buf(&[A])];
    check(
        mlir,
        BOTH_ABIS,
        &inputs,
        &[0.5 * (A * A) * (A * A) * (A * A)],
    );
}

/// The `map` block arguments used to share ids with `%0` and `%1`.
#[test]
fn map_block_args() {
    let mlir = r#"module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %2 = stablehlo.multiply %a, %b : tensor<f64>
      stablehlo.return %2 : tensor<f64>
    }) {dimensions = array<i64: 0>} : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1 = stablehlo.convert %0 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}"#;
    let inputs = [f64_buf(&[2.0, 3.0, 4.0]), f64_buf(&[5.0, 6.0, 7.0])];
    check(mlir, &[true], &inputs, &[10.0, 18.0, 28.0]);
}
