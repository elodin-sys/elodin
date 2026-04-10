// Reproducer for stablehlo.compare shape mismatch after inlining.
//
// The inner function has valid shapes (all scalar). After the InlinerPass
// inlines it into @main, the substituted value %sliced is tensor<1xi64>
// (not reshaped to scalar), producing tensor<1xf64> after convert. The
// inlined compare then has (tensor<1xf64>, tensor<f64>) operands.
//
// Discovered via customer sim_FT19.py: uint64 literals inside @el.map
// bodies trigger JAX type promotion (uint64+int64 -> float64).
module @module {
  func.func private @inner(%a: tensor<i64>, %b: tensor<ui64>) -> tensor<f64> {
    %af = stablehlo.convert %a : (tensor<i64>) -> tensor<f64>
    %bf = stablehlo.convert %b : (tensor<ui64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %cmp = stablehlo.compare EQ, %af, %cst, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %result = stablehlo.select %cmp, %bf, %af : tensor<i1>, tensor<f64>
    return %result : tensor<f64>
  }

  func.func public @main(%arg0: tensor<4xi64>) -> tensor<f64> {
    %sliced = stablehlo.slice %arg0 [0:1] : (tensor<4xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %sliced : (tensor<1xi64>) -> tensor<i64>
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %result = call @inner(%scalar, %c) : (tensor<i64>, tensor<ui64>) -> tensor<f64>
    return %result : tensor<f64>
  }
}
