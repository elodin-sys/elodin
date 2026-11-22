module @module {
  func.func public @main(%arg0: tensor<ui64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>, %arg3: tensor<f64>) -> (tensor<ui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>) {
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %0 = stablehlo.add %arg0, %c : tensor<ui64>
    %1 = stablehlo.subtract %arg2, %arg1 : tensor<2xf64>
    %dt = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %2 = stablehlo.multiply %1, %dt : tensor<2xf64>
    %3 = stablehlo.add %arg1, %2 : tensor<2xf64>
    return %0, %3, %arg2, %arg3 : tensor<ui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>
  }
}
