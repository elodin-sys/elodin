// Element-wise addition of two f64 tensors (critical for physics simulations).
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu --iree-vm-target-extension-f64 --iree-input-demote-f64-to-f32=false simple_add_f64.mlir -o <arch>/simple_add_f64.vmfb
// Omitting target triple compiles for the host architecture; see compile_fixtures.sh.
func.func @simple_add_f64(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf64>
  return %0 : tensor<4xf64>
}
