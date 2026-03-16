// Identity function for i64 tensors (pass-through).
// Tests integer type round-trip through the runtime.
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu identity_i64.mlir -o <arch>/identity_i64.vmfb
// Omitting target triple compiles for the host architecture; see compile_fixtures.sh.
func.func @identity_i64(%a: tensor<4xi64>) -> tensor<4xi64> {
  return %a : tensor<4xi64>
}
