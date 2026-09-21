//! Safe typed executor for UI display kernels compiled from StableHLO.

use crate::const_fold;
use crate::lower::{CompiledModule, compile_module};
use crate::parser;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

/// Compiled StableHLO module with a checked pointer ABI.
pub struct DisplayKernelExec {
    _compiled: CompiledModule,
    tick_fn: TickFn,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
    input_ptrs: Vec<*const u8>,
    output_ptrs: Vec<*mut u8>,
}

// Safety: the JIT module is immutable after `compile_module` returns, and
// pointer scratch is only rewritten through `&mut self` invoke. Matches the
// CraneliftExec contract used by simulations.
unsafe impl Send for DisplayKernelExec {}
unsafe impl Sync for DisplayKernelExec {}

impl DisplayKernelExec {
    pub fn compile(
        mlir: &str,
        input_bytes: &[usize],
        output_bytes: &[usize],
    ) -> Result<Self, String> {
        if input_bytes.iter().any(|&size| size == 0) || output_bytes.iter().any(|&size| size == 0) {
            return Err("display kernel tensor sizes must be non-zero".into());
        }
        let mut ir_module =
            parser::parse_module(mlir).map_err(|err| format!("MLIR parse failed: {err}"))?;
        const_fold::fold_module(&mut ir_module);
        let compiled =
            compile_module(&ir_module).map_err(|err| format!("Cranelift compile failed: {err}"))?;
        let fn_ptr = compiled.get_main_fn();
        if fn_ptr.is_null() {
            return Err("compiled module is missing a callable main entrypoint".into());
        }
        let tick_fn = unsafe { std::mem::transmute::<*const u8, TickFn>(fn_ptr) };
        Ok(Self {
            _compiled: compiled,
            tick_fn,
            input_bytes: input_bytes.to_vec(),
            output_bytes: output_bytes.to_vec(),
            input_ptrs: vec![std::ptr::null(); input_bytes.len()],
            output_ptrs: vec![std::ptr::null_mut(); output_bytes.len()],
        })
    }

    pub fn validate(mlir: &str) -> Result<(), String> {
        let mut ir_module =
            parser::parse_module(mlir).map_err(|err| format!("MLIR parse failed: {err}"))?;
        const_fold::fold_module(&mut ir_module);
        compile_module(&ir_module).map_err(|err| format!("Cranelift compile failed: {err}"))?;
        Ok(())
    }

    pub fn invoke(&mut self, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<(), String> {
        if inputs.len() != self.input_bytes.len() {
            return Err(format!(
                "expected {} inputs, got {}",
                self.input_bytes.len(),
                inputs.len()
            ));
        }
        if outputs.len() != self.output_bytes.len() {
            return Err(format!(
                "expected {} outputs, got {}",
                self.output_bytes.len(),
                outputs.len()
            ));
        }
        for (index, (buf, expected)) in inputs.iter().zip(&self.input_bytes).enumerate() {
            if buf.len() != *expected {
                return Err(format!(
                    "input {index} is {} bytes, expected {expected}",
                    buf.len()
                ));
            }
        }
        for (index, (buf, expected)) in outputs.iter().zip(&self.output_bytes).enumerate() {
            if buf.len() != *expected {
                return Err(format!(
                    "output {index} is {} bytes, expected {expected}",
                    buf.len()
                ));
            }
        }
        for (slot, buf) in self.input_ptrs.iter_mut().zip(inputs) {
            *slot = buf.as_ptr();
        }
        for (slot, buf) in self.output_ptrs.iter_mut().zip(outputs.iter_mut()) {
            *slot = buf.as_mut_ptr();
        }
        unsafe {
            (self.tick_fn)(self.input_ptrs.as_ptr(), self.output_ptrs.as_mut_ptr());
        }
        Ok(())
    }

    pub fn input_bytes(&self) -> &[usize] {
        &self.input_bytes
    }

    pub fn output_bytes(&self) -> &[usize] {
        &self.output_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sqrt_mlir() -> &'static str {
        r#"
module @module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#
    }

    #[test]
    fn compiles_and_evaluates_scalar_sqrt() {
        let mut exec = DisplayKernelExec::compile(sqrt_mlir(), &[8], &[8]).unwrap();
        let input = 16.0_f64.to_le_bytes();
        let mut output = [0u8; 8];
        exec.invoke(&[&input], &mut [&mut output]).unwrap();
        let value = f64::from_le_bytes(output);
        assert!((value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_buffer_size_mismatch() {
        let mut exec = DisplayKernelExec::compile(sqrt_mlir(), &[8], &[8]).unwrap();
        let input = [0u8; 4];
        let mut output = [0u8; 8];
        assert!(exec.invoke(&[&input], &mut [&mut output]).is_err());
    }
}
