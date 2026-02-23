use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use std::collections::HashSet;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::iree_exec::IREEExec;
use crate::system::{CompiledSystem, noxpr_to_callable};
use crate::utils::PrimTypeExt;
use crate::world::World;

pub fn compile_iree_module(
    py: Python<'_>,
    compiled_system: &CompiledSystem,
    world: &World,
) -> Result<IREEExec, Error> {
    let func = noxpr_to_callable(compiled_system.computation.func.clone());

    let mut input_arrays = vec![];
    let mut visited_ids = HashSet::new();

    for id in compiled_system
        .inputs
        .iter()
        .chain(compiled_system.outputs.iter())
    {
        if visited_ids.contains(id) {
            continue;
        }
        let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
        let elem_ty = col.schema.prim_type;
        let dtype = nox::jax::dtype(&elem_ty.to_element_type())?;
        let shape_vec: Vec<_> = std::iter::once(col.len() as u64)
            .chain(col.schema.shape().iter().copied())
            .collect();
        let jnp = py.import("jax.numpy")?;
        let arr = jnp.getattr("zeros")?.call((shape_vec, dtype), None)?;
        input_arrays.push(arr.into_py_any(py)?);
        visited_ids.insert(id);
    }

    let py_code = r#"
import jax
import os
import re
os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)

def compile_to_vmfb(func, input_arrays):
    from iree.compiler import compile_str

    jit_fn = jax.jit(func, keep_unused=True)
    lowered = jit_fn.lower(*input_arrays)
    stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
    stablehlo_mlir = str(stablehlo_module)

    # JAX names the MLIR module after the jitted function (e.g. @jit_fn).
    # Rename to @module so the VMFB function is always "module.main".
    stablehlo_mlir = re.sub(r'module @\S+', 'module @module', stablehlo_mlir, count=1)

    import platform, subprocess, tempfile, shutil, stat

    backend = "llvm-cpu"
    args = [
        "--iree-vm-target-extension-f64",
        "--iree-input-demote-f64-to-f32=false",
        "--iree-hal-indirect-command-buffers=false",
        "--iree-input-type=stablehlo",
        f"--iree-hal-target-backends={backend}",
    ]
    cleanup_files = []
    if platform.system() == "Darwin":
        # macOS embedded ELF linker can't resolve cos/sin/exp/log.
        # Provide them as linked bitcode using LLVM intrinsics.
        math_ll = (
            'declare double @llvm.cos.f64(double)\n'
            'declare double @llvm.sin.f64(double)\n'
            'declare double @llvm.exp.f64(double)\n'
            'declare double @llvm.log.f64(double)\n'
            'declare double @llvm.pow.f64(double, double)\n'
            'declare float @llvm.cos.f32(float)\n'
            'declare float @llvm.sin.f32(float)\n'
            'declare float @llvm.exp.f32(float)\n'
            'declare float @llvm.log.f32(float)\n'
            'declare float @llvm.pow.f32(float, float)\n'
            'define double @cos(double %x) { %r = call double @llvm.cos.f64(double %x) ret double %r }\n'
            'define double @sin(double %x) { %r = call double @llvm.sin.f64(double %x) ret double %r }\n'
            'define double @exp(double %x) { %r = call double @llvm.exp.f64(double %x) ret double %r }\n'
            'define double @log(double %x) { %r = call double @llvm.log.f64(double %x) ret double %r }\n'
            'define double @pow(double %x, double %y) { %r = call double @llvm.pow.f64(double %x, double %y) ret double %r }\n'
            'define float @cosf(float %x) { %r = call float @llvm.cos.f32(float %x) ret float %r }\n'
            'define float @sinf(float %x) { %r = call float @llvm.sin.f32(float %x) ret float %r }\n'
            'define float @expf(float %x) { %r = call float @llvm.exp.f32(float %x) ret float %r }\n'
            'define float @logf(float %x) { %r = call float @llvm.log.f32(float %x) ret float %r }\n'
            'define float @powf(float %x, float %y) { %r = call float @llvm.pow.f32(float %x, float %y) ret float %r }\n'
        )
        math_ll_fd, math_ll_path = tempfile.mkstemp(suffix=".ll", prefix="iree_math_")
        math_bc_path = math_ll_path.replace(".ll", ".bc")
        with os.fdopen(math_ll_fd, "w") as f:
            f.write(math_ll)
        cleanup_files.append(math_ll_path)
        cleanup_files.append(math_bc_path)
        llvm_as = shutil.which("llvm-as")
        if llvm_as:
            subprocess.run([llvm_as, math_ll_path, "-o", math_bc_path], check=True)
            args.append(f"--iree-link-bitcode={math_bc_path}")

    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        from iree.compiler import _mlir_libs
        import pathlib
        iree_compile = str(pathlib.Path(_mlir_libs.__file__).parent / "iree-compile")

    out_fd, out_path = tempfile.mkstemp(suffix=".vmfb")
    os.close(out_fd)
    cleanup_files.append(out_path)

    cmd = [iree_compile, "-", "-o", out_path] + args
    result = subprocess.run(
        cmd,
        input=stablehlo_mlir.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        for f in cleanup_files:
            try: os.unlink(f)
            except: pass
        raise RuntimeError(
            f"iree-compile failed (code {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    with open(out_path, "rb") as f:
        vmfb = f.read()
    for f in cleanup_files:
        try: os.unlink(f)
        except: pass

    return vmfb
"#;

    let module = PyModule::new(py, "iree_compile")?;
    let globals = module.dict();
    let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
    py.run(code_cstr.as_ref(), Some(&globals), None)?;
    let compile_fn: Py<PyAny> = module.getattr("compile_to_vmfb")?.into();

    let vmfb_bytes = compile_fn
        .call1(py, (func, input_arrays))?
        .extract::<PyObject>(py)?;
    let vmfb_bytes = vmfb_bytes
        .downcast_bound::<PyBytes>(py)
        .map_err(|_| Error::IreeCompilationFailed("VMFB result was not bytes".to_string()))?;
    let vmfb = vmfb_bytes.as_bytes().to_vec();

    let metadata = ExecMetadata {
        arg_ids: compiled_system.inputs.clone(),
        ret_ids: compiled_system.outputs.clone(),
    };

    IREEExec::new(&vmfb, metadata)
}
