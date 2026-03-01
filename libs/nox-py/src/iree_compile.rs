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
        input_arrays.push(arr.unbind());
        visited_ids.insert(id);
    }

    let py_code = r#"
import jax
import os
import re
import platform
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

    extra = [
        "--iree-vm-target-extension-f64",
        "--iree-input-demote-f64-to-f32=false",
        "--iree-input-type=stablehlo",
        "--iree-hal-indirect-command-buffers=false",
    ]

    import tempfile, stat, shutil, subprocess, pathlib
    from iree.compiler import _mlir_libs
    iree_tools_dir = str(pathlib.Path(_mlir_libs.__file__).parent)

    # The embedded ELF loader does not support importing symbols (log, cos,
    # sin, etc.).  Use the system-library loader on all platforms -- it
    # produces a .so / .dylib that gets loaded via dlopen which CAN resolve
    # libm symbols from the host process.
    extra.append('--iree-llvmcpu-link-embedded=false')

    if platform.system() == "Darwin":
        machine = platform.machine()
        arch = 'arm64' if machine == 'arm64' else 'x86_64'

        # macOS: IREE's system linker invocation passes -static which
        # macOS ld rejects. Create a wrapper that strips -static and maps
        # lld flags to cc flags.
        wrapper_path = tempfile.mktemp(suffix='.sh', prefix='iree_cc_')
        lines = [
            '#!/bin/bash',
            'filtered=()',
            'for arg in "$@"; do',
            '  case "$arg" in',
            '    -static) ;;',
            '    -dylib) filtered+=("-dynamiclib") ;;',
            '    -flat_namespace) filtered+=("-Wl,-flat_namespace") ;;',
            '    *) filtered+=("$arg") ;;',
            '  esac',
            'done',
            'exec /usr/bin/cc "${filtered[@]}"',
        ]
        with open(wrapper_path, 'w') as wf:
            wf.write('\n'.join(lines) + '\n')
        os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)

        extra.append('--iree-llvmcpu-target-triple=' + arch + '-apple-darwin')
        extra.append('--iree-llvmcpu-system-linker-path=' + wrapper_path)
        extra.append('--iree-opt-const-eval=false')

    iree_bin = shutil.which('iree-compile')
    if iree_bin is None:
        iree_bin = iree_tools_dir + '/iree-compile'

    import tempfile as _tempfile
    out_path = _tempfile.mktemp(suffix='.vmfb')
    cmd = [iree_bin, '-', '-o', out_path, '--iree-hal-target-backends=llvm-cpu'] + extra
    import sys
    print('[IREE] cmd: ' + ' '.join(cmd), file=sys.stderr, flush=True)
    result = subprocess.run(cmd, input=stablehlo_mlir.encode('utf-8'), capture_output=True)
    print('[IREE] exit: ' + str(result.returncode), file=sys.stderr, flush=True)
    if result.stderr:
        stderr_text = result.stderr.decode('utf-8', errors='replace')
        if 'error' in stderr_text.lower():
            print('[IREE] stderr: ' + stderr_text[:2000], file=sys.stderr, flush=True)
    if result.returncode != 0:
        try: os.unlink(out_path)
        except: pass
        raise RuntimeError(
            'iree-compile failed (code ' + str(result.returncode) + '):\n'
            + result.stderr.decode('utf-8', errors='replace')
        )
    with open(out_path, 'rb') as f:
        vmfb = f.read()
    os.unlink(out_path)
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
