use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};
use std::collections::HashSet;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::iree_diagnostics::{self, DiagnosticReport};
use crate::iree_exec::IREEExec;
use crate::system::{CompiledSystem, noxpr_to_callable};
use crate::utils::PrimTypeExt;
use crate::world::World;

#[derive(Debug, Clone, Default)]
pub struct IreeCompileStats {
    pub lower_ms: f64,
    pub stablehlo_emit_ms: f64,
    pub iree_compile_ms: f64,
    pub vmfb_size_bytes: usize,
}

pub struct IreeCompileResult {
    pub exec: IREEExec,
    pub stats: IreeCompileStats,
    pub report: DiagnosticReport,
}

pub fn compile_iree_module(
    py: Python<'_>,
    compiled_system: &CompiledSystem,
    world: &World,
    iree_device: &str,
    extra_iree_flags: &[String],
) -> Result<IreeCompileResult, Error> {
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
import json
import traceback
import time
import tempfile
import pathlib
import shlex
import shutil
import stat
import subprocess
os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)

def _artifact_dir():
    base_dir = os.environ.get('ELODIN_IREE_DUMP_DIR')
    if not base_dir:
        base_dir = os.path.join(tempfile.gettempdir(), 'elodin_iree_debug')
    os.makedirs(base_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    return tempfile.mkdtemp(prefix=f'{stamp}-', dir=base_dir)

def _write_versions(path, iree_bin):
    versions = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'jax': getattr(jax, '__version__', 'unknown'),
        'iree_compile_binary': iree_bin,
    }
    try:
        import jaxlib
        versions['jaxlib'] = getattr(jaxlib, '__version__', getattr(getattr(jaxlib, 'version', None), '__version__', 'unknown'))
    except Exception:
        versions['jaxlib'] = 'unknown'
    try:
        import elodin
        versions['elodin'] = getattr(elodin, '__version__', 'unknown')
    except Exception:
        versions['elodin'] = 'unknown'
    with open(os.path.join(path, 'versions.json'), 'w') as f:
        json.dump(versions, f, indent=2)

def _write_primary_artifacts(path, cmd_text, stablehlo_mlir, system_names, iree_bin):
    with open(os.path.join(path, 'iree_compile_cmd.sh'), 'w') as f:
        f.write('#!/usr/bin/env bash\n' + cmd_text + ' < stablehlo.mlir\n')
    with open(os.path.join(path, 'stablehlo.mlir'), 'w') as f:
        f.write(stablehlo_mlir)
    with open(os.path.join(path, 'system_names.txt'), 'w') as f:
        f.write('\n'.join(system_names or []))
    _write_versions(path, iree_bin)

def _detect_cuda_target():
    env_target = os.environ.get('ELODIN_CUDA_TARGET')
    if env_target:
        return env_target
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            timeout=2.0,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            return None
        first = result.stdout.strip().splitlines()[0].strip()
        major_minor = first.split('.', maxsplit=1)
        if len(major_minor) != 2:
            return None
        major = major_minor[0]
        minor = major_minor[1]
        if major.isdigit() and minor.isdigit():
            return f'sm_{major}{minor}'
    except Exception:
        return None
    return None

def _resolve_iree_device(requested_device):
    selected = os.environ.get('ELODIN_IREE_DEVICE', requested_device or 'auto').strip().lower()
    if selected in ('cpu', 'local-task', 'local-sync'):
        return 'cpu', 'local-task'
    if selected == 'cuda':
        return 'cuda', 'cuda'
    if selected == 'metal':
        return 'metal', 'metal'
    if selected != 'auto':
        raise RuntimeError(
            "stage=config\n"
            + f"invalid IREE device '{selected}': expected one of "
            + "'auto', 'cpu', 'cuda', 'metal', 'local-task', 'local-sync'"
        )
    if platform.system() == 'Linux' and shutil.which('nvidia-smi') is not None:
        return 'cuda', 'cuda'
    if platform.system() == 'Darwin':
        return 'metal', 'metal'
    return 'cpu', 'local-task'

def compile_to_vmfb(func, input_arrays, user_extra_flags, system_names, requested_device):
    try:
        lower_start = time.perf_counter()
        jit_fn = jax.jit(func, keep_unused=True)
        lowered = jit_fn.lower(*input_arrays)
        lower_ms = (time.perf_counter() - lower_start) * 1000.0
    except Exception:
        raise RuntimeError("stage=jax_lower\n" + traceback.format_exc())

    try:
        emit_start = time.perf_counter()
        stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
        stablehlo_mlir = str(stablehlo_module)
        stablehlo_emit_ms = (time.perf_counter() - emit_start) * 1000.0
    except Exception:
        raise RuntimeError("stage=stablehlo_emit\n" + traceback.format_exc())

    # JAX names the MLIR module after the jitted function (e.g. @jit_fn).
    # Rename to @module so the VMFB function is always "module.main".
    stablehlo_mlir = re.sub(r'module @\S+', 'module @module', stablehlo_mlir, count=1)

    compile_target, runtime_device = _resolve_iree_device(requested_device)

    extra = [
        "--iree-vm-target-extension-f64",
        "--iree-input-demote-f64-to-f32=false",
        "--iree-input-type=stablehlo",
        "--iree-hal-indirect-command-buffers=false",
    ]

    from iree.compiler import _mlir_libs
    iree_tools_dir = str(pathlib.Path(_mlir_libs.__file__).parent)

    wrapper_path = None
    cc_bin = (
        os.environ.get('ELODIN_IREE_CC')
        or shutil.which('clang')
        or os.environ.get('CC')
        or shutil.which('cc')
        or '/usr/bin/cc'
    )

    # Some CUDA/Metal compilations still route select executables through the
    # llvm-cpu linker path. Keep this enabled globally so trig/libm symbols
    # resolve consistently across mixed backend modules.
    extra.append('--iree-llvmcpu-link-embedded=false')
    if platform.system() in ("Darwin", "Linux"):
        machine = platform.machine()

        fd, wrapper_path = tempfile.mkstemp(suffix='.sh', prefix='iree_cc_')
        with os.fdopen(fd, 'w') as wf:
            if platform.system() == "Darwin":
                wf.write('\n'.join([
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
                    f'exec "{cc_bin}" "${{filtered[@]}}"',
                ]) + '\n')
            else:
                wf.write('\n'.join([
                    '#!/bin/bash',
                    'out=""',
                    'objs=()',
                    'passthrough=()',
                    'want_shared=0',
                    'args=("$@")',
                    'i=0',
                    'while [ $i -lt ${#args[@]} ]; do',
                    '  arg="${args[$i]}"',
                    '  case "$arg" in',
                    '    -o)',
                    '      i=$((i+1))',
                    '      out="${args[$i]}"',
                    '      ;;',
                    '    -shared|-dylib)',
                    '      want_shared=1',
                    '      ;;',
                    '    *.o|*.obj)',
                    '      objs+=("$arg")',
                    '      ;;',
                    '    -nostdlib|-static)',
                    '      ;;',
                    '    *)',
                    '      passthrough+=("$arg")',
                    '      ;;',
                    '  esac',
                    '  i=$((i+1))',
                    'done',
                    'if [ -z "$out" ]; then',
                    '  out="/tmp/iree_linked_$$.so"',
                    'fi',
                    'if [ ${#objs[@]} -eq 0 ]; then',
                    '  echo "IREE linker wrapper: no object files in args: $*" >&2',
                    '  exit 1',
                    'fi',
                    'rsp="$(mktemp /tmp/iree_objs_XXXXXX.rsp)"',
                    'trap \'rm -f "$rsp"\' EXIT',
                    '{',
                    '  printf "%s\\n" "${passthrough[@]}"',
                    '  if [ "$want_shared" -eq 1 ]; then',
                    '    printf "%s\\n" -shared',
                    '  fi',
                    '  printf "%s\\n" -o "$out"',
                    '  printf "%s\\n" "${objs[@]}"',
                    '  printf "%s\\n" -lm',
                    '} > "$rsp"',
                    'if [ "$want_shared" -eq 1 ]; then',
                    f'  exec "{cc_bin}" @"$rsp"',
                    'else',
                    f'  exec "{cc_bin}" @"$rsp"',
                    'fi',
                ]) + '\n')
        os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)
        extra.append('--iree-llvmcpu-system-linker-path=' + wrapper_path)
        extra.append('--iree-llvmcpu-embedded-linker-path=' + wrapper_path)

        if platform.system() == "Darwin":
            arch = 'arm64' if machine == 'arm64' else 'x86_64'
            extra.append('--iree-llvmcpu-target-triple=' + arch + '-apple-darwin')
            extra.append('--iree-opt-const-eval=false')
        else:
            arch = 'aarch64' if machine in ('aarch64', 'arm64') else 'x86_64'
            extra.append('--iree-llvmcpu-target-triple=' + arch + '-unknown-linux-gnu')
            extra.append('--iree-llvmcpu-link-static=false')
            extra.append('--iree-opt-const-eval=false')

    target_args = []
    compile_backend = compile_target
    if compile_target == 'cuda':
        cuda_target = _detect_cuda_target()
        if cuda_target:
            target_args.append(f'--iree-hal-target-device=cuda[--cuda-target={cuda_target}]')
        else:
            target_args.append('--iree-hal-target-device=cuda')
    elif compile_target == 'metal':
        target_args.append('--iree-hal-target-backends=metal-spirv')
    else:
        target_args.append('--iree-hal-target-backends=llvm-cpu')

    iree_bin = shutil.which('iree-compile')
    if iree_bin is None:
        iree_bin = iree_tools_dir + '/iree-compile'

    env_flags = shlex.split(os.environ.get('ELODIN_IREE_FLAGS', ''))
    extra.extend(env_flags)
    extra.extend(user_extra_flags or [])

    report_dir = _artifact_dir() if os.environ.get('ELODIN_IREE_DUMP_DIR') else None

    out_path = None
    try:
        fd, out_path = tempfile.mkstemp(suffix='.vmfb')
        os.close(fd)
        cmd = [iree_bin, '-', '-o', out_path] + target_args + extra
        cmd_text = shlex.join(cmd)
        if report_dir is not None:
            _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)

        compile_start = time.perf_counter()
        result = subprocess.run(cmd, input=stablehlo_mlir.encode('utf-8'), capture_output=True)
        stderr_text = result.stderr.decode('utf-8', errors='replace')
        # Some environments fail to link trig symbols (cos/sin) for llvm-cpu.
        # Retry with vmvx so users can still run on the IREE backend.
        if (
            compile_target == 'cpu'
            and result.returncode != 0
            and "undefined symbol" in stderr_text
            and "iree-lld" in stderr_text
        ):
            if report_dir is None:
                report_dir = _artifact_dir()
                _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
            vmvx_extra = [f for f in extra if not f.startswith('--iree-llvmcpu-')]
            vmvx_cmd = [iree_bin, '-', '-o', out_path, '--iree-hal-target-backends=vmvx'] + vmvx_extra
            with open(os.path.join(report_dir, 'iree_compile_cmd_vmvx.sh'), 'w') as f:
                f.write('#!/usr/bin/env bash\n' + shlex.join(vmvx_cmd) + ' < stablehlo.mlir\n')
            vmvx_result = subprocess.run(
                vmvx_cmd,
                input=stablehlo_mlir.encode('utf-8'),
                capture_output=True,
            )
            vmvx_stderr = vmvx_result.stderr.decode('utf-8', errors='replace')
            if vmvx_result.returncode == 0:
                result = vmvx_result
                compile_backend = 'vmvx'
                stderr_text = (
                    "Primary llvm-cpu compile failed; vmvx fallback succeeded.\n\n"
                    "=== llvm-cpu stderr ===\n"
                    + stderr_text
                    + "\n\n=== vmvx stderr ===\n"
                    + vmvx_stderr
                )
            else:
                stderr_text = (
                    "Primary llvm-cpu compile failed and vmvx fallback also failed.\n\n"
                    "=== llvm-cpu stderr ===\n"
                    + stderr_text
                    + "\n\n=== vmvx stderr ===\n"
                    + vmvx_stderr
                )

        # When auto mode selects CUDA but the CUDA lowering path is not
        # available, retry on llvm-cpu so default runs still succeed.
        selected_mode = os.environ.get('ELODIN_IREE_DEVICE', requested_device or 'auto').strip().lower()
        if compile_target == 'cuda' and selected_mode == 'auto' and result.returncode != 0:
            if report_dir is None:
                report_dir = _artifact_dir()
                _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
            cpu_cmd = [iree_bin, '-', '-o', out_path, '--iree-hal-target-backends=llvm-cpu'] + extra
            with open(os.path.join(report_dir, 'iree_compile_cmd_cpu_fallback.sh'), 'w') as f:
                f.write('#!/usr/bin/env bash\n' + shlex.join(cpu_cmd) + ' < stablehlo.mlir\n')
            cpu_result = subprocess.run(
                cpu_cmd,
                input=stablehlo_mlir.encode('utf-8'),
                capture_output=True,
            )
            cpu_stderr = cpu_result.stderr.decode('utf-8', errors='replace')
            if cpu_result.returncode == 0:
                result = cpu_result
                compile_backend = 'llvm-cpu'
                runtime_device = 'local-task'
                stderr_text = (
                    "Primary cuda compile failed; llvm-cpu fallback succeeded.\n\n"
                    "=== cuda stderr ===\n"
                    + stderr_text
                    + "\n\n=== llvm-cpu fallback stderr ===\n"
                    + cpu_stderr
                )
            else:
                stderr_text = (
                    "Primary cuda compile failed and llvm-cpu fallback also failed.\n\n"
                    "=== cuda stderr ===\n"
                    + stderr_text
                    + "\n\n=== llvm-cpu fallback stderr ===\n"
                    + cpu_stderr
                )

        compile_ms = (time.perf_counter() - compile_start) * 1000.0
        if result.returncode != 0 and report_dir is None:
            report_dir = _artifact_dir()
            _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
        if report_dir is not None:
            with open(os.path.join(report_dir, 'iree_compile_stderr.txt'), 'w') as f:
                f.write(stderr_text)
        if result.returncode != 0:
            raise RuntimeError(
                'stage=iree_compile\n'
                + 'iree-compile failed (code ' + str(result.returncode) + '):\n'
                + stderr_text
                + '\n\nDebug artifacts saved to: ' + report_dir
            )
        with open(out_path, 'rb') as f:
            vmfb = f.read()
        if report_dir is not None:
            with open(os.path.join(report_dir, 'module.vmfb'), 'wb') as f:
                f.write(vmfb)
        return {
            'vmfb': vmfb,
            'lower_ms': lower_ms,
            'stablehlo_emit_ms': stablehlo_emit_ms,
            'iree_compile_ms': compile_ms,
            'vmfb_size_bytes': len(vmfb),
            'report_dir': report_dir,
            'compile_backend': compile_backend,
            'runtime_device': runtime_device,
        }
    finally:
        if out_path:
            try: os.unlink(out_path)
            except: pass
        if wrapper_path:
            try: os.unlink(wrapper_path)
            except: pass
"#;

    let module = PyModule::new(py, "iree_compile")?;
    let globals = module.dict();
    let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
    py.run(code_cstr.as_ref(), Some(&globals), None)?;
    let compile_fn: Py<PyAny> = module.getattr("compile_to_vmfb")?.into();

    let result = compile_fn
        .call1(
            py,
            (
                func,
                input_arrays,
                extra_iree_flags.to_vec(),
                compiled_system.system_names.clone(),
                iree_device.to_string(),
            ),
        )
        .map_err(|e| {
            let msg = e.value(py).to_string();
            let report = iree_diagnostics::classify_failure(&msg);
            let mut rendered = format!(
                "{msg}\n\nFailure stage: {:?}\nFailure class: {:?}\n{}",
                report.stage, report.classification, report.suggestion
            );
            if report.should_suggest_jax_fallback() {
                rendered.push_str(
                    "\n\nThis simulation appears to use JAX features not yet supported by IREE.\n\
                     To temporarily unblock, run with backend=\"jax\" (slower).",
                );
            }
            Error::IreeCompilationFailed(rendered)
        })?;
    let result = result.extract::<PyObject>(py)?;
    let result = result.downcast_bound::<PyDict>(py).map_err(|_| {
        Error::IreeCompilationFailed("IREE compile helper returned non-dict".to_string())
    })?;

    let vmfb_bytes = result
        .get_item("vmfb")?
        .ok_or_else(|| Error::IreeCompilationFailed("compile result missing vmfb".to_string()))?
        .downcast_into::<PyBytes>()
        .map_err(|_| Error::IreeCompilationFailed("VMFB result was not bytes".to_string()))?;
    let vmfb = vmfb_bytes.as_bytes().to_vec();

    let stats = IreeCompileStats {
        lower_ms: result
            .get_item("lower_ms")?
            .and_then(|x| x.extract::<f64>().ok())
            .unwrap_or_default(),
        stablehlo_emit_ms: result
            .get_item("stablehlo_emit_ms")?
            .and_then(|x| x.extract::<f64>().ok())
            .unwrap_or_default(),
        iree_compile_ms: result
            .get_item("iree_compile_ms")?
            .and_then(|x| x.extract::<f64>().ok())
            .unwrap_or_default(),
        vmfb_size_bytes: result
            .get_item("vmfb_size_bytes")?
            .and_then(|x| x.extract::<usize>().ok())
            .unwrap_or_default(),
    };
    let report_dir = result
        .get_item("report_dir")?
        .and_then(|x| x.extract::<String>().ok());
    let runtime_device = result
        .get_item("runtime_device")?
        .and_then(|x| x.extract::<String>().ok())
        .unwrap_or_else(|| "local-task".to_string());

    let metadata = ExecMetadata {
        arg_ids: compiled_system.inputs.clone(),
        ret_ids: compiled_system.outputs.clone(),
    };
    let exec = IREEExec::new(&vmfb, metadata, Some(stats.clone()), &runtime_device)
        .map_err(|e| Error::IreeCompilationFailed(format!("stage=vmfb_load\n{e}")))?;

    let report = DiagnosticReport {
        stage: iree_diagnostics::FailureStage::IreeCompile,
        classification: iree_diagnostics::FailureClass::Unknown,
        raw_error: String::new(),
        matched_patterns: vec![],
        suggestion: String::new(),
        report_dir,
    };
    Ok(IreeCompileResult {
        exec,
        stats,
        report,
    })
}
