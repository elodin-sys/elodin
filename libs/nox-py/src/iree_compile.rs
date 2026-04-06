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

fn collect_noxpr_artifacts(compiled_system: &CompiledSystem) -> Vec<(String, String)> {
    let mut artifacts = vec![(
        "noxpr.txt".to_string(),
        format!("{}", compiled_system.computation.func.as_ref()),
    )];
    let mut seen_exprs = HashSet::new();
    let mut seen_call_comps = HashSet::new();
    let mut seen_scan_funcs = HashSet::new();
    let mut call_index = 0usize;
    let mut scan_index = 0usize;
    collect_noxpr_artifacts_from_expr(
        &compiled_system.computation.func.inner,
        &mut artifacts,
        &mut seen_exprs,
        &mut seen_call_comps,
        &mut seen_scan_funcs,
        &mut call_index,
        &mut scan_index,
    );
    artifacts
}

fn collect_noxpr_artifacts_from_expr(
    expr: &nox::Noxpr,
    artifacts: &mut Vec<(String, String)>,
    seen_exprs: &mut HashSet<nox::NoxprId>,
    seen_call_comps: &mut HashSet<nox::NoxprId>,
    seen_scan_funcs: &mut HashSet<nox::NoxprId>,
    call_index: &mut usize,
    scan_index: &mut usize,
) {
    if !seen_exprs.insert(expr.id()) {
        return;
    }

    match expr.node.as_ref() {
        nox::NoxprNode::Call(call) => {
            if seen_call_comps.insert(call.comp.id) {
                artifacts.push((
                    format!("noxpr_call_{:03}.txt", *call_index),
                    format!("{}", call.comp.func.as_ref()),
                ));
                *call_index += 1;
                collect_noxpr_artifacts_from_expr(
                    &call.comp.func.inner,
                    artifacts,
                    seen_exprs,
                    seen_call_comps,
                    seen_scan_funcs,
                    call_index,
                    scan_index,
                );
            }
            for arg in &call.args {
                collect_noxpr_artifacts_from_expr(
                    arg,
                    artifacts,
                    seen_exprs,
                    seen_call_comps,
                    seen_scan_funcs,
                    call_index,
                    scan_index,
                );
            }
        }
        nox::NoxprNode::Scan(scan) => {
            if seen_scan_funcs.insert(scan.scan_fn.inner.id()) {
                artifacts.push((
                    format!("noxpr_scan_{:03}.txt", *scan_index),
                    format!("{}", scan.scan_fn),
                ));
                *scan_index += 1;
                collect_noxpr_artifacts_from_expr(
                    &scan.scan_fn.inner,
                    artifacts,
                    seen_exprs,
                    seen_call_comps,
                    seen_scan_funcs,
                    call_index,
                    scan_index,
                );
            }
            for input in &scan.inputs {
                collect_noxpr_artifacts_from_expr(
                    input,
                    artifacts,
                    seen_exprs,
                    seen_call_comps,
                    seen_scan_funcs,
                    call_index,
                    scan_index,
                );
            }
            collect_noxpr_artifacts_from_expr(
                &scan.initial_state,
                artifacts,
                seen_exprs,
                seen_call_comps,
                seen_scan_funcs,
                call_index,
                scan_index,
            );
        }
        _ => {
            for child in expr.node.children() {
                collect_noxpr_artifacts_from_expr(
                    child,
                    artifacts,
                    seen_exprs,
                    seen_call_comps,
                    seen_scan_funcs,
                    call_index,
                    scan_index,
                );
            }
        }
    }
}

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
    compile_origin: &str,
) -> Result<IreeCompileResult, Error> {
    let func = noxpr_to_callable(compiled_system.computation.func.clone());
    let noxpr_artifacts = collect_noxpr_artifacts(compiled_system);

    let mut input_arrays = vec![];
    let mut visited_ids = HashSet::new();

    for slot in &compiled_system.input_slots {
        if !visited_ids.insert(slot.component_id) {
            continue;
        }
        let col = world
            .column_by_id(slot.component_id)
            .ok_or(Error::ComponentNotFound)?;
        let elem_ty = col.schema.prim_type;
        let dtype = nox::jax::dtype(&elem_ty.to_element_type())?;
        let shape_vec: Vec<_> = slot.shape.iter().map(|&dim| dim as u64).collect();
        let jnp = py.import("jax.numpy")?;
        let arr = jnp.getattr("zeros")?.call((shape_vec, dtype), None)?;
        input_arrays.push(arr.unbind());
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
import shlex
import shutil
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

def _write_text(path, filename, text):
    with open(os.path.join(path, filename), 'w') as f:
        f.write(text)

def _write_json(path, filename, payload):
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def _normalize_system_name(name):
    if ' at 0x' in name:
        name = name.split(' at 0x', maxsplit=1)[0]
        if not name.endswith('>'):
            name += '>'
    return name

def _input_summaries(input_arrays):
    summaries = []
    for arr in input_arrays or []:
        shape = None
        dtype = None
        try:
            shape = [int(dim) for dim in arr.shape]
        except Exception:
            pass
        try:
            dtype = str(arr.dtype)
        except Exception:
            pass
        summaries.append({'shape': shape, 'dtype': dtype})
    return summaries

def _update_compile_context(path, **updates):
    context_path = os.path.join(path, 'compile_context.json')
    context = {}
    if os.path.exists(context_path):
        try:
            with open(context_path, encoding='utf-8') as f:
                context = json.load(f)
        except Exception:
            context = {}
    context.update(updates)
    _write_json(path, 'compile_context.json', context)

def _write_base_artifacts(
    path,
    system_names,
    iree_bin,
    compile_origin,
    requested_device,
    has_singleton_lowering,
    user_extra_flags,
    env_flags,
    input_arrays,
    noxpr_artifacts,
):
    _write_text(path, 'system_names.txt', '\n'.join(system_names or []))
    artifact_names = []
    for filename, text in noxpr_artifacts or []:
        _write_text(path, filename, text)
        artifact_names.append(filename)
    _write_versions(path, iree_bin or 'unresolved')
    _update_compile_context(
        path,
        compile_origin=compile_origin,
        is_subsystem_diagnostic=(compile_origin == 'subsystem_diagnostic'),
        requested_device=requested_device,
        has_singleton_lowering=bool(has_singleton_lowering),
        system_names=system_names or [],
        display_system_names=[_normalize_system_name(name) for name in (system_names or [])],
        user_extra_flags=list(user_extra_flags or []),
        env_flags=list(env_flags or []),
        input_arrays=_input_summaries(input_arrays),
        noxpr_artifact=artifact_names[0] if artifact_names else None,
        noxpr_artifacts=artifact_names,
        report_stage='initialized',
    )

def _write_placeholder_compile_cmd(path, reason):
    _write_text(
        path,
        'iree_compile_cmd.sh',
        '#!/usr/bin/env bash\n' + '# unavailable: ' + reason + '\n',
    )

def _write_primary_artifacts(path, cmd_text, stablehlo_mlir, system_names, iree_bin):
    _write_text(path, 'iree_compile_cmd.sh', '#!/usr/bin/env bash\n' + cmd_text + ' < stablehlo.mlir\n')
    _write_text(path, 'stablehlo.mlir', stablehlo_mlir)
    _write_text(path, 'system_names.txt', '\n'.join(system_names or []))
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
    selected = (requested_device or 'auto').strip().lower()
    if selected in ('iree-cpu', 'iree'):
        selected = 'cpu'
    elif selected in ('iree-gpu', 'gpu'):
        selected = 'auto'
    if selected in ('cpu', 'local-sync'):
        return 'cpu', 'local-sync'
    if selected == 'local-task':
        return 'cpu', 'local-task'
    if selected == 'inline':
        return 'inline', 'local-sync'
    if selected == 'cuda':
        return 'cuda', 'cuda'
    if selected == 'metal':
        return 'metal', 'metal'
    if selected != 'auto':
        raise RuntimeError(
            "stage=config\n"
            + f"invalid IREE device '{selected}': expected one of "
            + "'auto', 'cpu', 'cuda', 'metal', 'inline', 'local-task', 'local-sync'"
        )
    if platform.system() == 'Linux' and shutil.which('nvidia-smi') is not None:
        return 'cuda', 'cuda'
    if platform.system() == 'Darwin':
        return 'metal', 'metal'
    return 'cpu', 'local-task'

def _prefer_indirect_command_buffers(stablehlo_mlir):
    # IREE 3.11 regressed in both directions here: threefry-based PRNG modules
    # fail when indirect command buffers are disabled, while rocket's
    # map_coordinates/dynamic-slice-heavy kernel fails when they stay enabled.
    return 'threefry' in stablehlo_mlir.lower()

def _extract_large_constants(mlir_text, threshold_bytes=None):
    """Scan the StableHLO MLIR for large constants and extract their raw bytes.

    The IREE compiler pass (promoteLargeConstants in StableHLOCustomCalls.cpp)
    handles the MLIR structural rewrite (adding parameters, threading call
    chains).  This function only extracts the byte data so the Elodin runtime
    can supply it as additional input buffers.

    The MLIR text is NOT modified -- the compiler pass handles that.

    Returns (mlir_text_unchanged, extracted_list) where extracted_list contains
    (raw_bytes, shape_list, dtype_str) tuples for each constant that the
    compiler pass will promote.
    """
    if threshold_bytes is None:
        env_val = os.environ.get('ELODIN_IREE_CONSTANT_PROMOTE_THRESHOLD')
        if env_val is not None:
            threshold_bytes = int(env_val)
            if threshold_bytes == 0:
                return mlir_text, []
        else:
            threshold_bytes = 1_048_576  # 1 MB default

    dense_pat = re.compile(
        r'\s+%\S+\s*=\s*stablehlo\.constant\s+dense<"0x([0-9a-fA-F]+)">\s*:\s*tensor<([^>]+)>'
    )
    resource_const_pat = re.compile(
        r'\s+%\S+\s*=\s*stablehlo\.constant\s+dense_resource<(\w+)>\s*:\s*tensor<([^>]+)>'
    )
    resource_blob_pat = re.compile(
        r'\s+(\w+)\s*:\s*"0x([0-9a-fA-F]+)"'
    )

    # First pass: collect resource blob hex data from the MLIR metadata
    # section (after {-#).  Scanning the module body would false-match
    # hex strings in backend_config or serialized constant attrs.
    resource_data = {}
    in_resource_section = False
    for line in mlir_text.split('\n'):
        if '{-#' in line:
            in_resource_section = True
        if not in_resource_section:
            continue
        blob_m = resource_blob_pat.match(line)
        if blob_m:
            resource_data[blob_m.group(1)] = blob_m.group(2)

    # Second pass: scan constant definitions in module order.
    # Threshold uses strictly-greater to match the compiler pass.
    extracted = []
    for line in mlir_text.split('\n'):
        m = dense_pat.match(line)
        if m:
            hex_data, shape_dtype = m.groups()
            raw_bytes = bytes.fromhex(hex_data)
            if len(raw_bytes) <= threshold_bytes:
                continue
            parts = shape_dtype.split('x')
            dtype_str = parts[-1]
            shape = [int(d) for d in parts[:-1]]
            extracted.append((raw_bytes, shape, dtype_str))
            continue
        m = resource_const_pat.match(line)
        if m:
            blob_name, shape_dtype = m.groups()
            hex_data = resource_data.get(blob_name)
            if not hex_data:
                continue
            raw_bytes = bytes.fromhex(hex_data)
            if len(raw_bytes) <= threshold_bytes:
                continue
            parts = shape_dtype.split('x')
            dtype_str = parts[-1]
            shape = [int(d) for d in parts[:-1]]
            extracted.append((raw_bytes, shape, dtype_str))

    if extracted:
        total_mb = sum(len(b) for b, _, _ in extracted) / 1e6
        import sys as _sys
        print(f'[elodin] found {len(extracted)} large constant(s) ({total_mb:.1f} MB) for promotion',
              file=_sys.stderr)

    return mlir_text, extracted

def compile_to_vmfb(
    func,
    input_arrays,
    user_extra_flags,
    system_names,
    requested_device,
    has_singleton_lowering=False,
    compile_origin='primary_system',
    noxpr_artifacts=(),
):
    env_flags = shlex.split(os.environ.get('ELODIN_IREE_FLAGS', ''))
    report_dir = _artifact_dir() if os.environ.get('ELODIN_IREE_DUMP_DIR') else None
    if report_dir is not None:
        _write_base_artifacts(
            report_dir,
            system_names,
            shutil.which('iree-compile'),
            compile_origin,
            requested_device,
            has_singleton_lowering,
            user_extra_flags,
            env_flags,
            input_arrays,
            noxpr_artifacts,
        )
    try:
        lower_start = time.perf_counter()
        jit_fn = jax.jit(func, keep_unused=True)
        lowered = jit_fn.lower(*input_arrays)
        lower_ms = (time.perf_counter() - lower_start) * 1000.0
    except Exception:
        traceback_text = traceback.format_exc()
        if report_dir is None:
            report_dir = _artifact_dir()
            _write_base_artifacts(
                report_dir,
                system_names,
                shutil.which('iree-compile'),
                compile_origin,
                requested_device,
                has_singleton_lowering,
                user_extra_flags,
                env_flags,
                input_arrays,
                noxpr_artifacts,
            )
        _write_text(report_dir, 'jax_lower_traceback.txt', traceback_text)
        _write_placeholder_compile_cmd(
            report_dir,
            'jax lowering failed before iree-compile command construction',
        )
        _update_compile_context(
            report_dir,
            report_stage='jax_lower_failed',
            failure_stage='jax_lower',
        )
        hint = ''
        lower_tb = traceback_text.lower()
        if 'scatter inputs have incompatible types' in lower_tb or (
            'index type must be an integer' in lower_tb and 'float64' in lower_tb
        ):
            hint = (
                '\n\n--- Unsigned integer type detected ---\n'
                'A system function uses unsigned integer types (e.g. jnp.uint64) which\n'
                'are incompatible with JAX type promotion. When uint64 values interact\n'
                'with int64, JAX promotes to float64, breaking index computations.\n\n'
                'Fix: change dtype=jnp.uint64 to dtype=jnp.int64 in your system function.\n'
                '     Values like state codes, entity IDs, and counters do not need unsigned types.\n\n'
                'Set ELODIN_IREE_DUMP_DIR and check the per-system diagnostic to identify\n'
                'which system function contains the unsigned type.\n'
            )
        raise RuntimeError(
            "stage=jax_lower\n"
            + traceback_text
            + hint
            + '\n\nDebug artifacts saved to: '
            + report_dir
        )

    try:
        emit_start = time.perf_counter()
        stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
        stablehlo_mlir = str(stablehlo_module)
        stablehlo_emit_ms = (time.perf_counter() - emit_start) * 1000.0
    except Exception:
        traceback_text = traceback.format_exc()
        if report_dir is None:
            report_dir = _artifact_dir()
            _write_base_artifacts(
                report_dir,
                system_names,
                shutil.which('iree-compile'),
                compile_origin,
                requested_device,
                has_singleton_lowering,
                user_extra_flags,
                env_flags,
                input_arrays,
                noxpr_artifacts,
            )
        _write_text(report_dir, 'stablehlo_emit_traceback.txt', traceback_text)
        _write_placeholder_compile_cmd(
            report_dir,
            'stablehlo emission failed before iree-compile command construction',
        )
        _update_compile_context(
            report_dir,
            report_stage='stablehlo_emit_failed',
            failure_stage='stablehlo_emit',
            lower_ms=lower_ms,
        )
        raise RuntimeError(
            "stage=stablehlo_emit\n"
            + traceback_text
            + '\n\nDebug artifacts saved to: '
            + report_dir
        )

    # JAX names the MLIR module after the jitted function (e.g. @jit_fn).
    # Rename to @module so the VMFB function is always "module.main".
    stablehlo_mlir = re.sub(r'module @\S+', 'module @module', stablehlo_mlir, count=1)

    # Extract large constant data for runtime-side upload.
    # The IREE compiler pass (promoteLargeConstants) handles the MLIR rewrite.
    stablehlo_mlir, promoted_constants = _extract_large_constants(stablehlo_mlir)

    # Integer signedness: PrimTypeExt maps all unsigned PrimTypes to signed
    # ElementTypes (S64 etc.) so that JAX's type promotion lattice never
    # mixes uint64+int64 (which promotes to float64).  StableHLO and IREE
    # use signless integers anyway.  Do NOT do a blanket text replacement
    # here -- it would corrupt JAX-internal threefry PRNG bit-manipulation.

    compile_target, runtime_device = _resolve_iree_device(requested_device)

    is_cpu_target = compile_target in ('cpu', 'inline')
    extra = [
        "--iree-vm-target-extension-f64",
        "--iree-input-demote-f64-to-f32=false",
        "--iree-input-type=stablehlo",
        "--iree-opt-level=O3" if is_cpu_target else "--iree-opt-level=O2",
        "--iree-dispatch-creation-enable-aggressive-fusion=true",
        "--iree-llvmcpu-enable-ukernels=all",
    ]
    if not is_cpu_target:
        extra.append("--iree-stream-partitioning-favor=max-concurrency")
    if is_cpu_target:
        extra.append("--iree-opt-strip-assertions=true")
    disable_indirect_command_buffers = not _prefer_indirect_command_buffers(stablehlo_mlir)
    if disable_indirect_command_buffers:
        extra.append("--iree-hal-indirect-command-buffers=false")
    if has_singleton_lowering:
        extra.append("--iree-flow-inline-constants-max-byte-length=0")
    if os.environ.get('ELODIN_IREE_DUMP_DIR'):
        extra.append("--iree-scheduling-dump-statistics-format=json")

    if platform.system() in ("Darwin", "Linux"):
        machine = platform.machine()
        if platform.system() == "Darwin":
            arch = 'arm64' if machine == 'arm64' else 'x86_64'
            extra.append('--iree-llvmcpu-target-triple=' + arch + '-apple-darwin')
            extra.append('--iree-llvmcpu-link-embedded=false')
            extra.append('--iree-opt-const-eval=false')
            cc = shutil.which('cc') or shutil.which('clang')
            if cc:
                wrapper = os.path.join(tempfile.gettempdir(), 'elodin_darwin_cc_wrapper.sh')
                with open(wrapper, 'w') as wf:
                    wf.write('#!/bin/sh\n')
                    wf.write('for a do shift; case "$a" in\n')
                    wf.write('  -static) ;;\n')
                    wf.write('  -dylib) set -- "$@" -dynamiclib ;;\n')
                    wf.write('  -flat_namespace) set -- "$@" -Wl,-flat_namespace ;;\n')
                    wf.write('  *) set -- "$@" "$a" ;;\n')
                    wf.write('esac; done\n')
                    wf.write('exec ' + shlex.quote(cc) + ' "$@"\n')
                os.chmod(wrapper, 0o755)
                extra.append('--iree-llvmcpu-system-linker-path=' + wrapper)
        else:
            arch = 'aarch64' if machine in ('aarch64', 'arm64') else 'x86_64'
            extra.append('--iree-llvmcpu-target-triple=' + arch + '-unknown-linux-gnu')
        extra.append('--iree-llvmcpu-target-cpu=host')

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
    elif compile_target == 'inline':
        target_args.append('--iree-hal-target-backends=llvm-cpu')
        extra.append('--iree-execution-model=inline-dynamic')
        compile_backend = 'llvm-cpu'
    else:
        target_args.append('--iree-hal-target-backends=llvm-cpu')

    iree_bin = None

    # 1. Baked-in path from nix-built elodin package
    try:
        import importlib.resources
        _ref = importlib.resources.files('elodin').joinpath('_iree_compiler_dir')
        if _ref.is_file():
            _dir = _ref.read_text().strip()
            _candidate = os.path.join(_dir, 'bin', 'iree-compile')
            if os.path.isfile(_candidate):
                iree_bin = _candidate
    except Exception:
        pass

    # 2. Explicit env var (for dev shell / maturin develop)
    if iree_bin is None:
        compiler_dir = os.environ.get('IREE_COMPILER_DIR', '')
        if compiler_dir:
            candidate = os.path.join(compiler_dir, 'bin', 'iree-compile')
            if os.path.isfile(candidate):
                iree_bin = candidate

    # 3. PATH fallback (venv/bin/iree-compile in nix shells)
    if iree_bin is None:
        candidate = shutil.which('iree-compile')
        if candidate:
            iree_bin = candidate

    if iree_bin is None:
        raise RuntimeError(
            "stage=config\n"
            "iree-compile not found. The source-built IREE compiler is required.\n"
            "Nix package: ensure elodin is installed via nix (the package bakes in the compiler path).\n"
            "Dev shell: set IREE_COMPILER_DIR to the iree-compiler-source nix store path."
        )

    extra.extend(env_flags)
    extra.extend(user_extra_flags or [])

    out_path = None
    try:
        fd, out_path = tempfile.mkstemp(suffix='.vmfb')
        os.close(fd)
        cmd = [iree_bin, '-', '-o', out_path] + target_args + extra
        cmd_text = shlex.join(cmd)
        if report_dir is not None:
            _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
            _update_compile_context(
                report_dir,
                report_stage='ready_to_compile',
                compile_target=compile_target,
                runtime_device=runtime_device,
                target_args=target_args,
                effective_iree_flags=extra,
                cmd_text=cmd_text,
                lower_ms=lower_ms,
                stablehlo_emit_ms=stablehlo_emit_ms,
                stablehlo_byte_len=len(stablehlo_mlir.encode('utf-8')),
                indirect_command_buffers_disabled=disable_indirect_command_buffers,
                singleton_inline_constants_disabled=bool(has_singleton_lowering),
            )

        compile_start = time.perf_counter()
        result = subprocess.run(cmd, input=stablehlo_mlir.encode('utf-8'), capture_output=True)
        stderr_text = result.stderr.decode('utf-8', errors='replace')

        # When auto mode selects CUDA but the CUDA lowering path is not
        # available, retry on llvm-cpu so default runs still succeed.
        selected_mode = (requested_device or 'auto').strip().lower()
        if selected_mode in ('iree-cpu', 'iree'):
            selected_mode = 'cpu'
        elif selected_mode in ('iree-gpu', 'gpu'):
            selected_mode = 'auto'
        if compile_target == 'cuda' and selected_mode == 'auto' and result.returncode != 0:
            if report_dir is None:
                report_dir = _artifact_dir()
                _write_base_artifacts(
                    report_dir,
                    system_names,
                    iree_bin,
                    compile_origin,
                    requested_device,
                    has_singleton_lowering,
                    user_extra_flags,
                    env_flags,
                    input_arrays,
                    noxpr_artifacts,
                )
                _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
                _update_compile_context(
                    report_dir,
                    report_stage='ready_to_compile',
                    compile_target=compile_target,
                    runtime_device=runtime_device,
                    target_args=target_args,
                    effective_iree_flags=extra,
                    cmd_text=cmd_text,
                    lower_ms=lower_ms,
                    stablehlo_emit_ms=stablehlo_emit_ms,
                    stablehlo_byte_len=len(stablehlo_mlir.encode('utf-8')),
                    indirect_command_buffers_disabled=disable_indirect_command_buffers,
                    singleton_inline_constants_disabled=bool(has_singleton_lowering),
                )
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
            _write_base_artifacts(
                report_dir,
                system_names,
                iree_bin,
                compile_origin,
                requested_device,
                has_singleton_lowering,
                user_extra_flags,
                env_flags,
                input_arrays,
                noxpr_artifacts,
            )
            _write_primary_artifacts(report_dir, cmd_text, stablehlo_mlir, system_names, iree_bin)
            _update_compile_context(
                report_dir,
                report_stage='ready_to_compile',
                compile_target=compile_target,
                runtime_device=runtime_device,
                target_args=target_args,
                effective_iree_flags=extra,
                cmd_text=cmd_text,
                lower_ms=lower_ms,
                stablehlo_emit_ms=stablehlo_emit_ms,
                stablehlo_byte_len=len(stablehlo_mlir.encode('utf-8')),
                indirect_command_buffers_disabled=disable_indirect_command_buffers,
                singleton_inline_constants_disabled=bool(has_singleton_lowering),
            )
        if report_dir is not None:
            with open(os.path.join(report_dir, 'iree_compile_stderr.txt'), 'w') as f:
                f.write(stderr_text)
            _update_compile_context(
                report_dir,
                report_stage='iree_compile_failed' if result.returncode != 0 else 'iree_compile_succeeded',
                compile_backend=compile_backend,
                runtime_device=runtime_device,
                iree_compile_ms=compile_ms,
            )
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
            _update_compile_context(
                report_dir,
                vmfb_size_bytes=len(vmfb),
                module_vmfb_path='module.vmfb',
            )
        return {
            'vmfb': vmfb,
            'lower_ms': lower_ms,
            'stablehlo_emit_ms': stablehlo_emit_ms,
            'iree_compile_ms': compile_ms,
            'vmfb_size_bytes': len(vmfb),
            'report_dir': report_dir,
            'compile_backend': compile_backend,
            'runtime_device': runtime_device,
            'promoted_constants': promoted_constants,
        }
    finally:
        if out_path:
            try: os.unlink(out_path)
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
                compiled_system.has_singleton_lowering,
                compile_origin.to_string(),
                noxpr_artifacts,
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
                     To temporarily unblock, run with backend=\"jax-cpu\" (slower).",
                );
            }
            if let Some(report_dir) = &report.report_dir {
                let stage_hint = match report.stage {
                    iree_diagnostics::FailureStage::JaxLower => {
                        "Inspect `jax_lower_traceback.txt` and `compile_context.json`."
                    }
                    iree_diagnostics::FailureStage::StablehloEmit => {
                        "Inspect `stablehlo_emit_traceback.txt` and `compile_context.json`."
                    }
                    iree_diagnostics::FailureStage::IreeCompile => {
                        "Inspect `iree_compile_stderr.txt`, `stablehlo.mlir`, and `compile_context.json`."
                    }
                    _ => "Inspect `compile_context.json` and the dumped artifacts in this directory.",
                };
                rendered.push_str(&format!(
                    "\n\nDebug artifacts directory: {report_dir}\n{stage_hint}"
                ));
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

    let mut promoted_constants = Vec::new();
    if let Some(pc_list) = result.get_item("promoted_constants")?
        && let Ok(py_list) = pc_list.downcast::<pyo3::types::PyList>()
    {
        for item in py_list.iter() {
            if let Ok(tup) = item.downcast::<pyo3::types::PyTuple>() {
                let data: Vec<u8> = tup.get_item(0)?.extract()?;
                let shape: Vec<i64> = tup.get_item(1)?.extract()?;
                let dtype_str: String = tup.get_item(2)?.extract()?;
                let element_type = match dtype_str.as_str() {
                    "f16" => nox::ElementType::F16,
                    "bf16" => nox::ElementType::Bf16,
                    "f32" => nox::ElementType::F32,
                    "f64" => nox::ElementType::F64,
                    "i1" | "i8" | "ui8" => nox::ElementType::S8,
                    "i16" | "ui16" => nox::ElementType::S16,
                    "i32" | "ui32" => nox::ElementType::S32,
                    "i64" | "ui64" => nox::ElementType::S64,
                    other => {
                        return Err(Error::IreeCompilationFailed(format!(
                            "unsupported dtype '{other}' in promoted constant"
                        )));
                    }
                };
                promoted_constants.push(crate::exec::ConstantSpec {
                    name: format!("promoted_{}", promoted_constants.len()),
                    data,
                    shape,
                    element_type,
                });
            }
        }
    }

    let metadata = ExecMetadata {
        arg_ids: compiled_system.inputs.clone(),
        ret_ids: compiled_system.outputs.clone(),
        arg_slots: compiled_system.input_slots.clone(),
        ret_slots: compiled_system.output_slots.clone(),
        has_singleton_lowering: compiled_system.has_singleton_lowering,
        promoted_constants,
    };
    let exec = IREEExec::new(&vmfb, metadata, Some(stats.clone()), &runtime_device, world)
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
