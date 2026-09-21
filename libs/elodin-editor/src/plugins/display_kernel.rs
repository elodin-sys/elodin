//! Fetch, verify, compile, and invoke content-addressed display kernels.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use cranelift_mlir::display_kernel::DisplayKernelExec;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{ConnectionAddr, EntityMap, TelemetryCache};
use impeller2_wkt::{
    ComponentValue, DISPLAY_KERNEL_ASSET_PREFIX, DisplayKernelArtifact, DisplayKernelBinding,
};
use sha2::{Digest, Sha256};

use crate::object_3d::{local_assets_root, resolve_db_asset_url};
use crate::plugins::kdl_document::InitialKdlPath;
use crate::ui::plot::data::EvaluatedSeries;

pub struct DisplayKernelPlugin;

impl Plugin for DisplayKernelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DisplayKernelCache>();
    }
}

#[derive(Resource, Default)]
pub struct DisplayKernelCache {
    modules: HashMap<String, Result<CompiledDisplayKernel, String>>,
}

pub struct CompiledDisplayKernel {
    pub artifact: DisplayKernelArtifact,
    pub scalar: DisplayKernelExec,
    pub batched: DisplayKernelExec,
}

#[derive(Clone, Copy)]
pub struct KernelFetchCtx<'a> {
    pub connection_addr: Option<SocketAddr>,
    pub local_root: Option<&'a Path>,
    pub kdl_dir: Option<&'a Path>,
}

impl DisplayKernelCache {
    pub fn compiled(
        &mut self,
        binding: &DisplayKernelBinding,
        fetch: KernelFetchCtx<'_>,
    ) -> Result<&mut CompiledDisplayKernel, String> {
        if !self.modules.contains_key(&binding.hash) {
            let loaded = load_compiled(binding, fetch);
            self.modules.insert(binding.hash.clone(), loaded);
        }
        match self.modules.get_mut(&binding.hash) {
            Some(Ok(compiled)) => Ok(compiled),
            Some(Err(err)) => Err(err.clone()),
            None => Err("display kernel cache insert failed".into()),
        }
    }
}

#[allow(dead_code)]
pub fn kernel_component_ids(binding: &DisplayKernelBinding) -> Vec<ComponentId> {
    let mut ids: Vec<_> = binding
        .inputs
        .iter()
        .map(|input| ComponentId::new(&input.component))
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

pub fn current_kernel_inputs(
    binding: &DisplayKernelBinding,
    entity_map: &EntityMap,
    values: &Query<&ComponentValue>,
) -> Option<Vec<Vec<u8>>> {
    let mut packed = Vec::with_capacity(binding.inputs.len());
    for input in &binding.inputs {
        let id = ComponentId::new(&input.component);
        let entity = entity_map.get(&id)?;
        let value = values.get(*entity).ok()?;
        packed.push(pack_component_value(value, &input.dtype).ok()?);
    }
    Some(packed)
}

pub fn invoke_scalar(
    compiled: &mut CompiledDisplayKernel,
    inputs: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, String> {
    let expected = compiled.artifact.input_nbytes(false)?;
    if inputs.len() != expected.len() {
        return Err(format!(
            "expected {} scalar inputs, got {}",
            expected.len(),
            inputs.len()
        ));
    }
    let input_refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    let mut outputs: Vec<Vec<u8>> = compiled
        .artifact
        .output_nbytes(false)?
        .into_iter()
        .map(|len| vec![0u8; len])
        .collect();
    let mut output_refs: Vec<&mut [u8]> = outputs.iter_mut().map(Vec::as_mut_slice).collect();
    compiled.scalar.invoke(&input_refs, &mut output_refs)?;
    Ok(outputs)
}

#[allow(clippy::too_many_lines)]
pub fn evaluate_kernel_series(
    cache: &TelemetryCache,
    compiled: &mut CompiledDisplayKernel,
    binding: &DisplayKernelBinding,
    range: std::ops::Range<Timestamp>,
    max_points: Option<usize>,
) -> Result<EvaluatedSeries, String> {
    let Some(driver_id) = binding
        .inputs
        .first()
        .map(|input| ComponentId::new(&input.component))
    else {
        return Ok(EvaluatedSeries {
            timestamps: Vec::new(),
            values: Vec::new(),
        });
    };
    let Some(driver) = cache.series(&driver_id) else {
        return Ok(EvaluatedSeries {
            timestamps: Vec::new(),
            values: Vec::new(),
        });
    };
    let sample_count = driver.range(range.clone()).count();
    let stride = max_points
        .filter(|&limit| limit > 0)
        .map(|limit| sample_count.div_ceil(limit))
        .unwrap_or(1)
        .max(1);
    let batch_size = compiled.artifact.batch_size as usize;
    let input_sizes = compiled.artifact.input_nbytes(false)?;
    let batched_in = compiled.artifact.input_nbytes(true)?;
    let batched_out = compiled.artifact.output_nbytes(true)?;

    let mut timestamps = Vec::new();
    let mut packed_inputs: Vec<Vec<Vec<u8>>> = vec![Vec::new(); compiled.artifact.inputs.len()];
    for (sample_index, (&timestamp, _)) in driver.range(range).enumerate() {
        if sample_index % stride != 0 {
            continue;
        }
        let mut row = Vec::with_capacity(binding.inputs.len());
        let mut missing = false;
        for (index, input) in binding.inputs.iter().enumerate() {
            let dtype = compiled
                .artifact
                .inputs
                .get(index)
                .map(|item| item.dtype.as_str())
                .unwrap_or(input.dtype.as_str());
            let id = ComponentId::new(&input.component);
            match cache.get_at_or_before(&id, timestamp) {
                Some(value) => match pack_component_value(value, dtype) {
                    Ok(bytes) if bytes.len() == input_sizes[index] => row.push(bytes),
                    _ => {
                        missing = true;
                        break;
                    }
                },
                None => {
                    missing = true;
                    break;
                }
            }
        }
        if missing {
            continue;
        }
        timestamps.push(timestamp);
        for (slot, bytes) in packed_inputs.iter_mut().zip(row) {
            slot.push(bytes);
        }
    }

    if timestamps.is_empty() {
        return Ok(EvaluatedSeries {
            timestamps,
            values: Vec::new(),
        });
    }

    let n = timestamps.len();
    let mut values = vec![Vec::with_capacity(n); flatten_value_count(&compiled.artifact.outputs)];

    let mut in_bufs: Vec<Vec<u8>> = batched_in.iter().map(|len| vec![0u8; *len]).collect();
    let mut out_bufs: Vec<Vec<u8>> = batched_out.iter().map(|len| vec![0u8; *len]).collect();

    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_len = (n - chunk_start).min(batch_size);
        for (input_index, samples) in packed_inputs.iter().enumerate() {
            let scalar = input_sizes[input_index];
            for row in 0..batch_size {
                let src = if row < chunk_len {
                    &samples[chunk_start + row]
                } else if chunk_len > 0 {
                    &samples[chunk_start + chunk_len - 1]
                } else {
                    continue;
                };
                let dest = &mut in_bufs[input_index][row * scalar..(row + 1) * scalar];
                dest.copy_from_slice(src);
            }
        }
        let input_refs: Vec<&[u8]> = in_bufs.iter().map(Vec::as_slice).collect();
        let mut output_refs: Vec<&mut [u8]> = out_bufs.iter_mut().map(Vec::as_mut_slice).collect();
        compiled.batched.invoke(&input_refs, &mut output_refs)?;
        unpack_batched_outputs(&compiled.artifact, &out_bufs, chunk_len, &mut values)?;
    }

    Ok(EvaluatedSeries { timestamps, values })
}

pub fn output_floats(bytes: &[u8], dtype: &str) -> Result<Vec<f64>, String> {
    match dtype {
        "f64" => {
            let (chunks, rest) = bytes.as_chunks::<8>();
            if !rest.is_empty() {
                return Err("f64 output is not 8-byte aligned".into());
            }
            Ok(chunks
                .iter()
                .map(|&chunk| f64::from_le_bytes(chunk))
                .collect())
        }
        "f32" => {
            let (chunks, rest) = bytes.as_chunks::<4>();
            if !rest.is_empty() {
                return Err("f32 output is not 4-byte aligned".into());
            }
            Ok(chunks
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk) as f64)
                .collect())
        }
        other => Err(format!("unsupported display kernel output dtype {other}")),
    }
}

pub fn kernel_fetch_ctx<'a>(
    connection_addr: Option<Res<'a, ConnectionAddr>>,
    initial_kdl: Option<Res<'a, InitialKdlPath>>,
) -> (Option<SocketAddr>, Option<PathBuf>, Option<PathBuf>) {
    let addr = connection_addr.map(|a| a.0);
    let local_root = local_assets_root(initial_kdl.as_deref());
    let kdl_dir = initial_kdl
        .and_then(|path| path.0.clone())
        .and_then(|path| path.parent().map(Path::to_path_buf));
    (addr, local_root, kdl_dir)
}

fn load_compiled(
    binding: &DisplayKernelBinding,
    fetch: KernelFetchCtx<'_>,
) -> Result<CompiledDisplayKernel, String> {
    let bytes = fetch_kernel_bytes(binding, fetch)?;
    let artifact = DisplayKernelArtifact::parse(&bytes)?;
    let (scalar_in, scalar_out, batched_in, batched_out) = verify_artifact(&artifact, binding)?;
    let scalar = DisplayKernelExec::compile(&artifact.scalar_mlir, &scalar_in, &scalar_out)?;
    let batched = DisplayKernelExec::compile(&artifact.batched_mlir, &batched_in, &batched_out)?;
    Ok(CompiledDisplayKernel {
        artifact,
        scalar,
        batched,
    })
}

type KernelBufferSizes = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

fn verify_artifact(
    artifact: &DisplayKernelArtifact,
    binding: &DisplayKernelBinding,
) -> Result<KernelBufferSizes, String> {
    if artifact.hash != binding.hash {
        return Err(format!(
            "display kernel hash mismatch: binding {} vs artifact {}",
            binding.hash, artifact.hash
        ));
    }
    let payload = artifact.hash_payload()?;
    let digest = hex::encode(Sha256::digest(&payload));
    if digest != artifact.hash {
        return Err("display kernel sidecar failed content-hash verification".into());
    }
    if artifact.batch_size == 0 || artifact.batch_size > 4096 {
        return Err("display kernel batch size is unbounded or zero".into());
    }
    let scalar_in = artifact.input_nbytes(false)?;
    let scalar_out = artifact.output_nbytes(false)?;
    let batched_in = artifact.input_nbytes(true)?;
    let batched_out = artifact.output_nbytes(true)?;
    if scalar_in.iter().any(|&n| n == 0 || n > 1 << 20)
        || scalar_out.iter().any(|&n| n == 0 || n > 1 << 20)
    {
        return Err("display kernel tensor is missing or unbounded".into());
    }
    Ok((scalar_in, scalar_out, batched_in, batched_out))
}

fn fetch_kernel_bytes(
    binding: &DisplayKernelBinding,
    fetch: KernelFetchCtx<'_>,
) -> Result<Vec<u8>, String> {
    let hash = binding
        .hash
        .rsplit('/')
        .next()
        .unwrap_or(&binding.hash)
        .to_string();
    if let Some(dir) = fetch.kdl_dir {
        let local = dir.join("kernels").join(&hash);
        if local.is_file() {
            return std::fs::read(&local).map_err(|err| format!("read {}: {err}", local.display()));
        }
    }
    let key = if binding.asset.starts_with(DISPLAY_KERNEL_ASSET_PREFIX)
        || binding.asset.starts_with("db:")
    {
        binding.asset.clone()
    } else {
        DisplayKernelBinding::asset_key(&hash)
    };
    if let Some(root) = fetch.local_root {
        let rel = key.strip_prefix("db:").unwrap_or(&key);
        let path = root.join(rel);
        if path.is_file() {
            return std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()));
        }
    }
    let url = if key.starts_with("http://") || key.starts_with("https://") {
        key
    } else if key.starts_with("db:") {
        resolve_db_asset_url(&key, fetch.connection_addr)
    } else {
        resolve_db_asset_url(&format!("db:{key}"), fetch.connection_addr)
    };
    reqwest::blocking::get(&url)
        .and_then(|resp| resp.error_for_status()?.bytes().map(|b| b.to_vec()))
        .map_err(|err| format!("fetch {url}: {err}"))
}

fn pack_component_value(value: &ComponentValue, dtype: &str) -> Result<Vec<u8>, String> {
    let floats = component_floats(value);
    write_floats(&floats, dtype)
}

fn component_floats(value: &ComponentValue) -> Vec<f64> {
    use nox::ArrayBuf;
    match value {
        ComponentValue::F64(array) => array.buf.as_buf().to_vec(),
        ComponentValue::F32(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::I64(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::I32(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::U64(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::U32(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::I16(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::U16(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::I8(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::U8(array) => array.buf.as_buf().iter().map(|&v| v as f64).collect(),
        ComponentValue::Bool(array) => array
            .buf
            .as_buf()
            .iter()
            .map(|&v| if v { 1.0 } else { 0.0 })
            .collect(),
    }
}

fn write_floats(values: &[f64], dtype: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    match dtype {
        "f64" | "" => {
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        "f32" => {
            for value in values {
                out.extend_from_slice(&(*value as f32).to_le_bytes());
            }
        }
        "i64" => {
            for value in values {
                out.extend_from_slice(&(*value as i64).to_le_bytes());
            }
        }
        "i32" => {
            for value in values {
                out.extend_from_slice(&(*value as i32).to_le_bytes());
            }
        }
        other => return Err(format!("unsupported display kernel input dtype {other}")),
    }
    Ok(out)
}

fn flatten_value_count(outputs: &[impeller2_wkt::DisplayKernelTensor]) -> usize {
    outputs
        .iter()
        .map(|tensor| {
            tensor
                .shape
                .iter()
                .fold(1usize, |acc, dim| acc.saturating_mul(*dim as usize))
                .max(1)
        })
        .sum()
}

fn unpack_batched_outputs(
    artifact: &DisplayKernelArtifact,
    out_bufs: &[Vec<u8>],
    chunk_len: usize,
    values: &mut [Vec<f32>],
) -> Result<(), String> {
    let mut series = 0;
    for (index, tensor) in artifact.outputs.iter().enumerate() {
        let width = impeller2_wkt::dtype_width(&tensor.dtype)?;
        let elems = tensor
            .shape
            .iter()
            .fold(1usize, |acc, dim| acc.saturating_mul(*dim as usize))
            .max(1);
        let scalar = elems * width;
        for row in 0..chunk_len {
            let start = row * scalar;
            let end = start + scalar;
            let sample = out_bufs
                .get(index)
                .and_then(|buf| buf.get(start..end))
                .ok_or_else(|| "batched kernel output is short".to_string())?;
            let floats = output_floats(sample, &tensor.dtype)?;
            if series + floats.len() > values.len() {
                return Err("kernel output series count mismatch".into());
            }
            for (offset, value) in floats.iter().enumerate() {
                values[series + offset].push(*value as f32);
            }
        }
        series += elems;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_wkt::{DisplayKernelInput, DisplayKernelTensor};
    use sha2::{Digest, Sha256};

    fn sample_artifact(hash: &str, batch_size: u32) -> DisplayKernelArtifact {
        DisplayKernelArtifact {
            version: 1,
            hash: hash.into(),
            batch_size,
            inputs: vec![DisplayKernelTensor {
                name: "x".into(),
                component: "drone.nav.speed".into(),
                shape: vec![],
                dtype: "f64".into(),
            }],
            outputs: vec![DisplayKernelTensor {
                name: String::new(),
                component: String::new(),
                shape: vec![],
                dtype: "f64".into(),
            }],
            scalar_mlir: "module {}".into(),
            batched_mlir: "module {}".into(),
        }
    }

    fn hashed_artifact(batch_size: u32) -> DisplayKernelArtifact {
        let mut artifact = sample_artifact("", batch_size);
        let digest = hex::encode(Sha256::digest(&artifact.hash_payload().unwrap()));
        artifact.hash = digest;
        artifact
    }

    fn binding_for(artifact: &DisplayKernelArtifact) -> DisplayKernelBinding {
        DisplayKernelBinding {
            hash: artifact.hash.clone(),
            asset: DisplayKernelBinding::asset_key(&artifact.hash),
            inputs: vec![DisplayKernelInput {
                component: "drone.nav.speed".into(),
                shape: vec![],
                dtype: "f64".into(),
            }],
        }
    }

    #[test]
    fn writes_and_reads_f64_payloads() {
        let bytes = write_floats(&[1.5, -2.0], "f64").unwrap();
        assert_eq!(output_floats(&bytes, "f64").unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn flatten_counts_matrix_outputs() {
        let outputs = [DisplayKernelTensor {
            name: String::new(),
            component: String::new(),
            shape: vec![3, 3],
            dtype: "f64".into(),
        }];
        assert_eq!(flatten_value_count(&outputs), 9);
    }

    #[test]
    fn kernel_component_ids_dedups() {
        let binding = DisplayKernelBinding {
            hash: "abc".into(),
            asset: "schematics/kernels/abc".into(),
            inputs: vec![
                DisplayKernelInput {
                    component: "drone.nav.speed".into(),
                    shape: vec![],
                    dtype: "f64".into(),
                },
                DisplayKernelInput {
                    component: "drone.nav.speed".into(),
                    shape: vec![],
                    dtype: "f64".into(),
                },
            ],
        };
        assert_eq!(kernel_component_ids(&binding).len(), 1);
    }

    #[test]
    fn content_hash_matches_binding() {
        let artifact = hashed_artifact(256);
        assert!(verify_artifact(&artifact, &binding_for(&artifact)).is_ok());
    }

    #[test]
    fn rejects_hash_mismatch() {
        let artifact = hashed_artifact(256);
        let mut binding = binding_for(&artifact);
        binding.hash = "deadbeef".into();
        let err = verify_artifact(&artifact, &binding).unwrap_err();
        assert!(err.contains("hash mismatch"));
    }

    #[test]
    fn rejects_tampered_sidecar_payload() {
        let mut artifact = hashed_artifact(256);
        artifact.scalar_mlir = "module @tampered {}".into();
        let err = verify_artifact(&artifact, &binding_for(&artifact)).unwrap_err();
        assert!(err.contains("content-hash"));
    }

    #[test]
    fn rejects_unbounded_batch_size() {
        let artifact = hashed_artifact(0);
        let err = verify_artifact(&artifact, &binding_for(&artifact)).unwrap_err();
        assert!(err.contains("batch size"));
        let artifact = hashed_artifact(8192);
        let err = verify_artifact(&artifact, &binding_for(&artifact)).unwrap_err();
        assert!(err.contains("batch size"));
    }

    #[test]
    fn unpacks_chunk_without_padding_rows() {
        let artifact = DisplayKernelArtifact {
            version: 1,
            hash: "x".into(),
            batch_size: 4,
            inputs: vec![],
            outputs: vec![DisplayKernelTensor {
                name: String::new(),
                component: String::new(),
                shape: vec![],
                dtype: "f64".into(),
            }],
            scalar_mlir: String::new(),
            batched_mlir: String::new(),
        };
        let mut buf = Vec::new();
        for value in [1.0_f64, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&value.to_le_bytes());
        }
        let mut values = vec![Vec::new()];
        unpack_batched_outputs(&artifact, &[buf], 3, &mut values).unwrap();
        assert_eq!(values[0], vec![1.0, 2.0, 3.0]);
    }
}
