use std::collections::HashSet;
use std::time::Instant;

use impeller2::types::ComponentId;
use tracing::trace;

use crate::error::Error;
use crate::exec::{ExecMetadata, ExecSlotMetadata};
use crate::iree_compile::IreeCompileStats;
use crate::profile::{Profiler, TickTimings};
use crate::utils::SchemaExt;

fn iree_debug_data() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ELODIN_IREE_DEBUG_DATA").is_ok())
}

fn dump_slot_f64(
    label: &str,
    slot_idx: usize,
    component_id: ComponentId,
    shape: &[i64],
    data: &[u8],
) {
    let n_f64 = data.len() / 8;
    let max_show = n_f64.min(8);
    let mut vals = Vec::with_capacity(max_show);
    let mut has_nan = false;
    let mut has_inf = false;
    let mut all_zero = true;
    for i in 0..n_f64 {
        let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap_or([0; 8]);
        let v = f64::from_le_bytes(bytes);
        if v.is_nan() {
            has_nan = true;
        }
        if v.is_infinite() {
            has_inf = true;
        }
        if v != 0.0 {
            all_zero = false;
        }
        if i < max_show {
            vals.push(v);
        }
    }
    let flag = if has_nan {
        " *** NAN ***"
    } else if has_inf {
        " *** INF ***"
    } else if all_zero {
        " (all zeros)"
    } else {
        ""
    };
    eprintln!(
        "[IREE_DEBUG] {label} slot={slot_idx} comp={component_id:?} shape={shape:?} bytes={} f64s={n_f64}{flag} vals={vals:?}",
        data.len()
    );
}
use crate::world::World;

fn nox_to_iree_element_type(ty: nox::ElementType) -> Result<iree_runtime::ElementType, Error> {
    match ty {
        nox::ElementType::Pred => Ok(iree_runtime::ElementType::Bool),
        nox::ElementType::S8 => Ok(iree_runtime::ElementType::Int8),
        nox::ElementType::S16 => Ok(iree_runtime::ElementType::Int16),
        nox::ElementType::S32 => Ok(iree_runtime::ElementType::Int32),
        nox::ElementType::S64 => Ok(iree_runtime::ElementType::Int64),
        nox::ElementType::U8 => Ok(iree_runtime::ElementType::Uint8),
        nox::ElementType::U16 => Ok(iree_runtime::ElementType::Uint16),
        nox::ElementType::U32 => Ok(iree_runtime::ElementType::Uint32),
        nox::ElementType::U64 => Ok(iree_runtime::ElementType::Uint64),
        nox::ElementType::F16 => Ok(iree_runtime::ElementType::Float16),
        nox::ElementType::F32 => Ok(iree_runtime::ElementType::Float32),
        nox::ElementType::Bf16 => Ok(iree_runtime::ElementType::BFloat16),
        nox::ElementType::F64 => Ok(iree_runtime::ElementType::Float64),
        nox::ElementType::C64 | nox::ElementType::C128 => Err(Error::IreeCompilationFailed(
            "complex types not supported by IREE".to_string(),
        )),
    }
}

pub struct IREEExec {
    pub metadata: ExecMetadata,
    pub compile_stats: Option<IreeCompileStats>,
    vmfb: Vec<u8>,
    device_uri: String,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    mutable_overlap: Vec<(usize, usize)>,
    input_arena: Option<iree_runtime::DeviceArena>,
    output_arena: Option<iree_runtime::DeviceArena>,
    output_views_scratch: Vec<iree_runtime::BufferView>,
    promoted_constant_views: Vec<iree_runtime::BufferView>,
    call: iree_runtime::Call,
    session: iree_runtime::Session,
    #[allow(dead_code)]
    instance: iree_runtime::Instance,
}

unsafe impl Send for IREEExec {}
unsafe impl Sync for IREEExec {}

impl IREEExec {
    pub fn new(
        vmfb: &[u8],
        metadata: ExecMetadata,
        compile_stats: Option<IreeCompileStats>,
        device_uri: &str,
        world: &World,
    ) -> Result<Self, Error> {
        let instance =
            iree_runtime::Instance::new().map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        let device = instance.create_device(device_uri).map_err(|err| {
            Error::IreeRuntimeError(format!(
                "failed to create requested IREE device '{device_uri}': {err}"
            ))
        })?;
        let session = iree_runtime::Session::new(&instance, &device)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        // Register the elodin_lapack module (provides SVD, Cholesky, LU, QR,
        // solve, eigh via OpenBLAS). Must be appended before loading the VMFB
        // so the VM can resolve elodin_lapack.* imports.
        if let Ok(lapack_module) =
            unsafe { iree_runtime::lapack::create_module(instance.vm_instance(), session.device()) }
        {
            let _ = unsafe { session.append_module(lapack_module) };
        }

        session
            .load_vmfb(vmfb)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        let call = session
            .call("module.main")
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        let mut output_ids = Vec::new();
        let mut seen_outputs = HashSet::new();
        for id in &metadata.ret_ids {
            if seen_outputs.insert(*id) {
                output_ids.push(*id);
            }
        }
        let mut input_ids = Vec::new();
        let mut input_specs = Vec::new();
        let mut seen_inputs = HashSet::new();
        for slot in &metadata.arg_slots {
            if !seen_inputs.insert(slot.component_id) {
                continue;
            }
            let spec = build_buffer_spec(world, slot)?;
            input_specs.push(spec);
            input_ids.push(slot.component_id);
        }

        let input_arena = if input_specs.is_empty() {
            None
        } else {
            Some(
                iree_runtime::DeviceArena::new(&session, &input_specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };

        let output_arena = if output_ids.is_empty() {
            None
        } else {
            let mut specs = Vec::with_capacity(output_ids.len());
            for id in &output_ids {
                let slot = metadata
                    .ret_slots
                    .iter()
                    .find(|slot| slot.component_id == *id)
                    .ok_or(Error::ComponentNotFound)?;
                specs.push(build_buffer_spec(world, slot)?);
            }
            Some(
                iree_runtime::DeviceArena::new(&session, &specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };

        if iree_debug_data() {
            eprintln!("[IREE_DEBUG] === IREEExec::new slot ordering ===");
            for (i, slot) in metadata.arg_slots.iter().enumerate() {
                let col = world.column_by_id(slot.component_id);
                let col_bytes = col.map(|c| c.column.len()).unwrap_or(0);
                let comp_name = world
                    .metadata
                    .component_map
                    .get(&slot.component_id)
                    .map(|(_, m)| m.name.as_str())
                    .unwrap_or("?");
                let deduped = input_ids.contains(&slot.component_id);
                eprintln!(
                    "[IREE_DEBUG]   arg_slot[{i}] comp={comp_name} id={:?} shape={:?} elided={} col_bytes={col_bytes} deduped_in={deduped}",
                    slot.component_id, slot.shape, slot.entity_axis_elided,
                );
            }
            eprintln!("[IREE_DEBUG]   input_ids (deduped, in push_input order):");
            for (i, id) in input_ids.iter().enumerate() {
                let comp_name = world
                    .metadata
                    .component_map
                    .get(id)
                    .map(|(_, m)| m.name.as_str())
                    .unwrap_or("?");
                let col = world.column_by_id(*id);
                let col_bytes = col.map(|c| c.column.len()).unwrap_or(0);
                let slot = metadata.arg_slots.iter().find(|s| s.component_id == *id);
                let shape = slot.map(|s| s.shape.as_slice()).unwrap_or(&[]);
                let expected_bytes = shape.iter().product::<i64>() as usize * 8;
                let mismatch = col_bytes != expected_bytes;
                eprintln!(
                    "[IREE_DEBUG]     [{i}] {comp_name} col_bytes={col_bytes} shape={shape:?} expected_bytes={expected_bytes}{}",
                    if mismatch {
                        " *** SIZE MISMATCH ***"
                    } else {
                        ""
                    }
                );
            }
            eprintln!("[IREE_DEBUG]   output_ids:");
            for (i, id) in output_ids.iter().enumerate() {
                let comp_name = world
                    .metadata
                    .component_map
                    .get(id)
                    .map(|(_, m)| m.name.as_str())
                    .unwrap_or("?");
                let col = world.column_by_id(*id);
                let col_bytes = col.map(|c| c.column.len()).unwrap_or(0);
                let slot = metadata.ret_slots.iter().find(|s| s.component_id == *id);
                let shape = slot.map(|s| s.shape.as_slice()).unwrap_or(&[]);
                let expected_bytes = shape.iter().product::<i64>() as usize * 8;
                let mismatch = col_bytes != expected_bytes;
                eprintln!(
                    "[IREE_DEBUG]     [{i}] {comp_name} col_bytes={col_bytes} shape={shape:?} expected_bytes={expected_bytes}{}",
                    if mismatch {
                        " *** SIZE MISMATCH ***"
                    } else {
                        ""
                    }
                );
            }
            eprintln!("[IREE_DEBUG] === end slot ordering ===");
        }

        let mut output_slot_by_id = std::collections::HashMap::new();
        for (slot, id) in output_ids.iter().enumerate() {
            output_slot_by_id.insert(*id, slot);
        }
        let mutable_overlap = input_ids
            .iter()
            .enumerate()
            .filter_map(|(input_slot, id)| {
                output_slot_by_id
                    .get(id)
                    .copied()
                    .map(|output_slot| (input_slot, output_slot))
            })
            .collect();

        let mut promoted_constant_views = Vec::new();
        for spec in &metadata.promoted_constants {
            let element_type = nox_to_iree_element_type(spec.element_type)?;
            let view = iree_runtime::BufferView::from_bytes(
                &session,
                &spec.data,
                &spec.shape,
                element_type,
            )
            .map_err(|e| {
                Error::IreeRuntimeError(format!("promoted constant '{}': {e}", spec.name))
            })?;
            promoted_constant_views.push(view);
        }

        Ok(Self {
            metadata,
            compile_stats,
            vmfb: vmfb.to_vec(),
            device_uri: device_uri.to_string(),
            input_ids,
            output_ids,
            mutable_overlap,
            input_arena,
            output_arena,
            output_views_scratch: Vec::new(),
            promoted_constant_views,
            call,
            session,
            instance,
        })
    }

    pub fn invoke_in_place(
        &mut self,
        world: &mut World,
        detailed: bool,
    ) -> Result<TickTimings, Error> {
        self.invoke_batch(world, 1, detailed)
    }

    pub fn invoke_batch(
        &mut self,
        world: &mut World,
        n: u64,
        detailed: bool,
    ) -> Result<TickTimings, Error> {
        let batch_ticks = n.max(1) as usize;
        let h2d_start = detailed.then(Instant::now);
        if let Some(arena) = &mut self.input_arena {
            for (slot, id) in self.input_ids.iter().enumerate() {
                let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
                if iree_debug_data() {
                    let comp_name = world
                        .metadata
                        .component_map
                        .get(id)
                        .map(|(_, m)| m.name.as_str())
                        .unwrap_or("?");
                    let slot_meta = self.compile_stats.as_ref().map(|_| "");
                    let arg_slot = std::iter::empty::<&ExecSlotMetadata>().next();
                    let _ = (slot_meta, arg_slot);
                    dump_slot_f64("INPUT", slot, *id, &[], col.column.as_slice());
                    eprintln!("[IREE_DEBUG]   ^ name={comp_name}");
                }
                arena
                    .write_slot(slot, col.column.as_slice())
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }
            arena
                .upload_staging(&self.session)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        }
        let h2d_upload_ms = h2d_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();

        let kernel_start = detailed.then(Instant::now);
        self.output_views_scratch.clear();
        self.output_views_scratch.reserve(self.output_ids.len());
        for batch_idx in 0..batch_ticks {
            self.call.reset();
            for slot in 0..self.input_ids.len() {
                let view = self
                    .input_arena
                    .as_ref()
                    .ok_or_else(|| Error::IreeRuntimeError("missing input arena".into()))?
                    .view(slot);
                self.call
                    .push_input(view)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }
            for cv in &self.promoted_constant_views {
                self.call
                    .push_input(cv)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }

            self.call
                .invoke()
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

            self.output_views_scratch.clear();
            let mut seen_outputs = HashSet::new();
            for ret_id in &self.metadata.ret_ids {
                let output = self
                    .call
                    .pop_output()
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                if seen_outputs.insert(*ret_id) {
                    self.output_views_scratch.push(output);
                }
            }

            if batch_idx + 1 < batch_ticks
                && let Some(input_arena) = &self.input_arena
            {
                for (input_slot, output_slot) in &self.mutable_overlap {
                    input_arena
                        .copy_slot_from_view(
                            &self.session,
                            *input_slot,
                            &self.output_views_scratch[*output_slot],
                        )
                        .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                }
            }
        }
        let kernel_invoke_ms = kernel_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();

        let d2h_start = detailed.then(Instant::now);
        if let Some(arena) = &mut self.output_arena {
            arena
                .copy_slots_from_views_batched(&self.session, &self.output_views_scratch)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            arena
                .download_all(&self.session)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            for (slot, id) in self.output_ids.iter().enumerate() {
                let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
                arena
                    .copy_slot_to_host(slot, &mut host.buffer)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                if iree_debug_data() {
                    let comp_name = world
                        .metadata
                        .component_map
                        .get(id)
                        .map(|(_, m)| m.name.as_str())
                        .unwrap_or("?");
                    dump_slot_f64("OUTPUT", slot, *id, &[], &host.buffer);
                    eprintln!("[IREE_DEBUG]   ^ name={comp_name}");
                }
            }
        } else {
            for (slot, id) in self.output_ids.iter().enumerate() {
                let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
                self.output_views_scratch[slot]
                    .download_into(&self.session, &mut host.buffer)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                if iree_debug_data() {
                    let comp_name = world
                        .metadata
                        .component_map
                        .get(id)
                        .map(|(_, m)| m.name.as_str())
                        .unwrap_or("?");
                    dump_slot_f64("OUTPUT", slot, *id, &[], &host.buffer);
                    eprintln!("[IREE_DEBUG]   ^ name={comp_name}");
                }
            }
        }
        let d2h_download_ms = d2h_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        Ok(TickTimings {
            h2d_upload_ms,
            call_setup_ms: 0.0,
            kernel_invoke_ms,
            d2h_download_ms,
        })
    }
}

fn build_buffer_spec(
    world: &World,
    slot: &ExecSlotMetadata,
) -> Result<iree_runtime::BufferSpec, Error> {
    let col = world
        .column_by_id(slot.component_id)
        .ok_or(Error::ComponentNotFound)?;
    let element_type = nox_to_iree_element_type(col.schema.element_type())?;
    Ok(iree_runtime::BufferSpec {
        byte_len: col.column.len(),
        shape: slot.shape.clone(),
        element_type,
    })
}

pub struct IREEWorldExec {
    pub world: World,
    pub tick_exec: IREEExec,
    pub startup_exec: Option<IREEExec>,
    pub profiler: Profiler,
}

impl IREEWorldExec {
    pub fn new(world: World, tick_exec: IREEExec, startup_exec: Option<IREEExec>) -> Self {
        Self {
            world,
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        }
    }

    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        let ticks_per_telemetry = self.world.ticks_per_telemetry();

        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.invoke_in_place(&mut self.world, self.profiler.detailed_timing)?;
        }

        let tick_start = Instant::now();
        let timings = self.tick_exec.invoke_batch(
            &mut self.world,
            ticks_per_telemetry,
            self.profiler.detailed_timing,
        )?;
        let tick_elapsed = tick_start.elapsed();
        if self.profiler.detailed_timing {
            let h2d = std::time::Duration::from_secs_f64(timings.h2d_upload_ms / 1000.0);
            let call_setup = std::time::Duration::from_secs_f64(timings.call_setup_ms / 1000.0);
            let kernel = std::time::Duration::from_secs_f64(timings.kernel_invoke_ms / 1000.0);
            let d2h = std::time::Duration::from_secs_f64(timings.d2h_download_ms / 1000.0);
            self.profiler.copy_to_client.observe_duration(h2d);
            self.profiler.execute_buffers.observe_duration(kernel);
            self.profiler.copy_to_host.observe_duration(d2h);
            self.profiler.h2d_upload.observe_duration(h2d);
            self.profiler.call_setup.observe_duration(call_setup);
            self.profiler.kernel_invoke.observe_duration(kernel);
            self.profiler.d2h_download.observe_duration(d2h);
            trace!(
                h2d_ms = timings.h2d_upload_ms,
                kernel_ms = timings.kernel_invoke_ms,
                d2h_ms = timings.d2h_download_ms,
                total_ms = tick_elapsed.as_secs_f64() * 1000.0,
                batch = ticks_per_telemetry,
                "iree tick",
            );
        } else {
            self.profiler.execute_buffers.observe_duration(tick_elapsed);
        }

        *start = Instant::now();
        for _ in 0..ticks_per_telemetry {
            self.world.advance_tick();
        }
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    pub fn fork(&self) -> Self {
        let tick_exec = IREEExec::new(
            &self.tick_exec.vmfb,
            self.tick_exec.metadata.clone(),
            self.tick_exec.compile_stats.clone(),
            &self.tick_exec.device_uri,
            &self.world,
        )
        .expect("failed to fork IREE exec");
        Self {
            world: self.world.clone(),
            tick_exec,
            startup_exec: None,
            profiler: self.profiler.clone(),
        }
    }

    pub fn profile(&self) -> std::collections::HashMap<&'static str, f64> {
        let mut profile = self.profiler.profile(
            self.world.sim_time_step().0,
            self.world.ticks_per_telemetry(),
        );
        if let Some(stats) = &self.tick_exec.compile_stats {
            profile.insert(
                "compile",
                stats.lower_ms + stats.stablehlo_emit_ms + stats.iree_compile_ms,
            );
            profile.insert("iree_lower", stats.lower_ms);
            profile.insert("iree_stablehlo_emit", stats.stablehlo_emit_ms);
            profile.insert("iree_compile", stats.iree_compile_ms);
            profile.insert("iree_vmfb_size_bytes", stats.vmfb_size_bytes as f64);
        }
        profile
    }
}
