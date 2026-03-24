use std::collections::HashSet;
use std::time::Instant;

use impeller2::types::ComponentId;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::iree_compile::IreeCompileStats;
use crate::profile::{Profiler, TickTimings};
use crate::utils::SchemaExt;
use crate::world::World;

fn nox_to_iree_element_type(ty: nox::ElementType) -> Result<iree_runtime::ElementType, Error> {
    match ty {
        nox::ElementType::Pred => Ok(iree_runtime::ElementType::Bool),
        nox::ElementType::S8 => Ok(iree_runtime::ElementType::Int8),
        nox::ElementType::S16 => Ok(iree_runtime::ElementType::Int16),
        nox::ElementType::S32 => Ok(iree_runtime::ElementType::Int32),
        nox::ElementType::S64 => Ok(iree_runtime::ElementType::Int64),
        // These arms are not reached in practice: PrimTypeExt::to_element_type()
        // maps all unsigned PrimTypes to signed ElementTypes upstream.  Kept as
        // correct unsigned mappings for defensive safety.
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
        for id in &metadata.arg_ids {
            if !seen_inputs.insert(*id) {
                continue;
            }
            let spec = build_buffer_spec(world, *id)?;
            input_specs.push(spec);
            input_ids.push(*id);
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
                specs.push(build_buffer_spec(world, *id)?);
            }
            Some(
                iree_runtime::DeviceArena::new(&session, &specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };

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
            }
        } else {
            for (slot, id) in self.output_ids.iter().enumerate() {
                let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
                self.output_views_scratch[slot]
                    .download_into(&self.session, &mut host.buffer)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
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

fn build_buffer_spec(world: &World, id: ComponentId) -> Result<iree_runtime::BufferSpec, Error> {
    let col = world.column_by_id(id).ok_or(Error::ComponentNotFound)?;
    let element_type = nox_to_iree_element_type(col.schema.element_type())?;
    let shape: Vec<i64> = if world.batch1 && col.len() <= 1 {
        col.schema.shape().iter().map(|&x| x as i64).collect()
    } else {
        std::iter::once(col.len() as i64)
            .chain(col.schema.shape().iter().map(|&x| x as i64))
            .collect()
    };
    Ok(iree_runtime::BufferSpec {
        byte_len: col.column.len(),
        shape,
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
