use std::collections::{HashMap, HashSet};
use std::time::Instant;

use impeller2::types::ComponentId;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::profile::{Profiler, TickTimings};
use crate::world::World;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

pub struct CraneliftExec {
    pub metadata: ExecMetadata,
    tick_fn: TickFn,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    output_keep_mask: Vec<bool>,
    mutable_overlap: Vec<(usize, usize)>,
    output_buffers: Vec<Vec<u8>>,
    dup_scratch: Vec<u8>,
    // Preallocated pointer scratch reused every tick to avoid per-tick
    // `Vec` allocations in the hot loop. Contents are rewritten every
    // invocation, but the allocations themselves are stable.
    input_ptrs: Vec<*const u8>,
    output_ptrs: Vec<*mut u8>,
    _compiled: cranelift_mlir::lower::CompiledModule,
    checkpoint_done: bool,
}

// Safety: `CraneliftExec` is wrapped by `WorldExec` which is stored in
// `PyExec`, a `#[pyclass]` — pyo3 requires pyclasses to be both `Send` and
// `Sync`. `cranelift_jit::JITModule` and the raw `*const/*mut u8` scratch
// pointers do not satisfy these automatically, so we assert them here.
//
// The safety invariants are:
// 1. `Send`: Ownership of the executor is moved once, from the Python-bound
//    thread into the dedicated 256 MB-stack sim runtime thread spawned in
//    `world_builder.rs`. After that move, no other thread accesses it.
// 2. `Sync`: We never actually hand out `&CraneliftExec` to multiple threads.
//    All mutation (scratch buffers, pointer vectors) goes through the `&mut
//    self` methods in `invoke_batch`, and the only method that takes `&self`
//    after finalization — `save_checkpoint_outputs` — reads immutable state.
//    The `JITModule` is read-only once `finalize_definitions` has run (which
//    happens inside `compile_module` before `CraneliftExec::new` returns).
//
// Sharing the raw function pointer across threads is safe because Cranelift
// emits position-independent native code that makes no use of thread-local
// state beyond what its inputs/outputs carry.
unsafe impl Send for CraneliftExec {}
unsafe impl Sync for CraneliftExec {}

impl CraneliftExec {
    pub fn new(
        metadata: ExecMetadata,
        compiled: cranelift_mlir::lower::CompiledModule,
        world: &World,
    ) -> Result<Self, Error> {
        let fn_ptr = compiled.get_main_fn();
        if fn_ptr.is_null() {
            return Err(Error::CraneliftBackend(
                "compiled module is missing a callable main entrypoint".into(),
            ));
        }
        let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

        let mut input_ids = Vec::new();
        let mut seen_inputs = HashSet::new();
        for slot in &metadata.arg_slots {
            if seen_inputs.insert(slot.component_id) {
                input_ids.push(slot.component_id);
            }
        }

        let mut output_ids = Vec::new();
        let mut seen_outputs = HashSet::new();
        let mut output_keep_mask = Vec::with_capacity(metadata.ret_ids.len());
        for id in &metadata.ret_ids {
            let is_new = seen_outputs.insert(*id);
            output_keep_mask.push(is_new);
            if is_new {
                output_ids.push(*id);
            }
        }

        let mut output_slot_by_id = HashMap::new();
        for (slot, id) in output_ids.iter().enumerate() {
            output_slot_by_id.insert(*id, slot);
        }
        let mutable_overlap: Vec<(usize, usize)> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(input_slot, id)| {
                output_slot_by_id
                    .get(id)
                    .copied()
                    .map(|output_slot| (input_slot, output_slot))
            })
            .collect();

        let mut output_buffers = Vec::new();
        let mut max_buf_size = 0usize;
        for id in &output_ids {
            let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
            max_buf_size = max_buf_size.max(col.column.len());
            output_buffers.push(vec![0u8; col.column.len()]);
        }
        let dup_scratch = vec![0u8; max_buf_size.max(1024)];

        let input_ptrs = vec![std::ptr::null(); input_ids.len()];
        let output_ptrs = vec![std::ptr::null_mut(); output_keep_mask.len()];

        Ok(Self {
            metadata,
            tick_fn,
            input_ids,
            output_ids,
            output_keep_mask,
            mutable_overlap,
            output_buffers,
            dup_scratch,
            input_ptrs,
            output_ptrs,
            _compiled: compiled,
            checkpoint_done: false,
        })
    }

    pub fn invoke_batch(
        &mut self,
        world: &mut World,
        n: u64,
        _detailed: bool,
    ) -> Result<TickTimings, Error> {
        let batch_ticks = n.max(1) as usize;

        for batch_idx in 0..batch_ticks {
            for (slot, id) in self.input_ids.iter().enumerate() {
                self.input_ptrs[slot] = world
                    .column_by_id(*id)
                    .ok_or(Error::ComponentNotFound)?
                    .column
                    .as_ptr();
            }

            let mut dedup_idx = 0usize;
            for (slot, keep) in self.output_keep_mask.iter().enumerate() {
                if *keep {
                    self.output_ptrs[slot] = self.output_buffers[dedup_idx].as_mut_ptr();
                    dedup_idx += 1;
                } else {
                    self.output_ptrs[slot] = self.dup_scratch.as_mut_ptr();
                }
            }

            let checkpoint_this_tick = !self.checkpoint_done
                && batch_idx == 0
                && std::env::var("ELODIN_CRANELIFT_DEBUG_DIR").is_ok();
            if checkpoint_this_tick {
                self.save_checkpoint_inputs(world);
            }

            unsafe {
                (self.tick_fn)(self.input_ptrs.as_ptr(), self.output_ptrs.as_mut_ptr());
            }

            if checkpoint_this_tick {
                self.save_checkpoint_outputs();
                self.checkpoint_done = true;
            }

            if batch_idx + 1 < batch_ticks {
                for &(input_slot, output_slot) in &self.mutable_overlap {
                    let id = self.input_ids[input_slot];
                    let host = world.host.get_mut(&id).ok_or(Error::ComponentNotFound)?;
                    let src = &self.output_buffers[output_slot];
                    if host.buffer.len() != src.len() {
                        return Err(Error::ValueSizeMismatch);
                    }
                    host.buffer.copy_from_slice(src);
                }
            }
        }

        for (slot, id) in self.output_ids.iter().enumerate() {
            let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
            let src = &self.output_buffers[slot];
            if host.buffer.len() != src.len() {
                return Err(Error::ValueSizeMismatch);
            }
            host.buffer.copy_from_slice(src);
        }

        Ok(TickTimings::default())
    }
}

impl CraneliftExec {
    fn save_checkpoint_inputs(&self, world: &World) {
        let Ok(dir) = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR") else {
            return;
        };
        let _ = std::fs::create_dir_all(&dir);
        for (i, id) in self.input_ids.iter().enumerate() {
            if let Some(col) = world.column_by_id(*id) {
                let data = &col.column;
                let path = format!("{dir}/input_{i}.bin");
                let _ = std::fs::write(&path, data);
            }
        }
        let mut meta = serde_json::Map::new();
        let mut inputs_meta = Vec::new();
        for (i, id) in self.input_ids.iter().enumerate() {
            let mut m = serde_json::Map::new();
            m.insert("index".into(), serde_json::Value::from(i));
            m.insert("component_id".into(), serde_json::Value::from(id.0));
            if let Some(col) = world.column_by_id(*id) {
                m.insert(
                    "byte_size".into(),
                    serde_json::Value::from(col.column.len()),
                );
            }
            inputs_meta.push(serde_json::Value::Object(m));
        }
        meta.insert("inputs".into(), serde_json::Value::Array(inputs_meta));

        let mut outputs_meta = Vec::new();
        for (i, id) in self.output_ids.iter().enumerate() {
            let mut m = serde_json::Map::new();
            m.insert("index".into(), serde_json::Value::from(i));
            m.insert("component_id".into(), serde_json::Value::from(id.0));
            m.insert(
                "byte_size".into(),
                serde_json::Value::from(self.output_buffers[i].len()),
            );
            outputs_meta.push(serde_json::Value::Object(m));
        }
        meta.insert("outputs".into(), serde_json::Value::Array(outputs_meta));
        meta.insert(
            "num_output_slots".into(),
            serde_json::Value::from(self.metadata.ret_ids.len()),
        );

        let path = format!("{dir}/checkpoint.json");
        if let Ok(json) = serde_json::to_string_pretty(&serde_json::Value::Object(meta)) {
            let _ = std::fs::write(&path, json);
        }
        eprintln!(
            "[elodin-cranelift] checkpoint: saved {} inputs to {dir}",
            self.input_ids.len()
        );
    }

    fn save_checkpoint_outputs(&self) {
        let Ok(dir) = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR") else {
            return;
        };
        for (i, buf) in self.output_buffers.iter().enumerate() {
            let path = format!("{dir}/cranelift_output_{i}.bin");
            let _ = std::fs::write(&path, buf);
        }
        eprintln!(
            "[elodin-cranelift] checkpoint: saved {} outputs to {dir}",
            self.output_buffers.len()
        );
    }
}

pub struct CraneliftWorldExec {
    pub world: World,
    pub tick_exec: CraneliftExec,
    pub startup_exec: Option<CraneliftExec>,
    pub profiler: Profiler,
}

impl CraneliftWorldExec {
    pub fn new(
        world: World,
        tick_exec: CraneliftExec,
        startup_exec: Option<CraneliftExec>,
    ) -> Self {
        Self {
            world,
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        let ticks_per_telemetry = self.world.ticks_per_telemetry();

        // Mirror `JaxWorldExec::run`: consume the startup exec (if any) with a
        // single invocation before entering the steady-state tick loop, so both
        // backends behave identically when a caller wires up a startup graph.
        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.invoke_batch(&mut self.world, 1, self.profiler.detailed_timing)?;
        }

        let tick_start = Instant::now();
        self.tick_exec.invoke_batch(
            &mut self.world,
            ticks_per_telemetry,
            self.profiler.detailed_timing,
        )?;
        let tick_elapsed = tick_start.elapsed();
        self.profiler.execute_buffers.observe_duration(tick_elapsed);

        let mut start = Instant::now();
        for _ in 0..ticks_per_telemetry {
            self.world.advance_tick();
        }
        self.profiler.add_to_history.observe(&mut start);
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(
            self.world.sim_time_step().0,
            self.world.ticks_per_telemetry(),
        )
    }
}
