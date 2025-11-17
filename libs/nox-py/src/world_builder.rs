use crate::*;
use ::s10::{GroupRecipe, SimRecipe, cli::run_recipe};
use clap::Parser;
use convert_case::Casing;
use impeller2::types::{PrimType, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata};
use miette::miette;
use nox_ecs::{ComponentSchema, IntoSystem, System as _, TimeStep, World, increment_sim_tick, nox};
use numpy::{PyArray, PyArrayMethods, ndarray::IntoDimension};
use pyo3::{
    IntoPyObjectExt,
    types::{PyDict, PyList},
    Py, PyAny,
};
use std::{
    collections::{HashMap, HashSet},
    iter,
    net::SocketAddr,
    path::{Path, PathBuf},
    time,
};
use tracing::{error, info};
use zerocopy::{FromBytes, TryFromBytes};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub enum Args {
    Run {
        #[arg(default_value = "[::]:2240")]
        addr: SocketAddr,
        #[arg(long, default_value = "false")]
        no_s10: bool,
        #[arg(long, default_value = None)]
        liveness_port: Option<u16>,
    },
    Plan {
        out_dir: PathBuf,
        #[arg(default_value = "[::]:2240")]
        addr: SocketAddr,
    },
    Components,
    #[clap(hide = true)]
    Bench {
        #[arg(long, default_value = "1000")]
        ticks: usize,
        #[arg(long, default_value = "false")]
        profile: bool,
    },
}

#[pyclass(subclass)]
#[derive(Default)]
pub struct WorldBuilder {
    pub world: World,
    pub recipes: HashMap<String, ::s10::Recipe>,
}

impl WorldBuilder {
    fn sim_recipe(&mut self, path: PathBuf, addr: SocketAddr, optimize: bool) -> ::s10::Recipe {
        let sim = SimRecipe {
            path,
            addr,
            optimize,
        };
        let group = GroupRecipe {
            refs: vec![],
            recipes: self
                .recipes
                .iter()
                .map(|(n, r)| (n.clone(), r.clone()))
                .chain(iter::once(("sim".to_string(), ::s10::Recipe::Sim(sim))))
                .collect(),
        };
        ::s10::Recipe::Group(group)
    }
}

fn is_snake_case(s: &str) -> bool {
    // This may look dumb and it is, but the [`is_case()` implementation][1] is
    // no better and less expressive in that you can't express boundaries.
    //
    // TODO: Don't allocate a string to test a string.
    //
    // [1]: https://github.com/rutrum/convert-case/blob/b9dd92b4394745e15943a604890cdc57fa6bd289/src/lib.rs#L366
    let b = s
        .without_boundaries(&convert_case::Boundary::digits())
        .to_case(convert_case::Case::Snake);
    b == s
}

#[pymethods]
impl WorldBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
    #[pyo3(signature = (spawnable, name=None, id=None))]
    pub fn spawn(
        &mut self,
        spawnable: Spawnable,
        name: Option<String>,
        id: Option<String>,
    ) -> Result<EntityId, Error> {
        let entity_id = EntityId {
            inner: impeller2::types::EntityId(self.world.entity_len()),
        };
        self.insert(entity_id, spawnable)?;
        self.world.metadata.entity_len += 1;
        let derived_id = match (&name, id) {
            (Some(name), None) => {
                let new_id = name
                    .without_boundaries(&convert_case::Boundary::digits())
                    .to_case(convert_case::Case::Snake);
                eprintln!("convert name {:?} to ID {:?}", &name, &new_id);
                info!("convert name {:?} to ID {:?}", &name, &new_id);
                Some(new_id)
            }
            (_, Some(id)) => Some(id),
            _ => None,
        };

        if let Some(derived_id) = derived_id {
            if !is_snake_case(&derived_id) {
                error!("the ID should be snake_case but was {:?}", derived_id);
            }
            self.world.metadata.entity_metadata.insert(
                entity_id.inner,
                EntityMetadata {
                    entity_id: entity_id.inner,
                    // TODO: Consider changing this `name` field to `id`.
                    // Perhaps add a human-readable field `display_name` or
                    // `name` after that.
                    name: derived_id,
                    metadata: Default::default(),
                },
            );
        }
        Ok(entity_id)
    }

    pub fn insert(&mut self, entity_id: EntityId, spawnable: Spawnable) -> Result<(), Error> {
        match spawnable {
            Spawnable::Archetypes(archetypes) => {
                for archetype in archetypes {
                    for (arr, component) in archetype.arrays.iter().zip(archetype.component_data) {
                        let component_id = ComponentId::new(&component.name);
                        let metadata = ComponentMetadata {
                            component_id,
                            name: component.name.clone(),
                            metadata: component.metadata.clone(),
                        };

                        self.world.metadata.component_map.insert(
                            component_id,
                            (ComponentSchema::from(component.clone()), metadata),
                        );
                        let buffer = self.world.host.entry(component_id).or_default();
                        let ty = component.ty.unwrap();
                        let prim_ty: PrimType = ty.ty.into();
                        let size = prim_ty.size();
                        let buf = unsafe { arr.buf(size) };
                        buffer.buffer.extend_from_slice(buf);
                        buffer
                            .entity_ids
                            .extend_from_slice(&entity_id.inner.0.to_le_bytes());
                        self.world.dirty_components.insert(component_id);
                    }
                }
                Ok(())
            }
        }
    }

    fn recipe(&mut self, py: Python<'_>, recipe_obj: PyObject) -> PyResult<()> {
        // Extract PyRecipe instead of Recipe directly
        let pyrecipe = recipe_obj.extract::<crate::s10::PyRecipe>(py)?;
        let name = pyrecipe.name();
        let rust_recipe = pyrecipe.to_json()?;
        // Parse from JSON since we can't access the inner Recipe directly
        let recipe: ::s10::Recipe = serde_json::from_str(&rust_recipe)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse recipe: {}", e)))?;
        self.recipes.insert(name, recipe);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        sys,
        sim_time_step = 1.0 / 120.0,
        run_time_step = None,
        default_playback_speed = 1.0,
        max_ticks = None,
        optimize = false,
        is_canceled = None,
        db_path = None,
    ))]
    pub fn run(
        &mut self,
        py: Python<'_>,
        sys: System,
        sim_time_step: f64,
        run_time_step: Option<f64>,
        default_playback_speed: f64,
        max_ticks: Option<u64>,
        optimize: bool,
        is_canceled: Option<PyObject>,
        db_path: Option<String>,
    ) -> Result<Option<String>, Error> {
        let _ = tracing_subscriber::fmt::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive("info".parse().expect("invalid filter"))
                    .from_env_lossy(),
            )
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
                "%Y-%m-%d %H:%M:%S%.3f".to_string(),
            ))
            .try_init();

        let args = py
            .import("sys")?
            .getattr("argv")?
            .extract::<Vec<String>>()?;
        let path = args.first().ok_or(Error::MissingArg("path".to_string()))?;
        let path = PathBuf::from(path);
        let args = Args::parse_from(args);

        match args {
            Args::Run {
                addr,
                no_s10,
                liveness_port,
            } => {
                let exec = self.build_uncompiled(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    default_playback_speed,
                    max_ticks,
                )?;
                let mut client = nox::Client::cpu()?;
                if !optimize {
                    client.disable_optimizations();
                }
                let recipes = self.recipes.clone();
                if !no_s10 {
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .map_err(|err| miette!("rt err {}", err))
                            .unwrap();
                        let group = ::s10::Recipe::Group(GroupRecipe {
                            recipes,
                            ..Default::default()
                        });
                        rt.block_on(run_recipe("world".to_string(), group, false, false))
                            .unwrap();
                    });
                }
                let exec = exec.compile(client.clone())?;
                if let Some(port) = liveness_port {
                    stellarator::struc_con::stellar(move || ::s10::liveness::monitor(port));
                }

                let db_path = match db_path {
                    Some(p) => PathBuf::from(p),
                    None => tempfile::tempdir()?.keep().join("db"),
                };
                py.allow_threads(|| {
                    stellarator::run(|| {
                        nox_ecs::impeller2_server::Server::new(
                            // Here we start the DB with an address.
                            elodin_db::Server::new(db_path, addr).unwrap(),
                            exec,
                        )
                        .run_with_cancellation({
                            move || {
                                if let Some(ref func) = is_canceled {
                                    Python::with_gil(|py| {
                                        func.call0(py)
                                            .and_then(|result| result.extract::<bool>(py))
                                            .unwrap_or_else(|_| {
                                                // If the function raises an exception or returns non-bool, treat as cancel
                                                py.check_signals().is_err()
                                            })
                                    })
                                } else {
                                    Python::with_gil(|py| py.check_signals().is_err())
                                }
                            }
                        })
                    })?;

                    Ok(None)
                })
            }
            Args::Plan { addr, out_dir } => {
                let recipe = self.sim_recipe(path, addr, optimize);
                let toml = toml::to_string_pretty(&recipe)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let plan_path = out_dir.join("s10.toml");
                std::fs::write(&plan_path, toml)?;
                Ok(None)
            }
            Args::Bench {
                ticks,
                profile: enable_profiling,
            } => {
                if !enable_profiling {
                    // Standard bench mode - just run and show timing
                    let mut exec = self.build(
                        py,
                        sys,
                        sim_time_step,
                        run_time_step,
                        default_playback_speed,
                        max_ticks,
                        optimize,
                        db_path,
                    )?;
                    exec.run(py, ticks, true, None)?;
                    let profile = exec.profile();
                    println!("copy_to_client time:  {:.3} ms", profile["copy_to_client"]);
                    println!("execute_buffers time: {:.3} ms", profile["execute_buffers"]);
                    println!("copy_to_host time:    {:.3} ms", profile["copy_to_host"]);
                    println!("add_to_history time:  {:.3} ms", profile["add_to_history"]);
                    println!("= tick time:          {:.3} ms", profile["tick"]);
                    println!("build time:           {:.3} ms", profile["build"]);
                    println!("compile time:         {:.3} ms", profile["compile"]);
                    println!("real_time_factor:     {:.3}", profile["real_time_factor"]);
                    Ok(None)
                } else {
                    // Profiling mode enabled - run full analysis with DOT graph export

                    // Set up output directory near the simulation file
                    let sim_file_dir = path.parent().unwrap_or_else(|| Path::new("."));
                    let output_dir = sim_file_dir.join("profile_output");
                    std::fs::create_dir_all(&output_dir)?;

                    // Set XLA flags for DOT graph output
                    let dot_dir = output_dir.join("graphs");
                    std::fs::create_dir_all(&dot_dir)?;

                    // Save existing XLA_FLAGS to restore later
                    let previous_xla_flags = std::env::var("XLA_FLAGS").ok();
                    let new_flags =
                        format!("--xla_dump_to={} --xla_dump_hlo_as_dot", dot_dir.display());

                    // Combine with existing flags if present
                    let combined_flags = if let Some(ref prev) = previous_xla_flags {
                        format!("{} {}", prev, new_flags)
                    } else {
                        new_flags
                    };

                    unsafe {
                        std::env::set_var("XLA_FLAGS", &combined_flags);
                    }

                    // Guard to restore XLA_FLAGS on scope exit (even on error)
                    struct XlaFlagsGuard(Option<String>);
                    impl Drop for XlaFlagsGuard {
                        fn drop(&mut self) {
                            unsafe {
                                match &self.0 {
                                    Some(prev) => std::env::set_var("XLA_FLAGS", prev),
                                    None => std::env::remove_var("XLA_FLAGS"),
                                }
                            }
                        }
                    }
                    let _xla_guard = XlaFlagsGuard(previous_xla_flags);

                    // Build and compile
                    let exec = self.build_uncompiled(
                        py,
                        sys,
                        sim_time_step,
                        run_time_step,
                        default_playback_speed,
                        max_ticks,
                    )?;

                    let mut client = nox::Client::cpu()?;
                    if !optimize {
                        client.disable_optimizations();
                    }

                    let build_time_ms = exec.profiler.build.mean();

                    let compile_start = time::Instant::now();
                    let mut compiled_exec = exec.compile(client)?;
                    let compile_time_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

                    // XLA_FLAGS automatically restored when _xla_guard is dropped here
                    drop(_xla_guard);

                    // Rename DOT files to simpler names while preserving module numbers
                    if let Ok(entries) = std::fs::read_dir(&dot_dir) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let path = entry.path();
                            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                // Extract module number if present (e.g., "module_0001" -> "0001")
                                let module_num = filename
                                    .strip_prefix("module_")
                                    .and_then(|s| s.split('.').next())
                                    .unwrap_or("");

                                // Simplify: module_0001.jit__unnamed_wrapped_function_.cpu_after_optimizations.dot
                                // → module_0001_cpu_after_optimizations.dot (or cpu_after_optimizations.dot if no module num)
                                let new_name = if filename.contains("before_optimizations")
                                    && filename.ends_with(".dot")
                                {
                                    if module_num.is_empty() {
                                        "before_optimizations.dot".to_string()
                                    } else {
                                        format!("module_{}_before_optimizations.dot", module_num)
                                    }
                                } else if filename.contains("cpu_after_optimizations")
                                    && filename.ends_with(".dot")
                                {
                                    if module_num.is_empty() {
                                        "cpu_after_optimizations.dot".to_string()
                                    } else {
                                        format!("module_{}_cpu_after_optimizations.dot", module_num)
                                    }
                                } else if filename.contains("ir-no-opt.ll") {
                                    if module_num.is_empty() {
                                        "llvm_ir_before_opt.ll".to_string()
                                    } else {
                                        format!("module_{}_llvm_ir_before_opt.ll", module_num)
                                    }
                                } else if filename.contains("ir-with-opt.ll") {
                                    if module_num.is_empty() {
                                        "llvm_ir_after_opt.ll".to_string()
                                    } else {
                                        format!("module_{}_llvm_ir_after_opt.ll", module_num)
                                    }
                                } else if filename.ends_with(".o") {
                                    if module_num.is_empty() {
                                        "compiled_module.o".to_string()
                                    } else {
                                        format!("module_{}_compiled.o", module_num)
                                    }
                                } else {
                                    continue; // Keep other files as-is
                                };

                                let new_path = dot_dir.join(new_name);
                                let _ = std::fs::rename(&path, &new_path);
                            }
                        }
                    }

                    // Get HLO text and save to file
                    let hlo_text = compiled_exec
                        .tick_exec
                        .hlo_module()
                        .computation()
                        .to_hlo_text()
                        .map_err(|e| {
                            Error::NoxEcs(nox_ecs::Error::Nox(nox_ecs::nox::Error::Xla(e)))
                        })?;

                    // Save HLO dump to output directory
                    let hlo_dump_path = output_dir.join("hlo_dump.txt");
                    std::fs::write(&hlo_dump_path, &hlo_text)?;

                    // === FULL PROFILING ANALYSIS ===

                    // Parse and analyze HLO
                    let mut op_counts = HashMap::<String, usize>::new();
                    let mut instruction_count = 0;
                    let mut op_details = HashMap::<String, Vec<String>>::new();
                    let mut source_line_ops = HashMap::<String, Vec<String>>::new();

                    // Build location map
                    let mut loc_map = HashMap::<String, String>::new();
                    let mut fused_map = HashMap::<String, Vec<String>>::new();

                    for line in hlo_text.lines() {
                        let trimmed = line.trim();

                        // Parse direct location definitions
                        if trimmed.starts_with("#loc") && trimmed.contains(" = loc(\"") {
                            if let Some(eq_pos) = trimmed.find(" = loc(\"") {
                                let loc_id = &trimmed[..eq_pos];
                                let after_eq = &trimmed[eq_pos + 8..];
                                if let Some(quote_end) = after_eq.find('"') {
                                    let file_path = &after_eq[..quote_end];
                                    let after_quote = &after_eq[quote_end + 1..];
                                    if file_path.contains(".py") && after_quote.starts_with(':') {
                                        let line_col = &after_quote[1..];
                                        if let Some(colon_pos) = line_col.find(':') {
                                            let line_num = &line_col[..colon_pos];
                                            let filename = if let Some(slash) = file_path.rfind('/')
                                            {
                                                &file_path[slash + 1..]
                                            } else {
                                                file_path
                                            };
                                            loc_map.insert(
                                                loc_id.to_string(),
                                                format!("{}:{}", filename, line_num),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        // Parse fused locations
                        else if trimmed.starts_with("#loc")
                            && trimmed.contains(" = loc(fused[")
                            && let Some(eq_pos) = trimmed.find(" = loc(fused[")
                        {
                            let loc_id = &trimmed[..eq_pos];
                            let after_fused = &trimmed[eq_pos + 13..];
                            if let Some(bracket_end) = after_fused.find("])") {
                                let locs_str = &after_fused[..bracket_end];
                                let fused_locs: Vec<String> =
                                    locs_str.split(',').map(|s| s.trim().to_string()).collect();
                                fused_map.insert(loc_id.to_string(), fused_locs);
                            }
                        }
                    }

                    // Resolve fused locations
                    let mut fused_resolutions = Vec::new();
                    for (fused_id, constituent_locs) in &fused_map {
                        for constituent_loc in constituent_locs {
                            if let Some(resolved) = loc_map.get(constituent_loc) {
                                fused_resolutions.push((fused_id.clone(), resolved.clone()));
                                break;
                            }
                        }
                    }
                    for (fused_id, resolved) in fused_resolutions {
                        loc_map.insert(fused_id, resolved);
                    }

                    // Complexity estimation helpers
                    let parse_tensor_dims = |text: &str| -> Vec<u64> {
                        let mut dims = Vec::new();
                        if let Some(tensor_start) = text.find("tensor<")
                            && let Some(tensor_end) = text[tensor_start..].find('>')
                        {
                            let tensor_spec = &text[tensor_start + 7..tensor_start + tensor_end];
                            for part in tensor_spec.split('x') {
                                if let Ok(dim) = part.parse::<u64>() {
                                    dims.push(dim);
                                } else {
                                    break;
                                }
                            }
                        }
                        dims
                    };

                    let estimate_op_complexity = |op_name: &str, hlo_line: &str| -> u64 {
                        let dims = parse_tensor_dims(hlo_line);
                        let tensor_size: u64 = if dims.is_empty() {
                            1
                        } else {
                            dims.iter().product()
                        };

                        let base_cost = if op_name.contains("dot_general") {
                            return tensor_size * 10;
                        } else if op_name.contains("dot") {
                            return tensor_size * 2;
                        } else if op_name.contains("convolution") {
                            return tensor_size * 20;
                        } else if op_name.contains("reduce") {
                            return tensor_size * 2;
                        } else if op_name.contains("sin")
                            || op_name.contains("cos")
                            || op_name.contains("exp")
                            || op_name.contains("log")
                            || op_name.contains("sqrt")
                            || op_name.contains("rsqrt")
                        {
                            return tensor_size * 10;
                        } else if op_name.contains("divide") {
                            5
                        } else if op_name == "call"
                            || op_name.contains("gather")
                            || op_name.contains("scatter")
                        {
                            3
                        } else if op_name.contains("select") {
                            2
                        } else if op_name.contains("add")
                            || op_name.contains("subtract")
                            || op_name.contains("multiply")
                            || op_name.contains("compare")
                            || op_name.contains("negate")
                        {
                            1
                        } else if op_name.contains("reshape")
                            || op_name.contains("transpose")
                            || op_name.contains("broadcast")
                            || op_name.contains("constant")
                            || op_name.contains("convert")
                        {
                            0
                        } else {
                            1
                        };

                        base_cost * tensor_size
                    };

                    // Analyze HLO: count ops and calculate complexity
                    let mut source_line_complexity = HashMap::<String, u64>::new();
                    let mut op_complexity = HashMap::<String, u64>::new();
                    let mut total_complexity = 0u64;

                    for line in hlo_text.lines() {
                        let trimmed = line.trim();
                        if trimmed.starts_with('%') {
                            instruction_count += 1;
                            if let Some(eq_pos) = trimmed.find('=') {
                                let after_eq = &trimmed[eq_pos + 1..].trim();
                                if let Some(op_name) = after_eq.split_whitespace().next() {
                                    *op_counts.entry(op_name.to_string()).or_insert(0) += 1;

                                    let complexity = estimate_op_complexity(op_name, trimmed);
                                    *op_complexity.entry(op_name.to_string()).or_insert(0) +=
                                        complexity;
                                    total_complexity += complexity;

                                    op_details
                                        .entry(op_name.to_string())
                                        .or_default()
                                        .push(trimmed.to_string());

                                    // Extract source location
                                    if let Some(loc_ref_start) = trimmed.rfind("loc(#loc")
                                        && let Some(loc_ref_end) =
                                            trimmed[loc_ref_start..].find(')')
                                    {
                                        let loc_ref = &trimmed
                                            [loc_ref_start + 4..loc_ref_start + loc_ref_end];
                                        if let Some(source_loc) = loc_map.get(loc_ref) {
                                            source_line_ops
                                                .entry(source_loc.clone())
                                                .or_default()
                                                .push(op_name.to_string());
                                            *source_line_complexity
                                                .entry(source_loc.clone())
                                                .or_insert(0) += complexity;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Sort operations
                    let mut op_vec: Vec<_> = op_counts.iter().collect();
                    op_vec.sort_by(|a, b| b.1.cmp(a.1));
                    let mut op_complexity_vec: Vec<_> = op_complexity.iter().collect();
                    op_complexity_vec.sort_by(|a, b| b.1.cmp(a.1));

                    // Analyze components memory
                    let element_size = |prim_type: impeller2::types::PrimType| -> usize {
                        match prim_type {
                            impeller2::types::PrimType::Bool
                            | impeller2::types::PrimType::U8
                            | impeller2::types::PrimType::I8 => 1,
                            impeller2::types::PrimType::U16 | impeller2::types::PrimType::I16 => 2,
                            impeller2::types::PrimType::U32
                            | impeller2::types::PrimType::I32
                            | impeller2::types::PrimType::F32 => 4,
                            impeller2::types::PrimType::U64
                            | impeller2::types::PrimType::I64
                            | impeller2::types::PrimType::F64 => 8,
                        }
                    };

                    let mut input_memory_bytes = 0usize;
                    let mut component_memory: Vec<(String, usize)> = Vec::new();

                    for id in &compiled_exec.tick_exec.metadata().arg_ids {
                        if let Some(col) = compiled_exec.world.column_by_id(*id) {
                            let shape_size: usize =
                                col.schema.shape().iter().map(|&x| x as usize).product();
                            let elem_size = element_size(col.schema.prim_type);
                            let mem_bytes = col.len() * shape_size * elem_size;
                            input_memory_bytes += mem_bytes;
                            component_memory.push((col.metadata.name.clone(), mem_bytes));
                        }
                    }

                    component_memory.sort_by(|a, b| b.1.cmp(&a.1));
                    let input_memory_kb = input_memory_bytes as f64 / 1024.0;

                    // Print analysis
                    println!("\n=== SYSTEM PROFILE ===");
                    println!("\n[Compilation]");
                    println!("  Build time:        {:.3} ms", build_time_ms);
                    println!("  Compile time:      {:.3} ms", compile_time_ms);

                    println!("\n[HLO Analysis]");
                    println!("  Total instructions: {}", instruction_count);
                    println!("  HLO text dump:      {}", hlo_dump_path.display());

                    // Check for DOT graph files
                    if let Ok(entries) = std::fs::read_dir(&dot_dir) {
                        let dot_files: Vec<_> = entries
                            .filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| p.extension().map(|ext| ext == "dot").unwrap_or(false))
                            .collect();

                        if !dot_files.is_empty() {
                            println!(
                                "  Graph visualization: {} ({} DOT file{})",
                                dot_dir.display(),
                                dot_files.len(),
                                if dot_files.len() == 1 { "" } else { "s" }
                            );

                            // Show example viewing command for the first optimized graph found
                            let example_file = dot_files
                                .iter()
                                .find(|p| {
                                    p.file_name()
                                        .and_then(|n| n.to_str())
                                        .map(|n| n.contains("after_optimizations"))
                                        .unwrap_or(false)
                                })
                                .or_else(|| dot_files.first());

                            if let Some(file) = example_file {
                                println!("    View with: xdot {}", file.display());
                                println!(
                                    "    Or render: dot -Tpng {} -o graph.png",
                                    file.display()
                                );
                            }
                        }
                    }

                    println!("\n[Operation Breakdown]");
                    println!("  By Complexity (estimated FLOPs):");
                    for (op, complexity) in op_complexity_vec.iter().take(10) {
                        let count = op_counts.get(*op).unwrap_or(&0);
                        let complexity_pct =
                            (**complexity as f64 / total_complexity as f64) * 100.0;
                        let count_pct = (*count as f64 / instruction_count as f64) * 100.0;
                        println!(
                            "  {:20} {:6} FLOPs ({:5.1}%) - {:4} ops ({:4.1}%)",
                            op, complexity, complexity_pct, count, count_pct
                        );
                    }
                    println!("\n  Total estimated FLOPs: {}", total_complexity);

                    println!("\n[Hot Spots by Complexity]");
                    if !source_line_ops.is_empty() {
                        let mut source_heat_map: Vec<_> = source_line_ops
                            .iter()
                            .map(|(loc, ops)| {
                                let op_count = ops.len();
                                let complexity =
                                    source_line_complexity.get(loc).copied().unwrap_or(0);
                                (loc, op_count, complexity)
                            })
                            .collect();
                        source_heat_map.sort_by(|a, b| b.2.cmp(&a.2));

                        println!("Top Python lines by computational cost:");
                        let sim_dir = path.parent().unwrap_or_else(|| Path::new("."));
                        let mut file_contents_cache = HashMap::<String, Vec<String>>::new();

                        for (source_loc, op_count, complexity) in source_heat_map.iter().take(10) {
                            let complexity_pct =
                                (*complexity as f64 / total_complexity as f64) * 100.0;
                            println!(
                                "\n  {} - {} ops, {} FLOPs ({:.1}%)",
                                source_loc, op_count, complexity, complexity_pct
                            );

                            if let Some(colon_pos) = source_loc.rfind(':') {
                                let file_name = &source_loc[..colon_pos];
                                let line_num_str = &source_loc[colon_pos + 1..];

                                if let Ok(line_num) = line_num_str.parse::<usize>() {
                                    let lines = file_contents_cache
                                        .entry(file_name.to_string())
                                        .or_insert_with(|| {
                                            let file_path = sim_dir.join(file_name);
                                            std::fs::read_to_string(&file_path)
                                                .ok()
                                                .map(|content| {
                                                    content.lines().map(String::from).collect()
                                                })
                                                .unwrap_or_default()
                                        });

                                    if line_num > 0 && line_num <= lines.len() {
                                        let code_line = lines[line_num - 1].trim();
                                        if !code_line.is_empty() {
                                            println!("    Code: {}", code_line);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    println!("\n[Top Components by Memory]");
                    for (name, bytes) in component_memory.iter().take(10) {
                        let kb = *bytes as f64 / 1024.0;
                        println!(
                            "  {:40} {:8.2} KB ({:.1}%)",
                            name,
                            kb,
                            (kb / input_memory_kb) * 100.0
                        );
                    }

                    // Runtime analysis follows
                    let db_dir = tempfile::tempdir()?;
                    let db_dir_path = db_dir.keep();
                    let db = elodin_db::DB::create(db_dir_path.join("db"))?;
                    nox_ecs::impeller2_server::init_db(
                        &db,
                        &mut compiled_exec.world,
                        impeller2::types::Timestamp::now(),
                    )?;

                    let mut exec_with_db = Exec {
                        exec: compiled_exec,
                        db: Box::new(db),
                    };
                    exec_with_db.run(py, ticks, true, None)?;
                    let profile = exec_with_db.profile();

                    println!("\n[Runtime Metrics]");
                    println!("copy_to_client time:  {:.3} ms", profile["copy_to_client"]);
                    println!("execute_buffers time: {:.3} ms", profile["execute_buffers"]);
                    println!("copy_to_host time:    {:.3} ms", profile["copy_to_host"]);
                    println!("build time:           {:.3} ms", profile["build"]);
                    println!("compile time:         {:.3} ms", profile["compile"]);
                    println!("real_time_factor:     {:.3}", profile["real_time_factor"]);

                    Ok(None)
                }
            }
            Args::Components => {
                // Discover components and entities without running the simulation
                let discovery_result = self.discover_components(py)?;

                // Convert to JSON and print
                let json_module = py.import("json")?;
                let json_str = json_module.call_method1("dumps", (discovery_result,))?;
                println!("{}", json_str.extract::<String>()?);

                Ok(None)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        system,
        sim_time_step = 1.0 / 120.0,
        run_time_step = None,
        default_playback_speed = 1.0,
        max_ticks = None,
        optimize = false,
        db_path = None,
    ))]
    pub fn build(
        &mut self,
        py: Python<'_>,
        system: System,
        sim_time_step: f64,
        run_time_step: Option<f64>,
        default_playback_speed: f64,
        max_ticks: Option<u64>,
        optimize: bool,
        db_path: Option<String>,
    ) -> Result<Exec, Error> {
        let exec = self.build_uncompiled(
            py,
            system,
            sim_time_step,
            run_time_step,
            default_playback_speed,
            max_ticks,
        )?;
        let mut client = nox::Client::cpu()?;
        if !optimize {
            client.disable_optimizations();
        }
        let mut exec = exec.compile(client.clone())?;
        let db_path = match db_path {
            Some(p) => PathBuf::from(p),
            None => tempfile::tempdir()?.keep().join("db"),
        };
        let db = elodin_db::DB::create(db_path)?;
        nox_ecs::impeller2_server::init_db(&db, &mut exec.world, Timestamp::now())?;
        Ok(Exec { exec, db: Box::new(db) })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (
        system,
        sim_time_step = 1.0 / 120.0,
        run_time_step = None,
        default_playback_speed = 1.0,
        max_ticks = None,
    ))]
    pub fn to_jax_func(
        &mut self,
        py: Python<'_>,
        system: System,
        sim_time_step: f64,
        run_time_step: Option<f64>,
        default_playback_speed: f64,
        max_ticks: Option<u64>,
    ) -> Result<
        (
            Py<PyAny>,
            Vec<u64>,
            Vec<u64>,
            Vec<Py<PyAny>>,
            Py<PyAny>,
            Py<PyAny>,
            Py<PyAny>,
        ),
        Error,
    > {
        let (python, in_id, out_id, state, dict, entity_dict, component_entity_dict) = self
            .build_jited(
                py,
                system,
                sim_time_step,
                run_time_step,
                default_playback_speed,
                max_ticks,
            )?;
        Ok((
            python,
            in_id,
            out_id,
            state,
            dict,
            entity_dict,
            component_entity_dict,
        ))
    }

    pub fn get_dict(&mut self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let dict = PyDict::new(py);
        for id in self.world.host.keys() {
            let component = self.world.column_by_id(*id).unwrap();
            let comp_name = component.metadata.name.clone();
            dict.set_item(comp_name, id.0)?;
        }
        Ok(dict.into_py_any(py)?)
    }

    /// Set the schematic to represent UI and visualization.
    ///
    /// If a `path` is given and the file exists, the file's contents will be
    /// used as the schematic.
    ///
    /// In all other cases, `default_content` is used as the schematic.
    ///
    /// Primarily this affords the code a means of specifying a default
    /// schematic and location for saving custom schematics. It is expected that
    /// the user may make changes and save the schematic to the given path, but
    /// this function itself does not write to the `path`.
    #[pyo3(signature = (default_content = None, path = None,))]
    pub fn schematic(&mut self, default_content: Option<String>, path: Option<String>) {
        self.world.metadata.schematic_path =
            // Don't use the ELODIN_KDL_DIR path here. We use that when we read
            // or write to the filesystem.
            //
            // path.map(|p| impeller2_kdl::env::schematic_file(&Path::new(&p)));
            path.map(PathBuf::from);
        let file_contents = self
            .world
            .metadata
            .schematic_path
            .as_ref()
            .map(|p| impeller2_kdl::env::schematic_file(Path::new(p)))
            .and_then(|path| {
                if path.exists() {
                    std::fs::read_to_string(&path)
                        .inspect(|_| info!("read schematic at {:?}", path.display()))
                        .inspect_err(|err| {
                            error!(
                                ?err,
                                "could not read schematic file at {:?}",
                                path.display()
                            )
                        })
                        .ok()
                } else {
                    None
                }
            });
        self.world.metadata.schematic = file_contents.or(default_content);
    }

    pub fn discover_components(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        // Create lists for components and entities
        let components = PyList::empty(py);
        let entities = PyList::empty(py);

        // Build a map of entity_id -> set of component names (using HashSet to avoid duplicates)
        let mut entity_components: HashMap<impeller2::types::EntityId, HashSet<String>> =
            HashMap::new();

        // Iterate through all components in the world
        for (component_id, (schema, metadata)) in &self.world.metadata.component_map {
            // Track which entities have this component
            if let Some(buffer) = self.world.host.get(component_id) {
                // Validate that entity_ids buffer is properly aligned
                if buffer.entity_ids.len() % 8 != 0 {
                    tracing::warn!(
                        "Component '{}' has misaligned entity_ids buffer (size: {}). \
                         Some entities may be missing from discovery.",
                        metadata.name,
                        buffer.entity_ids.len()
                    );
                }

                // Process entity IDs - using chunks_exact to ensure we only process complete IDs
                for chunk in buffer.entity_ids.chunks_exact(8) {
                    let entity_id = u64::from_le_bytes(chunk.try_into().unwrap());
                    let entity_id = impeller2::types::EntityId(entity_id);
                    entity_components
                        .entry(entity_id)
                        .or_default()
                        .insert(metadata.name.clone());
                }

                // Check if there's a remainder (incomplete ID) that was skipped
                let remainder = buffer.entity_ids.chunks_exact(8).remainder();
                if !remainder.is_empty() {
                    tracing::warn!(
                        "Component '{}' has {} bytes of incomplete entity ID data",
                        metadata.name,
                        remainder.len()
                    );
                }
            }

            // Extract type information
            let type_str = match schema.prim_type {
                PrimType::U8 => "u8",
                PrimType::U16 => "u16",
                PrimType::U32 => "u32",
                PrimType::U64 => "u64",
                PrimType::I8 => "i8",
                PrimType::I16 => "i16",
                PrimType::I32 => "i32",
                PrimType::I64 => "i64",
                PrimType::Bool => "bool",
                PrimType::F32 => "f32",
                PrimType::F64 => "f64",
            };

            // Extract shape if it's a tensor
            let shape_vec = schema.shape();
            let shape = if !shape_vec.is_empty() {
                Some(shape_vec.to_vec())
            } else {
                None
            };

            // Convert metadata to Python dict
            let py_metadata = PyDict::new(py);
            for (key, value) in &metadata.metadata {
                py_metadata.set_item(key, value)?;
            }

            let component_name = metadata.name.clone();

            // Create component dictionary for JSON
            let component_dict = PyDict::new(py);
            component_dict.set_item("name", component_name)?;
            component_dict.set_item("type", type_str)?;
            if let Some(shape) = shape {
                component_dict.set_item("shape", shape)?;
            }
            if !metadata.metadata.is_empty() {
                component_dict.set_item("metadata", py_metadata)?;
            }

            components.append(component_dict)?;
        }

        // Create entity dictionaries for JSON
        for (entity_id, entity_meta) in &self.world.metadata.entity_metadata {
            let entity_dict = PyDict::new(py);
            entity_dict.set_item("id", entity_id.0)?;
            entity_dict.set_item("name", &entity_meta.name)?;

            // Get components for this entity
            if let Some(component_names) = entity_components.get(entity_id) {
                // Convert HashSet to Vec for JSON serialization
                let component_list: Vec<String> = component_names.iter().cloned().collect();
                entity_dict.set_item("components", component_list)?;
            } else {
                entity_dict.set_item("components", Vec::<String>::new())?;
            }

            entities.append(entity_dict)?;
        }

        // Also include entities without metadata but with components
        for (entity_id, component_names) in &entity_components {
            if !self.world.metadata.entity_metadata.contains_key(entity_id) {
                let entity_dict = PyDict::new(py);
                entity_dict.set_item("id", entity_id.0)?;
                entity_dict.set_item("name", format!("entity_{}", entity_id.0))?;
                // Convert HashSet to Vec for JSON serialization
                let component_list: Vec<String> = component_names.iter().cloned().collect();
                entity_dict.set_item("components", component_list)?;

                entities.append(entity_dict)?;
            }
        }

        // Create result dictionary
        let result = PyDict::new(py);
        result.set_item("components", components)?;
        result.set_item("entities", &entities)?;
        result.set_item("total_components", self.world.metadata.component_map.len())?;
        result.set_item("total_entities", entities.len())?;

        Ok(result.into_py_any(py)?)
    }
}

impl WorldBuilder {
    fn build_uncompiled(
        &mut self,
        py: Python<'_>,
        sys: System,
        sim_time_step: f64,
        run_time_step: Option<f64>,
        default_playback_speed: f64,
        max_ticks: Option<u64>,
    ) -> Result<nox_ecs::WorldExec, Error> {
        let mut start = time::Instant::now();
        let ts = time::Duration::from_secs_f64(sim_time_step);
        self.world.metadata.sim_time_step = TimeStep(ts);
        self.world.metadata.default_playback_speed = default_playback_speed;
        if let Some(ts) = run_time_step {
            let ts = time::Duration::from_secs_f64(ts);
            self.world.metadata.run_time_step = Some(TimeStep(ts));
        }
        if let Some(max_ticks) = max_ticks {
            self.world.metadata.max_tick = max_ticks;
        }

        self.world.set_globals();

        let world = std::mem::take(&mut self.world);
        let xla_exec = increment_sim_tick.pipe(sys).compile(&world)?;
        let tick_exec = xla_exec.compile_hlo_module(py, &world)?;

        let mut exec = nox_ecs::WorldExec::new(world, tick_exec, None);
        exec.profiler.build.observe(&mut start);
        Ok(exec)
    }

    #[allow(clippy::type_complexity)]
    fn build_jited(
        &mut self,
        py: Python<'_>,
        sys: System,
        sim_time_step: f64,
        run_time_step: Option<f64>,
        default_playback_speed: f64,
        max_ticks: Option<u64>,
    ) -> Result<
        (
            Py<PyAny>,
            Vec<u64>,
            Vec<u64>,
            Vec<Py<PyAny>>,
            Py<PyAny>,
            Py<PyAny>,
            Py<PyAny>,
        ),
        Error,
    > {
        let ts = time::Duration::from_secs_f64(sim_time_step);
        self.world.metadata.sim_time_step = TimeStep(ts);
        self.world.metadata.default_playback_speed = default_playback_speed;
        if let Some(ts) = run_time_step {
            let ts = time::Duration::from_secs_f64(ts);
            self.world.metadata.run_time_step = Some(TimeStep(ts));
        }
        if let Some(max_ticks) = max_ticks {
            self.world.metadata.max_tick = max_ticks;
        }

        let mut input_id = Vec::<u64>::new();
        let mut output_id = Vec::<u64>::new();
        let mut state = Vec::<Py<PyAny>>::new();
        let dict = PyDict::new(py);
        let entity_dict = PyDict::new(py);
        let component_entity_dict = PyDict::new(py);
        self.world.set_globals();

        let world = std::mem::take(&mut self.world);
        let xla_exec = sys.compile(&world)?;

        for out_id in xla_exec.outputs.iter() {
            output_id.push(out_id.0);
        }

        let world_meta = world.entity_metadata();
        for (id, meta) in world_meta {
            entity_dict.set_item(&meta.name, id.0)?;
        }

        for id in xla_exec.inputs.iter() {
            input_id.push(id.0);
            let component = world.column_by_id(*id).unwrap();
            let schema = component.schema;
            let data = component.column;
            let mut dim = schema.dim.to_vec();
            dim.insert(0, component.entities.len() / 8);

            let comp_name = component.metadata.name.clone();
            dict.set_item(id.0, &comp_name)?;

            let mut entity_vec = Vec::<u64>::new();
            for entity_id in component.entity_ids() {
                entity_vec.push(entity_id.0);
            }

            component_entity_dict.set_item(&comp_name, entity_vec)?;

            match schema.prim_type {
                PrimType::U8 => {
                    let slice = <[u8]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::U16 => {
                    let slice = <[u16]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::U32 => {
                    let slice = <[u32]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::U64 => {
                    let slice = <[u64]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::I8 => {
                    let slice = <[i8]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::I16 => {
                    let slice = <[i16]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::I32 => {
                    let slice = <[i32]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::I64 => {
                    let slice = <[i64]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::F32 => {
                    let slice = <[f32]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::F64 => {
                    let slice = <[f64]>::ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
                PrimType::Bool => {
                    let slice = <[bool]>::try_ref_from_bytes(data).unwrap();
                    let py_array = PyArray::from_slice(py, slice)
                        .reshape(dim.into_dimension())
                        .unwrap();

                    state.push(py_array.into_py_any(py)?);
                }
            };
        }

        let jax_exec = xla_exec.compile_jax_module(py)?;
        let dictionary = dict.into_py_any(py)?;
        let entity_dict = entity_dict.into_py_any(py)?;
        let component_entity_dict = component_entity_dict.into_py_any(py)?;

        Ok((
            jax_exec,
            input_id,
            output_id,
            state,
            dictionary,
            entity_dict,
            component_entity_dict,
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use convert_case::Casing;

    #[test]
    fn test_snake_case() {
        assert!(!"e1".is_case(convert_case::Case::Snake));
        // We can't express this:
        // assert!("e1"
        //         .without_boundaries(&convert_case::Boundary::digits())
        //         .is_case(convert_case::Case::Snake));
        assert!("e_1".is_case(convert_case::Case::Snake));

        assert_eq!(
            "e1".without_boundaries(&convert_case::Boundary::digits())
                .to_case(convert_case::Case::Snake),
            String::from("e1")
        );
    }

    #[test]
    fn test_our_snake_case() {
        assert!(is_snake_case("e1"));
        assert!(is_snake_case("e_1"));
        assert!(!is_snake_case("E1"));
    }
}
