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
    },
    Profile {
        #[arg(long, default_value = "0")]
        ticks: usize,
        #[arg(long, default_value = "false")]
        deep: bool,
        #[arg(long, default_value = "false")]
        html: bool,
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
                py.allow_threads(|| {
                    stellarator::run(|| {
                        let tmpfile = tempfile::tempdir().unwrap().keep();
                        nox_ecs::impeller2_server::Server::new(
                            elodin_db::Server::new(tmpfile.join("db"), addr).unwrap(),
                            exec,
                        )
                        .run_with_cancellation(|| {
                            Python::with_gil(|py| py.check_signals().is_err())
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
            Args::Bench { ticks } => {
                let mut exec = self.build(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    default_playback_speed,
                    max_ticks,
                    optimize,
                )?;
                exec.run(py, ticks, true)?;
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
            Args::Profile { ticks, deep, html } => {
                // Set up output directory near the simulation file
                let sim_file_dir = path.parent().unwrap_or_else(|| Path::new("."));
                let output_dir = sim_file_dir.join("profile_output");
                std::fs::create_dir_all(&output_dir)?;

                // Set XLA flags for HTML output if requested
                let html_path = if html {
                    let html_dir = output_dir.join("html");
                    std::fs::create_dir_all(&html_dir)?;

                    // Set XLA_FLAGS to dump HTML (unsafe but necessary for XLA)
                    let xla_flags = format!(
                        "--xla_dump_to={} --xla_dump_hlo_as_html",
                        html_dir.display()
                    );
                    unsafe {
                        std::env::set_var("XLA_FLAGS", &xla_flags);
                    }

                    Some(html_dir)
                } else {
                    None
                };

                // Build uncompiled execution to get HLO module
                let exec = self.build_uncompiled(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    default_playback_speed,
                    max_ticks,
                )?;

                // Create client and compile to get compile time
                let mut client = nox::Client::cpu()?;
                client.disable_optimizations();

                let build_time_ms = exec.profiler.build.mean();

                let compile_start = time::Instant::now();
                let mut compiled_exec = exec.compile(client)?;
                let compile_time_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

                // Clear XLA_FLAGS after compilation
                if html {
                    unsafe {
                        std::env::remove_var("XLA_FLAGS");
                    }
                }

                // Get HLO text and save to file
                let hlo_text = compiled_exec
                    .tick_exec
                    .hlo_module()
                    .computation()
                    .to_hlo_text()
                    .map_err(|e| Error::NoxEcs(nox_ecs::Error::Nox(nox_ecs::nox::Error::Xla(e))))?;

                // Save HLO dump to output directory
                let hlo_dump_path = output_dir.join("hlo_dump.txt");
                std::fs::write(&hlo_dump_path, &hlo_text)?;

                // Deep analysis mode - save additional formats and capture more data
                let xla_flags = if deep {
                    // Capture XLA environment variables
                    let xla_flags: Vec<(String, String)> = std::env::vars()
                        .filter(|(k, _)| k.starts_with("XLA_") || k.starts_with("TF_XLA_"))
                        .collect();
                    xla_flags
                } else {
                    vec![]
                };

                // Count total instructions and categorize by operation type
                let mut op_counts = HashMap::<String, usize>::new();
                let mut instruction_count = 0;
                let mut op_details = HashMap::<String, Vec<String>>::new(); // Store details for deep mode
                let mut source_line_ops = HashMap::<String, Vec<String>>::new(); // Map source lines to operations

                // First pass: build location map (#loc123 -> "file.py:line" or vec of fused locs)
                let mut loc_map = HashMap::<String, String>::new();
                let mut fused_map = HashMap::<String, Vec<String>>::new();

                if deep {
                    for line in hlo_text.lines() {
                        let trimmed = line.trim();

                        // Parse direct location definitions: #loc123 = loc("file.py":456:0)
                        if trimmed.starts_with("#loc") && trimmed.contains(" = loc(\"") {
                            if let Some(eq_pos) = trimmed.find(" = loc(\"") {
                                let loc_id = &trimmed[..eq_pos];
                                let after_eq = &trimmed[eq_pos + 8..]; // Skip " = loc(\""
                                // Find closing quote for the path
                                if let Some(quote_end) = after_eq.find('"') {
                                    let file_path = &after_eq[..quote_end];
                                    // Now parse :line:col after the quote
                                    let after_quote = &after_eq[quote_end + 1..];
                                    if file_path.contains(".py") && after_quote.starts_with(':') {
                                        // Parse :line:col)
                                        let line_col = &after_quote[1..]; // Skip the first :
                                        if let Some(colon_pos) = line_col.find(':') {
                                            let line_num = &line_col[..colon_pos];
                                            // Extract just filename for readability
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
                        // Parse fused locations: #loc651 = loc(fused[#loc128, #loc129])
                        else if trimmed.starts_with("#loc")
                            && trimmed.contains(" = loc(fused[")
                            && let Some(eq_pos) = trimmed.find(" = loc(fused[")
                        {
                            let loc_id = &trimmed[..eq_pos];
                            let after_fused = &trimmed[eq_pos + 13..]; // Skip " = loc(fused["
                            if let Some(bracket_end) = after_fused.find("])") {
                                let locs_str = &after_fused[..bracket_end];
                                // Parse comma-separated loc refs
                                let fused_locs: Vec<String> =
                                    locs_str.split(',').map(|s| s.trim().to_string()).collect();
                                fused_map.insert(loc_id.to_string(), fused_locs);
                            }
                        }
                    }

                    // Resolve fused locations recursively (simple one-level resolution for now)
                    let mut fused_resolutions = Vec::new();
                    for (fused_id, constituent_locs) in &fused_map {
                        for constituent_loc in constituent_locs {
                            if let Some(resolved) = loc_map.get(constituent_loc) {
                                // Store the first resolved location from the fused set
                                fused_resolutions.push((fused_id.clone(), resolved.clone()));
                                break;
                            }
                        }
                    }
                    // Insert all fused resolutions
                    for (fused_id, resolved) in fused_resolutions {
                        loc_map.insert(fused_id, resolved);
                    }
                }

                // Second pass: count instructions and map to source
                for line in hlo_text.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with('%') {
                        instruction_count += 1;
                        // Extract operation type (the word after the '=')
                        if let Some(eq_pos) = trimmed.find('=') {
                            let after_eq = &trimmed[eq_pos + 1..].trim();
                            // Get the first word (operation type)
                            if let Some(op_name) = after_eq.split_whitespace().next() {
                                *op_counts.entry(op_name.to_string()).or_insert(0) += 1;

                                // In deep mode, collect operation details and source mapping
                                if deep {
                                    op_details
                                        .entry(op_name.to_string())
                                        .or_default()
                                        .push(trimmed.to_string());

                                    // Extract source location reference: loc(#loc123)
                                    if let Some(loc_ref_start) = trimmed.rfind("loc(#loc")
                                        && let Some(loc_ref_end) =
                                            trimmed[loc_ref_start..].find(')')
                                    {
                                        let loc_ref = &trimmed
                                            [loc_ref_start + 4..loc_ref_start + loc_ref_end];
                                        // Resolve to actual source location
                                        if let Some(source_loc) = loc_map.get(loc_ref) {
                                            source_line_ops
                                                .entry(source_loc.clone())
                                                .or_default()
                                                .push(op_name.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Sort operations by count (descending)
                let mut op_vec: Vec<_> = op_counts.iter().collect();
                op_vec.sort_by(|a, b| b.1.cmp(a.1));

                // Categorize operations for deep analysis
                let categorized_ops = if deep {
                    let mut categories = HashMap::<&str, Vec<(String, usize)>>::new();
                    for (op, count) in &op_counts {
                        let category = if op.contains("multiply") || op.contains("dot") {
                            "Arithmetic (Multiply/Dot)"
                        } else if op.contains("add") || op.contains("subtract") {
                            "Arithmetic (Add/Sub)"
                        } else if op.contains("reshape")
                            || op.contains("transpose")
                            || op.contains("broadcast")
                        {
                            "Shape Operations"
                        } else if op.contains("compare") || op.contains("select") {
                            "Control Flow"
                        } else if op.contains("constant") {
                            "Constants"
                        } else if op.contains("slice")
                            || op.contains("gather")
                            || op.contains("dynamic")
                        {
                            "Indexing"
                        } else if op == "call" {
                            "Function Calls"
                        } else {
                            "Other"
                        };
                        categories
                            .entry(category)
                            .or_default()
                            .push((op.clone(), *count));
                    }
                    Some(categories)
                } else {
                    None
                };

                // Calculate memory footprint with per-component tracking
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
                let mut output_memory_bytes = 0usize;
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

                for id in &compiled_exec.tick_exec.metadata().ret_ids {
                    if let Some(col) = compiled_exec.world.column_by_id(*id) {
                        let shape_size: usize =
                            col.schema.shape().iter().map(|&x| x as usize).product();
                        let elem_size = element_size(col.schema.prim_type);
                        let mem_bytes = col.len() * shape_size * elem_size;
                        output_memory_bytes += mem_bytes;
                    }
                }

                // Sort components by memory usage (descending)
                component_memory.sort_by(|a, b| b.1.cmp(&a.1));

                let input_memory_kb = input_memory_bytes as f64 / 1024.0;
                let output_memory_kb = output_memory_bytes as f64 / 1024.0;

                // Print formatted output
                println!("\n=== SYSTEM PROFILE ===");
                println!("\n[Compilation]");
                println!("  Build time:        {:.3} ms", build_time_ms);
                println!("  Compile time:      {:.3} ms", compile_time_ms);

                println!("\n[HLO Analysis]");
                println!("  Total instructions: {}", instruction_count);
                println!("  HLO text dump:      {}", hlo_dump_path.display());
                if let Some(html_dir) = &html_path {
                    // List HTML files that were generated
                    if let Ok(entries) = std::fs::read_dir(html_dir) {
                        let html_files: Vec<_> = entries
                            .filter_map(|e| e.ok())
                            .filter(|e| {
                                e.path().extension().and_then(|s| s.to_str()) == Some("html")
                            })
                            .collect();
                        if !html_files.is_empty() {
                            println!(
                                "  HTML visualization: {} (open in browser)",
                                html_dir.display()
                            );
                            println!("    Found {} HTML file(s)", html_files.len());
                        }
                    }
                }

                println!("\n[Operation Breakdown]");
                if instruction_count > 0 {
                    for (op, count) in op_vec.iter().take(10) {
                        println!(
                            "  {:20} {:6} ({:.1}%)",
                            op,
                            count,
                            (**count as f64 / instruction_count as f64) * 100.0
                        );
                    }
                    if op_vec.len() > 10 {
                        println!("  ... and {} more operation types", op_vec.len() - 10);
                    }
                } else {
                    println!("  No operations found in HLO");
                }

                println!("\n[Memory Footprint]");
                println!("  Input memory:       {:.2} KB", input_memory_kb);
                println!("  Output memory:      {:.2} KB", output_memory_kb);
                println!(
                    "  Total memory:       {:.2} KB",
                    input_memory_kb + output_memory_kb
                );

                println!("\n[Top Components by Memory]");
                if input_memory_kb > 0.0 {
                    for (name, bytes) in component_memory.iter().take(10) {
                        let kb = *bytes as f64 / 1024.0;
                        println!(
                            "  {:40} {:8.2} KB ({:.1}%)",
                            name,
                            kb,
                            (kb / input_memory_kb) * 100.0
                        );
                    }
                } else {
                    println!("  No input memory components found");
                }

                // Deep analysis output
                if deep {
                    println!("\n[Deep Analysis Mode]");

                    // XLA Flags
                    if !xla_flags.is_empty() {
                        println!("\n  Active XLA Flags:");
                        for (key, value) in &xla_flags {
                            println!("    {} = {}", key, value);
                        }
                    } else {
                        println!("\n  No XLA environment variables detected");
                    }

                    // Note about graph visualization
                    if !html {
                        println!("\n  Graph Visualization:");
                        println!("    Use --html flag for interactive HTML visualization");
                        println!("    Or XLA_FLAGS=\"--xla_dump_hlo_as_dot\" for GraphViz output");
                    }

                    // Operation categories
                    if let Some(ref categories) = categorized_ops {
                        println!("\n  Operation Categories:");
                        let mut cat_vec: Vec<_> = categories.iter().collect();
                        cat_vec.sort_by(|a, b| {
                            let sum_a: usize = a.1.iter().map(|(_, c)| c).sum();
                            let sum_b: usize = b.1.iter().map(|(_, c)| c).sum();
                            sum_b.cmp(&sum_a)
                        });

                        if instruction_count > 0 {
                            for (category, ops) in cat_vec {
                                let total: usize = ops.iter().map(|(_, c)| c).sum();
                                println!(
                                    "    {:30} {:6} ops ({:.1}%)",
                                    category,
                                    total,
                                    (total as f64 / instruction_count as f64) * 100.0
                                );
                            }
                        }
                    }

                    // Sample operations with shapes (first few of each type)
                    println!("\n  Sample Operations (with shapes):");
                    for (op, count) in op_vec.iter().take(5) {
                        if let Some(details) = op_details.get(*op) {
                            println!("    {} ({} instances):", op, count);
                            for detail in details.iter().take(2) {
                                // Extract and show just the interesting part (type/shape info)
                                if let Some(colon_pos) = detail.find(':') {
                                    let type_info =
                                        &detail[colon_pos..].chars().take(80).collect::<String>();
                                    println!("      {}", type_info);
                                }
                            }
                        }
                    }

                    // Theoretical FLOP estimate for multiply/dot operations
                    let mut total_flops_estimate = 0u64;
                    for line in hlo_text.lines() {
                        let trimmed = line.trim();
                        if trimmed.contains("mhlo.multiply") || trimmed.contains("mhlo.dot") {
                            // Very rough estimate: extract tensor dimensions if possible
                            if let Some(tensor_start) = trimmed.find("tensor<")
                                && let Some(tensor_end) = trimmed[tensor_start..].find('>')
                            {
                                let tensor_info =
                                    &trimmed[tensor_start + 7..tensor_start + tensor_end];
                                // Parse something like "1x6xf64" or "1xf64"
                                let dims: Vec<u64> = tensor_info
                                    .split('x')
                                    .filter_map(|s| {
                                        s.chars()
                                            .take_while(|c| c.is_ascii_digit())
                                            .collect::<String>()
                                            .parse()
                                            .ok()
                                    })
                                    .collect();
                                if !dims.is_empty() {
                                    total_flops_estimate += dims.iter().product::<u64>();
                                }
                            }
                        }
                    }

                    if total_flops_estimate > 0 {
                        println!("\n  Estimated Compute:");
                        println!(
                            "    Theoretical FLOPs: ~{} operations",
                            total_flops_estimate
                        );
                        println!("    (Very rough estimate from multiply/dot operations)");
                    }

                    // === OPTIMIZATION OPPORTUNITIES ===
                    println!("\n=== OPTIMIZATION OPPORTUNITIES ===");

                    // Analyze source code hot spots
                    println!("\n[Hot Spots in Python Code]");
                    if !source_line_ops.is_empty() {
                        let mut source_heat_map: Vec<_> = source_line_ops
                            .iter()
                            .map(|(loc, ops)| (loc, ops.len()))
                            .collect();
                        source_heat_map.sort_by(|a, b| b.1.cmp(&a.1));

                        println!("Python lines generating the most HLO operations:");

                        // Get the directory containing the simulation file
                        let sim_dir = path.parent().unwrap_or_else(|| Path::new("."));

                        // Cache file contents to avoid re-reading
                        let mut file_contents_cache = HashMap::<String, Vec<String>>::new();

                        for (source_loc, op_count) in source_heat_map.iter().take(10) {
                            println!("\n  {} - {} ops", source_loc, op_count);

                            // Parse filename and line number
                            if let Some(colon_pos) = source_loc.rfind(':') {
                                let file_name = &source_loc[..colon_pos];
                                let line_num_str = &source_loc[colon_pos + 1..];

                                if let Ok(line_num) = line_num_str.parse::<usize>() {
                                    // Try to read the file and show the line
                                    // Support both single-file and multi-file projects
                                    // Get or load file contents
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
                    } else {
                        println!("  (Location mapping requires source line metadata in HLO)");
                        println!(
                            "  Resolved {} Python source locations from HLO",
                            loc_map.len()
                        );
                    }

                    // Identify anti-patterns and optimization opportunities
                    println!("\n[Detected Optimization Patterns]");

                    let mut recommendations = Vec::new();

                    // 1. Excessive reshaping
                    if instruction_count > 0 {
                        let reshape_count = op_counts.get("mhlo.reshape").unwrap_or(&0);
                        let reshape_pct =
                            (*reshape_count as f64 / instruction_count as f64) * 100.0;
                        if reshape_pct > 15.0 {
                            recommendations.push((
                                reshape_pct,
                                "HIGH".to_string(),
                                format!("Excessive reshaping ({:.1}% of ops)", reshape_pct),
                                vec![
                                    "Consider preallocating arrays with correct shapes".to_string(),
                                    "Use jnp.reshape() sparingly - chain operations without intermediate reshapes".to_string(),
                                    "Check for implicit broadcasting causing extra reshapes".to_string(),
                                ]
                            ));
                        }

                        // 2. Shape operations overhead
                        let shape_ops_pct: f64 = if let Some(categories) = &categorized_ops {
                            if let Some(shape_ops) = categories.get("Shape Operations") {
                                let total: usize = shape_ops.iter().map(|(_, c)| c).sum();
                                (total as f64 / instruction_count as f64) * 100.0
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };

                        if shape_ops_pct > 20.0 {
                            recommendations.push((
                                shape_ops_pct,
                                "HIGH".to_string(),
                                format!("Heavy shape manipulation ({:.1}% of ops)", shape_ops_pct),
                                vec![
                                    "Review tensor shape consistency across function boundaries"
                                        .to_string(),
                                    "Use jax.jit with static_argnums for constant shapes"
                                        .to_string(),
                                    "Consider using einsum instead of reshape + matmul chains"
                                        .to_string(),
                                ],
                            ));
                        }
                    }

                    // 3. Memory hotspots
                    if input_memory_bytes > 0
                        && let Some((component_name, bytes)) = component_memory.first()
                    {
                        let pct = (*bytes as f64 / input_memory_bytes as f64) * 100.0;
                        if pct > 80.0 {
                            let kb = *bytes as f64 / 1024.0;
                            recommendations.push((
                                pct,
                                "MEDIUM".to_string(),
                                format!("Single component dominates memory: {} ({:.1}% = {:.2} KB)", component_name, pct, kb),
                                vec![
                                    format!("Review if {} buffer size is necessary", component_name),
                                    "Consider using smaller data types (f32 vs f64) if precision allows".to_string(),
                                    "Investigate if buffer can be reduced or windowed".to_string(),
                                ]
                            ));
                        }
                    }

                    // 4. Control flow overhead
                    if instruction_count > 0 {
                        let select_count = op_counts.get("mhlo.select").unwrap_or(&0);
                        let compare_count = op_counts.get("mhlo.compare").unwrap_or(&0);
                        let control_flow_count = select_count + compare_count;
                        let control_flow_pct =
                            (control_flow_count as f64 / instruction_count as f64) * 100.0;

                        if control_flow_pct > 5.0 {
                            recommendations.push((
                                control_flow_pct,
                                "LOW".to_string(),
                                format!("Conditional operations present ({:.1}% of ops)", control_flow_pct),
                                vec![
                                    "JAX jit() struggles with dynamic control flow - consider using jax.lax.cond".to_string(),
                                    "Static conditions can be hoisted outside @el.map functions".to_string(),
                                    "Review jax.lax.select usage for vectorization opportunities".to_string(),
                                ]
                            ));
                        }
                    }

                    // 5. Function call overhead
                    let call_count = op_counts.get("call").unwrap_or(&0);
                    if *call_count > 50 {
                        recommendations.push((
                            *call_count as f64,
                            "MEDIUM".to_string(),
                            format!(
                                "{} function calls - potential inlining opportunities",
                                call_count
                            ),
                            vec![
                                "JAX may benefit from inlining small functions".to_string(),
                                "Consider using jax.jit(inline=True) for helper functions"
                                    .to_string(),
                                "Review if scan/vmap could replace explicit calls".to_string(),
                            ],
                        ));
                    }

                    // Sort recommendations by severity/impact
                    recommendations.sort_by(|a, b| {
                        // First by priority, then by percentage
                        match (a.1.as_str(), b.1.as_str()) {
                            ("HIGH", "HIGH") => {
                                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            ("HIGH", _) => std::cmp::Ordering::Less,
                            (_, "HIGH") => std::cmp::Ordering::Greater,
                            ("MEDIUM", "MEDIUM") => {
                                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            ("MEDIUM", _) => std::cmp::Ordering::Less,
                            (_, "MEDIUM") => std::cmp::Ordering::Greater,
                            _ => b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal),
                        }
                    });

                    // Print recommendations
                    for (i, (_, priority, issue, suggestions)) in recommendations.iter().enumerate()
                    {
                        println!("\n{}. [{}] {}", i + 1, priority, issue);
                        println!("   Recommendations:");
                        for suggestion in suggestions {
                            println!("   • {}", suggestion);
                        }
                    }

                    if recommendations.is_empty() {
                        println!("\n✓ No major optimization opportunities detected!");
                        println!("  Your code is already well-optimized for JAX.");
                    }
                }

                // Optionally run execution analysis if ticks are specified
                if ticks > 0 {
                    println!("\n[Runtime Analysis - {} ticks]", ticks);

                    // Set up exec with database (similar to bench command)
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
                        db,
                    };

                    // Run the simulation
                    let exec_start = time::Instant::now();
                    exec_with_db.run(py, ticks, false)?;
                    let total_exec_time = exec_start.elapsed().as_secs_f64() * 1000.0;

                    // Get runtime profile metrics
                    let profile = exec_with_db.profile();

                    println!("  Execution time:");
                    println!("    Total:              {:.3} ms", total_exec_time);
                    println!(
                        "    Per tick (avg):     {:.3} ms",
                        total_exec_time / ticks as f64
                    );
                    println!(
                        "    copy_to_client:     {:.3} ms",
                        profile["copy_to_client"]
                    );
                    println!(
                        "    execute_buffers:    {:.3} ms",
                        profile["execute_buffers"]
                    );
                    println!("    copy_to_host:       {:.3} ms", profile["copy_to_host"]);
                    println!(
                        "    add_to_history:     {:.3} ms",
                        profile["add_to_history"]
                    );
                    println!("  Performance:");
                    println!(
                        "    Real-time factor:   {:.1}x",
                        profile["real_time_factor"]
                    );
                    println!(
                        "    Throughput:         {:.1} ticks/sec",
                        1000.0 / profile["tick"]
                    );
                }

                println!();
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
        let db_dir = tempfile::tempdir()?;
        let db_dir = db_dir.keep();
        let db = elodin_db::DB::create(db_dir.join("db"))?;
        nox_ecs::impeller2_server::init_db(&db, &mut exec.world, Timestamp::now())?;
        Ok(Exec { exec, db })
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
        self.world.metadata.run_time_step = TimeStep(ts);
        self.world.metadata.default_playback_speed = default_playback_speed;
        if let Some(ts) = run_time_step {
            let ts = time::Duration::from_secs_f64(ts);
            self.world.metadata.run_time_step = TimeStep(ts);
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
        self.world.metadata.run_time_step = TimeStep(ts);
        self.world.metadata.default_playback_speed = default_playback_speed;
        if let Some(ts) = run_time_step {
            let ts = time::Duration::from_secs_f64(ts);
            self.world.metadata.run_time_step = TimeStep(ts);
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
