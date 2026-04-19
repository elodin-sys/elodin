use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

fn load_bins(dir: &str, prefix: &str) -> Vec<Vec<u8>> {
    let mut results = Vec::new();
    for i in 0.. {
        let path = format!("{dir}/{prefix}{i}.bin");
        match std::fs::read(&path) {
            Ok(data) => results.push(data),
            Err(_) => break,
        }
    }
    results
}

fn compare_f64_buffers(got: &[u8], want: &[u8], output_idx: usize, rel_tol: f64, abs_tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "output {output_idx}: size mismatch (got {} bytes, want {} bytes)",
        got.len(),
        want.len()
    );
    let n = got.len() / 8;
    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    let mut first_fail = None;
    for i in 0..n {
        let g = f64::from_le_bytes(got[i * 8..(i + 1) * 8].try_into().unwrap());
        let w = f64::from_le_bytes(want[i * 8..(i + 1) * 8].try_into().unwrap());
        let abs_diff = (g - w).abs();
        let rel_diff = if w.abs() > 1e-15 {
            abs_diff / w.abs()
        } else {
            abs_diff
        };
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        if abs_diff > abs_tol && rel_diff > rel_tol && first_fail.is_none() {
            first_fail = Some((i, g, w, abs_diff, rel_diff));
        }
    }
    if let Some((idx, got_v, want_v, abs_d, rel_d)) = first_fail {
        panic!(
            "output {output_idx}: MISMATCH at element {idx}/{n}: got={got_v}, want={want_v}, \
             abs_diff={abs_d:.6e}, rel_diff={rel_d:.6e} (max_abs={max_abs_diff:.6e}, max_rel={max_rel_diff:.6e})"
        );
    }
    eprintln!(
        "output {output_idx}: OK ({n} elements, max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e})"
    );
}

#[test]
#[ignore = "requires ELODIN_CRANELIFT_DEBUG_DIR pointing to a generated checkpoint"]
fn verify_checkpoint() {
    // Run in a thread with a large stack to accommodate JIT functions with many stack slots.
    // Customer simulations with 194+ functions and deep call chains need 256MB+.
    let builder = std::thread::Builder::new().stack_size(256 * 1024 * 1024);
    let handle = builder
        .spawn(verify_checkpoint_impl)
        .expect("failed to spawn thread");
    handle
        .join()
        .expect("checkpoint verification thread panicked");
}

fn verify_checkpoint_impl() {
    let dir = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .expect("set ELODIN_CRANELIFT_DEBUG_DIR to the checkpoint directory");

    let mlir_path = format!("{dir}/stablehlo.mlir");
    let mlir = std::fs::read_to_string(&mlir_path)
        .unwrap_or_else(|e| panic!("cannot read {mlir_path}: {e}"));

    eprintln!("compiling MLIR from {mlir_path}...");
    let mut module = parse_module(&mlir).expect("MLIR parse failed");
    cranelift_mlir::const_fold::fold_module(&mut module);
    let compiled = compile_module(&module).expect("Cranelift compile failed");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };
    eprintln!("compilation OK");

    let inputs = load_bins(&dir, "input_");
    let xla_outputs = load_bins(&dir, "xla_output_");
    assert!(!inputs.is_empty(), "no input_*.bin files found in {dir}");
    assert!(
        !xla_outputs.is_empty(),
        "no xla_output_*.bin files found in {dir}"
    );
    eprintln!(
        "loaded {} inputs, {} XLA reference outputs",
        inputs.len(),
        xla_outputs.len()
    );

    let meta_path = format!("{dir}/checkpoint.json");
    let meta_str = std::fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("cannot read {meta_path}: {e}"));
    let meta: serde_json::Value = serde_json::from_str(&meta_str).expect("invalid JSON");
    let num_output_slots = meta["num_output_slots"]
        .as_u64()
        .unwrap_or(xla_outputs.len() as u64) as usize;

    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    let outputs_info: Vec<usize> = meta["outputs"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|o| o["byte_size"].as_u64().unwrap_or(0) as usize)
                .collect()
        })
        .unwrap_or_default();

    let mut cranelift_outputs: Vec<Vec<u8>> = outputs_info
        .iter()
        .map(|&sz| vec![0u8; sz.max(8)])
        .collect();
    while cranelift_outputs.len() < num_output_slots {
        cranelift_outputs.push(vec![0u8; 1024]);
    }

    let mut output_ptrs: Vec<*mut u8> = cranelift_outputs
        .iter_mut()
        .map(|b| b.as_mut_ptr())
        .collect();

    eprintln!(
        "executing tick function ({num_output_slots} output slots, {} output buffers)...",
        cranelift_outputs.len()
    );
    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };
    eprintln!("tick function OK");

    let n_compare = cranelift_outputs.len().min(xla_outputs.len());
    let mut failures = Vec::new();
    for i in 0..n_compare {
        let got = &cranelift_outputs[i][..xla_outputs[i].len()];
        let want = &xla_outputs[i];
        let result = std::panic::catch_unwind(|| {
            compare_f64_buffers(got, want, i, 1e-6, 1e-8);
        });
        if result.is_err() {
            failures.push(i);
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} of {n_compare} outputs FAILED: {failures:?}",
            failures.len()
        );
    }
    eprintln!("ALL {n_compare} outputs match XLA reference");
}

#[test]
#[ignore = "requires inner_375_standalone.mlir in testdata/checkpoints/cube-sat/"]
fn bisect_inner_375_standalone() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(bisect_inner_375_impl)
        .expect("failed to spawn thread");
    handle.join().expect("inner_375 bisection thread panicked");
}

fn bisect_inner_375_impl() {
    let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .unwrap_or_else(|_| "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string());

    let mlir_path = format!("{base}/inner_375_standalone.mlir");
    let mlir = std::fs::read_to_string(&mlir_path)
        .unwrap_or_else(|e| panic!("cannot read {mlir_path}: {e}"));

    eprintln!("compiling inner_375 standalone...");
    let module = parse_module(&mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };
    eprintln!("compilation OK");

    // inner_375 inputs from checkpoint:
    // %arg0 = main's %arg16 (tensor<4xf64>) = checkpoint input_16.bin
    // %arg1 = main's %13 (tensor<2x6xf64>) = computed, use initial wrench
    // %arg2 = main's %arg2 (tensor<2x7xf64>) = checkpoint input_2.bin
    // %arg3 = main's %arg23 (tensor<2x7xf64>) = checkpoint input_23.bin
    // %arg4 = main's %arg24 (tensor<f64>) = checkpoint input_24.bin
    let quat = std::fs::read(format!("{base}/input_16.bin")).expect("input_16.bin");
    let wrench: Vec<u8> = [
        -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let state1 = std::fs::read(format!("{base}/input_2.bin")).expect("input_2.bin");
    let state2 = std::fs::read(format!("{base}/input_23.bin")).expect("input_23.bin");
    let scalar = std::fs::read(format!("{base}/input_24.bin")).expect("input_24.bin");

    let inputs: Vec<&[u8]> = vec![&quat, &wrench, &state1, &state2, &scalar];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    // Outputs: (tensor<2x6xf64>, tensor<f64>) = 96 + 8 = 104 bytes
    let mut out_wrench = vec![0u8; 96];
    let mut out_scalar = vec![0u8; 8];
    let mut output_ptrs: Vec<*mut u8> = vec![out_wrench.as_mut_ptr(), out_scalar.as_mut_ptr()];

    eprintln!("executing inner_375...");
    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };
    eprintln!("inner_375 OK");

    // Parse the wrench output
    let wrench_out: Vec<f64> = out_wrench
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scalar_out = f64::from_le_bytes(out_scalar[..8].try_into().unwrap());

    eprintln!("wrench output (row 0): {:?}", &wrench_out[..6]);
    eprintln!("wrench output (row 1): {:?}", &wrench_out[6..12]);
    eprintln!("scalar output: {scalar_out}");

    // The force components (elements 3-5) should be significant for a satellite at 6778100m
    let force_x = wrench_out[3];
    let force_y = wrench_out[4];
    let force_z = wrench_out[5];
    eprintln!("force_x = {force_x:.10e}");
    eprintln!("force_y = {force_y:.10e}");
    eprintln!("force_z = {force_z:.10e}");

    // Expected: force_x ~ -8.68 (gravitational acceleration * direction)
    // If the bug is in inner_375, force_x will be ~-0.000236 (too small)
    // If the bug is in cross-ABI marshaling, force_x will be ~-8.68 (correct)
    let force_magnitude = (force_x * force_x + force_y * force_y + force_z * force_z).sqrt();
    eprintln!("force magnitude = {force_magnitude:.6e}");

    // This assertion catches the gravity magnitude bug
    assert!(
        force_magnitude > 1.0,
        "GRAVITY BUG REPRODUCED IN STANDALONE: force magnitude = {force_magnitude:.6e}, expected > 1.0"
    );
}

#[test]
#[ignore = "requires inner_375_bisect.mlir in testdata/checkpoints/cube-sat/"]
fn bisect_inner_375_reduce_scalars() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(bisect_reduce_scalars_impl)
        .expect("failed to spawn thread");
    handle.join().expect("bisection thread panicked");
}

fn bisect_reduce_scalars_impl() {
    let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .unwrap_or_else(|_| "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string());

    let mlir_path = format!("{base}/inner_375_bisect.mlir");
    let mlir = std::fs::read_to_string(&mlir_path)
        .unwrap_or_else(|e| panic!("cannot read {mlir_path}: {e}"));

    eprintln!("compiling inner_375 bisect variant...");
    let module = parse_module(&mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };
    eprintln!("compilation OK");

    let quat = std::fs::read(format!("{base}/input_16.bin")).expect("input_16.bin");
    let wrench: Vec<u8> = [
        -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let state1 = std::fs::read(format!("{base}/input_2.bin")).expect("input_2.bin");
    let state2 = std::fs::read(format!("{base}/input_23.bin")).expect("input_23.bin");
    let scalar = std::fs::read(format!("{base}/input_24.bin")).expect("input_24.bin");

    let inputs: Vec<&[u8]> = vec![&quat, &wrench, &state1, &state2, &scalar];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    // 6 outputs: wrench(96B) + scalar(8B) + 4 reduce scalars(8B each)
    let mut out_wrench = vec![0u8; 96];
    let mut out_scalar = vec![0u8; 8];
    let mut out_r96 = vec![0u8; 8];
    let mut out_r117 = vec![0u8; 8];
    let mut out_r144 = vec![0u8; 8];
    let mut out_r170 = vec![0u8; 8];
    let mut output_ptrs: Vec<*mut u8> = vec![
        out_wrench.as_mut_ptr(),
        out_scalar.as_mut_ptr(),
        out_r96.as_mut_ptr(),
        out_r117.as_mut_ptr(),
        out_r144.as_mut_ptr(),
        out_r170.as_mut_ptr(),
    ];

    eprintln!("executing inner_375 bisect...");
    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };
    eprintln!("inner_375 bisect OK");

    let r96 = f64::from_le_bytes(out_r96[..8].try_into().unwrap());
    let r117 = f64::from_le_bytes(out_r117[..8].try_into().unwrap());
    let r144 = f64::from_le_bytes(out_r144[..8].try_into().unwrap());
    let r170 = f64::from_le_bytes(out_r170[..8].try_into().unwrap());

    eprintln!("=== REDUCE SCALARS (force components before assembly) ===");
    eprintln!("  %96  (a_1 component) = {r96:.10e}");
    eprintln!("  %117 (a_2 component) = {r117:.10e}");
    eprintln!("  %144 (a_3 component) = {r144:.10e}");
    eprintln!("  %170 (a_4 radial)    = {r170:.10e}");
    eprintln!("  Expected order of magnitude: ~1e-4 to ~1e0 for gravity");

    let wrench_out: Vec<f64> = out_wrench
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eprintln!("  wrench force_x = {:.10e}", wrench_out[3]);
}

#[test]
#[ignore = "bisection"]
fn bisect_inner_375_scaling() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder.spawn(bisect_scaling_impl).expect("spawn");
    handle.join().expect("panicked");
}

fn bisect_scaling_impl() {
    let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .unwrap_or_else(|_| "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string());

    let mlir = std::fs::read_to_string(format!("{base}/inner_375_bisect2.mlir")).expect("read");
    let module = parse_module(&mlir).expect("parse");
    let compiled = compile_module(&module).expect("compile");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let quat = std::fs::read(format!("{base}/input_16.bin")).expect("quat");
    let wrench: Vec<u8> = [
        -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let state1 = std::fs::read(format!("{base}/input_2.bin")).expect("s1");
    let state2 = std::fs::read(format!("{base}/input_23.bin")).expect("s2");
    let scalar = std::fs::read(format!("{base}/input_24.bin")).expect("sc");

    let inputs: Vec<&[u8]> = vec![&quat, &wrench, &state1, &state2, &scalar];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    // 9 outputs: wrench(96) + scalar(8) + %11(8) + %23(8) + %24(8) + %25(8) + %26(8) + %46(8) + %53(520)
    let mut bufs: Vec<Vec<u8>> = vec![
        vec![0u8; 96],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 8],
        vec![0u8; 520],
    ];
    let mut output_ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    let f = |buf: &[u8]| f64::from_le_bytes(buf[..8].try_into().unwrap());
    let v11 = f(&bufs[2]); // %11 (orbit parameter from state)
    let r = f(&bufs[3]); // %23 (orbital radius)
    let dx = f(&bufs[4]); // %24 (direction cosine x)
    let dy = f(&bufs[5]); // %25 (direction cosine y)
    let dz = f(&bufs[6]); // %26 (direction cosine z)
    let gm_r = f(&bufs[7]); // %46 (GM/r)

    let rho: Vec<f64> = bufs[8]
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    eprintln!("=== INNER_375 SCALING PARAMETERS ===");
    eprintln!("  %11 (orbit param) = {v11:.10e}");
    eprintln!("  %23 (radius r)    = {r:.10e}  (expected ~6.778e6)");
    eprintln!("  %24 (dir_x)       = {dx:.10e}  (expected ~1.0)");
    eprintln!("  %25 (dir_y)       = {dy:.10e}  (expected ~0.0)");
    eprintln!("  %26 (dir_z)       = {dz:.10e}  (expected ~0.0)");
    eprintln!("  %46 (GM/r)        = {gm_r:.10e}  (expected ~5.88e7)");
    eprintln!("  rho[0..5]         = {:?}", &rho[..5]);
    eprintln!("  rho[63..65]       = {:?}", &rho[63..65]);

    let expected_gm_r = 3.986004418e14 / 6778100.0;
    eprintln!("  expected GM/r     = {expected_gm_r:.10e}");
    eprintln!("  GM/r ratio        = {:.6}", gm_r / expected_gm_r);
}

#[test]
#[ignore = "bisection"]
fn bisect_inner_375_row_sums() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder.spawn(bisect_row_sums_impl).expect("spawn");
    handle.join().expect("panicked");
}

fn bisect_row_sums_impl() {
    let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .unwrap_or_else(|_| "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string());

    let mlir = std::fs::read_to_string(format!("{base}/inner_375_bisect3.mlir")).expect("read");
    let module = parse_module(&mlir).expect("parse");
    let compiled = compile_module(&module).expect("compile");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let quat = std::fs::read(format!("{base}/input_16.bin")).expect("quat");
    let wrench: Vec<u8> = [
        -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let state1 = std::fs::read(format!("{base}/input_2.bin")).expect("s1");
    let state2 = std::fs::read(format!("{base}/input_23.bin")).expect("s2");
    let scalar = std::fs::read(format!("{base}/input_24.bin")).expect("sc");

    let inputs: Vec<&[u8]> = vec![&quat, &wrench, &state1, &state2, &scalar];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    // 5 outputs: wrench(96) + scalar(8) + row_sums(520) + cos(520) + sin(520)
    let mut bufs: Vec<Vec<u8>> = vec![
        vec![0u8; 96],
        vec![0u8; 8],
        vec![0u8; 520],
        vec![0u8; 520],
        vec![0u8; 520],
    ];
    let mut output_ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    let parse = |buf: &[u8]| -> Vec<f64> {
        buf.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let row_sums = parse(&bufs[2]);
    let cos_vec = parse(&bufs[3]);
    let sin_vec = parse(&bufs[4]);

    eprintln!("=== FIRST REDUCE CHAIN ROW SUMS (%95) ===");
    eprintln!("  row_sums[0..5] = {:?}", &row_sums[..5]);
    eprintln!("  row_sums[63..65] = {:?}", &row_sums[63..65]);
    let nz = row_sums.iter().filter(|&&x| x != 0.0).count();
    let total: f64 = row_sums.iter().sum();
    eprintln!("  nonzero count = {nz}, total sum = {total:.10e}");

    eprintln!("=== COS/SIN VECTORS (from while loop 3) ===");
    eprintln!("  cos[0..5] = {:?}", &cos_vec[..5]);
    eprintln!("  cos[63..65] = {:?}", &cos_vec[63..65]);
    eprintln!("  sin[0..5] = {:?}", &sin_vec[..5]);
    let cos_nz = cos_vec.iter().filter(|&&x| x != 0.0).count();
    let sin_nz = sin_vec.iter().filter(|&&x| x != 0.0).count();
    eprintln!("  cos nonzero = {cos_nz}, sin nonzero = {sin_nz}");
}

#[test]
#[ignore = "bisection"]
fn bisect_inner_375_matrices() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder.spawn(bisect_matrices_impl).expect("spawn");
    handle.join().expect("panicked");
}

fn bisect_matrices_impl() {
    let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .unwrap_or_else(|_| "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string());

    let mlir = std::fs::read_to_string(format!("{base}/inner_375_bisect4.mlir")).expect("read");
    let module = parse_module(&mlir).expect("parse");
    let compiled = compile_module(&module).expect("compile");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let quat = std::fs::read(format!("{base}/input_16.bin")).expect("quat");
    let wrench: Vec<u8> = [
        -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let state1 = std::fs::read(format!("{base}/input_2.bin")).expect("s1");
    let state2 = std::fs::read(format!("{base}/input_23.bin")).expect("s2");
    let scalar = std::fs::read(format!("{base}/input_24.bin")).expect("sc");

    let inputs: Vec<&[u8]> = vec![&quat, &wrench, &state1, &state2, &scalar];
    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();

    // 6 outputs: wrench(96) + scalar(8) + legendre_r0(520) + legendre_r2(520) + coeff_r0(520) + coeff_r2(520)
    let mut bufs: Vec<Vec<u8>> = vec![
        vec![0u8; 96],
        vec![0u8; 8],
        vec![0u8; 520],
        vec![0u8; 520],
        vec![0u8; 520],
        vec![0u8; 520],
    ];
    let mut output_ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    let parse = |buf: &[u8]| -> Vec<f64> {
        buf.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let leg_r0 = parse(&bufs[2]);
    let leg_r2 = parse(&bufs[3]);
    let coeff_r0 = parse(&bufs[4]);
    let coeff_r2 = parse(&bufs[5]);

    eprintln!("=== LEGENDRE MATRIX %42 ===");
    eprintln!("  row 0 [0..5] = {:?}", &leg_r0[..5]);
    eprintln!("  row 0 [63..65] = {:?}", &leg_r0[63..65]);
    let leg_r0_nz = leg_r0.iter().filter(|&&x| x != 0.0).count();
    eprintln!("  row 0 nonzero = {leg_r0_nz}");
    eprintln!("  row 2 [0..5] = {:?}", &leg_r2[..5]);
    let leg_r2_nz = leg_r2.iter().filter(|&&x| x != 0.0).count();
    eprintln!("  row 2 nonzero = {leg_r2_nz}");

    eprintln!("=== COEFFICIENT MATRIX %77 ===");
    eprintln!("  row 0 [0..5] = {:?}", &coeff_r0[..5]);
    eprintln!("  row 0 [63..65] = {:?}", &coeff_r0[63..65]);
    let coeff_r0_nz = coeff_r0.iter().filter(|&&x| x != 0.0).count();
    eprintln!("  row 0 nonzero = {coeff_r0_nz}");
    eprintln!("  row 2 [0..5] = {:?}", &coeff_r2[..5]);
    let coeff_r2_nz = coeff_r2.iter().filter(|&&x| x != 0.0).count();
    eprintln!("  row 2 nonzero = {coeff_r2_nz}");

    eprintln!("=== KEY: Is Legendre[0][0] correct? ===");
    eprintln!(
        "  Legendre[0][0] = {} (expected ~0.577 = 1/sqrt(3))",
        leg_r0[0]
    );
    eprintln!("  Legendre[2][2] = {} (expected ~4.63)", leg_r2[2]);
    eprintln!(
        "  Coeff[0][0] = {} (expected 0 since cos_shifted[0]=0)",
        coeff_r0[0]
    );
    eprintln!(
        "  Coeff[2][1] = {} (should be C_22 * cos_shifted[1] ≈ 2.44e-6)",
        coeff_r2[1]
    );
}

#[test]
#[ignore = "bisection"]
fn bisect_inner_375_nq2() {
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(|| {
            let base = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR").unwrap_or_else(|_| {
                "libs/cranelift-mlir/testdata/checkpoints/cube-sat".to_string()
            });
            let mlir =
                std::fs::read_to_string(format!("{base}/inner_375_bisect5.mlir")).expect("read");
            let module = parse_module(&mlir).expect("parse");
            let compiled = compile_module(&module).expect("compile");
            let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };
            let quat = std::fs::read(format!("{base}/input_16.bin")).unwrap();
            let wrench: Vec<u8> = [
                -0.002f64, -0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
            let s1 = std::fs::read(format!("{base}/input_2.bin")).unwrap();
            let s2 = std::fs::read(format!("{base}/input_23.bin")).unwrap();
            let sc = std::fs::read(format!("{base}/input_24.bin")).unwrap();
            let inputs: Vec<&[u8]> = vec![&quat, &wrench, &s1, &s2, &sc];
            let iptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
            let mut bufs: Vec<Vec<u8>> = (0..8).map(|_| vec![0u8; 8]).collect();
            let mut optrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { tick_fn(iptrs.as_ptr(), optrs.as_mut_ptr()) };
            let f = |i: usize| f64::from_le_bytes(bufs[i][..8].try_into().unwrap());
            eprintln!("=== REDUCE SCALARS ===");
            eprintln!("  a_1 (%96)  = {:.10e} (XLA: 3.4484e-5)", f(0));
            eprintln!("  a_2 (%117) = {:.10e} (XLA: -1.1884e-5)", f(1));
            eprintln!("  a_3 (%144) = {:.10e} (XLA: 1.5583e-5)", f(2));
            eprintln!("  a_4 (%170) = {:.10e} (XLA: -8.6802e0)", f(3));
            eprintln!("=== NQ MATRICES ===");
            eprintln!("  nq2[0,0] = {:.10e}", f(4));
            eprintln!("  nq2[1,1] = {:.10e}", f(5));
            eprintln!("  nq2[2,2] = {:.10e}", f(6));
            eprintln!("  nq1[0,0] = {:.10e}", f(7));
        })
        .unwrap();
    handle.join().unwrap();
}
