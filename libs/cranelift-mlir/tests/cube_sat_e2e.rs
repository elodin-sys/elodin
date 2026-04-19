use cranelift_mlir::parser::parse_module;

#[test]
fn parse_cube_sat_mlir() {
    let mlir = include_str!("../testdata/cube-sat.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    assert!(module.main_func().is_some(), "no main function found");
}

#[test]
fn compile_cube_sat_mlir() {
    let mlir = include_str!("../testdata/cube-sat.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    let compiled = cranelift_mlir::lower::compile_module(&module).expect("failed to compile");
    assert!(!compiled.get_main_fn().is_null());
}

#[ignore = "needs valid cube-sat input state"]
#[test]
fn run_cube_sat_single_tick() {
    let mlir = include_str!("../testdata/cube-sat.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    let main_func = module.main_func().expect("no main");

    let compiled = cranelift_mlir::lower::compile_module(&module).expect("failed to compile");
    type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let mut input_bufs: Vec<Vec<u8>> = Vec::new();
    for (_vid, ty) in &main_func.params {
        let byte_sz = ty.byte_size();
        let buf = vec![0u8; byte_sz];
        input_bufs.push(buf);
    }

    // Set satellite position to [0,0,0,1, 6778100,0,0] in the world_pos parameter
    // main params: %arg0:i64, %arg1:2x6, %arg2:2x7, %arg3:3, %arg4:6x3, %arg5:6x1,
    //   %arg6:6, %arg7:3, %arg8:3, %arg9:3, %arg10:2x6, %arg11:3, %arg12:4, %arg13:3,
    //   %arg14:6x6, %arg15:3, %arg16:4, %arg17:6, %arg18:3x3, %arg19:3x6, %arg20:3,
    //   %arg21:3, %arg22:3x3, %arg23:2x7, %arg24:f64, %arg25:3, %arg26:2x6, %arg27:f64
    // %arg2 = tensor<2x7xf64> (first state: world_pos-like)
    // Set row 0 = [0,0,0,1, 6778100,0,0]
    {
        let pos_buf = &mut input_bufs[2]; // %arg2: tensor<2x7xf64>
        let pos: [f64; 14] = [
            0.0, 0.0, 0.0, 1.0, 6778100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (i, &v) in pos.iter().enumerate() {
            pos_buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    // %arg16 = tensor<4xf64> (quaternion) = [0,0,0,1]
    {
        let q_buf = &mut input_bufs[16]; // %arg16: tensor<4xf64>
        let q: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
        for (i, &v) in q.iter().enumerate() {
            q_buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    // Set inertia for 3kg satellite: mass in the last element of appropriate param
    // %arg23 = tensor<2x7xf64> (second state)
    // Set same position for consistency
    {
        let pos_buf = &mut input_bufs[23]; // %arg23: tensor<2x7xf64>
        let pos: [f64; 14] = [
            0.0, 0.0, 0.0, 1.0, 6778100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (i, &v) in pos.iter().enumerate() {
            pos_buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    let input_ptrs: Vec<*const u8> = input_bufs.iter().map(|b| b.as_ptr()).collect();

    let mut output_bufs: Vec<Vec<u8>> = Vec::new();
    for ty in &main_func.result_types {
        output_bufs.push(vec![0u8; ty.byte_size()]);
    }
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    // Output 0: tensor<2x6xf64> (updated wrench)
    let wrench_bytes = &output_bufs[0];
    let wrench: Vec<f64> = wrench_bytes
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eprintln!("wrench output: {:?}", &wrench);

    // The gravity force components are in the linear part of the wrench
    // wrench = [torque_x, torque_y, torque_z, force_x, force_y, force_z, ...]
    let force_x = wrench[3]; // row 0, element 3 (linear x)
    eprintln!("force_x = {force_x}");

    // With position at [6778100, 0, 0], the gravity force should be significant
    // Expected: approximately -24.5 N (including monopole) or ~-0.04 N (perturbation only)
    // The key test: force_x should NOT be near-zero
    assert!(
        force_x.abs() > 0.01,
        "gravity force_x = {force_x} is too small, expected at least 0.01 in magnitude"
    );
}
