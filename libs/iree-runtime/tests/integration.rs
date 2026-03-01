use std::path::PathBuf;

use iree_runtime::{BufferView, ElementType, Instance, Session};
use zerocopy::FromBytes;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn setup_session(vmfb_filename: &str) -> (Instance, Session) {
    let instance = Instance::new().expect("failed to create instance");
    let device = instance
        .create_device("local-sync")
        .expect("failed to create device");
    let session = Session::new(&instance, &device).expect("failed to create session");
    session
        .load_vmfb_file(&fixture_path(vmfb_filename))
        .expect("failed to load VMFB");
    (instance, session)
}

#[test]
fn test_simple_mul() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    let input0: [f32; 4] = [1.0, 1.1, 1.2, 1.3];
    let input1: [f32; 4] = [10.0, 100.0, 1000.0, 10000.0];
    let expected: [f32; 4] = [10.0, 110.0, 1200.0, 13000.0];

    let buf0 = BufferView::from_bytes(&session, as_bytes(&input0), &[4], ElementType::Float32)
        .expect("failed to create buffer view 0");
    let buf1 = BufferView::from_bytes(&session, as_bytes(&input1), &[4], ElementType::Float32)
        .expect("failed to create buffer view 1");

    let mut call = session
        .call("module.simple_mul")
        .expect("failed to create call");
    call.push_input(&buf0).expect("failed to push input 0");
    call.push_input(&buf1).expect("failed to push input 1");
    call.invoke().expect("invoke failed");

    let output = call.pop_output().expect("failed to pop output");
    let output_bytes = output.to_bytes().expect("failed to read output");
    let result: &[f32] = from_bytes(&output_bytes);

    assert_eq!(result.len(), 4);
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "element {i}: got {got}, expected {want}"
        );
    }
}

#[test]
fn test_f64_support() {
    let (_instance, session) = setup_session("simple_add_f64.vmfb");

    let input0: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let input1: [f64; 4] = [0.1, 0.2, 0.3, 0.4];
    let expected: [f64; 4] = [1.1, 2.2, 3.3, 4.4];

    let buf0 = BufferView::from_bytes(&session, as_bytes(&input0), &[4], ElementType::Float64)
        .expect("failed to create f64 buffer view 0");
    let buf1 = BufferView::from_bytes(&session, as_bytes(&input1), &[4], ElementType::Float64)
        .expect("failed to create f64 buffer view 1");

    let mut call = session
        .call("module.simple_add_f64")
        .expect("failed to create call");
    call.push_input(&buf0).expect("failed to push input 0");
    call.push_input(&buf1).expect("failed to push input 1");
    call.invoke().expect("invoke failed");

    let output = call.pop_output().expect("failed to pop output");
    let output_bytes = output.to_bytes().expect("failed to read output");
    let result: &[f64] = from_bytes(&output_bytes);

    assert_eq!(result.len(), 4);
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "element {i}: got {got}, expected {want}"
        );
    }
}

#[test]
fn test_buffer_view_metadata() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = BufferView::from_bytes(&session, as_bytes(&data), &[4], ElementType::Float32)
        .expect("failed to create buffer view");

    assert_eq!(buf.shape(), vec![4]);
    assert_eq!(buf.element_type(), Some(ElementType::Float32));
    assert_eq!(buf.element_count(), 4);
}

// Note: there is no test_error_on_bad_vmfb because IREE's FlatBuffer verifier
// calls abort() on completely invalid data rather than returning an error status.
// This is a C-level assertion in IREE that we cannot catch from Rust.

#[test]
fn test_multiple_invocations() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    for multiplier in [1.0f32, 2.0, 3.0, 10.0] {
        let input0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input1: [f32; 4] = [multiplier; 4];

        let buf0 = BufferView::from_bytes(&session, as_bytes(&input0), &[4], ElementType::Float32)
            .unwrap();
        let buf1 = BufferView::from_bytes(&session, as_bytes(&input1), &[4], ElementType::Float32)
            .unwrap();

        let mut call = session.call("module.simple_mul").unwrap();
        call.push_input(&buf0).unwrap();
        call.push_input(&buf1).unwrap();
        call.invoke().unwrap();

        let output = call.pop_output().unwrap();
        let output_bytes = output.to_bytes().unwrap();
        let result: &[f32] = from_bytes(&output_bytes);

        for (i, &val) in result.iter().enumerate() {
            let expected = input0[i] * multiplier;
            assert!(
                (val - expected).abs() < 1e-6,
                "multiplier={multiplier}, element {i}: got {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_multidimensional_shapes() {
    let (_instance, session) = setup_session("matmul.vmfb");

    // A: 3x4 matrix (row-major)
    #[rustfmt::skip]
    let a: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    // B: 4x2 matrix (row-major)
    #[rustfmt::skip]
    let b: [f32; 8] = [
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ];
    // Expected C = A @ B: 3x2 matrix
    // Row 0: 1*1+2*3+3*5+4*7=50, 1*2+2*4+3*6+4*8=60
    // Row 1: 5*1+6*3+7*5+8*7=114, 5*2+6*4+7*6+8*8=140
    // Row 2: 9*1+10*3+11*5+12*7=178, 9*2+10*4+11*6+12*8=220
    let expected: [f32; 6] = [50.0, 60.0, 114.0, 140.0, 178.0, 220.0];

    let buf_a = BufferView::from_bytes(&session, as_bytes(&a), &[3, 4], ElementType::Float32)
        .expect("failed to create A");
    let buf_b = BufferView::from_bytes(&session, as_bytes(&b), &[4, 2], ElementType::Float32)
        .expect("failed to create B");

    assert_eq!(buf_a.shape(), vec![3, 4]);
    assert_eq!(buf_b.shape(), vec![4, 2]);

    let mut call = session
        .call("module.matmul")
        .expect("failed to create call");
    call.push_input(&buf_a).unwrap();
    call.push_input(&buf_b).unwrap();
    call.invoke().expect("matmul invoke failed");

    let output = call.pop_output().expect("failed to pop output");
    assert_eq!(output.shape(), vec![3, 2]);
    assert_eq!(output.element_type(), Some(ElementType::Float32));

    let output_bytes = output.to_bytes().unwrap();
    let result: &[f32] = from_bytes(&output_bytes);
    assert_eq!(result.len(), 6);
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "matmul element {i}: got {got}, expected {want}"
        );
    }
}

#[test]
fn test_multiple_outputs() {
    let (_instance, session) = setup_session("multi_output.vmfb");

    let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    let expected_sum: [f32; 4] = [11.0, 22.0, 33.0, 44.0];
    let expected_prod: [f32; 4] = [10.0, 40.0, 90.0, 160.0];

    let buf_a = BufferView::from_bytes(&session, as_bytes(&a), &[4], ElementType::Float32).unwrap();
    let buf_b = BufferView::from_bytes(&session, as_bytes(&b), &[4], ElementType::Float32).unwrap();

    let mut call = session
        .call("module.multi_output")
        .expect("failed to create call");
    call.push_input(&buf_a).unwrap();
    call.push_input(&buf_b).unwrap();
    call.invoke().expect("multi_output invoke failed");

    let out_sum = call.pop_output().expect("failed to pop first output (sum)");
    let out_prod = call
        .pop_output()
        .expect("failed to pop second output (prod)");

    let sum_bytes = out_sum.to_bytes().unwrap();
    let sum_result: &[f32] = from_bytes(&sum_bytes);
    for (i, (&got, &want)) in sum_result.iter().zip(expected_sum.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "sum element {i}: got {got}, expected {want}"
        );
    }

    let prod_bytes = out_prod.to_bytes().unwrap();
    let prod_result: &[f32] = from_bytes(&prod_bytes);
    for (i, (&got, &want)) in prod_result.iter().zip(expected_prod.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "prod element {i}: got {got}, expected {want}"
        );
    }
}

#[test]
fn test_integer_roundtrip() {
    let (_instance, session) = setup_session("identity_i64.vmfb");

    let input: [i64; 4] = [1, -42, i64::MAX, i64::MIN];

    let buf = BufferView::from_bytes(&session, as_bytes(&input), &[4], ElementType::Int64).unwrap();
    assert_eq!(buf.element_type(), Some(ElementType::Int64));

    let mut call = session
        .call("module.identity_i64")
        .expect("failed to create call");
    call.push_input(&buf).unwrap();
    call.invoke().expect("identity_i64 invoke failed");

    let output = call.pop_output().unwrap();
    let output_bytes = output.to_bytes().unwrap();
    let result: &[i64] = from_bytes(&output_bytes);

    assert_eq!(result, &input, "i64 round-trip must be exact byte-for-byte");
}

#[test]
fn test_load_vmfb_from_bytes() {
    let vmfb_bytes = std::fs::read(fixture_path("simple_mul.vmfb")).expect("failed to read VMFB");

    let instance = Instance::new().unwrap();
    let device = instance.create_device("local-sync").unwrap();
    let session = Session::new(&instance, &device).unwrap();
    session
        .load_vmfb(&vmfb_bytes)
        .expect("load_vmfb from bytes failed");

    let input0: [f32; 4] = [2.0, 3.0, 4.0, 5.0];
    let input1: [f32; 4] = [10.0, 10.0, 10.0, 10.0];

    let buf0 =
        BufferView::from_bytes(&session, as_bytes(&input0), &[4], ElementType::Float32).unwrap();
    let buf1 =
        BufferView::from_bytes(&session, as_bytes(&input1), &[4], ElementType::Float32).unwrap();

    let mut call = session.call("module.simple_mul").unwrap();
    call.push_input(&buf0).unwrap();
    call.push_input(&buf1).unwrap();
    call.invoke()
        .expect("invoke after load_vmfb from bytes failed");

    let output = call.pop_output().unwrap();
    let output_bytes = output.to_bytes().unwrap();
    let result: &[f32] = from_bytes(&output_bytes);
    assert_eq!(result, &[20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn test_error_wrong_function_name() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    let result = session.call("module.nonexistent_function");
    assert!(
        result.is_err(),
        "calling a nonexistent function should return an error"
    );
}

#[test]
fn test_error_wrong_input_count() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf =
        BufferView::from_bytes(&session, as_bytes(&input), &[4], ElementType::Float32).unwrap();

    let mut call = session.call("module.simple_mul").unwrap();
    call.push_input(&buf).unwrap();
    // Only 1 input pushed to a 2-input function

    let result = call.invoke();
    assert!(
        result.is_err(),
        "invoking with wrong input count should return an error"
    );
}

#[test]
fn test_large_buffer() {
    let (_instance, session) = setup_session("simple_mul.vmfb");

    // simple_mul expects tensor<4xf32>, so we can't use 1024 elements directly.
    // Instead, test that BufferView handles large allocations correctly.
    // Create a large buffer, read it back, verify fidelity.
    let large_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let buf = BufferView::from_bytes(
        &session,
        as_bytes(&large_data),
        &[1024],
        ElementType::Float32,
    )
    .expect("failed to create large buffer view");

    assert_eq!(buf.shape(), vec![1024]);
    assert_eq!(buf.element_count(), 1024);

    let round_trip = buf.to_bytes().unwrap();
    let result: &[f32] = from_bytes(&round_trip);
    assert_eq!(result.len(), 1024);
    for (i, (&got, &want)) in result.iter().zip(large_data.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "large buffer element {i}: got {got}, expected {want}"
        );
    }
}

fn as_bytes<T: zerocopy::IntoBytes + zerocopy::Immutable>(slice: &[T]) -> &[u8] {
    zerocopy::IntoBytes::as_bytes(slice)
}

fn from_bytes<T: zerocopy::FromBytes + zerocopy::Immutable>(bytes: &[u8]) -> &[T] {
    <[T]>::ref_from_bytes(bytes).expect("alignment or size mismatch in from_bytes")
}
