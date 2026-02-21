use iree_runtime::{BufferView, Device, ElementType, Instance, Session};

const SIMPLE_MUL_VMFB: &[u8] = include_bytes!("fixtures/simple_mul.vmfb");
const SIMPLE_ADD_F64_VMFB: &[u8] = include_bytes!("fixtures/simple_add_f64.vmfb");

fn setup_session(vmfb: &[u8]) -> (Instance, Device, Session) {
    let instance = Instance::new().expect("failed to create instance");
    let device = instance
        .create_device("local-sync")
        .expect("failed to create device");
    let session = Session::new(&instance, &device).expect("failed to create session");
    session.load_vmfb(vmfb).expect("failed to load VMFB");
    (instance, device, session)
}

#[test]
fn test_simple_mul() {
    let (_instance, _device, session) = setup_session(SIMPLE_MUL_VMFB);

    let input0: [f32; 4] = [1.0, 1.1, 1.2, 1.3];
    let input1: [f32; 4] = [10.0, 100.0, 1000.0, 10000.0];
    let expected: [f32; 4] = [10.0, 110.0, 1200.0, 13000.0];

    let buf0 = BufferView::from_bytes(&session, bytemuck(&input0), &[4], ElementType::Float32)
        .expect("failed to create buffer view 0");

    let buf1 = BufferView::from_bytes(&session, bytemuck(&input1), &[4], ElementType::Float32)
        .expect("failed to create buffer view 1");

    let mut call = session
        .call("module.simple_mul")
        .expect("failed to create call");
    call.push_input(&buf0).expect("failed to push input 0");
    call.push_input(&buf1).expect("failed to push input 1");
    call.invoke().expect("invoke failed");

    let output = call.pop_output().expect("failed to pop output");
    let output_bytes = output.to_bytes().expect("failed to read output");
    let result: &[f32] = bytecast(&output_bytes);

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
    let (_instance, _device, session) = setup_session(SIMPLE_ADD_F64_VMFB);

    let input0: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let input1: [f64; 4] = [0.1, 0.2, 0.3, 0.4];
    let expected: [f64; 4] = [1.1, 2.2, 3.3, 4.4];

    let buf0 = BufferView::from_bytes(&session, bytemuck(&input0), &[4], ElementType::Float64)
        .expect("failed to create f64 buffer view 0");

    let buf1 = BufferView::from_bytes(&session, bytemuck(&input1), &[4], ElementType::Float64)
        .expect("failed to create f64 buffer view 1");

    let mut call = session
        .call("module.simple_add_f64")
        .expect("failed to create call");
    call.push_input(&buf0).expect("failed to push input 0");
    call.push_input(&buf1).expect("failed to push input 1");
    call.invoke().expect("invoke failed");

    let output = call.pop_output().expect("failed to pop output");
    let output_bytes = output.to_bytes().expect("failed to read output");
    let result: &[f64] = bytecast(&output_bytes);

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
    let (_instance, _device, session) = setup_session(SIMPLE_MUL_VMFB);

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = BufferView::from_bytes(&session, bytemuck(&data), &[4], ElementType::Float32)
        .expect("failed to create buffer view");

    assert_eq!(buf.shape(), vec![4]);
    assert_eq!(buf.element_type(), Some(ElementType::Float32));
    assert_eq!(buf.element_count(), 4);
}

#[test]
fn test_error_on_bad_vmfb() {
    let instance = Instance::new().expect("failed to create instance");
    let device = instance
        .create_device("local-task")
        .expect("failed to create device");
    let session = Session::new(&instance, &device).expect("failed to create session");

    let result = session.load_vmfb(b"this is not a valid vmfb");
    assert!(result.is_err(), "expected error from invalid VMFB data");
}

#[test]
fn test_multiple_invocations() {
    let (_instance, _device, session) = setup_session(SIMPLE_MUL_VMFB);

    for multiplier in [1.0f32, 2.0, 3.0, 10.0] {
        let input0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input1: [f32; 4] = [multiplier; 4];

        let buf0 = BufferView::from_bytes(&session, bytemuck(&input0), &[4], ElementType::Float32)
            .unwrap();
        let buf1 = BufferView::from_bytes(&session, bytemuck(&input1), &[4], ElementType::Float32)
            .unwrap();

        let mut call = session.call("module.simple_mul").unwrap();
        call.push_input(&buf0).unwrap();
        call.push_input(&buf1).unwrap();
        call.invoke().unwrap();

        let output = call.pop_output().unwrap();
        let output_bytes = output.to_bytes().unwrap();
        let result: &[f32] = bytecast(&output_bytes);

        for (i, &val) in result.iter().enumerate() {
            let expected = input0[i] * multiplier;
            assert!(
                (val - expected).abs() < 1e-6,
                "multiplier={multiplier}, element {i}: got {val}, expected {expected}"
            );
        }
    }
}

fn bytemuck<T: Copy>(slice: &[T]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn bytecast<T: Copy>(bytes: &[u8]) -> &[T] {
    let ptr = bytes.as_ptr() as *const T;
    let len = bytes.len() / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
