use super::*;

#[test]
fn test_new_builder() {
    XlaBuilder::new("test");
}

#[test]
fn test_cpu_client() {
    PjRtClient::cpu().expect("client create failed");
}

#[test]
fn test_compile() {
    let client = PjRtClient::cpu().expect("client create failed");
    let builder = XlaBuilder::new("test");
    let a = builder
        .parameter(0, crate::ElementType::F32, &[], "a")
        .unwrap();
    let b = builder
        .parameter(1, crate::ElementType::F32, &[], "b")
        .unwrap();
    let add = a.add(&b);
    let comp = builder.build(&add).unwrap();
    client.compile(&comp).unwrap();
}

#[test]
fn test_exec() {
    let client = PjRtClient::cpu().expect("client create failed");
    let builder = XlaBuilder::new("test");
    let a = builder
        .parameter(0, crate::ElementType::F32, &[], "a")
        .unwrap();
    let b = builder
        .parameter(1, crate::ElementType::F32, &[], "b")
        .unwrap();
    let add = a.add(&b);
    let comp = builder.build(&add).unwrap();
    let exec = client.compile(&comp).unwrap();
    let mut args = BufferArgsRef::default();
    let a = client.copy_host_buffer(&[1.0f32], &[]).unwrap();
    let b = client.copy_host_buffer(&[2.0f32], &[]).unwrap();
    args.push(&a);
    args.push(&b);
    let mut res = exec.execute_buffers(&args).unwrap();
    let out = res.pop().unwrap();
    let lit = out.to_literal_sync().unwrap();
    assert_eq!(lit.typed_buf::<f32>().unwrap(), &[3.0f32]);
}

#[test]
fn add_op() -> Result<()> {
    let client = crate::PjRtClient::cpu()?;
    let builder = crate::XlaBuilder::new("test");
    let cst42 = builder.constant(42f32);
    let cst43 = builder.constant_vector(&[43f32; 2]);
    let sum = cst42 + cst43;
    let computation = sum.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute_buffers(&BufferArgsRef::default())?;
    let result = result[0].to_literal_sync()?;
    assert_eq!(result.element_count(), 2);
    assert_eq!(
        result.shape()?,
        crate::Shape::Array(crate::ArrayShape::new::<f32>(vec![2]))
    );
    assert_eq!(result.typed_buf::<f32>()?, [85., 85.]);
    Ok(())
}

#[test]
fn tuple_op() -> Result<()> {
    let client = crate::PjRtClient::cpu()?;
    let builder = crate::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1], "x")?;
    let y = builder.parameter(1, f32::TY, &[2], "x")?;
    let tuple = builder.tuple(&[x.as_ref(), y.as_ref()]).build()?;
    let tuple = client.compile(&tuple)?;
    let x = crate::Literal::scalar(3.1f32);
    let y = crate::Literal::vector(&[4.2f32, 1.337f32]);
    let x = client.copy_literal(&x)?;
    let y = client.copy_literal(&y)?;
    let result = tuple.execute_buffers(BufferArgsRef::from([&x, &y]))?;
    let result = result[0].to_literal_sync()?;
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let mut result = result;
    let result = result.decompose_tuple()?;
    assert_eq!(result[1].typed_buf::<f32>()?, &[4.2, 1.337]);
    assert_eq!(result[0].typed_buf::<f32>()?, &[3.1]);
    Ok(())
}

// #[test]
// fn tuple_literal() -> Result<()> {
//     let x = crate::Literal::scalar(3.1f32);
//     let y = crate::Literal::vector(&[4.2f32, 1.337f32]);
//     let result = crate::Literal::tuple(vec![x, y]);
//     assert_eq!(result.shape()?.tuple_size(), Some(2));
//     let mut result = result;
//     let result = result.decompose_tuple()?;
//     assert_eq!(result[1].to_vec::<f32>()?, [4.2, 1.337]);
//     assert_eq!(result[0].to_vec::<f32>()?, [3.1]);
//     Ok(())
// }
