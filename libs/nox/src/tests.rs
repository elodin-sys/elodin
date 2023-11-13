use nalgebra::matrix;
use std::ops::Add;

use super::*;

#[test]
fn test_add() {
    let client = Client::cpu().unwrap();
    let comp = Matrix::add.build().unwrap();
    let exec = comp.compile(&client).unwrap();
    let out = exec
        .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
        .unwrap();
    assert_eq!(out, matrix![3.0, 5.0]);
}

#[test]
fn test_map() {
    let client = Client::cpu().unwrap();
    fn add_one(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 4> {
        mat.map(|x: Scalar<f32>| x + 1f32).unwrap()
    }
    let comp = add_one.build().unwrap();
    let exec = match comp.compile(&client) {
        Ok(exec) => exec,
        Err(xla::Error::XlaError { msg, .. }) => {
            println!("{}", msg);
            panic!();
        }
        Err(e) => {
            panic!("{:?}", e);
        }
    };
    let out = exec.run(&client, matrix![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(out, matrix![2.0, 3.0, 4.0, 5.0])
}

#[test]
fn test_fixed_slice() {
    let client = Client::cpu().unwrap();
    fn slice(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 1> {
        mat.fixed_slice::<1, 1>(0, 2)
    }
    let comp = slice.build().unwrap();
    let exec = match comp.compile(&client) {
        Ok(exec) => exec,
        Err(xla::Error::XlaError { msg, .. }) => {
            println!("{}", msg);
            panic!();
        }
        Err(e) => {
            panic!("{:?}", e);
        }
    };
    let out = exec.run(&client, matrix![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(out, matrix![3.0])
}
