//! Micro-benches for rustc-compiled tensor_rt loops.
//!
//! ```text
//! cargo test -p cranelift-mlir --release --test bench_tensor_rt -- --ignored --nocapture
//! ```

use std::hint::black_box;
use std::time::Instant;

use cranelift_mlir::tensor_rt::{tensor_matmul_f64, tensor_reduce_sum_f64};

fn time_iters(name: &str, iters: u32, mut body: impl FnMut()) {
    for _ in 0..5 {
        body();
    }
    let start = Instant::now();
    for _ in 0..iters {
        body();
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / f64::from(iters);
    println!("{name}: {ns:.1} ns/iter ({iters} iters, {elapsed:?})");
}

#[test]
#[ignore]
fn bench_matmul_65() {
    const N: usize = 65;
    let a: Vec<f64> = (0..N * N).map(|i| (i % 17) as f64 * 0.1).collect();
    let b: Vec<f64> = (0..N * N).map(|i| (i % 13) as f64 * 0.2).collect();
    let mut dst = vec![0.0f64; N * N];
    time_iters("tensor_matmul_f64 65x65x65", 2_000, || {
        tensor_matmul_f64(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N, N, N);
        black_box(&dst);
    });
}

#[test]
#[ignore]
fn bench_reduce_sum_65x65() {
    const OUTER: usize = 65;
    const INNER: usize = 65;
    let src: Vec<f64> = (0..OUTER * INNER).map(|i| (i % 19) as f64 * 0.05).collect();
    let mut dst = vec![0.0f64; OUTER];
    time_iters("tensor_reduce_sum_f64 65x65", 50_000, || {
        tensor_reduce_sum_f64(dst.as_mut_ptr(), src.as_ptr(), OUTER, INNER);
        black_box(&dst);
    });
}

#[test]
#[ignore]
fn bench_reduce_sum_1x100k() {
    const OUTER: usize = 1;
    const INNER: usize = 100_000;
    let src: Vec<f64> = (0..INNER).map(|i| (i % 23) as f64 * 0.01).collect();
    let mut dst = vec![0.0f64; OUTER];
    time_iters("tensor_reduce_sum_f64 1x100k", 5_000, || {
        tensor_reduce_sum_f64(dst.as_mut_ptr(), src.as_ptr(), OUTER, INNER);
        black_box(&dst);
    });
}
