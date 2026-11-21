mod common;

use common::{CHUNKED, f64_buf, run_mlir_workers};
use cranelift_mlir::parser::parse_module;

#[test]
fn independent_calls_form_one_group() {
    let mut module = parse_module(CHUNKED).unwrap();
    let groups = cranelift_mlir::parallel::group_module_calls(&mut module, |f| f.name == "main");
    assert_eq!(groups["main"].len(), 1);
    assert_eq!(groups["main"][0].len(), 6);
}

#[test]
fn workers_reproduce_the_serial_bits() {
    let input = f64_buf(&(0..128).map(|i| i as f64 * 0.37 - 20.0).collect::<Vec<_>>());
    let sizes = [128 * 8; 3];
    let serial = run_mlir_workers(CHUNKED, &[&input], &sizes, 1);
    for _ in 0..20 {
        assert_eq!(run_mlir_workers(CHUNKED, &[&input], &sizes, 4), serial);
    }
}
