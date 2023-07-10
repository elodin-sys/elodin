use super::*;
use approx::assert_relative_eq;

#[test]
fn test_const_rk4() {
    let mut int = 0.0;
    for _ in 0..10 {
        int += rk4_step(|_| 1., 0., 0.1);
    }
    assert_relative_eq!(int, 1.0);
}
