pub fn calculate_strides(shape: &[usize]) -> impl Iterator<Item = usize> + '_ {
    shape.iter().rev().scan(1, |acc, &x| {
        let res = *acc;
        *acc *= x;
        Some(res)
    })
}
