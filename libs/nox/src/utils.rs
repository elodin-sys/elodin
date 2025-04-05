pub fn calculate_strides(shape: &[usize]) -> smallvec::SmallVec<[usize; 4]> {
    let mut strides: smallvec::SmallVec<[usize; 4]> = shape
        .iter()
        .rev()
        .scan(1, |acc, &x| {
            let res = *acc;
            *acc *= x;
            Some(res)
        })
        .collect();
    strides.reverse();
    strides
}
