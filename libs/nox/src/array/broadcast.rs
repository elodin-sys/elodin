use smallvec::SmallVec;

use crate::Error;

pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<SmallVec<[usize; 4]>, Error> {
    let rank = left.len().max(right.len());
    let mut shape = SmallVec::with_capacity(rank);

    for axis in 0..rank {
        let left_axis = axis
            .checked_sub(rank - left.len())
            .map(|i| left[i])
            .unwrap_or(1);
        let right_axis = axis
            .checked_sub(rank - right.len())
            .map(|i| right[i])
            .unwrap_or(1);

        if left_axis == right_axis {
            shape.push(left_axis);
        } else if left_axis == 1 {
            shape.push(right_axis);
        } else if right_axis == 1 {
            shape.push(left_axis);
        } else {
            return Err(Error::BroadcastShapeMismatch {
                left: left.iter().copied().collect(),
                right: right.iter().copied().collect(),
            });
        }
    }

    Ok(shape)
}

pub fn can_broadcast(left: &[usize], right: &[usize]) -> bool {
    broadcast_shape(left, right).is_ok()
}

pub(crate) fn cobroadcast_dims(output: &mut [usize], other: &[usize]) -> bool {
    for (output, other) in output.iter_mut().rev().zip(other.iter().rev()) {
        if *output == *other || *other == 1 {
            continue;
        }
        if *output == 1 {
            *output = *other;
        } else {
            return false;
        }
    }
    true
}
