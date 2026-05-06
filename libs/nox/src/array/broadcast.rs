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
                left: left.to_vec(),
                right: right.to_vec(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_shape_handles_scalar_vector_matrix_and_higher_rank() {
        assert_eq!(broadcast_shape(&[], &[3]).unwrap().as_slice(), &[3]);
        assert_eq!(broadcast_shape(&[3], &[]).unwrap().as_slice(), &[3]);
        assert_eq!(broadcast_shape(&[3], &[2, 3]).unwrap().as_slice(), &[2, 3]);
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap().as_slice(), &[2, 3]);
        assert_eq!(
            broadcast_shape(&[2, 3], &[2, 1]).unwrap().as_slice(),
            &[2, 3]
        );
        assert_eq!(
            broadcast_shape(&[1, 3], &[2, 1, 3]).unwrap().as_slice(),
            &[2, 1, 3]
        );
    }

    #[test]
    fn broadcast_shape_rejects_incompatible_shapes() {
        let err = broadcast_shape(&[2, 3], &[3, 2]).unwrap_err();

        assert!(matches!(
            err,
            crate::Error::BroadcastShapeMismatch { left, right }
                if left == [2, 3] && right == [3, 2]
        ));
    }
}
