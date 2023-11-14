use xla::XlaOp;

use crate::AsOp;

pub trait Tensor: Sized + AsOp {
    fn from_op(op: XlaOp) -> Self;

    fn sqrt(&self) -> Self {
        Self::from_op(self.as_op().sqrt().unwrap())
    }
}
