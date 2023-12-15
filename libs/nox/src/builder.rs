use crate::{Op, ScalarDim, Tensor};
use std::{cell::UnsafeCell, sync::atomic::AtomicI64};

pub struct Builder {
    pub(crate) inner: xla::XlaBuilder,
    pub(crate) param_count: AtomicI64,
    pub(crate) mut_params: boxcar::Vec<UnsafeCell<Tensor<f32, ScalarDim, Op>>>,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            inner: xla::XlaBuilder::new(name),
            param_count: AtomicI64::default(),
            mut_params: boxcar::Vec::new(),
        }
    }
}
