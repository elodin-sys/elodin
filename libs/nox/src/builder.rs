use crate::{Noxpr, Op, ScalarDim, Tensor};
use std::cell::{RefCell, UnsafeCell};

pub struct Builder {
    pub(crate) params: RefCell<Vec<Noxpr>>,
    pub(crate) mut_params: boxcar::Vec<UnsafeCell<Tensor<f32, ScalarDim, Op>>>,
    pub(crate) aliased_indexes: Vec<(u64, u64)>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            params: RefCell::new(vec![]),
            mut_params: boxcar::Vec::new(),
            aliased_indexes: vec![],
        }
    }

    pub fn setup_alias(&mut self, param_index: u64, alias_index: u64) {
        self.aliased_indexes.push((param_index, alias_index));
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
