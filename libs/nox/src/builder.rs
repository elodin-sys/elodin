use std::sync::atomic::AtomicI64;

pub struct Builder {
    pub(crate) inner: xla::XlaBuilder,
    pub(crate) param_count: AtomicI64,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            inner: xla::XlaBuilder::new(name),
            param_count: AtomicI64::default(),
        }
    }
}
