use std::ops::Deref;

use crate::Error;

#[derive(Clone)]
pub struct Client {
    pjrt_client: xla::PjRtClient,
    compile_options: xla::CompileOptions,
}

impl Deref for Client {
    type Target = xla::PjRtClient;

    fn deref(&self) -> &Self::Target {
        &self.pjrt_client
    }
}

impl Client {
    fn new(pjrt_client: xla::PjRtClient) -> Self {
        Client {
            pjrt_client,
            compile_options: xla::CompileOptions::default(),
        }
    }

    pub fn disable_optimizations(&mut self) {
        self.compile_options.disable_optimizations();
    }

    pub fn compile(
        &self,
        comp: &xla::XlaComputation,
    ) -> Result<xla::PjRtLoadedExecutable, xla::Error> {
        self.pjrt_client
            .compile_with_options(comp, self.compile_options.clone())
    }

    /// Create a new [`Client`] using the CPU based backend.
    pub fn cpu() -> Result<Self, Error> {
        xla::PjRtClient::cpu().map(Client::new).map_err(Error::from)
    }

    /// Create a new [`Client`] using a GPU based backend.
    /// By default the backend is either CUDA or Metal depending on your OS.
    ///
    /// This functions uses the default memory fraction of `0.95`,
    /// and does not preallocate any memory
    pub fn gpu() -> Result<Self, Error> {
        const DEFAULT_MEMORY_PERCENT: f64 = 0.95;
        xla::PjRtClient::gpu(DEFAULT_MEMORY_PERCENT, false)
            .map(Client::new)
            .map_err(Error::from)
    }

    /// Create a new [`Client`] using a GPU based backend, with the specified memory percent.
    /// By default the backend is either CUDA or Metal depending on your OS.
    ///
    /// This function allows you to customize the memory limit, and preallocation behavior of the backend.
    /// The first argument is the memory limit in the range [0..1.0].
    /// The second paremeter is whether to preallocate memory or not.
    pub fn gpu_with_memory_limit(mem_limit: f64, prealloc: bool) -> Result<Self, Error> {
        xla::PjRtClient::gpu(mem_limit, prealloc)
            .map(Client::new)
            .map_err(Error::from)
    }
}
