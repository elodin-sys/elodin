//! Provides functionality for managing a client that interfaces with the XLA (Accelerated Linear Algebra) library.
use std::ops::Deref;

use crate::Error;

/// Represents a high-level client for the XLA library, encapsulating compile options and client behaviors.
#[derive(Clone)]
pub struct Client {
    pjrt_client: xla::PjRtClient,
    compile_options: xla::CompileOptions,
}

impl Deref for Client {
    type Target = xla::PjRtClient;

    /// Returns a reference to the underlying PJRT client.
    fn deref(&self) -> &Self::Target {
        &self.pjrt_client
    }
}

impl Client {
    /// Creates a new `Client` with a provided PJRT client.
    fn new(pjrt_client: xla::PjRtClient) -> Self {
        Client {
            pjrt_client,
            compile_options: xla::CompileOptions::default(),
        }
    }

    /// Disables XLA optimizations.
    pub fn disable_optimizations(&mut self) {
        self.compile_options.disable_optimizations();
    }

    /// Compiles an XLA computation into a kernel using the client's compile options.
    pub fn compile(
        &self,
        comp: &xla::XlaComputation,
    ) -> Result<xla::PjRtLoadedExecutable, xla::Error> {
        self.pjrt_client
            .compile_with_options(comp, self.compile_options.clone())
    }

    /// Creates a new `Client` using the default CPU backend.
    pub fn cpu() -> Result<Self, Error> {
        xla::PjRtClient::cpu().map(Client::new).map_err(Error::from)
    }

    /// Creates a new [`Client`] using the GPU backend with default memory settings
    /// By default the backend is either CUDA or Metal depending on your OS.
    ///
    /// This function uses a default memory fraction of `0.95` and does not preallocate any memory.
    pub fn gpu() -> Result<Self, Error> {
        const DEFAULT_MEMORY_PERCENT: f64 = 0.95;
        xla::PjRtClient::gpu(DEFAULT_MEMORY_PERCENT, false)
            .map(Client::new)
            .map_err(Error::from)
    }

    /// Creates a new `Client` using the GPU backend with custom memory settings.
    /// By default the backend is either CUDA or Metal depending on your OS.
    ///
    /// # Parameters
    /// - `mem_limit`: Memory limit in the range [0..1.0].
    /// - `prealloc`: Whether to preallocate memory or not.
    pub fn gpu_with_memory_limit(mem_limit: f64, prealloc: bool) -> Result<Self, Error> {
        xla::PjRtClient::gpu(mem_limit, prealloc)
            .map(Client::new)
            .map_err(Error::from)
    }
}
