use crate::{
    ArrayElement, CompileOptions, Error, Literal, PjRtBuffer, PjRtLoadedExecutable, Result, Status,
    XlaComputation,
};
use cpp::{cpp, cpp_class};
use std::pin::Pin;

cpp! {{
   #pragma GCC diagnostic ignored "-Wignored-attributes"
   #include "xla/pjrt/pjrt_api.h"
   #include "xla/pjrt/pjrt_c_api_client.h"
   #include "xla/pjrt/pjrt_client.h"
   #include "xla/pjrt/pjrt_stream_executor_client.h"
   #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
   #include "xla/pjrt/gpu/gpu_helpers.h"
   #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
   #include "xla/service/custom_call_target_registry.h"
   #include "jaxlib/cpu/lapack_kernels.h"
   #include "xla/service/custom_call_target_registry.h"
   #include "tsl/platform/cpu_info.h"

   #ifdef EL_CUDA
   #include "xla/ffi/api/api.h"
   #include "xla/ffi/ffi_api.h"
   #include "jaxlib/gpu/blas_kernels.h"
   #include "jaxlib/gpu/cholesky_update_kernel.h"
   #include "jaxlib/gpu/lu_pivot_kernels.h"
   #include "jaxlib/gpu/prng_kernels.h"
   #include "jaxlib/gpu/rnn_kernels.h"
   #include "jaxlib/gpu/solver_kernels.h"
   #include "jaxlib/gpu/sparse_kernels.h"
   //#include "jaxlib/gpu/triton_kernels.h"
   //#include "jaxlib/gpu/vendor.h"
   #endif

   #pragma clang diagnostic ignored "-Wundefined-var-template"
   namespace ffi = xla::ffi;
}}

cpp_class!(pub unsafe struct PjRtClient as "std::shared_ptr<PjRtClient>");

impl PjRtClient {
    pub fn cpu() -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "absl::Status*"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                auto status = GetXlaPjrtCpuClient(CpuClientOptions());
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(client)
    }

    #[cfg(feature = "cuda")]
    #[allow(unused_variables)]
    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        init_cpu_lapack();
        let client = unsafe {
            cpp!([out_status as "__attribute__((unused)) absl::Status*", memory_fraction as "__attribute__((unused)) double", preallocate as "__attribute__((unused)) bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                #ifdef EL_CUDA
                auto reg = CustomCallTargetRegistry::Global();
                xla::ffi::Ffi::RegisterStaticHandler(
                    xla::ffi::GetXlaFfiApi(),
                    "cu_lu_pivots_to_permutation",
                    "CUDA",
                    reinterpret_cast<XLA_FFI_Handler*>(jax::cuda::LuPivotsToPermutation)
                );
                auto syms = reg->registered_symbols("CUDA");
                auto is_null = CustomCallTargetRegistry::Global()->Lookup("cu_lu_pivots_to_permutation", std::string("CUDA"));
                GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
                GpuClientOptions options = {
                    .allocator_config = allocator,
                    .platform_name = "CUDA"
                };
                auto status = GetStreamExecutorGpuClient(options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
                #else
                return std::shared_ptr<PjRtClient>();
                #endif
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(client)
    }

    pub fn copy_host_buffer<T: ArrayElement>(&self, buf: &[T], dims: &[i64]) -> Result<PjRtBuffer> {
        let element_count: usize = dims.iter().product::<i64>() as usize;
        if element_count != buf.len() {
            return Err(Error::WrongElementCount {
                dims: dims.to_vec(),
                element_count,
            });
        }
        let buf_ptr = buf.as_ptr();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = T::TY.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "absl::Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                    nullptr, memory_space, nullptr
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn copy_raw_host_buffer(
        &self,
        ty: super::ElementType,
        buf: &[u8],
        dims: &[i64],
    ) -> Result<PjRtBuffer> {
        let element_count: usize = dims.iter().product::<i64>() as usize;
        let element_size_in_bytes = ty.element_size_in_bytes();
        if element_count * element_size_in_bytes != buf.len() {
            Err(Error::WrongElementCount {
                dims: dims.to_vec(),
                element_count,
            })?
        }
        let buf_ptr = buf.as_ptr();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = ty.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "absl::Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                    nullptr, memory_space, nullptr
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn copy_literal(&self, literal: &Literal) -> Result<PjRtBuffer> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", literal as "const std::shared_ptr<Literal>*", out_status as "absl::Status*"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostLiteral(*literal->get(), memory_space);
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn compile_with_options(
        &self,
        comp: &XlaComputation,
        options: CompileOptions,
    ) -> Result<PjRtLoadedExecutable> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let mut options = options.0;
        let exec = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", mut options as "CompileOptions", comp as "const XlaComputation*", out_status as "absl::Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "", tsl::port::MaxParallelism());
                options.executable_build_options.set_compile_thread_pool(thread_pool.get());
                auto status = client->CompileAndLoad(*comp, options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
                    return std::shared_ptr<PjRtLoadedExecutable>();
                }
            })
        };
        out_status.to_result()?;
        if exec.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(exec)
    }

    pub fn compile_with_default_options(
        &self,
        comp: &XlaComputation,
    ) -> Result<PjRtLoadedExecutable> {
        self.compile_with_options(comp, Default::default())
    }

    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtClient>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn to_host_vec(&self, buffer: &PjRtBuffer) -> Result<Vec<u8>> {
        let mut dst = Vec::default();
        self.copy_into_host_vec(buffer, &mut dst)?;
        Ok(dst)
    }

    pub fn copy_into_host_vec(&self, buffer: &PjRtBuffer, dst: &mut Vec<u8>) -> Result<()> {
        // NOTE: The previous optimisation that used `UnsafeBufferPointer` for CPU buffers
        // started returning partially updated data with newer PJRT/XLA builds (see
        // https://github.com/openxla/xla/pull/11438). Always materialise the literal on the
        // host instead, which keeps the simulation deterministically in sync.
        buffer.copy_into(dst)
    }
}
