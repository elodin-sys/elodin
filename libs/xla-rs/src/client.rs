use crate::{
    ArrayElement, Error, Literal, PjRtBuffer, PjRtLoadedExecutable, Result, Status, XlaComputation,
};
use cpp::{cpp, cpp_class};
use std::pin::Pin;

cpp! {{
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    #include "xla/pjrt/pjrt_stream_executor_client.h"
    #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
    #include "xla/pjrt/gpu/gpu_helpers.h"
    #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct PjRtClient as "std::shared_ptr<PjRtClient>");

impl PjRtClient {
    pub fn cpu() -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "Status*"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                auto status = xla::GetTfrtCpuClient(false);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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

    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "Status*", memory_fraction as "double", preallocate as "bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
                auto status = GetStreamExecutorGpuClient(false, allocator, 0, 0);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {}, device
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {}, device
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", literal as "const std::shared_ptr<Literal>*", out_status as "Status*"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostLiteral(*literal->get(), device);
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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

    pub fn compile(&self, comp: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let exec = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", comp as "const XlaComputation*", out_status as "Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                CompileOptions options;
                auto status = client->Compile(*comp, options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
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

    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtClient>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }
}
